import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.args = args
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, x in enumerate(tqdm(train_loader, desc="Training")):
            x = x.to(self.args.device)
            
            # Forward pass
            logits, z_indices_masked = self.model(x)
            
            # Get original z_indices (ground truth)
            with torch.no_grad():
                z_indices_gt, _ = self.model.encode_to_z(x)
                # Apply same preprocessing as in forward pass
                original_batch_size = x.shape[0]
                if len(z_indices_gt.shape) > 2:
                    z_indices_gt = z_indices_gt.flatten(1)
                elif len(z_indices_gt.shape) == 1:
                    # If 1D, reshape to proper batch size
                    z_indices_gt = z_indices_gt.view(original_batch_size, -1)
                z_indices_gt = z_indices_gt.long()
                if len(z_indices_gt.shape) > 1 and z_indices_gt.shape[1] != self.model.num_image_tokens:
                    z_indices_gt = z_indices_gt[:, :self.model.num_image_tokens]
                z_indices_gt = torch.clamp(z_indices_gt, 0, self.model.mask_token_id - 1)
            
            # Calculate cross entropy loss only on masked positions
            mask = (z_indices_masked == self.model.mask_token_id)
            if mask.sum() > 0:
                loss = F.cross_entropy(
                    logits[mask].view(-1, logits.size(-1)), 
                    z_indices_gt[mask].view(-1)
                )
            else:
                loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Handle remaining gradients
        if num_batches % self.args.accum_grad != 0:
            self.optim.step()
            self.optim.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x in tqdm(val_loader, desc="Validation"):
                x = x.to(self.args.device)
                
                # Forward pass
                logits, z_indices_masked = self.model(x)
                
                # Get original z_indices (ground truth)
                z_indices_gt, _ = self.model.encode_to_z(x)
                # Apply same preprocessing as in forward pass
                original_batch_size = x.shape[0]
                if len(z_indices_gt.shape) > 2:
                    z_indices_gt = z_indices_gt.flatten(1)
                elif len(z_indices_gt.shape) == 1:
                    # If 1D, reshape to proper batch size
                    z_indices_gt = z_indices_gt.view(original_batch_size, -1)
                z_indices_gt = z_indices_gt.long()
                if len(z_indices_gt.shape) > 1 and z_indices_gt.shape[1] != self.model.num_image_tokens:
                    z_indices_gt = z_indices_gt[:, :self.model.num_image_tokens]
                z_indices_gt = torch.clamp(z_indices_gt, 0, self.model.mask_token_id - 1)
                
                # Calculate cross entropy loss only on masked positions
                mask = (z_indices_masked == self.model.mask_token_id)
                if mask.sum() > 0:
                    loss = F.cross_entropy(
                        logits[mask].view(-1, logits.size(-1)), 
                        z_indices_gt[mask].view(-1)
                    )
                else:
                    loss = torch.tensor(0.0, device=self.args.device)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def configure_optimizers(self):
        lr = self.args.learning_rate if hasattr(self.args, 'learning_rate') else 1e-4
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        return optimizer, scheduler
    
    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        
        # Load model state
        self.model.transformer.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Return epoch info
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded. Starting from epoch {start_epoch + 1}")
        return start_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)
    
    # Load checkpoint if specified
    start_epoch = args.start_from_epoch
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_epoch = train_transformer.load_checkpoint(args.checkpoint_path)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(start_epoch+1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training phase
        train_loss = train_transformer.train_one_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation phase
        val_loss = train_transformer.eval_one_epoch(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Update learning rate scheduler
        train_transformer.scheduler.step()
        
        # Save checkpoint
        if epoch % args.save_per_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': train_transformer.model.transformer.state_dict(),
                'optimizer_state_dict': train_transformer.optim.state_dict(),
                'scheduler_state_dict': train_transformer.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f"transformer_checkpoints/epoch_{epoch}.pt")
            print(f"Checkpoint saved at epoch {epoch}")
        
        # Save latest checkpoint
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': train_transformer.model.transformer.state_dict(),
            'optimizer_state_dict': train_transformer.optim.state_dict(),
            'scheduler_state_dict': train_transformer.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(latest_checkpoint, "transformer_checkpoints/latest.pt")
    
    print("Training completed!")