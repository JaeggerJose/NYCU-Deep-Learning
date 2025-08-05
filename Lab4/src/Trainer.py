import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
import json

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) + 1e-8  # Add small value to avoid log(0)
    # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    logvar = logvar.clamp(min=-10, max=10)  # Clamp logvar to prevent numerical issues
    # KLD = -0.5 * torch.sum(1 + logvar - mu
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size  
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        self.beta = 0.0
        
    def update(self):
        # TODO
        self.current_epoch += 1
        if self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(start=0.0, stop=1.0)
        elif self.kl_anneal_type == 'Linear':
            self.beta = min(1.0, self.beta + self.kl_anneal_ratio * self.current_epoch / self.kl_anneal_cycle)
        elif self.kl_anneal_type == 'None':
            self.beta = 1.0
        else:
            print(f"Unknown kl_anneal_type: {self.kl_anneal_type}")
            raise NotImplementedError
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, start=0.0, stop=1.0):
        # TODO
        cycle_progress = (self.current_epoch % self.kl_anneal_cycle) / self.kl_anneal_cycle
        # Apply ratio to determine how much of the cycle to use for annealing
        if cycle_progress <= self.kl_anneal_ratio:
            # Scale progress to full range within the annealing portion
            scaled_progress = cycle_progress / self.kl_anneal_ratio
            return start + (stop - start) * scaled_progress
        else:
            # Stay at max value for remaining portion of cycle
            return stop

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[15, 30, 60, 90, 120], gamma=0.5)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.writer = SummaryWriter(log_dir=self.args.save_root)
        
        # History tracking for plotting curves
        self.train_loss_history = []
        self.val_loss_history = []
        self.tfr_history = []
        self.beta_history = []
        self.psnr_per_frame_history = []  # Store PSNR for each frame in validation
        
        
    def forward(self, img, label, mode='train'):
        # VAE forward pass
        img_feature = self.frame_transformation(img)
        label_feature = self.label_transformation(label)
        z, mu, logvar = self.Gaussian_Predictor(img_feature, label_feature)
        if mode == 'test':
            z = torch.randn_like(z)
        parm = self.Decoder_Fusion(img_feature, label_feature, z)
        output = self.Generator(parm)
        # Clamp output to valid range to prevent numerical issues
        output = torch.clamp(output, 0.0, 1.0)
        return output, mu, logvar
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            epoch_train_losses = []
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=160)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                adapt_TeacherForcing = True if random.random() < self.tfr else False
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_train_losses.append(loss.detach().cpu().item())
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            # Store epoch metrics
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            self.train_loss_history.append(avg_train_loss)
            self.tfr_history.append(self.tfr)
            self.beta_history.append(self.kl_annealing.get_beta())
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            val_loss = self.eval()
            self.val_loss_history.append(val_loss)
            
            # Save training history
            self.save_training_history()
            
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
        # Auto-generate plots after training completion
        print(f"\n🎉 Training completed after {self.args.num_epoch} epochs!")
        self.auto_generate_plots()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        epoch_val_losses = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            epoch_val_losses.append(loss.detach().cpu().item())
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else 0
        return avg_val_loss
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        #  sequential training step
        self.optim.zero_grad()
        seq_len = img.size(1)
        generated_frames = []
        all_loss = 0
        for i in range(1, seq_len):
            if i == 1:
                prev_img = img[:, 0] # at the first step, use the first frame as previous frame
            else:
                if adapt_TeacherForcing:
                    prev_img = img[:, i - 1] # use the true previous frame  
                else:
                    prev_img = generated_frames[-1].detach()

            current_label = label[:, i]
            target_frame = img[:, i]

            # generate the current frame
            output, mu, logvar = self.forward(prev_img, current_label, mode='train')
            generated_frames.append(output)
            # calculate the loss
            mse_loss = self.mse_criterion(output, target_frame)
            kl_loss = kl_criterion(mu, logvar, img.size(0))
            frame_loss = mse_loss + self.kl_annealing.get_beta() * kl_loss
            
            # Check for NaN in training loss
            if torch.isnan(frame_loss) or torch.isinf(frame_loss):
                print(f"Warning: NaN or Inf in training loss at epoch {self.current_epoch}, frame {i}")
                return torch.tensor(0.0, device=img.device, requires_grad=True)
                
        all_loss += frame_loss

        # If the loop completed without NaN
        if all_loss > 0:
            # Average the loss over the sequence length
            avg_loss = all_loss / (seq_len - 1)
            avg_loss.backward()
            self.optimizer_step()
            return avg_loss
        else:
            # This case might happen if seq_len is 1 or a NaN was hit on the first frame
            return torch.tensor(0.0, device=img.device)
    def val_one_step(self, img, label):
        seq_len = img.size(1)
        generated_frames = []
        all_loss = 0
        frame_psnrs = []  # Store PSNR for each frame

        for i in range(1, seq_len):
            if i == 1:
                prev_img = img[:, 0]  # 第一幀使用真實幀
            else:
                prev_img = generated_frames[-1]  # 總是使用生成的前一幀

            current_label = label[:, i]
            target_frame = img[:, i]

            # 生成當前幀
            output, mu, logvar = self.forward(prev_img, current_label, mode='val')
            generated_frames.append(output)

            # 計算損失（不需要backward）
            mse_loss = self.mse_criterion(output, target_frame)
            kl_loss = kl_criterion(mu, logvar, img.size(0))
            frame_loss = mse_loss + self.kl_annealing.get_beta() * kl_loss
            
            # Check for NaN in validation loss
            if torch.isnan(frame_loss) or torch.isinf(frame_loss):
                print(f"Warning: NaN or Inf in validation loss at epoch {self.current_epoch}, frame {i}")
                continue
                
            all_loss += frame_loss
            
            # Calculate and store PSNR for this frame
            psnr_value = Generate_PSNR(output, target_frame)
            frame_psnrs.append(psnr_value.item())
            self.writer.add_scalar("PSNR/"+str(self.current_epoch), psnr_value, i)
            
        # Store PSNR per frame for this epoch
        if frame_psnrs:
            self.psnr_per_frame_history.append(frame_psnrs)
        
        return all_loss / (seq_len - 1)  # 返回平均損失
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # 從第 tfr_sde 個epoch開始減少teacher forcing ratio
        if self.current_epoch >= self.tfr_sde:
            # Decay teacher forcing ratio
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)
            print(f"Update teacher forcing ratio to {self.tfr}")
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[30, 50, 65], gamma=0.5)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        # Check for NaN or Inf gradients
        has_nan_grad = False
        for param in self.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Warning: NaN or Inf gradients detected at epoch {self.current_epoch}, skipping this step")
                    has_nan_grad = True
                    break
        
        if not has_nan_grad:
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optim.step()

    def save_training_history(self):
        """Save training history to JSON file for plotting curves"""
        history = {
            'epoch': list(range(len(self.train_loss_history))),
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'teacher_forcing_ratio': self.tfr_history,
            'kl_beta': self.beta_history,
            'psnr_per_frame': self.psnr_per_frame_history
        }
        
        history_path = os.path.join(self.args.save_root, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        return history
    
    def plot_learning_loss_curve(self, history, save_path):
        """Plot learning loss curve showing train and validation loss over epochs"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = history['epoch']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Loss over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning loss curve saved to: {save_path}")

    def plot_teacher_forcing_curve(self, history, save_path):
        """Plot teacher forcing ratio curve over epochs"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = history['epoch']
        tfr = history['teacher_forcing_ratio']
        
        ax.plot(epochs, tfr, 'g-', label='Teacher Forcing Ratio', linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Teacher Forcing Ratio', fontsize=12)
        ax.set_title('Teacher Forcing Ratio over Epochs', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Teacher forcing curve saved to: {save_path}")

    def plot_combined_curves(self, history, save_path):
        """Plot combined view of loss and teacher forcing ratio"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = history['epoch']
        
        # Loss subplot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Learning Loss and Teacher Forcing Ratio over Epochs', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Teacher forcing subplot
        ax2.plot(epochs, history['teacher_forcing_ratio'], 'g-', label='Teacher Forcing Ratio', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Teacher Forcing Ratio', fontsize=12)
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined curves saved to: {save_path}")

    def plot_kl_beta_curve(self, history, save_path):
        """Plot KL beta annealing curve over epochs"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = history['epoch']
        kl_beta = history['kl_beta']
        
        ax.plot(epochs, kl_beta, 'purple', label='KL Beta', linewidth=2, marker='o', markersize=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('KL Beta Value', fontsize=12)
        ax.set_title('KL Beta Annealing over Epochs', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"KL beta curve saved to: {save_path}")

    def plot_psnr_per_frame(self, history, save_path):
        """Plot PSNR-per-frame diagram for validation dataset"""
        if not history['psnr_per_frame'] or len(history['psnr_per_frame']) == 0:
            print("No PSNR per frame data available for plotting")
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get data for multiple epochs (e.g., early, middle, late training)
        num_epochs = len(history['psnr_per_frame'])
        epochs_to_plot = []
        
        if num_epochs >= 3:
            # Plot early, middle, and late epochs
            epochs_to_plot = [0, num_epochs//2, num_epochs-1]
            epoch_labels = ['Early Training', 'Mid Training', 'Late Training']
        elif num_epochs == 2:
            epochs_to_plot = [0, num_epochs-1]
            epoch_labels = ['Early Training', 'Late Training']
        else:
            epochs_to_plot = [0]
            epoch_labels = ['Training']
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, epoch_idx in enumerate(epochs_to_plot):
            if epoch_idx < len(history['psnr_per_frame']):
                psnr_values = history['psnr_per_frame'][epoch_idx]
                frame_numbers = list(range(1, len(psnr_values) + 1))
                
                ax.plot(frame_numbers, psnr_values, 
                       color=colors[i % len(colors)], 
                       label=f'{epoch_labels[i]} (Epoch {epoch_idx})',
                       linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR per Frame in Validation Dataset', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PSNR per frame diagram saved to: {save_path}")

    def auto_generate_plots(self):
        """Automatically generate all training curves after training completion"""
        print("\n" + "="*50)
        print("🎨 Auto-generating training curves...")
        print("="*50)
        
        # Save and get history
        history = self.save_training_history()
        
        # Create plots directory
        plots_dir = os.path.join(self.args.save_root, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate all plots
        self.plot_learning_loss_curve(history, os.path.join(plots_dir, 'learning_loss_curve.png'))
        self.plot_teacher_forcing_curve(history, os.path.join(plots_dir, 'teacher_forcing_curve.png'))
        self.plot_kl_beta_curve(history, os.path.join(plots_dir, 'kl_beta_curve.png'))
        self.plot_psnr_per_frame(history, os.path.join(plots_dir, 'psnr_per_frame.png'))
        self.plot_combined_curves(history, os.path.join(plots_dir, 'combined_curves.png'))
        
        print(f"\n✅ All training curves saved to: {plots_dir}/")
        print("📊 Generated plots:")
        print("  - learning_loss_curve.png")
        print("  - teacher_forcing_curve.png") 
        print("  - kl_beta_curve.png")
        print("  - psnr_per_frame.png")
        print("  - combined_curves.png")
        print("="*50)



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.0005,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=60,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=50,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1, help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='None',       help="Type of KL annealing, 'Cyclical' or 'Linear'", choices=['Cyclical', 'Linear', 'None'])
    parser.add_argument('--kl_anneal_cycle',    type=int, default=20,               help="Number of cycles for KL annealing")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,            help="Final KL weight ratio")

    # 固定seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parser.parse_args()
    
    main(args)
