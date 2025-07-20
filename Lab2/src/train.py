import argparse
from oxford_pet import load_dataset
from models.unet import MyUNet
from models.resnet34_unet import MyResNet34UNet
from evaluate import evaluate
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_training_curves(train_losses, valid_losses, dice_scores, args=None, model_name=None):
    save_path = '../results'
    os.makedirs(save_path, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2.plot(epochs, dice_scores, 'g-', label='Dice Score', linewidth=2)
    ax2.set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    best_dice_epoch = np.argmax(dice_scores) + 1
    best_dice_score = np.max(dice_scores)
    ax2.axhline(y=best_dice_score, color='orange', linestyle='--', alpha=0.7, label=f'Best: {best_dice_score:.4f}')
    ax2.axvline(x=best_dice_epoch, color='orange', linestyle='--', alpha=0.7)
    ax2.text(best_dice_epoch, best_dice_score + 0.02, f'Epoch {best_dice_epoch}', 
             ha='center', va='bottom', fontsize=10, color='orange')
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_path, f'training_curves_lr{args.learning_rate}_batch{args.batch_size}_model{model_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")

def train(args):
    # 如果沒有給 seed，就使用隨機 seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(1, 1000000)
    
    set_seed(seed)
    
    dataset = load_dataset(args.data_path, mode='train')
    dataset_valid = load_dataset(args.data_path, mode='valid')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        worker_init_fn=lambda worker_id: set_seed(seed + worker_id)
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    if args.model == 'unet':
        model = MyUNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet34_unet':
        model = MyResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_valid_loss = float('inf')
    best_dice_score = 0.0
    
    train_losses = []
    valid_losses = []
    dice_scores = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader_valid, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                valid_loss += loss.item() * images.size(0)

        epoch_valid_loss = valid_loss / len(dataloader_valid.dataset)
        valid_losses.append(epoch_valid_loss)
        
        avg_dice_score, dice_scores_list = evaluate(model, dataloader_valid, device)
        dice_scores.append(avg_dice_score)
        
        print(f'Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}, Dice Score: {avg_dice_score:.4f}')

        scheduler.step(epoch_valid_loss)
        
        if avg_dice_score > best_dice_score:
            best_dice_score = avg_dice_score
            # save model path 
            if os.path.exists('../saved_models'):
                model_path = os.path.join('../saved_models', f'{args.model}_best.pth')
            else:
                os.makedirs('../saved_models', exist_ok=True)
                model_path = os.path.join('../saved_models', f'{args.model}_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved (dice-based) with dice score: {best_dice_score:.4f}, Valid Loss: {epoch_valid_loss:.4f}')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Best Validation Loss: {best_valid_loss:.4f}")
    print(f"Best Dice Score: {best_dice_score:.4f}")
    print(f"Training seed: {seed}")
    print("="*50)
    
    plot_training_curves(train_losses, valid_losses, dice_scores, args, model_name=args.model)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model', '-m', type=str, default='unet', help='model to use', choices=['unet', 'resnet34_unet'])
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
