import argparse
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from models.unet import MyUNet
from models.resnet34_unet import MyResNet34UNet
from evaluate import evaluate
from oxford_pet import load_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_results_full(original_image, pred_mask, pred_prob, output_dir, image_name, gt_mask):
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存預測遮罩
    mask_image = (pred_mask * 255).astype(np.uint8)
    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    Image.fromarray(mask_image).save(mask_path, optimize=True)
    
    # 保存概率圖
    prob_image = (pred_prob * 255).astype(np.uint8)
    prob_path = os.path.join(output_dir, f"{image_name}_prob.png")
    Image.fromarray(prob_image).save(prob_path, optimize=True)
    
    # create visualization image (4 in 1: original image, ground truth mask, predicted mask, probability map)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 真實遮罩
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # 預測遮罩
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # 概率圖
    axes[3].imshow(pred_prob, cmap='hot')
    axes[3].set_title('Prediction Probability')
    axes[3].axis('off')
    
    # 保存可視化結果
    vis_path = os.path.join(output_dir, f"{image_name}_visualization.png")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()

def test_model(model, data_path, device, output_dir, batch_size=8, seed=None):
    # 使用跟 inference.py 一樣的方式載入測試資料集
    test_dataset = load_dataset(data_path, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 直接讀取 test.txt 檔案來獲取檔案名稱列表
    csv_path = data_path + '/annotations/test.txt'
    with open(csv_path) as f:
        split_data = f.read().strip("\n").split("\n")
    filenames = [x.split(" ")[0] for x in split_data]
    
    total_samples = len(filenames)
    print(f'total samples: {total_samples}')
    
    
    # calculate the average dice score of the model on the test dataset
    avg_dice_score, dice_scores = evaluate(model, test_loader, device)
    
    print(f"   平均 Dice Score: {avg_dice_score:.4f}")
    print(f"   最小 Dice Score: {min(dice_scores):.4f}")
    print(f"   最大 Dice Score: {max(dice_scores):.4f}")
    print(f"   標準差: {np.std(dice_scores):.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=total_samples, desc="saving results", unit="units", ncols=100, colour='blue')

        for file_idx, filename in enumerate(filenames):
            # 載入原始圖像和遮罩
            image_path = os.path.join(data_path, 'images', filename + '.jpg')
            mask_path = os.path.join(data_path, 'annotations', 'trimaps', filename + '.png')
            
            # 載入和預處理圖像
            original_image = np.array(Image.open(image_path).convert("RGB"))
            original_image_resized = np.array(Image.fromarray(original_image).resize((256, 256), Image.BILINEAR))
            original_image_normalized = (np.moveaxis(original_image_resized, -1, 0) / 255.0).astype(np.float32)
            
            # 載入真實遮罩
            trimap = np.array(Image.open(mask_path))
            gt_mask = trimap.astype(np.float32)
            gt_mask[gt_mask == 2.0] = 0.0
            gt_mask[(gt_mask == 1.0) | (gt_mask == 3.0)] = 1.0
            gt_mask = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST))
            
            # 模型預測
            input_tensor = torch.tensor(original_image_normalized).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            pred_probs = torch.sigmoid(outputs)
            
            # 後處理
            pred_prob = pred_probs[0].squeeze().cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.float32)
            
            # 轉換圖像格式用於保存
            original_image_for_save = np.moveaxis(original_image_normalized, 0, -1)  # CHW -> HWC
            
            dice = dice_scores[file_idx]
            
            sample_name = f"test_{file_idx:04d}_dice_{dice:.3f}"
            save_results_full(original_image_for_save, pred_mask, pred_prob, output_dir, sample_name, gt_mask)
            
            # 更新進度條
            pbar.set_postfix({
                'Dice': f'{dice:.3f}',
                'Avg': f'{avg_dice_score:.3f}',
            })
            pbar.update(1)
        
        pbar.close()
    
    # 保存結果統計
    stats_path = os.path.join(output_dir, 'test_results.txt')
    with open(stats_path, 'w') as f:
        f.write("測試集推理結果統計\n")
        f.write("="*40 + "\n")
        f.write(f"總圖像數: {total_samples}\n")
        f.write(f"平均 Dice Score: {avg_dice_score:.4f}\n")
        f.write(f"最小 Dice Score: {min(dice_scores):.4f}\n")
        f.write(f"最大 Dice Score: {max(dice_scores):.4f}\n")
        f.write(f"標準差: {np.std(dice_scores):.4f}\n\n")
        
        f.write("每張圖像的詳細結果:\n")
        f.write("-" * 40 + "\n")
        for i, dice in enumerate(dice_scores):
            f.write(f"test_{i:04d}_dice_{dice:.3f}: {dice:.4f}\n")
    
    print(f"Result save at: {stats_path}")
    
    # 顯示推理完成和使用的 seed
    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Average Dice Score: {avg_dice_score:.4f}")
    print(f"Min Dice Score: {min(dice_scores):.4f}")
    print(f"Max Dice Score: {max(dice_scores):.4f}")
    print(f"Standard Deviation: {np.std(dice_scores):.4f}")
    print(f"Inference seed: {seed}")
    print("="*50)

def get_args():
    parser = argparse.ArgumentParser(description='Test UNet model on Oxford Pet test dataset')
    parser.add_argument('--model', '-m', type=str, default='unet', 
                       help='model to use', choices=['unet', 'resnet34_unet'])
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet', 
                       help='path to Oxford Pet dataset')
    parser.add_argument('--output_dir', type=str, default='../predictions', 
                       help='output directory for test results')
    parser.add_argument('--batch_size', '-b', type=int, default=8, 
                       help='batch size for testing')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # 如果沒有給 seed，就使用隨機 seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(1, 1000000)
    
    set_seed(seed)

    model_path = os.path.join('../saved_models', f'{args.model}_best.pth')
    if not os.path.exists(model_path):
        print(f" Error: Model file {model_path} does not exist!")
        exit(1)
    if args.model == 'unet':
        model = MyUNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet34_unet':
        model = MyResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f" Error loading model: {e}")
        exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    test_model(model, args.data_path, device, args.output_dir, args.batch_size, seed)