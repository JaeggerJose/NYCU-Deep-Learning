import torch
import numpy as np

def dice_score(pred_mask, gt_mask):
    with torch.no_grad():
        pred_binary = (pred_mask > 0.5).float()
        
        pred_np = pred_binary.cpu().numpy().flatten()
        gt_np = gt_mask.cpu().numpy().flatten()
        
        # 計算交集 (intersection)
        intersection = np.sum(pred_np * gt_np)
        
        # 計算各自的面積
        pred_area = np.sum(pred_np)
        gt_area = np.sum(gt_np)
        
        # 計算Dice係數
        # Dice = (2 * intersection) / (pred_area + gt_area)
        # 加入小的epsilon避免除零錯誤
        epsilon = 1e-7
        dice = (2.0 * intersection + epsilon) / (pred_area + gt_area + epsilon)
        
        return dice
