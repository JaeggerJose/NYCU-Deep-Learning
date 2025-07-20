import torch
from torch.utils.data import DataLoader
from utils import dice_score

def evaluate(net, data, device):
    net.eval()  # set to evaluation mode
    score = 0
    pic_num = 0
    dice_scores = []  # record the dice score of each image
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(data):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            outputs = net(image)
            # 將logits轉換為 probability
            outputs = torch.sigmoid(outputs)
            
            for i in range(outputs.shape[0]):
                dice = dice_score(outputs[i], mask[i])
                dice_scores.append(dice)
                score += dice
            pic_num += outputs.shape[0]
    
    avg_score = score / pic_num
    return avg_score, dice_scores
        