#!/usr/bin/env python
# coding: utf-8

"""
Test code with ZSC integration for bounding box generation
"""

import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim
# Import ZSC functions and dependencies
from ZSC import generate_class_prototype, select_relevant_patches, get_boxes
from config import cfg
from models import build_model
from prompt import fn_prompt

parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation with ZSC")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int, default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float, default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float, default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
parser.add_argument("--prompt-method", type=str, default='BLIP-LLM', help="Prompt generation method for ZSC")
parser.add_argument("--cfg", type=str, default="./config/test.yaml", help="Path to ZSC config file")
args = parser.parse_args()

# Config and device setup
cfg.merge_from_file(args.cfg)
cfg.DIR.output_dir = os.path.join(cfg.DIR.snapshot, cfg.DIR.exp)
os.makedirs(cfg.DIR.output_dir, exist_ok=True)
cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)
cfg.VAL.resume = os.path.join(cfg.DIR.output_dir, cfg.VAL.resume)

device = torch.device('cuda:0')
torch.manual_seed(cfg.TRAIN.seed)
np.random.seed(cfg.TRAIN.seed)

# Load ZSC model
zsc_model = build_model(cfg).to(device)
zsc_model.eval()
checkpoint = torch.load(cfg.VAL.resume, map_location='cpu', weights_only=False)
zsc_model.load_state_dict(checkpoint['model'])

# Load counting model
resnet50_conv = Resnet50FPN().to(device)
if device.type != "cpu": resnet50_conv.to(device)
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean').to(device)
regressor.load_state_dict(torch.load(args.model_path))
if device.type != "cpu": regressor.to(device)
regressor.eval()

# Data paths
data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    exit(-1)

# Load annotations
with open(anno_file) as f:
    annotations = json.load(f)
with open(data_split_file) as f:
    data_split = json.load(f)
    
image_classes = {}
class_file = os.path.join(args.data_path, 'ImageClasses_FSC147.txt')
if not exists(class_file):
    print(f"Class file {class_file} not found. Using default class name.")
else:
    with open(class_file, 'r') as f:
        for line in f:
            im_id, class_name = line.strip().split('\t')
            image_classes[im_id] = class_name

cnt = 0
SAE = 0  # Sum of absolute errors
SSE = 0  # Sum of square errors

print(f"Evaluation on {args.test_split} data")
im_ids = data_split[args.test_split]
pbar = tqdm(im_ids)
for im_id in pbar:
    anno = annotations[im_id]
    print('-> img id', im_id)
    dots = np.array(anno['points'])

    # Load image
    image = Image.open(f'{im_dir}/{im_id}').convert('RGB')
    
    class_name = image_classes.get(im_id)
    
    # Generate ZSC boxes
    prompt = fn_prompt(image, class_name, args.prompt_method)
    class_prototype, query_proposals, query_features, query_boxes = generate_class_prototype(prompt, image, zsc_model, device)
    _, _, relevant_boxes = select_relevant_patches(query_proposals, query_boxes, class_prototype, zsc_model, device)
    
    # Convert ZSC boxes to test.py format [y1, x1, y2, x2]
    rects = [[box[1], box[0], box[3], box[2]] for box in relevant_boxes]

    # Transform image and boxes
    sample = {'image': image, 'lines_boxes': rects}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    if device.type != "cpu":
        image = image.to('cuda')
        boxes = boxes.to('cuda')

    # Extract features and count
    with torch.no_grad():
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

    if not args.adapt:
        with torch.no_grad():
            output = regressor(features)
    else:
        features.requires_grad = True
        adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
        for step in range(args.gradient_steps):
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = args.weight_mincount * MincountLoss(output, boxes)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            Loss = lCount + lPerturbation
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
        features.requires_grad = False
        output = adapted_regressor(features)

    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    cnt += 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    pbar.set_description(f'{im_id:<8}: actual-predicted: {gt_cnt:6d}, {pred_cnt:6.1f}, error: {err:6.1f}. Current MAE: {SAE/cnt:5.2f}, RMSE: {(SSE/cnt)**0.5:5.2f}')
    print("")

print(f'On {args.test_split} data, MAE: {SAE/cnt:6.2f}, RMSE: {(SSE/cnt)**0.5:6.2f}')