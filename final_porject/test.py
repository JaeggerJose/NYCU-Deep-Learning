#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
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
import random

# SynGen imports
try:
    from syngen_diffusion_pipeline import SynGenDiffusionPipeline
    SYNGEN_AVAILABLE = True
except ImportError:
    print("Warning: SynGen not available. Install required dependencies to use --use_syngen")
    SYNGEN_AVAILABLE = False

parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
# seed 
parser.add_argument("-s", "--seed", type=int, default=42, help="seed for the random number generator")
# SynGen parameters
parser.add_argument("--use_syngen", action='store_true', help="Use SynGen to generate exemplar images")
parser.add_argument("--syngen_num_images", type=int, default=3, help="Number of exemplar images to generate with SynGen")
parser.add_argument("--syngen_step_size", type=float, default=20.0, help="SynGen step size for linguistic binding")
parser.add_argument("--syngen_steps", type=int, default=25, help="Number of SynGen intervention steps")
args = parser.parse_args()

def load_image_classes(data_path):
    """Load image classes from FSC147 dataset"""
    classes_file = data_path + 'ImageClasses_FSC147.txt'
    image_classes = {}
    if exists(classes_file):
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_classes[parts[0]] = parts[1]
    return image_classes

def generate_syngen_exemplars(im_id, image_classes, syngen_pipe, args):
    """Generate exemplar images using SynGen"""
    # Get object class for this image
    object_class = image_classes.get(im_id, "objects")
    prompt = f"a photo of {object_class}"
    
    # Generate exemplar images using SynGen
    try:
        images = syngen_pipe(
            prompt,
            num_images_per_prompt=args.syngen_num_images,
            syngen_step_size=args.syngen_step_size,
            num_intervention_steps=args.syngen_steps,
            num_inference_steps=50
        ).images
        
        # Save generated images
        output_dir = "./demo/sd_exemplars"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, img in enumerate(images):
            filename = f"{im_id}_{object_class.replace(' ', '_')}_{i}.jpg"
            save_path = os.path.join(output_dir, filename)
            img.save(save_path)
            saved_paths.append(save_path)
            
        return images, saved_paths
    except Exception as e:
        print(f"Error generating SynGen exemplars for {im_id}: {e}")
        return [], []

def images_to_bboxes(images, target_size=(384, 384)):
    """Convert generated images to bboxes format used by the model"""
    rects = []
    
    for img in images:
        # Resize image to target size if needed
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Use the entire image as a bbox (can be improved with object detection)
        h, w = target_size
        # Create multiple patches from the generated image
        patch_size = min(w, h) // 4  # Use quarter size patches
        
        # Generate several patches from different parts of the image
        for i in range(min(3, len(images))):  # Up to 3 patches per image
            x = (i * patch_size) % (w - patch_size)
            y = (i * patch_size) % (h - patch_size)
            
            # Format: [y1, x1, y2, x2]
            rect = [y, x, y + patch_size, x + patch_size]
            rects.append(rect)
            
            if len(rects) >= 3:  # Limit to 3 exemplars total
                break
    
    return rects if rects else [[0, 0, 50, 50]]  # Fallback bbox

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Initialize SynGen pipeline if requested
syngen_pipe = None
if args.use_syngen:
    if not SYNGEN_AVAILABLE:
        print("Error: SynGen not available. Please install required dependencies.")
        exit(-1)
    
    print("Initializing SynGen pipeline...")
    try:
        device_name = f"cuda:{args.gpu_id}" if use_gpu else "cpu"
        syngen_pipe = SynGenDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if use_gpu else torch.float32,
        ).to(device_name)
        print("SynGen pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing SynGen pipeline: {e}")
        exit(-1)

data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'

# Load image classes for SynGen prompts
image_classes = load_image_classes(data_path)

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
if use_gpu: resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')
regressor.load_state_dict(torch.load(args.model_path))
if use_gpu: regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
pbar = tqdm(im_ids)
for im_id in pbar:
    anno = annotations[im_id]
    dots = np.array(anno['points'])

    # Use SynGen to generate exemplars if requested
    if args.use_syngen and syngen_pipe is not None:
        generated_images, saved_paths = generate_syngen_exemplars(im_id, image_classes, syngen_pipe, args)
        if generated_images:
            rects = images_to_bboxes(generated_images)
            print(f"Generated {len(generated_images)} SynGen exemplars for {im_id}, saved to: {saved_paths}")
        else:
            # Fallback to original bboxes if SynGen fails
            bboxes = anno['box_examples_coordinates']
            rects = list()
            for bbox in bboxes:
                x1, y1 = bbox[0][0], bbox[0][1]
                x2, y2 = bbox[2][0], bbox[2][1]
                rects.append([y1, x1, y2, x2])
            print(f"SynGen failed for {im_id}, using original exemplars")
    else:
        # Use original bboxes from dataset
        bboxes = anno['box_examples_coordinates']
        rects = list()
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    sample = {'image': image, 'lines_boxes': rects}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    if use_gpu:
        image = image.cuda()
        boxes = boxes.cuda()

    with torch.no_grad(): features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

    if not args.adapt:
        with torch.no_grad(): output = regressor(features)
    else:
        features.required_grad = True
        adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
        for step in range(0, args.gradient_steps):
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = args.weight_mincount * MincountLoss(output, boxes)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            Loss = lCount + lPerturbation
            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
        features.required_grad = False
        output = adapted_regressor(features)

    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
    print("")

print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
