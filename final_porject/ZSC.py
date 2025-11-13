import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import os
from config import cfg
from FSC147_dataset import get_image_classes
from models import build_model
from models.regressor import get_regressor
from generate_exemplar import generate_image_from_prompt
from RPN import get_rpn_proposals, display_patches
from prompt import fn_prompt
torch.cuda.empty_cache()

# convert to imagenet embeddings
def extract_features(image, model, device, size=(128, 128)):
    transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.backbone(image) 
    return features.cpu().numpy()

# calculate distances between sd patch and query patch
def compute_euclidean_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

# generate the class prototype
def generate_class_prototype(prompt, query_image, model, device, num_sd_images=5, top_k=5):
    # generate prototype
    sd_images = generate_image_from_prompt(prompt, num_sd_images)
    sd_features = []
    for i, image in enumerate(sd_images):
        # take every patches from sd_images with RPN
        proposals, boxes = get_rpn_proposals(image, num_proposals=5)
        display_patches(image, proposals, boxes, output_file=f'./demo/sd_images_patches_{i}.jpg')
        for patch in proposals:
            # move patch into embedings spaces
            features = extract_features(patch, model, device)
            sd_features.append(features)
    
    # take every patches from query_images with RPN and move to imagenet embeddings
    query_proposals, query_boxes = get_rpn_proposals(query_image, num_proposals=100)
    display_patches(query_image, query_proposals, query_boxes, output_file=f'./demo/query_proposal.jpg')
    query_features = [extract_features(patch, model, device) for patch in query_proposals]
    
    # calculate the distances between sd and query embeddings
    distances = []
    for sd_feat in sd_features:
        avg_distance = np.mean([compute_euclidean_distance(sd_feat, qf) for qf in query_features])
        distances.append((sd_feat, avg_distance))
    
    distances.sort(key=lambda x: x[1])
    top_k_features = [feat for feat, _ in distances[:top_k]]
    class_prototype = np.mean(top_k_features, axis=0)
    return class_prototype, query_proposals, query_features, query_boxes

# select the relevant patches    
def select_relevant_patches(query_patches, query_boxes, class_prototype, model, device, num_patches=3, cosine_threshold=0.85):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    query_patches_tensor = torch.stack([transform(p) for p in query_patches]).to(device)
    patch_features = model.backbone(query_patches_tensor)

    patch_features_pooled = model.EPF_extractor.avgpool(patch_features).flatten(1)
    class_prototype_tensor = torch.tensor(class_prototype).to(device)
    
    # Ensure class_prototype_tensor has at least 3 dimensions for avgpool
    if class_prototype_tensor.dim() < 3:
        if class_prototype_tensor.dim() == 2:
            class_prototype_tensor = class_prototype_tensor.unsqueeze(0)  # Add batch dim
        elif class_prototype_tensor.dim() == 1:
            class_prototype_tensor = class_prototype_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    class_prototype_pooled = model.EPF_extractor.avgpool(class_prototype_tensor).flatten(1).to(device)

    # 1. Euclidean distance
    euclidean_dist = ((patch_features_pooled - class_prototype_pooled) ** 2).sum(dim=1)
    sorted_indices = euclidean_dist.argsort()

    # 2. Cosine similarity
    patch_norm = F.normalize(patch_features_pooled, dim=1)
    class_norm = F.normalize(class_prototype_pooled, dim=1)
    cosine_sim = torch.matmul(patch_norm, class_norm.T).squeeze(1)

    # 3. Dari Euclidean paling dekat, cek cosine threshold
    selected_patches = []
    selected_features = []
    selected_boxes = []

    for idx in sorted_indices:
        if cosine_sim[idx] >= cosine_threshold:
            selected_patches.append(query_patches[idx])
            selected_features.append(patch_features[idx])
            selected_boxes.append(query_boxes[idx])
        if len(selected_patches) == num_patches:
            break

    # Fallback kalau tidak ada yang lolos threshold
    if len(selected_patches) == 0:
        print("No patch passed cosine threshold. Using closest by Euclidean only.")
        selected_indices = sorted_indices[:num_patches]
        selected_patches = [query_patches[i] for i in selected_indices.cpu()]
        selected_features = [patch_features[i] for i in selected_indices]
        selected_boxes = [query_boxes[i] for i in selected_indices.cpu()]
    else:
        selected_features = torch.stack(selected_features)

    print("Final cosine similarities:", [cosine_sim[i].item() for i in sorted_indices[:num_patches]])
    print("Final euclidean distances:", [euclidean_dist[i].item() for i in sorted_indices[:num_patches]])

    return selected_features, selected_patches, selected_boxes

def get_boxes(boxes, output_file):
    with open(output_file, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            f.write(f"{y1} {x1} {y2} {x2}\n")

def plot_patches(patches, cols=5, figsize=(15, 8), titles=None, save_path=None):
    rows = (len(patches) + cols - 1) // cols
    plt.figure(figsize=figsize)
    
    for i, patch in enumerate(patches):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(patch)
        plt.axis('off')
        if titles:
            plt.title(titles[i], fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def process(img, cls, prompt_methods='BLIP-LLM'):

    query_img = Image.open(img).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(query_img).to(device)
    with torch.no_grad():
        img_tensor = img.unsqueeze(0).to(device)
        features = model.backbone(img_tensor)
        features = model.input_proj(features)

        prompt = fn_prompt(query_img, cls, prompt_methods)
        class_prototype, query_proposals, query_features, query_boxes = generate_class_prototype(prompt, query_img, model, device)
        relevant_features, relevant_patches, relevant_boxes = select_relevant_patches(query_proposals, query_boxes, class_prototype, model, device)
        display_patches(query_img, relevant_patches, relevant_boxes, output_file='./demo/relevant_patches.jpg')
        plot_patches(relevant_patches, cols=4, save_path="./demo/selected_patches.png")
        get_boxes(relevant_boxes, './demo/box.txt')
           


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exemplar finder")
    parser.add_argument('--image', required=True, type=str, help='Path to the input image.')
    parser.add_argument('--class-name', required=True, type=str, help='List of object class names to detect.')
    parser.add_argument('--prompt-method', default='original', type=str, help='Prompt generation method to use.')
    parser.add_argument("--cfg", default="./config/test.yaml", metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()

    # config
    cfg.merge_from_file(args.cfg)
    cfg.DIR.output_dir = os.path.join(cfg.DIR.snapshot, cfg.DIR.exp)
    os.makedirs(cfg.DIR.output_dir, exist_ok=True)
    cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)
    cfg.VAL.resume = os.path.join(cfg.DIR.output_dir, cfg.VAL.resume)

    # init
    device = torch.device(cfg.TRAIN.device)
    torch.manual_seed(cfg.TRAIN.seed)
    np.random.seed(cfg.TRAIN.seed)
    model = build_model(cfg).to(device)
    model.eval()
    checkpoint = torch.load(cfg.VAL.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    process(args.image, args.class_name, args.prompt_method)

    # run: python ZSC.py --image ./demo/demo.jpg --class-name apple --prompt-method BLIP-LLM