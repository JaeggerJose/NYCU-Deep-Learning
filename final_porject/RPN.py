import torch, math
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.ops as ops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RPN_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
RPN_model.eval()

def get_rpn_proposals(image, num_proposals=5, score_thresh=0, iou_thresh=0.3):
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = RPN_model(image_tensor)
    
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    
    # Filter berdasarkan threshold confidence
    keep = scores > score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    # Terapkan NMS
    keep = ops.nms(boxes, scores, iou_threshold=iou_thresh)
    boxes = boxes[keep]
    scores = scores[keep]

    # Ambil top-N berdasarkan score tertinggi
    top_idx = scores.argsort(descending=True)[:num_proposals]
    boxes = boxes[top_idx].cpu().numpy()
    scores = scores[top_idx].cpu().numpy()

    patches = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        patch = image.crop((x1, y1, x2, y2))
        patches.append(patch)
        
    return patches, boxes

# display patches for analyzing
def display_patches(query_image, relevant_patches, relevant_boxes, cols=4, output_file='patches.jpg'):
    num_patches = len(relevant_patches)
    total_images = num_patches + 1  
    rows = math.ceil(total_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    # Gambar utama (query image)
    axes[0].imshow(query_image)
    for box in relevant_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title("Image", fontsize=12 * cols)
    axes[0].axis('off')

    # Sisanya adalah patch
    for i, patch in enumerate(relevant_patches):
        axes[i + 1].imshow(patch)
        axes[i + 1].set_title(f"Patch {i + 1}", fontsize=12 * cols)
        axes[i + 1].axis('off')

    # Kosongkan sisa axis jika ada
    for i in range(total_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

if __name__ == '__main__':
    image = Image.open("kiwi.jpg").convert('RGB')
    patches, boxes = get_rpn_proposals(image, 100)
    display_patches(image, patches, boxes, filename='query_patches')