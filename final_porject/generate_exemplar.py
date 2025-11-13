from syngen_diffusion_pipeline import SynGenDiffusionPipeline
import torch
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use SynGen pipeline instead of standard Stable Diffusion
pipe = SynGenDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

# function for generate number of images base on given prompt with linguistic binding
def generate_image_from_prompt(prompt, num_image=5, save=False, img_id_ext='inference.jpg'):
    output_dir = "./demo/sd_exemplars"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use SynGen with linguistic binding to avoid semantic bias
    images = pipe(
        prompt, 
        num_images_per_prompt=num_image,
        syngen_step_size=20.0,        # Controls linguistic binding strength
        num_intervention_steps=25,    # Number of SynGen intervention steps
        num_inference_steps=50        # Total diffusion steps
    ).images
    
    if save:
        for i, img in enumerate(images):
            file_name = prompt.replace(" ", "_") + f"_{i}_" + img_id_ext
            save_path = os.path.join(output_dir, file_name).replace("\n", "")
            img.save(save_path)
    # pipe.to('cpu')
    return images


if __name__ == '__main__':
    generate_image_from_prompt('a photo of apple', num_image=3, save=True, img_id_ext='testing.jpg')