from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, requests
from io import BytesIO
torch.manual_seed(42) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer phi3 #facebook/opt-350m, EleutherAI/gpt-neo-125M, microsoft/phi3-8b-instruct
llm_model_id = "EleutherAI/gpt-neo-1.3B"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=torch.float16, )
llm_model = llm_model.to(device)

# load model and processor BLIP
blip_model_id = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(blip_model_id, use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id) 
blip_model = blip_model.to(device)

# fn_prompt
def fn_prompt(image, class_name, methods='original', output_file='./demo/prompt.txt'):
    if methods == 'original':
        prompt = f'a photo of {class_name}'
        print("Original class:", class_name)
        with open(output_file, 'w') as f:
            f.write(prompt)
        return prompt
    
    elif methods == 'BLIP-LLM':
        image_caption = get_image_caption(image)
        refined_class = refine_prompt(image_caption, class_name)
        prompt = f'a photo of {refined_class}'
        print("Image caption:", image_caption)
        print("Original class:", class_name)
        print("Refined Prompt:", prompt)
        with open(output_file, 'w') as f:
            f.write(prompt)
        return prompt
    
    else:
        raise NotImplementedError(f"Method '{methods}' is not implemented.")

# with blip (image should use pil module)
def get_image_caption(image_query):
    blip_model.eval()
    inputs = blip_processor(images=image_query, return_tensors="pt").to(device)  # <-- input ke device
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    # blip_model = blip_model.to("cpu")
    return caption

# with instruction llm
def refine_prompt(image_caption, class_name):
    # Simplified prompt without chat format for better compatibility with gpt-neo
    prompt = f"""Given an image caption and a class name, refine the class name to be more specific.

Image caption: {image_caption}
Class name: {class_name}

Refined class name:"""
    
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, 
            max_new_tokens=20,  # Reduced to get shorter, more focused responses
            do_sample=True, 
            temperature=0.7,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )
    
    # Get only the generated part
    generated_text = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean and extract the refined class name
    refined_class_name = generated_text.strip()
    
    # Split by common delimiters and take the first meaningful part
    if '\n' in refined_class_name:
        refined_class_name = refined_class_name.split('\n')[0].strip()
    
    # Remove common prefixes and artifacts
    prefixes_to_remove = ['a photo of', 'an image of', 'the', 'Refined class name:', 'Answer:', 'Output:']
    for prefix in prefixes_to_remove:
        if refined_class_name.lower().startswith(prefix.lower()):
            refined_class_name = refined_class_name[len(prefix):].strip()
    
    # Remove punctuation at the end
    refined_class_name = refined_class_name.rstrip('.,!?;:')
    
    # If the result is empty or too long, fallback to original
    if not refined_class_name or len(refined_class_name) > 100:
        refined_class_name = class_name
        
    return refined_class_name

if __name__ == '__main__':
    # input
    url = 'https://cdn.britannica.com/84/65384-050-A9528785/Sandhill-cranes.jpg'
    image_path = BytesIO(requests.get(url).content)
    # image_path = "kiwi.jpg"
    class_name = "bird"

    # pipeline
    image = Image.open(image_path).convert("RGB")
    prompt = fn_prompt(image, class_name, methods='BLIP-LLM')

    
    
