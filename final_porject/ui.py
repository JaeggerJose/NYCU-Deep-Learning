from PIL import Image
import streamlit as st
import subprocess
import os

# Streamlit app
st.title("Zero-Shot Object Counting Demo")
st.write("Upload an image and specify class names (separated by semicolons for multiple classes).")
main_dir = './demo'

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Class name input
class_input = st.text_input("Enter class name")

# Prompt method radio button
prompt_method = st.radio("Select prompt method", options=['original', 'BLIP-LLM'])

# Show uploaded image preview
if uploaded_file is not None:
    st.subheader("Original Image Preview")
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

# Process button
if st.button("Count Objects"):
    if uploaded_file is not None and class_input.strip() != "":
        with st.spinner("Processing... Please wait."):
            # Save uploaded image
            image_path = f"{main_dir}/demo.jpg"
            image.save(image_path)

            # Run ZSC.py
            subprocess.run([
                "python", "ZSC.py",
                "--image", image_path,
                "--class-name", class_input,
                "--prompt-method", prompt_method
            ])

            # Run demo.py
            subprocess.run([
                "python", "demo.py",
                "--input-image", image_path,
                "--bbox-file", f"{main_dir}/box.txt",
                "--output-dir", main_dir,
                "-g", "5"
            ])

            # Show prompt used
            st.success("Processing Complete!")
            with open('./demo/prompt.txt', 'r') as f:
                prompt = f.read().strip()
            st.markdown(f"### Prompt Used ({prompt_method}): {prompt}")

            # Show output image
            with open('./demo/count.txt', 'r') as f:
                count = f.read().strip()
            output_image_path = os.path.join(main_dir, "demo_out.png")
            if os.path.exists(output_image_path):
                st.subheader(f"Counted: {count}")
                st.image(output_image_path, caption='Detected and Counted Image', use_container_width=True)
            else:
                st.warning("Output image not found.")

            # Show more details in expandable
            with st.expander("Show Details"):
                st.markdown("#### Detailed Visualizations")

            # List of detail images
            detail_images = [
                ["query_proposal.jpg", 'Proposal Patch from Query Image'],
                ["sd_images_patches_0.jpg", 'SD images patches (1)'],
                ["sd_images_patches_1.jpg", 'SD images patches (2)'],
                ["sd_images_patches_2.jpg", 'SD images patches (3)'],
                ["relevant_patches.jpg", 'Selected Patches as Exemplar'],
            ]

            for img_name, caption in detail_images:
                img_path = os.path.join(main_dir, img_name)
                if os.path.exists(img_path):
                    st.image(img_path, caption=caption, use_container_width=True)
                else:
                    st.info(f"{img_name} not found.")
    else:
        st.error("Please upload an image and specify the class names.")
