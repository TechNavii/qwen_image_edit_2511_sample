import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Global pipeline variable
pipeline = None
current_device = None


def load_model(progress=gr.Progress()):
    """Load the Qwen Image Edit model onto MPS device."""
    global pipeline, current_device
    
    if pipeline is not None:
        return f"Model already loaded on {current_device}!"
    
    progress(0, desc="Loading model from Hugging Face...")
    
    # Check for MPS availability
    if torch.backends.mps.is_available():
        current_device = "mps"
        # Use float32 for MPS stability (float16 can cause black outputs)
        dtype = torch.float32
    elif torch.cuda.is_available():
        current_device = "cuda"
        dtype = torch.bfloat16
    else:
        current_device = "cpu"
        dtype = torch.float32
    
    progress(0.2, desc=f"Using device: {current_device}")
    
    # Load the pipeline
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=dtype
    )
    
    progress(0.7, desc="Moving model to device...")
    pipeline.to(current_device)
    
    progress(1.0, desc="Model loaded successfully!")
    return f"Model loaded on {current_device} with {dtype}"


def prepare_image(image):
    """Convert and prepare image for the pipeline."""
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def edit_image(
    image,
    reference_image,
    prompt,
    true_cfg_scale,
    num_inference_steps,
    guidance_scale,
    seed,
    progress=gr.Progress()
):
    """Edit an image using the loaded model with optional reference image."""
    global pipeline, current_device
    
    if pipeline is None:
        return None, "Please load the model first!"
    
    if image is None:
        return None, "Please upload a main image!"
    
    if not prompt or prompt.strip() == "":
        return None, "Please enter a prompt!"
    
    progress(0, desc="Preparing images...")
    
    # Prepare main image
    main_image = prepare_image(image)
    
    # Build image list (main image + optional reference images)
    images = [main_image]
    
    # Add reference image if provided
    if reference_image is not None:
        ref_img = prepare_image(reference_image)
        images.append(ref_img)
        progress(0.05, desc=f"Processing {len(images)} images...")
    
    progress(0.1, desc="Starting inference...")
    
    # Set up generator for reproducibility (always on CPU for MPS compatibility)
    if seed == -1:
        generator = None
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    
    # Prepare inputs
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": float(true_cfg_scale),
        "negative_prompt": " ",
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "num_images_per_prompt": 1,
    }
    
    num_imgs = len(images)
    progress(0.2, desc=f"Running {int(num_inference_steps)} steps with {num_imgs} image(s)...")
    
    try:
        # Run inference with no_grad for MPS compatibility
        with torch.no_grad():
            output = pipeline(**inputs)
        
        # Sync MPS to ensure computation is complete
        if current_device == "mps":
            torch.mps.synchronize()
        
        progress(1.0, desc="Done!")
        output_image = output.images[0]
        
        return output_image, f"Image editing complete! (Used {num_imgs} input image(s))"
    
    except Exception as e:
        return None, f"Error during inference: {str(e)}"


# Create the Gradio interface
with gr.Blocks(title="Qwen Image Edit 2511") as demo:
    gr.Markdown("# Qwen Image Edit 2511")
    gr.Markdown("Edit images using AI with the Qwen-Image-Edit-2511 model. Optimized for Apple Silicon (MPS).")
    
    with gr.Row():
        load_btn = gr.Button("Load Model", variant="primary", scale=1)
        model_status = gr.Textbox(label="Model Status", interactive=False, scale=2)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Images")
            with gr.Row():
                input_image = gr.Image(label="Main Image (Figure 1)", type="pil")
                reference_image = gr.Image(label="Reference Image (Figure 2) - Optional", type="pil")
            
            prompt_input = gr.Textbox(
                label="Edit Prompt",
                placeholder="e.g., 'Replace the person's face in Figure 1 with the face from Figure 2' or 'Put the person from Figure 1 next to the person from Figure 2'",
                lines=3
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                true_cfg_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                    label="True CFG Scale"
                )
                num_inference_steps = gr.Slider(
                    minimum=10, maximum=100, value=40, step=5,
                    label="Number of Inference Steps"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0, maximum=5.0, value=1.0, step=0.1,
                    label="Guidance Scale"
                )
                seed = gr.Number(
                    value=-1,
                    label="Seed (-1 for random)"
                )
            
            edit_btn = gr.Button("Edit Image", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_image = gr.Image(label="Output Image", type="pil")
            output_status = gr.Textbox(label="Status", interactive=False)
    
    # Wire up events
    load_btn.click(fn=load_model, outputs=model_status)
    edit_btn.click(
        fn=edit_image,
        inputs=[
            input_image,
            reference_image,
            prompt_input,
            true_cfg_scale,
            num_inference_steps,
            guidance_scale,
            seed
        ],
        outputs=[output_image, output_status]
    )
    
    gr.Markdown("""
    ## Tips
    - Click "Load Model" first (downloads ~20GB on first run)
    - **Single image edit**: Upload main image only, describe your edit
    - **Multi-image edit**: Upload main image + reference image for:
      - Face swapping: "Replace the face in Figure 1 with the face from Figure 2"
      - Combining people: "Put the person from Figure 1 on the left and person from Figure 2 on the right"
      - Style transfer: "Apply the style of Figure 2 to Figure 1"
    - Reference images as "Figure 1" (main) and "Figure 2" (reference) in your prompt
    - Higher inference steps = better quality but slower
    """)


if __name__ == "__main__":
    demo.launch()
