import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from styleid_pipeline import StyleIDPipeline
import os

def load_image(image_path):
    """Load and prepare image from path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img[:, :, ::-1]  # Convert BGR to RGB

def main():
    parser = argparse.ArgumentParser(description="StyleID: Style transfer through diffusion")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save output image")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--gamma", type=float, default=0.75, help="Content preservation strength (0-1)")
    parser.add_argument("--temperature", type=float, default=1.5, help="Attention temperature")
    parser.add_argument("--no_adain", action="store_true", help="Disable initial latent AdaIN")
    parser.add_argument("--no_attn", action="store_true", help="Disable attention-based style injection")
    parser.add_argument("--model", type=str, default="1.5", 
                        choices=["1.5", "2.0", "2.1-base", "2.1"], 
                        help="Stable Diffusion model version")
    parser.add_argument("--save_intermediates", action="store_true", help="Save intermediate results")
    parser.add_argument("--intermediates_dir", type=str, default="results", 
                        help="Directory to save intermediate results")
    
    args = parser.parse_args()
    
    # Map model version to model ID
    model_map = {
        "1.5": "runwayml/stable-diffusion-v1-5",
        "2.0": "stabilityai/stable-diffusion-2-base",
        "2.1-base": "stabilityai/stable-diffusion-2-1-base",
        "2.1": "stabilityai/stable-diffusion-2-1"
    }
    model_id = model_map[args.model]
    
    # Load images
    print(f"Loading content image from {args.content}")
    content_image = load_image(args.content)
    
    print(f"Loading style image from {args.style}")
    style_image = load_image(args.style)
    
    # Create StyleID pipeline
    print(f"Loading StyleID pipeline with model {model_id}")
    pipeline = StyleIDPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Run style transfer
    print("Running style transfer...")
    output = pipeline.style_transfer(
        content_image=content_image,
        style_image=style_image,
        num_inference_steps=args.steps,
        gamma=args.gamma,
        temperature=args.temperature,
        without_init_adain=args.no_adain,
        without_attn_injection=args.no_attn,
        save_intermediates_dir=args.intermediates_dir if args.save_intermediates else None
    )
    
    # Save output
    print(f"Saving output to {args.output}")
    
    # Extract filenames from paths
    # Get style and content image names without path and extension
    style_name = os.path.splitext(os.path.basename(args.style))[0]
    content_name = os.path.splitext(os.path.basename(args.content))[0]
    
    # Handle output directory or path
    if os.path.isdir(args.output) or args.output.endswith('/') or not os.path.splitext(args.output)[1]:
        # It's a directory path
        os.makedirs(args.output, exist_ok=True)
        output_filename = f"{style_name}_stylized_{content_name}.png"
        output_path = os.path.join(args.output, output_filename)
    else:
        # It's a file path
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_path = args.output
        # Add .png extension if no extension is provided
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            output_path += '.png'
    
    print(f"Saving final image to: {output_path}")
    output.images[0].save(output_path)
    print("Done!")

if __name__ == "__main__":
    main() 