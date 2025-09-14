#./run_styleid_v2v.py
import os
import argparse
import glob
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
import random
import time
from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline

def main(args):
    """
    Main execution function
    """
    # --- 1. Prepare paths and data ---
    # Build paths for content frames and style image
    content_folder_path = os.path.join(args.data_root, 'content', args.content_name)
    style_image_path = os.path.join(args.data_root, 'style', args.style_name)
    
    # Check if paths exist
    if not os.path.isdir(content_folder_path):
        raise FileNotFoundError(f"Content folder not found: {content_folder_path}")
    if not os.path.isfile(style_image_path):
        raise FileNotFoundError(f"Style image not found: {style_image_path}")

    # Load content frames
    # Use glob to find all jpg files and sort them to ensure correct frame order
    frame_paths = sorted(glob.glob(os.path.join(content_folder_path, '*.jpg')) + glob.glob(os.path.join(content_folder_path, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    if not frame_paths:
        raise ValueError(f"No .jpg frames found in {content_folder_path}")
        
    print(f"Found {len(frame_paths)} content frames.")
    
    # Read all frames using cv2 and convert to RGB format
    content_frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]
    
    # Load style image
    style_image = cv2.cvtColor(cv2.imread(style_image_path), cv2.COLOR_BGR2RGB)
    print(f"Loaded style image: {args.style_name}")

    # --- 2. Initialize pipeline ---
    print(f"Loading model from '{args.model_path}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use fp16 to save VRAM
    pipe = StyleIDVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    print("Model loaded, pipeline initialized.")

    # --- 3. Execute style transfer ---
    print("Starting video style transfer...")
    start_time = time.time()  # Start timer
    result = pipe.style_transfer_video(
        content_frames=content_frames,
        style_image=style_image,
        num_inference_steps=args.ddim_steps,
        gamma=args.gamma,
        temperature=args.temperature,
        mask_strength=args.mask_strength,
        without_init_adain=args.without_init_adain,
    )
    
    output_frames = result['images'] # Returns a list of PIL Images
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Style transfer completed! Total time taken: {elapsed_time:.2f} seconds.")

    # --- 4. Save results ---
    # Create output directory, named according to the required format
    style_name_base = os.path.splitext(args.style_name)[0]
    output_dir_name = f"{style_name_base}_stylized_{args.content_name}"
    output_path = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving results to: {output_path}")
    
    # Save each frame
    for i, frame_pil in enumerate(output_frames):
        frame_path = os.path.join(output_path, f"{i+1:04d}.png")
        frame_pil.save(frame_path)
        
    print(f"All stylized frames successfully saved in the folder.")


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(description="Perform video style transfer with StyleIDVideoPipeline")
    
    # --- Path and model parameters ---
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--content_name", type=str, required=True, help="Name of the content video folder under 'data/content/' (e.g., 'car')")
    parser.add_argument("--style_name", type=str, required=True, help="Filename of the style image under 'data/style/' (e.g., 'wave.png')")
    parser.add_argument("--output_dir", type=str, default="./results", help="Root directory to save results")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to the pre-trained Stable Diffusion model or HuggingFace name")

    # --- Core hyperparameters ---
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM inversion and sampling steps")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query preservation strength (controls content preservation)")
    parser.add_argument("--temperature", type=float, default=1.5, help="Attention temperature coefficient (controls stylization strength)")
    parser.add_argument("--mask_strength", type=float, default=1.0, help="Fusion strength for PA Fusion")
    
    # --- Other options ---
    parser.add_argument("--without_init_adain", action="store_true", help="Disable AdaIN operation on initial latent")
    

    args = parser.parse_args()
    main(args)