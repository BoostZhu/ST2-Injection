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
import json  # --- ADDED: Import the json library ---

from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline

def main(args):
    """
    Main execution function
    """
    # --- 1. Prepare paths and data ---
    # (This section remains unchanged)
    content_folder_path = os.path.join(args.data_root, 'content', args.content_name)
    style_image_path = os.path.join(args.data_root, 'style', args.style_name)
    
    if not os.path.isdir(content_folder_path):
        raise FileNotFoundError(f"Content folder not found: {content_folder_path}")
    if not os.path.isfile(style_image_path):
        raise FileNotFoundError(f"Style image not found: {style_image_path}")

    frame_paths = sorted(glob.glob(os.path.join(content_folder_path, '*.jpg')) + glob.glob(os.path.join(content_folder_path, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    if not frame_paths:
        raise ValueError(f"No .jpg or .png frames found in {content_folder_path}")
        
    print(f"Found {len(frame_paths)} content frames.")
    
    content_frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]
    style_image = cv2.cvtColor(cv2.imread(style_image_path), cv2.COLOR_BGR2RGB)
    print(f"Loaded style image: {args.style_name}")

    # --- 2. Initialize pipeline ---
    # (This section remains unchanged)
    print(f"Loading model from '{args.model_path}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StyleIDVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    print("Model loaded, pipeline initialized.")

    # --- 3. Execute style transfer ---
    # (This section remains unchanged)
    print("Starting video style transfer with the following settings:")
    print(f"  - Fusion Strategy: {args.fusion_strategy}")
    print(f"  - Fusion Window: {args.fusion_start_percent*100:.0f}% -> {args.fusion_end_percent*100:.0f}%")

    start_time = time.time()
    
    result = pipe.style_transfer_video(
        content_frames=content_frames,
        style_image=style_image,
        num_inference_steps=args.ddim_steps,
        gamma=args.gamma,
        temperature=args.temperature,
        mask_strength=args.mask_strength,
        without_init_adain=args.without_init_adain,
        fusion_strategy=args.fusion_strategy,
        fusion_start_percent=args.fusion_start_percent,
        fusion_end_percent=args.fusion_end_percent,
    )
    
    output_frames = result['images']
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Style transfer completed! Total time taken: {elapsed_time:.2f} seconds.")

    # --- 4. Save results ---
    style_name_base = os.path.splitext(args.style_name)[0]
    output_dir_name = f"{style_name_base}_stylized_{args.content_name}"
    output_path = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving results to: {output_path}")

    # --- ADDED: Save parameters to a JSON file ---
    params_path = os.path.join(output_path, 'parameters.json')
    # Convert argparse Namespace to a dictionary
    params_dict = vars(args)
    # Also save the execution time for complete record-keeping
    params_dict['elapsed_time_seconds'] = round(elapsed_time, 2)
    
    print(f"Saving parameters to: {params_path}")
    with open(params_path, 'w', encoding='utf-8') as f:
        # Use indent=4 for a human-readable format
        json.dump(params_dict, f, indent=4)
    # --- END ADDED ---
    
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

    # --- Fusion control parameters ---
    parser.add_argument(
        "--fusion-strategy", 
        type=str, 
        default="anchor_only", 
        choices=["anchor_and_prev", "anchor_only"],
        help="The fusion strategy to use."
    )
    parser.add_argument(
        "--fusion-start-percent", 
        type=float, 
        default=0.5,
        help="At what percentage of the denoising steps to start the fusion (e.g., 0.3 for 30%%)."
    )
    parser.add_argument(
        "--fusion-end-percent", 
        type=float, 
        default=0.8,
        help="At what percentage of the denoising steps to end the fusion (e.g., 0.8 for 80%%)."
    )
    
    # --- Other options ---
    parser.add_argument("--without_init_adain", action="store_true", help="Disable AdaIN operation on initial latent")
    
    args = parser.parse_args()
    main(args)