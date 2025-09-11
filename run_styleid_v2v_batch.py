#./run_styleid_v2v_batch.py (Final simplified version)
import os
import argparse
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import random
import itertools
from tqdm import tqdm
from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline

def run_style_transfer_for_pair(pipe, content_name, style_name, args):
    """
    Performs style transfer for a single content-style pair.
    This version fully delegates all preprocessing (cropping, scaling) to the Pipeline.
    """
    try:
        # 1. Build the expected output folder path based on the naming convention
        style_name_base = os.path.splitext(os.path.basename(style_name))[0]
        output_dir_name = f"{style_name_base}_stylized_{content_name}"
        output_path = os.path.join(args.output_dir, output_dir_name)

        # 2. Check if the path exists and is a directory; if so, skip
        if os.path.isdir(output_path):
            tqdm.write(f"Result already exists, skipping: {output_dir_name}")
            return

        tqdm.write("-" * 50)
        tqdm.write(f"Starting to process: Content='{content_name}', Style='{style_name}'")

        content_folder_path = os.path.join(args.data_root, 'content', content_name)
        style_image_path = os.path.join(args.data_root, 'style', style_name)

        if not os.path.isdir(content_folder_path):
            tqdm.write(f"Error: Content folder not found '{content_folder_path}', skipping.")
            return
        if not os.path.isfile(style_image_path):
            tqdm.write(f"Error: Style image not found '{style_image_path}', skipping.")
            return

        frame_paths = sorted(glob.glob(os.path.join(content_folder_path, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if not frame_paths:
            tqdm.write(f"Warning: No .jpg frames found in folder '{content_folder_path}', skipping.")
            return

        
        content_frames = []
        for p in frame_paths:
            frame_bgr = cv2.imread(p)
            if frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                content_frames.append(frame_rgb)
                 # Directly append the original BGR NumPy array

        if not content_frames:
            tqdm.write(f"Error: Could not read any valid frames from '{content_folder_path}', skipping.")
            return
            
        style_image_bgr = cv2.imread(style_image_path)
        if style_image_bgr is None:
            tqdm.write(f"Error: Could not load style image '{style_image_path}', skipping.")
            return
        style_image = cv2.cvtColor(style_image_bgr, cv2.COLOR_BGR2RGB)
        # --- Call the core pipeline ---
        # The Pipeline will now handle all sizing issues itself
        result = pipe.style_transfer_video(
            content_frames=content_frames,
            style_image=style_image,
            num_inference_steps=args.ddim_steps,
            gamma=args.gamma,
            temperature=args.temperature,
            mask_strength=args.mask_strength,
            without_init_adain=args.without_init_adain,
        )
        
        output_frames = result['images']
        
        # --- Save results ---
        os.makedirs(output_path, exist_ok=True)
        
        for i, frame_pil in enumerate(output_frames):
            frame_path = os.path.join(output_path, f"{i+1:04d}.png")
            frame_pil.save(frame_path)
            
        tqdm.write(f"Success: Processed '{content_name}' and '{style_name}', saved to {output_path}")

    except Exception as e:
        tqdm.write(f"\nSerious error occurred while processing '{content_name}' and '{style_name}': {e}")
        import traceback
        tqdm.write(traceback.format_exc())
        tqdm.write("Will continue to the next pair.\n")

def main(args):
    content_root = os.path.join(args.data_root, 'content')
    style_root = os.path.join(args.data_root, 'style')
    if not os.path.isdir(content_root): raise FileNotFoundError(f"Content data root directory not found: {content_root}")
    if not os.path.isdir(style_root): raise FileNotFoundError(f"Style data root directory not found: {style_root}")
    
    all_content_names = [d for d in os.listdir(content_root) if os.path.isdir(os.path.join(content_root, d))]
    all_style_names_png = glob.glob(os.path.join(style_root, '*.png'))
    all_style_names_jpg = glob.glob(os.path.join(style_root, '*.jpg'))
    all_style_names_jpeg = glob.glob(os.path.join(style_root, '*.jpeg'))
    all_style_names = all_style_names_png + all_style_names_jpg + all_style_names_jpeg

    if not all_content_names or not all_style_names: 
        print("Error: No content or style files found.")
        return
        
    job_pairs = list(itertools.product(all_content_names, all_style_names))
    
    print("="*50)
    print(f"Found {len(all_content_names)} content videos.")
    print(f"Found {len(all_style_names)} style images.")
    print(f"Total of {len(job_pairs)} style transfer tasks to perform.")
    print("="*50)
    
    print(f"Loading model from '{args.model_path}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StyleIDVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    print("Model loaded, pipeline initialized.")
    
    print("\nStarting to process all tasks in batch...")
    for content_name, style_name in tqdm(job_pairs, desc="Overall progress"):
        run_style_transfer_for_pair(pipe, content_name, os.path.basename(style_name), args)
        
    print("\nAll style transfer tasks have been completed!")

if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(description="Batch video style transfer using StyleIDVideoPipeline")
    parser.add_argument("--data_root", type=str, default="./data", help="Root data directory, should contain 'content' and 'style' subdirectories")
    parser.add_argument("--output_dir", type=str, default="./results", help="Root directory for saving results")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to a pre-trained Stable Diffusion model or a HuggingFace name")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of steps for DDIM inversion and sampling")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query preservation strength")
    parser.add_argument("--temperature", type=float, default=1.5, help="Attention temperature coefficient")
    parser.add_argument("--mask_strength", type=float, default=1.0, help="PA Fusion blend strength")
    parser.add_argument("--without_init_adain", action="store_true", help="Disable the AdaIN operation on the initial latent")
    args = parser.parse_args()
    main(args)