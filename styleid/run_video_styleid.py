import os
import argparse
import cv2
import torch
import glob
from tqdm import tqdm
from styleid_pipeline import StyleIDPipeline

def load_image_from_path(image_path):
    """
    Loads an image from a given path and converts it from BGR to RGB.
    Returns a numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return img[:, :, ::-1]  # Convert BGR to RGB

def main():
    """
    Main function to handle video style transfer using the StyleID pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Video Style Transfer Baseline using StyleID. \n"
                    "Processes all videos in a content directory with all styles in a style directory."
    )
    parser.add_argument(
        "--content_path", type=str, required=True,
        help="Path to the content directory (containing subfolders of video frames) or a single video folder."
    )
    parser.add_argument(
        "--style_path", type=str, required=True,
        help="Path to the style image or a directory of style images."
    )
    parser.add_argument(
        "--output_dir", type=str, default="video_results",
        help="Directory to save the stylized video frames."
    )
    parser.add_argument(
        "--sd_model", type=str, default="1.5", choices=["1.5", "2.0", "2.1-base", "2.1"],
        help="Stable Diffusion model version to use."
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50,
        help="Number of DDIM inference steps."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.75,
        help="Query preservation strength for style injection."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5,
        help="Attention temperature scaling."
    )
    parser.add_argument(
        "--without_init_adain", action="store_true",
        help="Disable the initial AdaIN layer for latent alignment."
    )
    parser.add_argument(
        "--without_attn_injection", action="store_true",
        help="Disable attention-based style injection."
    )
    
    args = parser.parse_args()

    # --- 1. Setup and Path Validation ---
    print("--- Initializing Video Style Transfer ---")
    
    # Create the main output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved in: {args.output_dir}")

    # --- 2. Collect Style and Content Paths ---
    # Get a list of style images
    style_paths = []
    if os.path.isfile(args.style_path):
        style_paths.append(args.style_path)
    elif os.path.isdir(args.style_path):
        style_paths = sorted(glob.glob(os.path.join(args.style_path, "*[.jpg,.jpeg,.png]")))
    
    if not style_paths:
        raise FileNotFoundError(f"No style images found in {args.style_path}")
    print(f"Found {len(style_paths)} style image(s).")

    # Get a list of content video folders
    content_folders = []
    if os.path.isdir(args.content_path):
        # Check if the path itself is a video folder (contains images)
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(args.content_path)):
             content_folders.append(args.content_path)
        else:
             # Assume it's a parent directory of video folders
            content_folders = sorted([os.path.join(args.content_path, d) for d in os.listdir(args.content_path) if os.path.isdir(os.path.join(args.content_path, d))])

    if not content_folders:
        raise FileNotFoundError(f"No content video folders found in {args.content_path}")
    print(f"Found {len(content_folders)} content video folder(s).")

    # --- 3. Load the StyleID Pipeline ---
    sd_model_map = {
        "1.5": "runwayml/stable-diffusion-v1-5",
        "2.0": "stabilityai/stable-diffusion-2-base",
        "2.1-base": "stabilityai/stable-diffusion-2-1-base",
        "2.1": "stabilityai/stable-diffusion-2-1"
    }
    model_id = sd_model_map[args.sd_model]
    
    print(f"Loading StyleID pipeline with model: {model_id}")
    try:
        pipeline = StyleIDPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda")
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # --- 4. Main Processing Loop ---
    # This baseline processes each frame independently.
    # While less efficient (re-inverts style for each frame), it's a robust
    # way to evaluate the core algorithm on video without temporal logic.

    for style_path in tqdm(style_paths, desc="Total Styles"):
        try:
            style_image_rgb = load_image_from_path(style_path)
            style_name = os.path.splitext(os.path.basename(style_path))[0]
        except Exception as e:
            print(f"Skipping style {style_path} due to loading error: {e}")
            continue
        print(f"\n[Style Pre-computation] Inverting style '{style_name}' once...")
        pipeline.state.reset()
        pipeline.state.to_invert_style()
        text_embeddings = pipeline.get_text_embedding() # get an empty text embedding
        style_latent = pipeline.encode_image(style_image_rgb)
        pipeline.ddim_inversion(style_latent, text_embeddings)
        print(f"Style '{style_name}' inverted and features are now cached.")
        for content_folder in tqdm(content_folders, desc=f"Processing Content for '{style_name}'", leave=False):
            content_name = os.path.basename(content_folder)
            
            # Create the specific output folder for this (style, content) pair
            result_folder_name = f"{style_name}_stylized_{content_name}"
            result_path = os.path.join(args.output_dir, result_folder_name)
            os.makedirs(result_path, exist_ok=True)
            
            # Get all frames in the content folder, sorted numerically/alphabetically
            frame_paths = sorted(glob.glob(os.path.join(content_folder, "*[.jpg,.jpeg,.png]")))

            if not frame_paths:
                print(f"Warning: No frames found in {content_folder}, skipping.")
                continue

            print(f"\nProcessing: {content_name} ({len(frame_paths)} frames) with style {style_name}")

            for frame_path in tqdm(frame_paths, desc=f"Stylizing '{content_name}'", leave=False):
                try:
                    content_frame_rgb = load_image_from_path(frame_path)
                    
                    # call a modified transfer function ===
                    # This function will reuse the cached style features
                    stylized_output = pipeline.transfer_with_cached_style(
                        content_image=content_frame_rgb,
                        style_image_for_adain=style_image_rgb, # 仅用于初始AdaIN
                        num_inference_steps=args.ddim_steps,
                        gamma=args.gamma,
                        temperature=args.temperature,
                        without_init_adain=args.without_init_adain,
                        without_attn_injection=args.without_attn_injection,
                        output_type="pil"
                    )
                    
                    # Save the stylized frame
                    output_image = stylized_output.images[0]
                    frame_basename = os.path.basename(frame_path)
                    output_frame_path = os.path.join(result_path, f"{os.path.splitext(frame_basename)[0]}.png")
                    output_image.save(output_frame_path)

                except Exception as e:
                    print(f"\nError processing frame {frame_path}: {e}")
                    print("Skipping frame and continuing...")
                    continue


    print("\n--- Video style transfer complete! ---")


if __name__ == "__main__":
    main()