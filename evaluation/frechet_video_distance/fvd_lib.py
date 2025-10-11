# File: evaluation/frechet_video_distance/fvd_lib.py

import torch
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm

# We can import `compute_fvd` because it's in the same directory
from calculate_fvd import compute_fvd 

def _load_video_from_folder(folder_path, image_size=(224, 224)): # Keep default, but we'll override
    """Loads all frames from a folder into a single video tensor."""
    
    # --- CHANGE 1: Set image size back to 512x512 ---
    REPLICA_IMAGE_SIZE = (512, 512)

    frame_files = sorted(glob.glob(os.path.join(folder_path, '*.[jp][pn]g')))
    if not frame_files:
        return None
    
    # The old script did not have this padding logic, so we can comment it out
    # if len(frame_files) < 16:
    #     frame_files.extend([frame_files[-1]] * (16 - len(frame_files)))
        
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB").resize(REPLICA_IMAGE_SIZE) # Use the old size
        frames.append(np.array(img))
    
    # --- CHANGE 2: Add back the frame duplication line ---
    if frames:
        frames = frames * 2

    video_array = np.stack(frames, axis=0)
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2)
    return video_tensor.float()

def evaluate_fvd_for_method(method_name, method_path, original_videos_path, device, batch_size, local_model_path):
    """Evaluates the FVD score for a single method against the original videos."""
    print(f"\n--- Evaluating FVD for method: {method_name} ---")

    # Load original videos only once (if not cached)
    if not hasattr(evaluate_fvd_for_method, "y_true_tensor"):
        print("INFO: Loading and caching original videos for FVD...")
        original_video_folders = [os.path.join(original_videos_path, d) for d in sorted(os.listdir(original_videos_path)) if os.path.isdir(os.path.join(original_videos_path, d))]
        original_videos = [_load_video_from_folder(f) for f in tqdm(original_video_folders, desc="  -> Loading original")]
        evaluate_fvd_for_method.y_true_tensor = torch.stack([v for v in original_videos if v is not None])
        print(f"INFO: Cached {len(evaluate_fvd_for_method.y_true_tensor)} original videos.")

    y_true = evaluate_fvd_for_method.y_true_tensor
    
    # Load all stylized videos for the current method
    stylized_video_folders = [os.path.join(method_path, d) for d in sorted(os.listdir(method_path)) if os.path.isdir(os.path.join(method_path, d))]
    stylized_videos = [_load_video_from_folder(f) for f in tqdm(stylized_video_folders, desc=f"  -> Loading {method_name}")]
    y_pred = torch.stack([v for v in stylized_videos if v is not None])
    print(f"INFO: Loaded {y_pred.shape[0]} stylized videos for {method_name}.")

    # Repeat original videos to match the number of predicted videos
    num_pred, num_true = y_pred.shape[0], y_true.shape[0]
    if num_pred % num_true != 0:
        raise ValueError("Number of predicted videos must be a multiple of original videos.")
    num_repeats = num_pred // num_true
    y_true_expanded = y_true.repeat(num_repeats, 1, 1, 1, 1)

    # Calculate FVD
    fvd_score = compute_fvd(
        y_true=y_true_expanded, y_pred=y_pred, max_items=num_pred,
        device=device, batch_size=batch_size, local_model_path=local_model_path
    )
    return fvd_score