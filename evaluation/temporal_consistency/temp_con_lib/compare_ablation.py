import os
import csv
import argparse
import traceback
from tqdm import tqdm
import torch
from collections import defaultdict

# We reuse the library file from before, no changes needed there.
from temporal_consistency_lib import FeatureExtractor, calculate_temporal_consistency

def run_ablation_study(args):
    """
    Main function to run the ablation study comparison.
    Compares a 'baseline' against 'ours' using CLIP and DINOv3.
    """
    # 1. Define the methods and their corresponding directory paths
    method_paths = {
        "baseline": args.baseline,
        "ours": args.ours
    }

    # --- 2. Load only the required models (CLIP and DINOv3) ---
    extractors = {}
    try:
        print("Loading CLIP and DINOv3 models to GPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_to_load = ["CLIP", "DINOv3"]
        
        for model_name in tqdm(models_to_load, desc="Loading Models"):
            model_type_map = {"CLIP": "clip", "DINOv3": "dino3"}
            extractors[model_name] = FeatureExtractor(model_type=model_type_map[model_name], device=device)
        
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        print(traceback.format_exc())
        return

    # --- 3. Process all subfolders and calculate scores ---
    results = defaultdict(dict)
    
    for method_name, base_dir in method_paths.items():
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' does not exist. Skipping method '{method_name}'.")
            continue
            
        subdirs = [os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not subdirs:
            print(f"Warning: No subdirectories found in '{base_dir}'. Skipping method '{method_name}'.")
            continue

        print(f"\nProcessing method: {method_name} ({len(subdirs)} folders)")
        for folder_path in tqdm(subdirs, desc=f"Processing {method_name}"):
            folder_name = os.path.basename(folder_path)
            
            for model_name, extractor in extractors.items():
                try:
                    score = calculate_temporal_consistency(folder_path, extractor, args.batch_size)
                    results[folder_name][f"{method_name}_{model_name}"] = score
                except Exception:
                    tqdm.write(f"Error processing {folder_name} with {model_name}.")
                    results[folder_name][f"{method_name}_{model_name}"] = "ERROR"

    # --- 4. Write the results to a CSV file ---
    if not results:
        print("No results were collected. CSV file will not be generated.")
        return

    # Build the CSV header
    header = ["Folder", "baseline_CLIP", "baseline_DINOv3", "ours_CLIP", "ours_DINOv3"]

    try:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write data row by row, sorted by folder name
            for folder_name in sorted(results.keys()):
                row = [folder_name]
                for col_name in header[1:]: # Skip the "Folder" column title
                    score = results[folder_name].get(col_name, 'N/A')
                    if isinstance(score, float):
                        row.append(f"{score:.4f}")
                    else:
                        row.append(score)
                writer.writerow(row)
        
        print("-" * 50)
        print(f"\nAblation study complete! Results saved to: {args.output}")

    except IOError as e:
        print(f"Error: Could not write to file {args.output}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an ablation study comparing temporal consistency.')
    
    # Input directory arguments
    parser.add_argument('--baseline', type=str, required=True, help='Path to the directory with baseline result frames.')
    parser.add_argument('--ours', type=str, required=True, help='Path to the directory with "ours" result frames.')
    
    # Output and configuration arguments
    parser.add_argument('--output', type=str, default="ablation_study_results.csv", help='Path for the output CSV file.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images to process in a single batch.')
    
    args = parser.parse_args()
    run_ablation_study(args)