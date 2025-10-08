# compare_scores.py
import os
import csv
import argparse
import traceback
from tqdm import tqdm

# 假設 AestheticPredictor 類在同目錄的 aesthetic_scorer.py 文件中
# Import our optimized predictor class
try:
    from aesthetic_scorer import AestheticPredictor
except ImportError:
    print("Error: 'aesthetic_scorer.py' not found.")
    print("Please make sure the file containing the AestheticPredictor class is in the same directory.")
    exit()

def compare_ablation_studies(ours_dir, baseline_dir, mlp_model_path, output_file, batch_size=32):
    """
    Compares the average aesthetic scores of corresponding subfolders 
    in two main directories (e.g., "Ours" vs "Baseline").
    """
    print("--- Ablation Study Comparison ---")
    print(f"Method 'Ours': {ours_dir}")
    print(f"Method 'Baseline': {baseline_dir}")
    print(f"Output CSV: {output_file}")
    
    # --- CORE OPTIMIZATION: Load models only ONCE ---
    try:
        print("\nLoading aesthetic predictor models (CLIP ViT-L/14 and MLP)...")
        predictor = AestheticPredictor(mlp_model_path=mlp_model_path)
        print("Models loaded successfully.")
    except Exception as e:
        print("\nFATAL ERROR: Could not load the prediction models.")
        print(traceback.format_exc())
        return

    # Find subfolders in the 'Ours' directory to use as the reference
    try:
        # We use sorted list to ensure a consistent order
        ours_subdirs = sorted([d for d in os.listdir(ours_dir) if os.path.isdir(os.path.join(ours_dir, d))])
        if not ours_subdirs:
            print(f"Error: No subdirectories found in the 'Ours' directory: {ours_dir}")
            return
        print(f"Found {len(ours_subdirs)} subfolders in the 'Ours' directory to compare.")
    except FileNotFoundError:
        print(f"Error: The 'Ours' directory was not found: {ours_dir}")
        return

    # --- Start Processing ---
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write the CSV header
            writer.writerow(["Folder", "Ours_Avg_Score", "Baseline_Avg_Score"])
            
            progress_bar = tqdm(ours_subdirs, desc="Comparing folders")
            for folder_name in progress_bar:
                ours_folder_path = os.path.join(ours_dir, folder_name)
                baseline_folder_path = os.path.join(baseline_dir, folder_name)
                
                ours_score_str = "N/A"
                baseline_score_str = "N/A"

                # Check if the corresponding baseline folder exists
                if not os.path.isdir(baseline_folder_path):
                    tqdm.write(f"Warning: Baseline folder '{folder_name}' not found. Skipping.")
                    baseline_score_str = "FOLDER_NOT_FOUND"
                else:
                    # Calculate baseline score
                    try:
                        results = predictor.predict_folder_scores(baseline_folder_path, batch_size)
                        if results and results['image_count'] > 0:
                            baseline_score_str = f"{results['mean_score']:.4f}"
                        else:
                            baseline_score_str = "NO_IMAGES"
                    except Exception as e:
                        tqdm.write(f"--- ERROR processing Baseline folder {folder_name} ---")
                        tqdm.write(traceback.format_exc())
                        baseline_score_str = "ERROR"
                
                # Calculate 'Ours' score
                try:
                    results = predictor.predict_folder_scores(ours_folder_path, batch_size)
                    if results and results['image_count'] > 0:
                        ours_score_str = f"{results['mean_score']:.4f}"
                    else:
                        ours_score_str = "NO_IMAGES"
                except Exception as e:
                    tqdm.write(f"--- ERROR processing Ours folder {folder_name} ---")
                    tqdm.write(traceback.format_exc())
                    ours_score_str = "ERROR"

                # Write the combined results to the CSV
                writer.writerow([folder_name, ours_score_str, baseline_score_str])
                
        print("-" * 50)
        print(f"Comparison complete. Results saved to: {output_file}")

    except Exception as e:
        print("\nAn unrecoverable error occurred during file processing or writing.")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare aesthetic scores between two sets of result directories for ablation studies.")
    
    parser.add_argument("--ours_dir", type=str, required=True, 
                        help="Path to the main directory of your method's results.")
    parser.add_argument("--baseline_dir", type=str, required=True, 
                        help="Path to the main directory of the baseline method's results.")
    parser.add_argument("--mlp_model_path", type=str, 
                        default="/root/autodl-tmp/video_style_transfer/evaluation/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
                        help="Path to the trained aesthetic MLP model (.pth file). Default is the pre-set path.")
    parser.add_argument("--output_file", type=str, default="ablation_comparison_scores.csv", 
                        help="Path for the output CSV file (optional).")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Number of images to process in a single batch.")
    
    args = parser.parse_args()
    
    compare_ablation_studies(
        ours_dir=args.ours_dir,
        baseline_dir=args.baseline_dir,
        mlp_model_path=args.mlp_model_path,
        output_file=args.output_file,
        batch_size=args.batch_size
    )