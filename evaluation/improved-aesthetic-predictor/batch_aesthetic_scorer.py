import os
import csv
import argparse
import traceback
from tqdm import tqdm

# Import our optimized predictor class
from aesthetic_scorer import AestheticPredictor

def run_batch_prediction(base_directory, mlp_model_path, output_file=None, batch_size=32):
    """
    Finds all subdirectories in a base directory and runs aesthetic prediction on them.
    """
    if output_file is None:
        output_file = os.path.join(base_directory, "aesthetic_scores.csv")
    
    # Find all subfolders to process
    subdirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    if not subdirs:
        print(f"Error: No subdirectories found in {base_directory}")
        return

    print(f"Found {len(subdirs)} folders to process.")

    try:
        # --- CORE OPTIMIZATION: Load models only ONCE ---
        print("Loading aesthetic predictor models (CLIP ViT-L/14 and MLP)...")
        predictor = AestheticPredictor(mlp_model_path=mlp_model_path)
        print("Models loaded successfully.")

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write the CSV header
            writer.writerow(["Folder", "Average_Score", "Max_Score", "Min_Score", "Image_Count"])
            
            progress_bar = tqdm(sorted(subdirs), desc="Processing folders")
            for folder in progress_bar:
                folder_name = os.path.basename(folder)
                
                try:
                    # Get the dictionary of scores for the folder
                    results = predictor.predict_folder_scores(folder, batch_size)
                    
                    if results:
                        writer.writerow([
                            folder_name,
                            f"{results['mean_score']:.4f}",
                            f"{results['max_score']:.4f}",
                            f"{results['min_score']:.4f}",
                            results['image_count']
                        ])
                    else:
                        writer.writerow([folder_name, "NO_IMAGES_FOUND", "N/A", "N/A", 0])

                except Exception as e:
                    tqdm.write(f"--- ERROR processing folder {folder_name} ---")
                    tqdm.write(traceback.format_exc())
                    writer.writerow([folder_name, "ERROR", "ERROR", "ERROR", 0])

        print("-" * 50)
        print(f"Processing complete. Results saved to: {output_file}")

    except Exception as e:
        print("\nAn unrecoverable error occurred during setup or file writing.")
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch calculate aesthetic scores for images in subfolders.")
    parser.add_argument("base_directory", type=str, help="The base directory containing subfolders of images.")
    parser.add_argument("--mlp_model_path", type=str, required=True, help="Path to the trained aesthetic MLP model (.pth file).")
    parser.add_argument("--output_file", type=str, default=None, help="Path for the output CSV file (optional).")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to process in a single batch.")
    
    args = parser.parse_args()
    
    run_batch_prediction(
        base_directory=args.base_directory,
        mlp_model_path=args.mlp_model_path,
        output_file=args.output_file,
        batch_size=args.batch_size
    )