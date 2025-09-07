import os
import argparse
from temporal_consistency_unified import calculate_with_both_models

'''python evaluation/temporal_consistency/batch_temporal_consistency.py trial/foe'''

def process_folders(base_dir, output_file=None):
    """
    Process all subfolders in the base directory to calculate temporal consistency.
    
    Args:
        base_dir: Base directory containing subfolders with frames
        output_file: Optional output file path. If None, will create a results.txt in the base_dir
    """
    # If no output file is specified, create one in the base directory
    if output_file is None:
        output_file = os.path.join(base_dir, "temporal_consistency_results.txt")
    
    # Get list of subdirectories
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        return
    
    print(f"Found {len(subdirs)} folders to process")
    
    # Open the output file for writing results
    with open(output_file, 'w') as f:
        f.write("Folder\tDINO Score\tCLIP Score\tAverage Score\n")
        f.write("-" * 70 + "\n")
        
        # Process each folder
        for folder in subdirs:
            folder_name = os.path.basename(folder)
            print(f"\nProcessing: {folder_name}")
            
            try:
                # Calculate temporal consistency with both models
                results = calculate_with_both_models(folder)
                
                # Write results to the file
                f.write(f"{folder_name}\t{results['dino']:.4f}\t\t{results['clip']:.4f}\t\t{results['average']:.4f}\n")
                
                # Also save results in the individual folder
                folder_result_file = os.path.join(folder, "temporal_consistency.txt")
                with open(folder_result_file, 'w') as folder_f:
                    folder_f.write(f"Folder: {folder_name}\n")
                    folder_f.write(f"DINO Score: {results['dino']:.4f}\n")
                    folder_f.write(f"CLIP Score: {results['clip']:.4f}\n")
                    folder_f.write(f"Average Score: {results['average']:.4f}\n")
                
            except Exception as e:
                error_msg = f"Error processing {folder_name}: {str(e)}"
                print(error_msg)
                f.write(f"{folder_name}\tERROR\tERROR\tERROR\n")
    
    print(f"\nProcessing complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate temporal consistency for multiple folders')
    parser.add_argument('base_directory', type=str, help='Base directory containing subfolders with frames')
    parser.add_argument('--output', type=str, default=None, help='Output file path (optional)')
    
    args = parser.parse_args()
    
    process_folders(args.base_directory, args.output) 