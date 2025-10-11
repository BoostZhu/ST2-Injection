#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_full_evaluation.py

A unified script for comprehensive quantitative evaluation of video style transfer methods,
adapted to work with the user's specific directory structure.
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import warnings

# --- Dynamic Path Insertion to Match Your File Structure ---
# This makes the script find your library files without moving them.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'temporal_consistency/temp_con_lib'))
sys.path.insert(0, os.path.join(script_dir, 'improved-aesthetic-predictor'))
sys.path.insert(0, os.path.join(script_dir, 'csd'))
sys.path.insert(0, os.path.join(script_dir, 'frechet_video_distance'))

# --- Import from your existing and new library files ---
from temporal_consistency_lib import FeatureExtractor, calculate_temporal_consistency
from aesthetic_scorer import AestheticPredictor
from csd_lib import CSDScorer
from fvd_lib import evaluate_fvd_for_method

warnings.filterwarnings("ignore", message="xFormers is not available")

def main(args):
    """Main orchestration function."""
    print("="*60 + "\n🚀 Starting Unified Quantitative Evaluation Script 🚀\n" + "="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Using device: {DEVICE}")

    method_paths = {"Ours": args.ours_dir}
    for d in os.listdir(args.baselines_dir):
        path = os.path.join(args.baselines_dir, d)
        if os.path.isdir(path) and d != "quantitative_results": 
            method_paths[d] = path

    if args.ablation_dir:
        if os.path.isdir(args.ablation_dir):
            # We'll use the folder's name (e.g., "ablation_styleid") as the method name
            ablation_name = os.path.basename(args.ablation_dir)
            method_paths[ablation_name] = args.ablation_dir
            print(f"INFO: Added ablation study '{ablation_name}' to the evaluation.")
        else:
            print(f"[WARN] Ablation directory provided but not found: {args.ablation_dir}")
    print(f"INFO: Found {len(method_paths)} methods to evaluate: {list(method_paths.keys())}")
    
    print("\n--- Loading all required models... ---")
    try:
        tc_clip_extractor = FeatureExtractor(model_type="clip", device=DEVICE)
        tc_dino_extractor = FeatureExtractor(model_type="dino3", device=DEVICE)
        aesthetic_predictor = AestheticPredictor(mlp_model_path=args.aesthetic_model_path)
        csd_scorer = CSDScorer(device=DEVICE)
        print("✅ All non-FVD models loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load models. Aborting. Error: {e}"); return

    print("\n--- Calculating per-video metrics (TC, Aesthetic, CSD)... ---")
    all_video_results = []
    for method_name, method_path in method_paths.items():
        print(f"\nProcessing Method: {method_name}")
        video_folders = [d for d in sorted(os.listdir(method_path)) if os.path.isdir(os.path.join(method_path, d))]
        
        for folder_name in tqdm(video_folders, desc=f"  -> {method_name}", unit="video"):
            folder_path = os.path.join(method_path, folder_name)
            result_row = {"Method": method_name, "Video": folder_name}
            try:
                result_row["TC_CLIP"] = calculate_temporal_consistency(folder_path, tc_clip_extractor, args.batch_size)
                result_row["TC_DINOv3"] = calculate_temporal_consistency(folder_path, tc_dino_extractor, args.batch_size)
                scores = aesthetic_predictor.predict_folder_scores(folder_path, args.batch_size)
                result_row["Aesthetic"] = scores['mean_score'] if scores else 0.0
                cos_sim, l2_dist = csd_scorer.calculate_style_similarity_for_folder(folder_path, args.style_dir)
                result_row["CSD_CosSim"] = cos_sim
                result_row["CSD_L2"] = l2_dist
            except Exception as e:
                tqdm.write(f"  [WARN] Failed on '{folder_name}' with method '{method_name}'. Error: {e}")
                for key in ["TC_CLIP", "TC_DINOv3", "Aesthetic", "CSD_CosSim", "CSD_L2"]:
                    if key not in result_row: result_row[key] = "ERROR"
            all_video_results.append(result_row)
            
    results_df = pd.DataFrame(all_video_results)
    
    fvd_scores = {}
    for method_name, method_path in method_paths.items():
        try:
            score = evaluate_fvd_for_method(method_name, method_path, args.content_dir, DEVICE, args.batch_size, args.i3d_model_path)
            fvd_scores[method_name] = score
        except Exception as e:
            print(f"  [ERROR] FVD failed for {method_name}: {e}"); fvd_scores[method_name] = float('inf')

# --- 5. Final Reporting and Saving (CORRECTED) ---
    print("\n" + "="*60)
    print("📊 Evaluation Complete. Generating Reports. 📊")
    print("="*60)
    
    if not all_video_results:
        print("WARNING: No results were collected, cannot generate a report.")
        return

    # --- Step A: Generate the Detailed Per-Video Wide Report ---
    try:
        # Create the pivot table for per-video scores
        wide_df = results_df.pivot_table(
            index='Video', 
            columns='Method', 
            values=["TC_DINOv3", "TC_CLIP", "CSD_CosSim", "CSD_L2", "Aesthetic"]
        )
        
        # Reorder columns for consistency
        method_order = [m for m in method_paths.keys() if m in wide_df.columns.get_level_values(1)]
        metric_order = ["TC_DINOv3", "TC_CLIP", "CSD_CosSim", "CSD_L2", "Aesthetic"]
        wide_df = wide_df.reindex(columns=metric_order, level=0)
        wide_df = wide_df.reindex(columns=method_order, level=1)

        # Calculate and append the average row for per-video metrics
        averages_per_video_metrics = wide_df.mean(numeric_only=True)
        averages_per_video_metrics.name = 'Average'
        wide_df.loc['Average'] = averages_per_video_metrics
        
        # Save the detailed wide-format report
        output_wide_path = os.path.join(args.output_dir, "comparison_report_wide.csv")
        wide_df.to_csv(output_wide_path, float_format="%.4f")
        
        print(f"\n✅ Detailed per-video report saved successfully.")
        print(f"   -> File: {output_wide_path}")

    except Exception as e:
        print(f"❌ ERROR: Failed to create the wide-format report: {e}")

    # --- Step B: Generate the Final Summary Report (with FVD) ---
    try:
        # Use the calculated averages as the base for our summary
        # .unstack() transforms the multi-level series into a clean DataFrame
        summary_df = averages_per_video_metrics.unstack(level=0)
        
        # Now, add the FVD scores to this summary table
        # .map() correctly aligns the fvd_scores dict with the method names in the index
        summary_df['FVD'] = summary_df.index.map(fvd_scores)
        
        # Add direction arrows for clarity (↑ Higher is better, ↓ Lower is better)
        summary_df.rename(columns={
            "TC_DINOv3": "TC_DINOv3 (↑)", 
            "TC_CLIP": "TC_CLIP (↑)", 
            "CSD_CosSim": "CSD_CosSim (↑)",
            "CSD_L2": "CSD_L2 (↓)", 
            "Aesthetic": "Aesthetic (↑)", 
            "FVD": "FVD (↓)"
        }, inplace=True)

        print("\n--- Overall Average Scores per Method (including FVD) ---")
        print(summary_df.to_string(float_format="%.4f"))
        
        # Save the final summary report
        output_summary_path = os.path.join(args.output_dir, "all_metrics_summary.csv")
        summary_df.to_csv(output_summary_path, float_format="%.4f")
        
        print(f"\n✅ Final summary report (with FVD) saved successfully.")
        print(f"   -> File: {output_summary_path}")

    except Exception as e:
        print(f"❌ ERROR: Failed to create the final summary report: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for video style transfer evaluation.")
    parser.add_argument('--baselines_dir', type=str, required=True, help='Directory of baseline method folders (e.g., CCPL).')
    parser.add_argument('--ours_dir', type=str, required=True, help='Directory for your method\'s results.')
    parser.add_argument('--content_dir', type=str, required=True, help='Directory with original content video frames (for FVD).')
    parser.add_argument('--style_dir', type=str, required=True, help='Directory with style reference images (for CSD).')
    parser.add_argument('--output_dir', type=str, default="./quant_results/", help='Directory to save the final CSV reports.')
    parser.add_argument('--aesthetic_model_path', type=str, required=True, help='Path to the aesthetic MLP model (.pth).')
    parser.add_argument('--i3d_model_path', type=str, required=True, help='Path to the I3D model for FVD (.pt).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for GPU calculations.')
    parser.add_argument('--ablation_dir', type=str, required=False, help='(Optional) Path to the ablation study results directory.')
    args = parser.parse_args()
    main(args)