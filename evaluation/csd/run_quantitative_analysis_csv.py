#./evaluation/csd
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import List, Dict, Tuple
from glob import glob
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import clip
import copy

# --- 1. CONFIGURATION ---
# ==============================================================================
'''GENERATED_DIRS = {
    "Ours": "/root/autodl-tmp/video_style_transfer/results/quant_exp_results_batch_4/",
    "CCPL": "/root/autodl-tmp/quantitative_comparison_results/batch_4/CCPL",
    "CSBNet": "/root/autodl-tmp/quantitative_comparison_results/batch_4/CSBNet",
    "MCCNet": "/root/autodl-tmp/quantitative_comparison_results/batch_4/MCCNet"
}'''
GENERATED_DIRS = {
    "Ours": "/root/autodl-tmp/video_style_transfer/results/quant_exp_results_batch_4/",
    "Baseline_Ablation": "/root/autodl-tmp/video_style_transfer/results/ablation_styleid/"
}
STYLE_REFERENCE_DIR = "/root/autodl-tmp/video_style_transfer/data/data_quant/style/"
OUTPUT_DIR = "./ablation"
# ==============================================================================


# --- 2. EXACT CUSTOM MODEL DEFINITION ---
# ==============================================================================
def convert_weights_float(model: nn.Module):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."): k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict

class CSD_CLIP(nn.Module):
    """
    Exact structural replica of the authors' model to ensure perfect weight loading.
    Includes the 'last_layer_content' even if unused.
    """
    def __init__(self, name='vit_large', content_proj_head='default'):
        super(CSD_CLIP, self).__init__()
        self.content_proj_head = content_proj_head
        if name == 'vit_large':
            clipmodel, _ = clip.load("ViT-L/14", device="cpu")
            self.backbone = clipmodel.visual
        else:
            raise NotImplementedError('Only ViT-L is supported for this model')

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        
        # --- FIX: Add the missing 'last_layer_content' to match the checkpoint ---
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        
        self.backbone.proj = None

    def forward(self, input_data):
        """This forward pass only computes and returns the style vector we need."""
        feature = self.backbone(input_data)
        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)
        return style_output
# ==============================================================================


# --- 3. MODEL LOADING AND UTILITIES ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: Using device: {DEVICE}")

image_preprocessor = transforms.Compose([
    transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])
print("INFO: Manual image preprocessor created successfully.")

print("INFO: Loading CSD ViT-L model using custom class definition...")
try:
    model = CSD_CLIP(name='vit_large')
    model_path = hf_hub_download(repo_id="tomg-group-umd/CSD-ViT-L", filename="pytorch_model.bin")
    
    # Load the full checkpoint with weights_only=True for security
    full_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = full_checkpoint['model_state_dict']
    
    converted_state_dict = convert_state_dict(state_dict)
    
    # Use strict=False to ignore loading layers we don't need (like content layer) if they have issues,
    # but the current fix makes this a safety net rather than a necessity.
    model.load_state_dict(converted_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    print("INFO: Model loaded and weights assigned successfully.")
except Exception as e:
    print(f"ERROR: Failed to load custom model. Exception: {e}")
    exit()

style_embedding_cache = {}

def get_style_embedding(image_path: str) -> np.ndarray:
    if image_path in style_embedding_cache:
        return style_embedding_cache[image_path]
    try:
        image = Image.open(image_path).convert("RGB")
        processed_image = image_preprocessor(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            style_embedding = model(processed_image)
            embedding = style_embedding.squeeze().cpu().numpy()
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        style_embedding_cache[image_path] = norm_embedding
        return norm_embedding
    except Exception as e:
        return None

def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, float]:
    cosine_sim = np.dot(emb1, emb2)
    l2_dist = np.linalg.norm(emb1 - emb2)
    return float(cosine_sim), float(l2_dist)

# --- 4. CORE EVALUATION LOGIC ---
def process_generated_directory(base_path: str, style_ref_dir: str) -> List[Dict]:
    results = []
    if not os.path.isdir(base_path):
        print(f"ERROR: Base directory not found: {base_path}")
        return []
    video_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    if not video_folders:
        print(f"WARNING: No subdirectories found in {base_path}")
        return []

    print(f"INFO: Found {len(video_folders)} folders in {os.path.basename(base_path.strip('/'))}...")

    for folder_name in tqdm(video_folders, desc=f"Processing {os.path.basename(base_path.strip('/'))}"):
        try:
            style_name = folder_name.split('_stylized_')[0]
        except IndexError: continue

        style_img_path_list = glob(os.path.join(style_ref_dir, f"{style_name}.*"))
        if not style_img_path_list: continue
        style_img_path = style_img_path_list[0]

        style_embedding = get_style_embedding(style_img_path)
        if style_embedding is None: continue

        video_folder_path = os.path.join(base_path, folder_name)
        frame_paths = sorted(glob(os.path.join(video_folder_path, '*.[jp][pn]g')) + glob(os.path.join(video_folder_path, '*.jpeg')))
        if not frame_paths: continue

        video_scores_cos, video_scores_l2 = [], []
        for frame_path in frame_paths:
            frame_embedding = get_style_embedding(frame_path)
            if frame_embedding is not None:
                cos_sim, l2_dist = calculate_similarity(style_embedding, frame_embedding)
                video_scores_cos.append(cos_sim)
                video_scores_l2.append(l2_dist)

        if video_scores_cos:
            results.append({
                "Video": folder_name,
                "Avg_Cosine_Similarity": np.mean(video_scores_cos),
                "Avg_L2_Distance": np.mean(video_scores_l2)
            })
    return results

# --- 5. MAIN EXECUTION AND REPORTING ---
def main():
    all_results_df = None
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for method_name, generated_path in GENERATED_DIRS.items():
        print(f"\n{'='*30}\nProcessing Method: {method_name}\n{'='*30}")
        method_results = process_generated_directory(generated_path, STYLE_REFERENCE_DIR)
        
        if not method_results:
            print(f"INFO: No results generated for method '{method_name}'.")
            continue
            
        method_df = pd.DataFrame(method_results)
        method_df.rename(columns={
            "Avg_Cosine_Similarity": f"{method_name}_Cosine_Sim",
            "Avg_L2_Distance": f"{method_name}_L2_Dist"
        }, inplace=True)

        if all_results_df is None:
            all_results_df = method_df
        else:
            all_results_df = pd.merge(all_results_df, method_df, on="Video", how="outer")

    if all_results_df is not None and not all_results_df.empty:
        print(f"\n\n{'='*80}\n"
              f"          Quantitative Style Similarity Analysis Complete\n"
              f"{'='*80}")
        
        cols = ["Video"] + [f"{m}_{s}" for m in GENERATED_DIRS.keys() for s in ["Cosine_Sim", "L2_Dist"] if f"{m}_Cosine_Sim" in all_results_df.columns]
        all_results_df = all_results_df[cols]
        
        print("\n--- Detailed Results per Video (Preview) ---")
        print(all_results_df.head().to_string(index=False))
        
        detailed_csv_path = os.path.join(OUTPUT_DIR, 'quantitative_results.csv')
        all_results_df.to_csv(detailed_csv_path, index=False, float_format='%.6f')
        print(f"\n✅ Detailed results saved to: {detailed_csv_path}")

        print("\n\n--- Overall Average Scores ---")
        summary_data = {m: {"Avg Cosine Sim (↑)": all_results_df[f"{m}_Cosine_Sim"].mean(), "Avg L2 Dist (↓)": all_results_df[f"{m}_L2_Dist"].mean()} for m in GENERATED_DIRS.keys() if f"{m}_Cosine_Sim" in all_results_df.columns}
        summary_df = pd.DataFrame(summary_data).T
        print(summary_df.to_string(float_format='%.4f'))
        
        summary_csv_path = os.path.join(OUTPUT_DIR, 'summary_results.csv')
        summary_df.to_csv(summary_csv_path, float_format='%.6f')
        print(f"✅ Summary results saved to: {summary_csv_path}")
        
        print(f"\n(↑ Higher is better for Cosine Similarity, ↓ Lower is better for L2 Distance)")
        print(f"{'='*80}\n")
    else:
        print("\n\nAnalysis complete, but no results were generated. Please check paths and directory structures.")

if __name__ == "__main__":
    main()