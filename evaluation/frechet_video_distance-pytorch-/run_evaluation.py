import torch
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

# 假设 frechet_video_distance.py 和 pytorch_i3d_model 文件夹都在当前工作目录下
from frechet_video_distance import frechet_video_distance

# --- 1. 定义常量 ---
ORIGINAL_VIDEOS_PATH = "/root/autodl-tmp/video_style_transfer/data/data_quant/content"
METHOD_PATHS = {
    "Ours": "/root/autodl-tmp/video_style_transfer/results/quant_exp_results_batch_4",
    "Baseline_1_CCPL": "/root/autodl-tmp/quantitative_comparison_results/batch_4/CCPL",
    "Baseline_2_CSBNet": "/root/autodl-tmp/quantitative_comparison_results/batch_4/CSBNet",
    "Baseline_3_MCCNet": "/root/autodl-tmp/quantitative_comparison_results/batch_4/MCCNet"
}
PATH_TO_MODEL_WEIGHTS = "./pytorch_i3d_model/models/rgb_imagenet.pt" # 确保模型权重在此路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 # 根据你的GPU显存调整



# --- 2. 视频加载辅助函数  ---
def load_videos_from_path(path, num_frames=8, resolution=(512, 512)):
    """
    从指定路径加载视频帧序列。
    - 只会扫描子文件夹。
    - 同时支持 .png 和 .jpg 格式的帧。
    - 返回 float32 类型的张量。
    - 将8帧视频扩展到16帧以满足I3D模型要求。
    """
    video_folders = sorted([p for p in glob.glob(os.path.join(path, '*')) if os.path.isdir(p)])
    
    all_videos = []
    print(f"Loading videos from: {path}")
    if not video_folders:
         raise ValueError(f"No video folders found in path: {path}")
            
    for video_folder in tqdm(video_folders, desc=f"Loading videos"):
        png_files = glob.glob(os.path.join(video_folder, '*.png'))
        jpg_files = glob.glob(os.path.join(video_folder, '*.jpg'))
        frame_files = sorted(png_files + jpg_files)
        
        if len(frame_files) < num_frames:
            print(f"Warning: Folder {video_folder} has {len(frame_files)} frames (less than required {num_frames}), skipping.")
            continue
            
        frames = []
        for frame_path in frame_files[:num_frames]:
            img = Image.open(frame_path).convert('RGB')
            if img.size != resolution:
                img = img.resize(resolution)
            frames.append(np.array(img))
        
        video_tensor = np.stack(frames, axis=0) # Shape: (8, H, W, C)

        # 直接将视频帧复制一次
        video_tensor = np.concatenate([video_tensor, video_tensor], axis=0) # Shape: (16, H, W, C)

        all_videos.append(video_tensor)
        
    if not all_videos:
        raise ValueError(f"No valid videos with enough frames were loaded from: {path}")
        
    return torch.from_numpy(np.stack(all_videos, axis=0)).float()

# --- 3. 主执行逻辑 ---
def main():
    print(f"Using device: {DEVICE}")

    # --- 加载并准备原始视频 ---
    print("\n--- Loading Original Videos ---")
    try:
        y_true_tensor = load_videos_from_path(ORIGINAL_VIDEOS_PATH)
        print(f"Loaded {y_true_tensor.shape[0]} original videos.")
        
        # 扩展原始视频以匹配生成视频的数量
        # 假设每个原始视频对应10种风格
        num_styles = 10 
        y_true_expanded = y_true_tensor.repeat(num_styles, 1, 1, 1, 1)
        print(f"Expanded original videos from {y_true_tensor.shape[0]} to {y_true_expanded.shape[0]} to match stylized videos.")
        y_true_expanded = y_true_expanded.to(DEVICE)
    except Exception as e:
        print(f"Error loading original videos: {e}")
        return

    # --- 循环评估每种方法 ---
    results = {}
    for name, path in METHOD_PATHS.items():
        print(f"\n--- Evaluating method: {name} ---")
        try:
            # 加载风格化视频
            y_pred_tensor = load_videos_from_path(path)
            y_pred_tensor = y_pred_tensor.to(DEVICE)
            print(f"Loaded {y_pred_tensor.shape[0]} stylized videos.")
            
            if y_pred_tensor.shape[0] != y_true_expanded.shape[0]:
                print(f"Error: Mismatch in video count. Expected {y_true_expanded.shape[0]}, but found {y_pred_tensor.shape[0]}.")
                continue

            # 计算FVD
            print("Calculating FVD score...")
            fvd_score = frechet_video_distance(y_true_expanded, y_pred_tensor, PATH_TO_MODEL_WEIGHTS,device=DEVICE)
            results[name] = fvd_score
            print(f"FVD Score for {name}: {fvd_score:.4f}")

        except Exception as e:
            print(f"Failed to evaluate {name}. Error: {e}")
    
    # --- 总结结果 ---
    print("\n\n--- FVD Evaluation Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")
    print("---------------------------------")
    print("*Lower FVD scores are better.*")


if __name__ == "__main__":
    main()