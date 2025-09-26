import os
import glob
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


from calculate_fvd import compute_fvd

# --- 辅助函数：从帧文件夹加载视频 ---
def load_video_from_folder(folder_path, image_size=(512, 512)):
    """
    从给定的文件夹路径中加载所有帧，并将它们堆叠成一个视频张量。

    Args:
        folder_path (str): 包含视频帧的文件夹路径。
        image_size (tuple): 图像的目标尺寸。

    Returns:
        torch.Tensor: 一个形状为 (C, T, H, W) 的视频张量。
    """
    # 查找所有图片文件并按名称排序，确保帧顺序正确
    frame_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        img = img.resize(image_size)
        # 转换为 (H, W, C) 的 numpy 数组
        frames.append(np.array(img))
    frames = frames * 2
    # 堆叠成 (T, H, W, C)
    video_array = np.stack(frames, axis=0)
    
    # 转换为 PyTorch 张量并调整维度为 (C, T, H, W)
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2)
    

    return video_tensor.float()

# --- 主评估函数 ---
def evaluate_method(method_name, method_path, original_videos_tensor, device, batch_size=8, local_model_path=None):
    """
    评估单个方法的FVD分数。

    Args:
        method_name (str): 方法的名称 (e.g., "Ours")。
        method_path (str): 该方法生成的视频所在的根目录。
        original_videos_tensor (torch.Tensor): 形状为 (N_orig, C, T, H, W) 的原始视频张量。
        device (torch.device): 计算设备 (e.g., "cuda")。
        batch_size (int): FVD计算的批处理大小。

    Returns:
        float: 计算出的FVD分数。
    """
    print(f"--- Evaluating method: {method_name} ---")
    print(f"Loading stylized videos from: {method_path}")

    # 获取该方法生成的所有视频文件夹路径
    stylized_video_folders = [os.path.join(method_path, d) for d in os.listdir(method_path) if os.path.isdir(os.path.join(method_path, d))]
    
    # 加载所有风格化视频
    stylized_videos = []
    for folder in tqdm(stylized_video_folders, desc=f"Loading videos for {method_name}"):
        stylized_videos.append(load_video_from_folder(folder))
    
    # y_pred: 将视频列表堆叠成一个大的张量
    y_pred = torch.stack(stylized_videos)
    print(f"Loaded {y_pred.shape[0]} stylized videos.")

    # y_true: 这是关键！扩展原始视频张量以匹配生成视频的数量
    y_true = original_videos_tensor
    num_pred_videos = y_pred.shape[0]
    num_true_videos = y_true.shape[0]

    if num_pred_videos % num_true_videos != 0:
        raise ValueError("Number of predicted videos must be a multiple of the number of original videos.")
    
    num_repeats = num_pred_videos // num_true_videos
    y_true_expanded = y_true.repeat(num_repeats, 1, 1, 1, 1)
    print(f"Expanded original videos from {num_true_videos} to {y_true_expanded.shape[0]} to match.")

    # 计算FVD
    print("Calculating FVD score...")
    fvd_score = compute_fvd(
        y_true=y_true_expanded,
        y_pred=y_pred,
        max_items=num_pred_videos, # 使用所有视频
        device=device,
        batch_size=batch_size,
        local_model_path=local_model_path
    )
    
    print(f"FVD Score for {method_name}: {fvd_score:.4f}\n")
    return fvd_score


if __name__ == "__main__":
    # --- 1. 配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    LOCAL_I3D_MODEL_PATH = "/root/autodl-tmp/video_style_transfer/evaluation/frechet_video_distance/loaded_models/i3d_torchscript.pt"
    
    ORIGINAL_VIDEOS_PATH = "/root/autodl-tmp/video_style_transfer/data/data_quant/content" 
    
    
    METHOD_PATHS = {
        "Ours": "/root/autodl-tmp/video_style_transfer/results/quant_exp_results_batch_4", 
        "Baseline_1": "/root/autodl-tmp/quantitative_comparison_results/batch_4/CCPL", 
        "Baseline_2": "/root/autodl-tmp/quantitative_comparison_results/batch_4/CSBNet", 
        "Baseline_3": "/root/autodl-tmp/quantitative_comparison_results/batch_4/MCCNet"
    }
    
    BATCH_SIZE = 32 # 根据你的GPU显存大小调整，如果显存不足，可以调小

    # --- 2. 加载原始视频 (只加载一次) ---
    print("--- Loading Original Videos ---")
    original_video_folders = [os.path.join(ORIGINAL_VIDEOS_PATH, d) for d in os.listdir(ORIGINAL_VIDEOS_PATH) if os.path.isdir(os.path.join(ORIGINAL_VIDEOS_PATH, d))]
    
    original_videos = []
    for folder in tqdm(original_video_folders, desc="Loading original videos"):
        original_videos.append(load_video_from_folder(folder))
    
    # 制作 y_true 张量
    y_true_tensor = torch.stack(original_videos)
    print(f"Loaded {y_true_tensor.shape[0]} original videos.\n")

    # --- 3. 循环评估每个方法 ---
    results = {}
    for name, path in METHOD_PATHS.items():
        if not os.path.exists(path):
            print(f"Warning: Path not found for {name}: {path}. Skipping.")
            continue
        score = evaluate_method(name, path, y_true_tensor, DEVICE, BATCH_SIZE, local_model_path=LOCAL_I3D_MODEL_PATH)
        results[name] = score
    # --- 4. 打印最终结果 ---
    print("--- Final FVD Results ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")