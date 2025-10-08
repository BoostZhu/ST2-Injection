import os
import argparse
import traceback
from tqdm import tqdm
import torch
import pandas as pd
from collections import defaultdict

# 我们的库文件保持不变，直接复用
from style_similarity_lib import FeatureExtractor, calculate_style_similarity

def run_ablation_comparison(methods, style_dir, output_path, batch_size):
    """
    专门用于运行消融实验对比的脚本。
    """
    print("开始消融实验风格相似度评估...")
    
    # --- 1. 一次性加载所有模型 (DINOv2, CLIP, DINOv3) ---
    extractors = {}
    try:
        print("正在加载模型到 GPU... (此过程可能需要一些时间)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractors["dino"] = FeatureExtractor(model_type="dino", device=device)
        extractors["clip"] = FeatureExtractor(model_type="clip", device=device)
        extractors["dinov3"] = FeatureExtractor(model_type="dinov3", device=device)
    except Exception as e:
        print("模型加载失败，请检查网络或库配置。")
        print(traceback.format_exc())
        return

    # --- 2. 准备数据结构和文件夹列表 ---
    results_data = defaultdict(dict)
    
    # 以 'Ours' 方法的文件夹为基准，获取所有视频文件夹的名称
    base_method_name = "Ours"
    base_method_path = methods[base_method_name]
    try:
        video_folders = sorted([d for d in os.listdir(base_method_path) if os.path.isdir(os.path.join(base_method_path, d))])
        print(f"找到 {len(video_folders)} 个视频文件夹作为基准进行对比。")
    except FileNotFoundError:
        print(f"错误：基准方法 '{base_method_name}' 的路径不存在: {base_method_path}")
        return

    # --- 3. 遍历两个方法 (Ours, Baseline_Ablation) 和视频文件夹进行计算 ---
    for method_name, method_path in methods.items():
        print(f"\n--- 正在处理方法: {method_name} ---")
        if not os.path.isdir(method_path):
            print(f"警告：方法 '{method_name}' 的路径不存在: {method_path}，将跳过此方法。")
            continue
            
        for folder in tqdm(video_folders, desc=f"计算 {method_name}"):
            frames_folder_path = os.path.join(method_path, folder)
            
            # 使用 '_stylized_' 分割逻辑来找到风格图片名称
            style_image_name_base = folder.split('_stylized_')[0]

            possible_exts = ['.png', '.jpg', '.jpeg', '.webp']
            style_image_path = None
            for ext in possible_exts:
                path = os.path.join(style_dir, style_image_name_base + ext)
                if os.path.exists(path):
                    style_image_path = path
                    break
            
            if not style_image_path:
                tqdm.write(f"警告: 找不到文件夹 '{folder}' 对应的风格图片 '{style_image_name_base}', 跳过。")
                results_data[folder][f'{method_name}_DINO_Score'] = 'N/A'
                results_data[folder][f'{method_name}_CLIP_Score'] = 'N/A'
                results_data[folder][f'{method_name}_DINOv3_Score'] = 'N/A'
                continue

            try:
                dino_score = calculate_style_similarity(frames_folder_path, style_image_path, extractors["dino"], batch_size)
                clip_score = calculate_style_similarity(frames_folder_path, style_image_path, extractors["clip"], batch_size)
                dinov3_score = calculate_style_similarity(frames_folder_path, style_image_path, extractors["dinov3"], batch_size)

                results_data[folder][f'{method_name}_DINO_Score'] = f"{dino_score:.4f}"
                results_data[folder][f'{method_name}_CLIP_Score'] = f"{clip_score:.4f}"
                results_data[folder][f'{method_name}_DINOv3_Score'] = f"{dinov3_score:.4f}"
            
            except Exception as e:
                tqdm.write(f"处理文件夹 {frames_folder_path} 时发生错误: {e}")
                results_data[folder][f'{method_name}_DINO_Score'] = 'ERROR'
                results_data[folder][f'{method_name}_CLIP_Score'] = 'ERROR'
                results_data[folder][f'{method_name}_DINOv3_Score'] = 'ERROR'

    # --- 4. 将结果写入CSV文件 ---
    df = pd.DataFrame.from_dict(results_data, orient='index')
    df.index.name = 'Folder'
    df = df.sort_index()
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    df.to_csv(output_path)
    print("-" * 50)
    print(f"\n处理完成！消融实验对比结果已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为消融实验批量计算风格相似度并生成对比报告')
    
    parser.add_argument('--style_dir', type=str, required=True, help='包含所有风格参考图片的文件夹路径')
    parser.add_argument('--output_path', type=str, default='./ablation_style_similarity.csv', help='最终生成的CSV报告的路径')
    parser.add_argument('--batch_size', type=int, default=16, help='每个GPU批次处理的图像数量')
    
    args = parser.parse_args()

    # --- 核心修改：定义要对比的两个文件夹 ---
    method_directories = {
        "Ours": "/root/autodl-tmp/video_style_transfer/results/quant_exp_results_batch_4",
        "Baseline_Ablation": "/root/autodl-tmp/video_style_transfer/results/ablation_styleid"
    }
    
    run_ablation_comparison(
        methods=method_directories,
        style_dir=args.style_dir,
        output_path=args.output_path,
        batch_size=args.batch_size
    )