import os
import csv
import argparse
import traceback
from tqdm import tqdm
import torch
from collections import defaultdict

# 从我们更新后的库中导入所需模块
from temporal_consistency_lib import FeatureExtractor, calculate_temporal_consistency

def run_comparison(args):
    """
    主函数，用于执行跨方法、跨模型的时序一致性评估。
    """
    # 定义方法名称和对应的路径
    method_paths = {
        "ours": args.ours,
        "CCPL": args.ccpl,
        "MCCNet": args.mccnet,
        "CSBNet": args.csbnet,
    }

    # --- 1. 一次性加载所有模型 ---
    extractors = {}
    try:
        print("正在加载所有模型到 GPU... (此过程只需一次)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_to_load = ["DINOv2", "CLIP", "DINOv3"]
        
        for model_name in tqdm(models_to_load, desc="加载模型"):
            model_type_map = {"DINOv2": "dino", "CLIP": "clip", "DINOv3": "dino3"}
            extractors[model_name] = FeatureExtractor(model_type=model_type_map[model_name], device=device)
        
        print("所有模型加载完毕。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print(traceback.format_exc())
        return

    # --- 2. 收集所有子文件夹并计算分数 ---
    results = defaultdict(dict) # 使用 defaultdict 简化代码
    
    for method_name, base_dir in method_paths.items():
        if not os.path.isdir(base_dir):
            print(f"警告: 目录 '{base_dir}' 不存在，跳过方法 '{method_name}'")
            continue
            
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not subdirs:
            print(f"警告: 在 '{base_dir}' 中未找到子目录，跳过方法 '{method_name}'")
            continue

        print(f"\n正在处理方法: {method_name} ({len(subdirs)} 个文件夹)")
        for folder_path in tqdm(sorted(subdirs), desc=f"处理 {method_name}"):
            folder_name = os.path.basename(folder_path)
            
            for model_name, extractor in extractors.items():
                try:
                    score = calculate_temporal_consistency(folder_path, extractor, args.batch_size)
                    results[folder_name][f"{method_name}_{model_name}"] = score
                except Exception:
                    tqdm.write(f"处理 {folder_name} ({model_name}) 时出错")
                    results[folder_name][f"{method_name}_{model_name}"] = "ERROR"

    # --- 3. 将结果写入 CSV 文件 ---
    if not results:
        print("没有收集到任何结果，无法生成 CSV 文件。")
        return

    # 构建CSV表头
    header = ["Folder"]
    for method_name in method_paths.keys():
        for model_name in models_to_load:
            header.append(f"{method_name}_{model_name}")

    try:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # 按文件夹名称排序写入
            for folder_name in sorted(results.keys()):
                row = [folder_name]
                for col_name in header[1:]: # 跳过 "Folder" 列
                    score = results[folder_name].get(col_name, 'N/A')
                    # 格式化数字，保留错误信息
                    if isinstance(score, float):
                        row.append(f"{score:.4f}")
                    else:
                        row.append(score)
                writer.writerow(row)
        
        print("-" * 50)
        print(f"\n处理完成！结果已保存到: {args.output}")

    except IOError as e:
        print(f"错误: 无法写入文件 {args.output}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='比较多种视频风格化方法的时序一致性')
    
    # 输入目录参数
    parser.add_argument('--ours', type=str, required=True, help='包含 "ours" 方法结果帧的目录')
    parser.add_argument('--ccpl', type=str, required=True, help='包含 "CCPL" 方法结果帧的目录')
    parser.add_argument('--mccnet', type=str, required=True, help='包含 "MCCNet" 方法结果帧的目录')
    parser.add_argument('--csbnet', type=str, required=True, help='包含 "CSBNet" 方法结果帧的目录')
    
    # 输出和配置参数
    parser.add_argument('--output', type=str, default="temporal_consistency_comparison.csv", help='输出CSV文件的路径')
    parser.add_argument('--batch_size', type=int, default=16, help='每个GPU批次处理的图像数量 (8帧图片设8或16即可)')
    
    args = parser.parse_args()
    run_comparison(args)