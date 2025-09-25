import os
import csv
import argparse
import traceback
from tqdm import tqdm
import torch

# 从我们优化好的库中导入所需模块
from temporal_consistency_optimized import FeatureExtractor, calculate_temporal_consistency

def process_folders(base_dir, output_file=None, batch_size=64):
    """
    使用优化后的流程批量处理文件夹。
    """
    if output_file is None:
        output_file = os.path.join(base_dir, "temporal_consistency_results_optimized.csv")
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"在 {base_dir} 中没有找到子目录")
        return
    
    print(f"找到 {len(subdirs)} 个文件夹待处理")
    
    # --- 核心优化：在所有循环开始前加载模型 ---
    try:
        print("正在加载模型到 GPU... (此过程只需一次)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dino_extractor = FeatureExtractor(model_type="dino", device=device)
        clip_extractor = FeatureExtractor(model_type="clip", device=device)
        print("模型加载完毕。")
    except Exception as e:
        print("模型加载失败，请检查网络或PyTorch Hub配置。")
        print(traceback.format_exc())
        return

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Folder", "DINO Score", "CLIP Score", "Average Score"])
            
            # 使用tqdm包装循环
            for folder in tqdm(sorted(subdirs), desc="处理文件夹"):
                folder_name = os.path.basename(folder)
                
                try:
                    # --- 调用优化后的函数，并传入已加载的模型提取器 ---
                    dino_score = calculate_temporal_consistency(folder, dino_extractor, batch_size)
                    clip_score = calculate_temporal_consistency(folder, clip_extractor, batch_size)
                    avg_score = (dino_score + clip_score) / 2
                    
                    writer.writerow([
                        folder_name,
                        f"{dino_score:.4f}",
                        f"{clip_score:.4f}",
                        f"{avg_score:.4f}"
                    ])
                
                except Exception as e:
                    # 打印详细错误，帮助定位问题（例如空文件夹、坏图片等）
                    error_msg = f"处理文件夹 {folder_name} 时发生错误:"
                    tqdm.write(error_msg)
                    tqdm.write(traceback.format_exc())
                    writer.writerow([folder_name, "ERROR", "ERROR", "ERROR"])
            
        print("-" * 50)
        print(f"\n处理完成。结果已保存到 {output_file}")

    except IOError as e:
        print(f"无法写入文件 {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量、高效地计算多个文件夹的时间一致性')
    parser.add_argument('base_directory', type=str, help='包含帧序列子文件夹的基础目录')
    parser.add_argument('--output', type=str, default=None, help='CSV输出文件的路径 (可选)')
    parser.add_argument('--batch_size', type=int, default=32, help='每个GPU批次处理的图像数量')
    
    args = parser.parse_args()
    
    # 每个文件夹只有8张图，batch_size设为8或16即可
    # 如果你的图片数量更多，可以增大batch_size
    process_folders(args.base_directory, args.output, batch_size=args.batch_size)