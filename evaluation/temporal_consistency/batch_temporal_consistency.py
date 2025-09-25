import os
import argparse
import csv
from temporal_consistency_unified import calculate_with_both_models
from tqdm import tqdm # 导入 tqdm

'''
使用示例:
python batch_temporal_consistency.py /path/to/your/base_directory
'''

def process_folders(base_dir, output_file=None):
    """
    处理基础目录中的所有子文件夹以计算时间一致性，并将结果保存为 CSV 文件。
    
    Args:
        base_dir: 包含帧序列子文件夹的基础目录
        output_file: 可选的输出文件路径。如果为 None，将在基础目录中创建一个 results.csv
    """
    # 如果未指定输出文件，则在基础目录中创建一个
    if output_file is None:
        output_file = os.path.join(base_dir, "temporal_consistency_results.csv")
    
    # 获取子目录列表
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
             if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"在 {base_dir} 中没有找到子目录")
        return
    
    print(f"找到 {len(subdirs)} 个文件夹待处理")
    
    # 打开输出文件以写入结果
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入CSV文件的表头
            writer.writerow(["Folder", "DINO Score", "CLIP Score", "Average Score"])
            
            # 使用 tqdm 包装循环以显示进度条
            # 对文件夹排序以保证结果顺序一致
            for folder in tqdm(sorted(subdirs), desc="正在处理文件夹"): 
                folder_name = os.path.basename(folder)
                
                # 使用 tqdm 时可以减少自定义的打印输出，因为进度条已经很清晰了
                # print("-" * 50) 
                
                try:
                    # 使用两种模型计算时间一致性
                    results = calculate_with_both_models(folder)
                    
                    # 将结果写入CSV文件
                    writer.writerow([
                        folder_name,
                        f"{results['dino']:.4f}",
                        f"{results['clip']:.4f}",
                        f"{results['average']:.4f}"
                    ])
                    
                    # (可选) 同样在每个单独的文件夹中保存结果
                    folder_result_file = os.path.join(folder, "temporal_consistency.txt")
                    with open(folder_result_file, 'w') as folder_f:
                        folder_f.write(f"Folder: {folder_name}\n")
                        folder_f.write(f"DINO Score: {results['dino']:.4f}\n")
                        folder_f.write(f"CLIP Score: {results['clip']:.4f}\n")
                        folder_f.write(f"Average Score: {results['average']:.4f}\n")
                
                except Exception as e:
                    error_msg = f"处理 {folder_name} 时出错: {str(e)}"
                    # 使用 tqdm.write 来打印，避免与进度条冲突
                    tqdm.write(error_msg) 
                    writer.writerow([folder_name, "ERROR", "ERROR", "ERROR"])
            
        print("-" * 50)
        print(f"\n处理完成。结果已保存到 {output_file}")

    except IOError as e:
        print(f"无法写入文件 {output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量计算多个文件夹的时间一致性并生成CSV报告')
    parser.add_argument('base_directory', type=str, help='包含帧序列子文件夹的基础目录')
    parser.add_argument('--output', type=str, default=None, help='CSV输出文件的路径 (可选)')
    
    args = parser.parse_args()
    
    process_folders(args.base_directory, args.output)