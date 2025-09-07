import os
import argparse
import csv
# 确保这个模块在您的 Python 环境中是可用的
from temporal_consistency_unified import calculate_with_both_models

'''
示例命令:
python evaluation/temporal_consistency/folder_consistency.py trial
'''
def find_frame_folders(base_dir):
    """
    递归地查找所有包含图片帧的文件夹。
    
    参数:
        base_dir: 开始搜索的基础目录。
        
    返回:
        一个包含所有图片文件夹完整路径的列表。
    """
    frame_folders = []
    # 常见的图片文件后缀名
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    
    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录中是否有任何文件是图片
        if any(os.path.splitext(file)[1].lower() in image_extensions for file in files):
            frame_folders.append(root)
            
    return frame_folders

def process_folders(base_dir, output_file='base_dir/temporal_consistency_results.csv'):
    """
    处理基础目录中的所有子文件夹，计算时间一致性。
    
    参数:
        base_dir: 包含帧文件夹的基础目录。
        output_file: (可选) 输出的 CSV 文件路径。如果为 None, 则在基础目录中创建 results.csv。
    """
    # 如果未指定输出文件，则在基础目录中创建一个
    if output_file is None:
        output_file = os.path.join(base_dir, "temporal_consistency_results.csv")
    
    # 递归地获取所有包含图片帧的文件夹列表
    folders_to_process = find_frame_folders(base_dir)
    
    if not folders_to_process:
        print(f"在 {base_dir} 中没有找到包含图片的文件夹")
        return
        
    print(f"找到 {len(folders_to_process)} 个文件夹待处理")
    
    # 打开输出文件准备写入结果
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["Folder Path", "DINO Score", "CLIP Score", "Average Score"])
        
        # 处理每个文件夹
        for folder in sorted(folders_to_process): # 排序以保证结果顺序一致
            folder_name = os.path.basename(folder)
            # 打印完整路径，让用户知道当前进度
            print(f"\n正在处理: {os.path.abspath(folder)}")
            
            try:
                # 使用两个模型计算时间一致性
                results = calculate_with_both_models(folder)
                
                # 获取文件夹的绝对路径用于写入CSV
                full_folder_path = os.path.abspath(folder)
                
                # 将结果写入 CSV 文件
                writer.writerow([full_folder_path, f"{results['dino']:.4f}", f"{results['clip']:.4f}", f"{results['average']:.4f}"])
                
                # 同时，在每个图片文件夹内也保存一份结果（保留原始功能）
                folder_result_file = os.path.join(folder, "temporal_consistency.txt")
                with open(folder_result_file, 'w') as folder_f:
                    folder_f.write(f"Folder: {folder_name}\n")
                    folder_f.write(f"Path: {full_folder_path}\n")
                    folder_f.write(f"DINO Score: {results['dino']:.4f}\n")
                    folder_f.write(f"CLIP Score: {results['clip']:.4f}\n")
                    folder_f.write(f"Average Score: {results['average']:.4f}\n")
                    
            except Exception as e:
                error_msg = f"处理 {folder} 时出错: {str(e)}"
                print(error_msg)
                writer.writerow([os.path.abspath(folder), "ERROR", "ERROR", "ERROR"])
    
    print(f"\n处理完成。结果已保存至 {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='在一个基础目录中递归地为所有图片文件夹计算时间一致性。')
    parser.add_argument('base_directory', type=str, help='需要递归搜索图片文件夹的基础目录。')
    parser.add_argument('--output', type=str, default=None, help='输出的 CSV 文件路径。')
    
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.base_directory, 'temporal_consistency_results.csv')
    process_folders(args.base_directory, args.output)