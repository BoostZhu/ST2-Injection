#./run_styleid_v2v_batch.py (最终简化版)
import os
import argparse
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import random
import itertools
from tqdm import tqdm
from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline

def run_style_transfer_for_pair(pipe, content_name, style_name, args):
    """
    对单个内容-风格对执行风格迁移。
    此版本将所有预处理（裁切、缩放）完全交给Pipeline处理。
    """
    try:
        # 1. 根据命名规则，构建预期的输出文件夹路径
        style_name_base = os.path.splitext(os.path.basename(style_name))[0]
        output_dir_name = f"{style_name_base}_stylized_{content_name}"
        output_path = os.path.join(args.output_dir, output_dir_name)

        # 2. 检查该路径是否存在且是一个文件夹，如果存在则跳过
        if os.path.isdir(output_path):
            tqdm.write(f"结果已存在，跳过: {output_dir_name}")
            return

        tqdm.write("-" * 50)
        tqdm.write(f"开始处理: 内容='{content_name}', 风格='{style_name}'")

        content_folder_path = os.path.join(args.data_root, 'content', content_name)
        style_image_path = os.path.join(args.data_root, 'style', style_name)

        if not os.path.isdir(content_folder_path):
            tqdm.write(f"错误: 内容文件夹未找到 '{content_folder_path}', 跳过。")
            return
        if not os.path.isfile(style_image_path):
            tqdm.write(f"错误: 风格图未找到 '{style_image_path}', 跳过。")
            return

        frame_paths = sorted(glob.glob(os.path.join(content_folder_path, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if not frame_paths:
            tqdm.write(f"警告: 文件夹 '{content_folder_path}' 中没有找到 .jpg 格式的帧, 跳过。")
            return

        # --- 简化的预处理流程 ---
        # 只读取原始图像数据，不进行任何裁切或缩放
        content_frames = []
        for p in frame_paths:
            frame = cv2.imread(p)
            if frame is not None:
                content_frames.append(frame) # 直接添加原始的BGR NumPy数组

        if not content_frames:
            tqdm.write(f"错误: 无法从 '{content_folder_path}' 读取任何有效帧, 跳过。")
            return
            
        style_image = cv2.imread(style_image_path)
        if style_image is None:
            tqdm.write(f"错误: 无法加载风格图 '{style_image_path}', 跳过。")
            return
        
        # --- 调用核心流程 ---
        # Pipeline现在会自己处理所有尺寸问题
        result = pipe.style_transfer_video(
            content_frames=content_frames,
            style_image=style_image,
            num_inference_steps=args.ddim_steps,
            gamma=args.gamma,
            temperature=args.temperature,
            mask_strength=args.mask_strength,
            without_init_adain=args.without_init_adain,
        )
        
        output_frames = result['images']
        
        # --- 保存结果 ---
        os.makedirs(output_path, exist_ok=True)
        
        for i, frame_pil in enumerate(output_frames):
            frame_path = os.path.join(output_path, f"{i+1:04d}.png")
            frame_pil.save(frame_path)
            
        tqdm.write(f"成功: '{content_name}' 与 '{style_name}' 处理完毕, 已保存到 {output_path}")

    except Exception as e:
        tqdm.write(f"\n处理 '{content_name}' 与 '{style_name}' 时发生严重错误: {e}")
        import traceback
        tqdm.write(traceback.format_exc())
        tqdm.write("将继续处理下一对。\n")

def main(args):
    content_root = os.path.join(args.data_root, 'content')
    style_root = os.path.join(args.data_root, 'style')
    if not os.path.isdir(content_root): raise FileNotFoundError(f"内容数据根目录未找到: {content_root}")
    if not os.path.isdir(style_root): raise FileNotFoundError(f"风格数据根目录未找到: {style_root}")
    
    all_content_names = [d for d in os.listdir(content_root) if os.path.isdir(os.path.join(content_root, d))]
    all_style_names_png = glob.glob(os.path.join(style_root, '*.png'))
    all_style_names_jpg = glob.glob(os.path.join(style_root, '*.jpg'))
    all_style_names_jpeg = glob.glob(os.path.join(style_root, '*.jpeg'))
    all_style_names = all_style_names_png + all_style_names_jpg + all_style_names_jpeg

    if not all_content_names or not all_style_names: 
        print("错误: 未找到任何内容或风格文件。")
        return
        
    job_pairs = list(itertools.product(all_content_names, all_style_names))
    
    print("="*50)
    print(f"发现 {len(all_content_names)} 个内容视频。")
    print(f"发现 {len(all_style_names)} 个风格图像。")
    print(f"总共将执行 {len(job_pairs)} 次风格迁移任务。")
    print("="*50)
    
    print(f"正在从 '{args.model_path}' 加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StyleIDVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    print("模型加载完毕，管线已初始化。")
    
    print("\n开始批量处理所有任务...")
    for content_name, style_name in tqdm(job_pairs, desc="总体进度"):
        run_style_transfer_for_pair(pipe, content_name, os.path.basename(style_name), args)
        
    print("\n所有风格迁移任务已全部完成！")

if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(description="使用StyleIDVideoPipeline进行批量视频风格迁移")
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录，应包含 'content' 和 'style' 子目录")
    parser.add_argument("--output_dir", type=str, default="./results", help="保存结果的根目录")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="预训练的Stable Diffusion模型路径或HuggingFace名称")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM反转和采样的步数")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query保留强度")
    parser.add_argument("--temperature", type=float, default=1.5, help="注意力温度系数")
    parser.add_argument("--mask_strength", type=float, default=1.0, help="PA Fusion的融合强度")
    parser.add_argument("--without_init_adain", action="store_true", help="禁用初始latent的AdaIN操作")
    args = parser.parse_args()
    main(args)