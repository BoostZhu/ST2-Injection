#./run_styleid_v2v_batch.py (Corrected Version)
import os
import argparse
import glob
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
import random

from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline

def run_style_transfer_for_pair(pipe, content_name, style_name, args):
    """
    对单个内容-风格对执行风格迁移。
    这个函数包含了原 `main` 函数的核心逻辑。

    Args:
        pipe: 预先加载的 StyleIDVideoPipeline 对象。
        content_name (str): 内容视频的文件夹名称。
        style_name (str): 风格图像的文件名。
        args: 从 argparse 解析的参数。
    """
    try:
        print("-" * 50)
        print(f"开始处理: 内容='{content_name}', 风格='{style_name}'")

        # --- 1. 准备路径和数据 ---
        content_folder_path = os.path.join(args.data_root, 'content', content_name)
        style_image_path = os.path.join(args.data_root, 'style', style_name)

        if not os.path.isdir(content_folder_path):
            print(f"错误: 内容文件夹未找到: {content_folder_path}, 跳过此对。")
            return
        if not os.path.isfile(style_image_path):
            print(f"错误: 风格图未找到: {style_image_path}, 跳过此对。")
            return

        # 加载内容帧
        frame_paths = sorted(glob.glob(os.path.join(content_folder_path, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if not frame_paths:
            print(f"警告: 在 {content_folder_path} 中没有找到 .jpg 格式的帧, 跳过此对。")
            return
        
        print(f"找到 {len(frame_paths)} 个内容帧。")
        
        # <--- FIX: 修正了此处的变量名以确保它们匹配 --->
        # 使用 'frame_path'作为循环变量，并确保在 cv2.imread 中也使用它
        content_frames = [cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB) for frame_path in frame_paths]
        
        # 加载风格图
        style_image = cv2.cvtColor(cv2.imread(style_image_path), cv2.COLOR_BGR2RGB)
        print(f"已加载风格图: {style_name}")

        # --- 2. 执行风格迁移 (模型已在外部加载) ---
        print("开始视频风格迁移...")
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
        print("风格迁移完成！")

        # --- 3. 保存结果 ---
        style_name_base = os.path.splitext(style_name)[0]
        output_dir_name = f"{style_name_base}_stylized_{content_name}"
        output_path = os.path.join(args.output_dir, output_dir_name)
        os.makedirs(output_path, exist_ok=True)
        
        print(f"正在将结果保存到: {output_path}")
        
        for i, frame_pil in enumerate(output_frames):
            frame_path = os.path.join(output_path, f"{i+1:04d}.png")
            frame_pil.save(frame_path)
            
        print(f"成功保存 {len(output_frames)} 帧。")
        print(f"完成处理: 内容='{content_name}', 风格='{style_name}'")

    except Exception as e:
        print(f"\n处理 '{content_name}' 与 '{style_name}' 时发生严重错误: {e}")
        print("将继续处理下一对。\n")


def main(args):
    """
    主执行函数，负责模型加载和任务分发。
    """
    # --- 1. 发现所有的内容和风格 ---
    content_root = os.path.join(args.data_root, 'content')
    style_root = os.path.join(args.data_root, 'style')

    if not os.path.isdir(content_root):
        raise FileNotFoundError(f"内容数据根目录未找到: {content_root}")
    if not os.path.isdir(style_root):
        raise FileNotFoundError(f"风格数据根目录未找到: {style_root}")

    all_content_names = [d for d in os.listdir(content_root) if os.path.isdir(os.path.join(content_root, d))]
    all_style_names = [f for f in os.listdir(style_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not all_content_names:
        print(f"错误: 在 '{content_root}' 中没有找到任何内容文件夹。")
        return
    if not all_style_names:
        print(f"错误: 在 '{style_root}' 中没有找到任何支持的风格图像。")
        return

    print("="*50)
    print(f"发现 {len(all_content_names)} 个内容视频: {all_content_names}")
    print(f"发现 {len(all_style_names)} 个风格图像: {all_style_names}")
    print(f"总共将执行 {len(all_content_names) * len(all_style_names)} 次风格迁移任务。")
    print("="*50)


    # --- 2. 初始化管线 (只执行一次) ---
    print(f"正在从 '{args.model_path}' 加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StyleIDVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    print("模型加载完毕，管线已初始化。")

    # --- 3. 遍历所有组合并执行风格迁移 ---
    for content_name in all_content_names:
        for style_name in all_style_names:
            run_style_transfer_for_pair(pipe, content_name, style_name, args)

    print("\n所有风格迁移任务已全部完成！")


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(description="使用StyleIDVideoPipeline进行批量视频风格迁移")
    
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--output_dir", type=str, default="./results/pa_anchor_ma1", help="保存结果的根目录")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="预训练的Stable Diffusion模型路径或HuggingFace名称")

    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM反转和采样的步数")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query保留强度 (控制内容保留程度)")
    parser.add_argument("--temperature", type=float, default=1.5, help="注意力温度系数 (控制风格化强度)")
    parser.add_argument("--mask_strength", type=float, default=1.0, help="PA Fusion的融合强度")
    
    parser.add_argument("--without_init_adain", action="store_true", help="禁用初始latent的AdaIN操作")
    
    args = parser.parse_args()
    main(args)