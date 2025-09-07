#./run_styleid_v2v.py
import os
import argparse
import glob
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
import random
# 导入我们自定义的视频管线
# 假设此脚本与 styleid/ 文件夹在同一目录下
from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline

def main(args):
    """
    主执行函数
    """
    # --- 1. 准备路径和数据 ---
    # 构建内容帧和风格图的路径
    content_folder_path = os.path.join(args.data_root, 'content', args.content_name)
    style_image_path = os.path.join(args.data_root, 'style', args.style_name)
    
    # 检查路径是否存在
    if not os.path.isdir(content_folder_path):
        raise FileNotFoundError(f"内容文件夹未找到: {content_folder_path}")
    if not os.path.isfile(style_image_path):
        raise FileNotFoundError(f"风格图未找到: {style_image_path}")

    # 加载内容帧
    # 使用glob找到所有jpg文件并排序，以确保帧顺序正确
    frame_paths = sorted(glob.glob(os.path.join(content_folder_path, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    if not frame_paths:
        raise ValueError(f"在 {content_folder_path} 中没有找到 .jpg 格式的帧")
        
    print(f"找到 {len(frame_paths)} 个内容帧。")
    
    # 使用cv2读取所有帧，并转换为RGB格式
    content_frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]
    
    # 加载风格图
    style_image = cv2.cvtColor(cv2.imread(style_image_path), cv2.COLOR_BGR2RGB)
    print(f"已加载风格图: {args.style_name}")

    # --- 2. 初始化管线 ---
    print(f"正在从 '{args.model_path}' 加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用fp16以节省显存
    pipe = StyleIDVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    print("模型加载完毕，管线已初始化。")

    # --- 3. 执行风格迁移 ---
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
    
    output_frames = result['images'] # 返回的是PIL Image列表
    print("风格迁移完成！")

    # --- 4. 保存结果 ---
    # 创建输出目录，根据要求的格式命名
    style_name_base = os.path.splitext(args.style_name)[0]
    output_dir_name = f"{style_name_base}_stylized_{args.content_name}"
    output_path = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"正在将结果保存到: {output_path}")
    
    # 保存每一帧
    for i, frame_pil in enumerate(output_frames):
        frame_path = os.path.join(output_path, f"{i+1:04d}.png")
        frame_pil.save(frame_path)
        
    print(f"所有风格化帧已成功保存在文件夹中。")


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(description="使用StyleIDVideoPipeline进行视频风格迁移")
    
    # --- 路径和模型参数 ---
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--content_name", type=str, required=True, help="在 'data/content/' 下的内容视频文件夹名称 (例如 'car')")
    parser.add_argument("--style_name", type=str, required=True, help="在 'data/style/' 下的风格图文件名 (例如 'wave.png')")
    parser.add_argument("--output_dir", type=str, default="./results", help="保存结果的根目录")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="预训练的Stable Diffusion模型路径或HuggingFace名称")

    # --- 核心超参数 ---
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM反转和采样的步数")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query保留强度 (控制内容保留程度)")
    parser.add_argument("--temperature", type=float, default=1.5, help="注意力温度系数 (控制风格化强度)")
    parser.add_argument("--mask_strength", type=float, default=1.0, help="PA Fusion的融合强度")
    
    # --- 其他选项 ---
    parser.add_argument("--without_init_adain", action="store_true", help="禁用初始latent的AdaIN操作")
    

    args = parser.parse_args()
    main(args)
