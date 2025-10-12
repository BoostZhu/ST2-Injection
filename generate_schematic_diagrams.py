import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
import argparse

# 导入你的核心 pipeline 类和辅助工具
from styleid_v2v.styleid_v2v_pipeline import StyleIDVideoPipeline, get_warped_and_mask, blur

# --- 辅助函数 ---

def save_tensor_image(tensor, filename, output_dir="."):
    """
    将像素空间的 PyTorch tensor (范围 [-1, 1]) 解码并保存为图像文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_tensor = (tensor.squeeze(0).permute(1, 2, 0).cpu() / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.numpy() * 255).round().astype("uint8")
    
    image_pil = Image.fromarray(image_np)
    save_path = os.path.join(output_dir, filename)
    image_pil.save(save_path)
    print(f"Saved DECODED image: {save_path}")

def save_latent_image(latent_tensor, filename, output_dir="."):
    """
    [新功能] 将4通道的 Latent tensor 直接可视化并保存为图像文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 取前3个通道进行可视化
    latent_rgb = latent_tensor.squeeze(0)[:3]

    # 对每个通道进行独立的归一化，以最大化视觉对比度
    normalized_channels = []
    for channel in latent_rgb:
        min_val = torch.min(channel)
        max_val = torch.max(channel)
        channel_normalized = (channel - min_val) / (max_val - min_val + 1e-5)
        normalized_channels.append(channel_normalized)
    
    # 重新组合通道
    normalized_latent = torch.stack(normalized_channels)
    
    # 转换为 NumPy 数组并保存
    image_np = (normalized_latent.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    
    image_pil = Image.fromarray(image_np)
    save_path = os.path.join(output_dir, filename)
    image_pil.save(save_path)
    print(f"Saved LATENT visualization: {save_path}")


def save_mask_image(mask_tensor, filename, output_dir="."):
    """
    将单通道的 mask tensor (范围 [0, 1]) 保存为灰度图。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).round().astype("uint8")
    mask_pil = Image.fromarray(mask_np, 'L')
    save_path = os.path.join(output_dir, filename)
    mask_pil.save(save_path)
    print(f"Saved MASK image: {save_path}")


@torch.no_grad()
def generate_visuals(args):
    """
    主函数，用于生成所有示意图。
    """
    # --- 1. 初始化和加载模型 ---
    print("Loading StyleID Video Pipeline...")
    pipe = StyleIDVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler.set_timesteps(args.ddim_steps, device="cuda")
    
    # --- 2. 加载和预处理图像 ---
    print("Loading and preprocessing images...")
    
    content_frame_0_np = cv2.imread(args.content_frame_0)[:, :, ::-1]
    content_frame_1_np = cv2.imread(args.content_frame_1)[:, :, ::-1]
    style_image_np = cv2.imread(args.style_image)[:, :, ::-1]

    # 保存原始输入图像
    os.makedirs(args.output_dir, exist_ok=True)
    Image.fromarray(content_frame_0_np).save(os.path.join(args.output_dir, "0_input_content_frame_0.png"))
    Image.fromarray(content_frame_1_np).save(os.path.join(args.output_dir, "0_input_content_frame_1.png"))
    Image.fromarray(style_image_np).save(os.path.join(args.output_dir, "0_input_style_image.png"))

    content_frame_0 = pipe._preprocess_image(content_frame_0_np)
    content_frame_1 = pipe._preprocess_image(content_frame_1_np)
    style_image = pipe._preprocess_image(style_image_np)
    text_embeddings = pipe.get_text_embedding()
    
    early_step_index = int(args.ddim_steps * args.early_step_percent)
    late_step_index = int(args.ddim_steps * args.late_step_percent)
    print(f"Will visualize states at denoising steps: {early_step_index} and {late_step_index}")

    # --- 3. 预计算风格特征 ---
    print("\nStep A: Pre-computing style features...")
    style_cache = pipe.precompute_style(style_image, args.ddim_steps)
    pipe.state.style_features = style_cache["style_features"]
    style_noisy_latent_T = style_cache["style_latents"][0]

    # --- 4. 处理锚点帧 (Frame 0) ---
    print("\nStep B: Processing anchor frame (Frame 0)...")
    pipe.state.content_features.clear()
    pipe.state.to_invert_content()
    content_latent_0 = pipe.encode_image(content_frame_0)
    content_latents_0 = pipe.ddim_inversion(content_latent_0, text_embeddings)
    initial_latent_0 = (content_latents_0[0] - content_latents_0[0].mean(dim=(2,3), keepdim=True)) / (content_latents_0[0].std(dim=(2,3), keepdim=True) + 1e-4) * style_noisy_latent_T.std(dim=(2,3), keepdim=True) + style_noisy_latent_T.mean(dim=(2,3), keepdim=True)
    final_latent_0 = pipe._denoise_loop(initial_latent_0, text_embeddings)
    stylized_anchor_frame_tensor = pipe.decode_latent(final_latent_0)
    save_tensor_image(stylized_anchor_frame_tensor, "stylized_frame_0.png", args.output_dir)

    # --- 5. 对目标帧 (Frame 1) 进行详细步骤的可视化 ---
    print("\n" + "="*20)
    print("Step C: Visualizing pipeline for Frame 1")
    print("="*20)

    # C1: DDIM Inversion
    print("\nC1: DDIM Inversion on Content Frame 1...")
    pipe.state.content_features.clear()
    pipe.state.to_invert_content()
    content_latent_1 = pipe.encode_image(content_frame_1)
    content_latents_1_all = pipe.ddim_inversion(content_latent_1, text_embeddings)
    
    inverted_latent_early = content_latents_1_all[early_step_index]
    inverted_latent_late = content_latents_1_all[late_step_index]
    
    # [修改] 调用 save_latent_image，不解码
    save_latent_image(inverted_latent_early, f"c1_inverted_latent_at_step_{early_step_index}.png", args.output_dir)
    save_latent_image(inverted_latent_late, f"c1_inverted_latent_at_step_{late_step_index}.png", args.output_dir)
    
    # C2: Denoising
    print("\nC2: Denoising process visualization...")
    initial_latent_1 = (content_latents_1_all[0] - content_latents_1_all[0].mean(dim=(2,3), keepdim=True)) / (content_latents_1_all[0].std(dim=(2,3), keepdim=True) + 1e-4) * style_noisy_latent_T.std(dim=(2,3), keepdim=True) + style_noisy_latent_T.mean(dim=(2,3), keepdim=True)
    
    pipe.state.to_transfer()
    current_latent = initial_latent_1.clone()
    
    denoised_latent_early, denoised_latent_late = None, None

    for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="Manual Denoising")):
        if i == early_step_index:
            denoised_latent_early = current_latent.clone()
        if i == late_step_index:
            denoised_latent_late = current_latent.clone()

        pipe.state.set_timestep(t.item())
        noise_pred = pipe.unet(current_latent, t, encoder_hidden_states=text_embeddings).sample
        current_latent = pipe.scheduler.step(noise_pred, t, current_latent).prev_sample

    if denoised_latent_early is not None:
        # [修改] 调用 save_latent_image，不解码
        save_latent_image(denoised_latent_early, f"c2_denoised_latent_at_step_{early_step_index}.png", args.output_dir)
    if denoised_latent_late is not None:
        # [修改] 调用 save_latent_image，不解码
        save_latent_image(denoised_latent_late, f"c2_denoised_latent_at_step_{late_step_index}.png", args.output_dir)

    # C3: Warp & Fuse 结果 和 Occlusion Mask
    print("\nC3: Warp, Fuse, and Occlusion Mask visualization...")
    base_stylized_frame_tensor_1 = pipe.decode_latent(current_latent)
    
    warped_anchor, bwd_occ_anchor, _ = get_warped_and_mask(
        pipe.flow_model, content_frame_0, content_frame_1, stylized_anchor_frame_tensor, device="cuda"
    )
    
    blend_mask_anchor = blur(F.max_pool2d(bwd_occ_anchor.unsqueeze(1), kernel_size=9, stride=1, padding=4))
    blend_mask_anchor = torch.clamp(blend_mask_anchor + bwd_occ_anchor.unsqueeze(1), 0, 1)
    
    save_mask_image(blend_mask_anchor, "c3_occlusion_mask.png", args.output_dir)
    
    blend_results_image = (1 - blend_mask_anchor) * warped_anchor + blend_mask_anchor * base_stylized_frame_tensor_1
    save_tensor_image(blend_results_image, "c3_warp_and_fuse_result.png", args.output_dir)
    
    # C4: 对应噪声水平的 latent feature
    print("\nC4: Noisy fusion target visualization...")
    xtrg = pipe.fidelity_oriented_encode(blend_results_image.to(pipe.vae.dtype))
    
    timestep_early = pipe.scheduler.timesteps[early_step_index]
    timestep_late = pipe.scheduler.timesteps[late_step_index]
    
    noise = torch.randn_like(xtrg)
    noisy_fusion_target_latent_early = pipe.scheduler.add_noise(xtrg, noise, timestep_early)
    noisy_fusion_target_latent_late = pipe.scheduler.add_noise(xtrg, noise, timestep_late)
    
    # [修改] 调用 save_latent_image，不解码
    save_latent_image(noisy_fusion_target_latent_early, f"c4_noisy_fusion_target_at_step_{early_step_index}.png", args.output_dir)
    save_latent_image(noisy_fusion_target_latent_late, f"c4_noisy_fusion_target_at_step_{late_step_index}.png", args.output_dir)
    
    print("\nAll schematic diagrams have been generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate schematic diagrams for the StyleID Video pipeline.")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to the pre-trained Stable Diffusion model.")
    parser.add_argument("--content_frame_0", type=str, required=True, help="Path to the first content frame (anchor frame).")
    parser.add_argument("--content_frame_1", type=str, required=True, help="Path to the second content frame (target for visualization).")
    parser.add_argument("--style_image", type=str, required=True, help="Path to the style image.")
    parser.add_argument("--output_dir", type=str, default="schematic_results", help="Directory to save the output images.")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps for inversion and sampling.")
    parser.add_argument("--early_step_percent", type=float, default=0.5, help="Percentage for the early visualization step (e.g., 0.5 for 50%).")
    parser.add_argument("--late_step_percent", type=float, default=0.8, help="Percentage for the late visualization step (e.g., 0.8 for 80%).")
    
    args = parser.parse_args()
    
    generate_visuals(args)

