# ./styleid_v2v/styleid_v2v_pipeline.py

import os
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 Diffusers 和 Transformers 的核心组件
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel,DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# 导入我们自己的父类 Pipeline 和辅助函数
from styleid.styleid_pipeline import StyleIDPipeline, normalize, denormalize

# 导入 GMFlow
import sys
from pathlib import Path


gmflow_path = str(Path(__file__).resolve().parents[1] / 'gmflow')



if gmflow_path not in sys.path:
    sys.path.append(gmflow_path)

try:
    from gmflow.gmflow import GMFlow
    GMFlow_installed = True
except ImportError as e:
    print(f"Warning: GMFlow is not installed. Temporal consistency features will not be available.")
    print(f"ImportError details: {e}") 
    GMFlow_installed = False

blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))

# === 光流计算相关辅助函数 (来自 Rerender-A-Video) ===
def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()
    return grid[None].repeat(b, 1, 1, 1)

def bilinear_sample(img, sample_coords, mode="bilinear", padding_mode="zeros", return_mask=False):
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    b, _, h, w = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        return img, mask
    return img

def flow_warp(feature, flow, mask=False, mode="bilinear", padding_mode="zeros"):
    b, c, h, w = feature.size()
    if flow.size(1) != 2:
         flow = flow.permute(0,3,1,2)
    grid = coords_grid(b, h, w, device=flow.device) + flow
    grid = grid.to(feature.dtype)
    return bilinear_sample(feature, grid, mode=mode, padding_mode=padding_mode, return_mask=mask)

def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)
    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)
    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)
    threshold = alpha * flow_mag + beta
    fwd_occ = (diff_fwd > threshold).float()
    bwd_occ = (diff_bwd > threshold).float()
    return fwd_occ, bwd_occ

class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


@torch.no_grad()
def get_warped_and_mask(flow_model, image1, image2, image3=None, pixel_consistency=False, device=None):
    if image3 is None:
        image3 = image1
    padder = InputPadder(image1.shape, padding_factor=8)
    image1, image2 = padder.pad(image1[None].to(device), image2[None].to(device))
    results_dict = flow_model(
        image1, image2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True
    )
    flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
    fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(
            bwd_occ + (abs(image2 - warped_image1).mean(dim=1) > 255 * 0.25).float(), 0, 1
        ).unsqueeze(0)
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ, bwd_flow
    
# =========================================================================================
# === Video Style Transfer Pipeline ===
# =========================================================================================

class StyleIDVideoPipeline(StyleIDPipeline):
    def __init__(
        self, 
        vae, 
        text_encoder, 
        tokenizer, 
        unet, 
        scheduler,
        safety_checker=None, 
        feature_extractor=None, 
        image_encoder=None,
        requires_safety_checker: bool = True, #  must have this
    ):
        # Pass all arguments up to the parent StyleIDPipeline
        super().__init__(
            vae=vae, 
            text_encoder=text_encoder, 
            tokenizer=tokenizer, 
            unet=unet, 
            scheduler=scheduler,
            safety_checker=safety_checker, 
            feature_extractor=feature_extractor,
            image_encoder=image_encoder, 
            requires_safety_checker=requires_safety_checker #  Pass it along
        )
        
        # 2. add GMFlow optical flow pretrained model for image preprocess
        if not GMFlow_installed:
            self.flow_model = None
            return
            
        print("Loading GMFlow model for video processing...")
        try:
            # 步骤 1: 在 CPU 上创建模型实例 (注意：这里不要 .to(self.device))
            flow_model = GMFlow(
                feature_channels=128, num_scales=1, upsample_factor=8, num_head=1,
                attention_type="swin", ffn_dim_expansion=4, num_transformer_layers=6,
            ).to(self.device)

            # 步骤 2: 从文件加载权重，并确保权重数据也放在 CPU 上
            local_gmflow_path = "gmflow/pretrained/gmflow_sintel-0c07dcb3.pth"
            print(f"Loading GMFlow model from local path: {local_gmflow_path}")
            checkpoint = torch.load(local_gmflow_path, map_location=lambda storage, loc: storage,
        )

            weights = checkpoint["model"] if "model" in checkpoint else checkpoint
            
            # 步骤 3: 将 CPU 上的权重加载到 CPU 上的模型里
            flow_model.load_state_dict(weights, strict=False)

            # 步骤 4: 所有东西都准备好后，将完整的模型一次性移动到 GPU
            flow_model.to(self.device)
            flow_model.eval()
            self.flow_model=flow_model
            print(f"GMFlow model loaded successfully and is on device: {next(self.flow_model.parameters()).device}")

        except Exception as e:
            print(f"ERROR: Could not load GMFlow model weights. Temporal consistency will fail. Error: {e}")
            self.flow_model = None

    def to(self, torch_device):
        super().to(torch_device)
        if self.flow_model is not None:
            self.flow_model.to(torch_device)
        return self
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the StyleIDVideoPipeline from a pretrained model."""
        # This pattern is robust: load the base pipeline from the grandparent or a known source
        # to get all the standard, pre-trained components correctly.
        # Since StyleIDPipeline's from_pretrained already does this by loading StableDiffusionImg2ImgPipeline,
        # we can leverage that. Or we can be explicit here for clarity.
        
        # Load all base components from the original diffuser's pipeline
        base_pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Now, create an instance of *our* specific class (cls will be StyleIDVideoPipeline)
        # This ensures the correct __init__ method (the one that loads GMFlow) is called.
        video_pipe = cls(
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            unet=base_pipeline.unet,
            scheduler=base_pipeline.scheduler,
            safety_checker=getattr(base_pipeline, 'safety_checker', None),
            feature_extractor=getattr(base_pipeline, 'feature_extractor', None),
            image_encoder=getattr(base_pipeline, 'image_encoder', None),
            requires_safety_checker=getattr(base_pipeline.config, 'requires_safety_checker', True),
        )

        # The DDIMScheduler check is crucial for inversion, let's ensure it's here too.
        from diffusers import DDIMScheduler
        if not isinstance(video_pipe.scheduler, DDIMScheduler):
            scheduler = DDIMScheduler.from_config(base_pipeline.scheduler.config)
            video_pipe.scheduler = scheduler

        return video_pipe
    
    @torch.no_grad()
    def _denoise_pure_styleid(self, initial_latent: torch.Tensor, text_embeddings: torch.Tensor):
        self.state.to_transfer()
        current_latent = initial_latent
        for t in tqdm(self.scheduler.timesteps, desc="Denoising (Pure StyleID)"):
            self.state.set_timestep(t.item())
            noise_pred = self.unet(current_latent, t, encoder_hidden_states=text_embeddings).sample
            current_latent = self.scheduler.step(noise_pred, t, current_latent).prev_sample
        return current_latent
    
    @torch.no_grad()
    def _calculate_fusion_target(
        self,
        base_stylized_current_frame_tensor: torch.Tensor,
        original_anchor_frame_tensor: torch.Tensor,
        original_prev_frame_tensor: torch.Tensor,
        original_current_frame_tensor: torch.Tensor,
        stylized_anchor_frame_tensor: torch.Tensor,
        stylized_prev_frame_tensor: torch.Tensor
    ):
        # Warp reference frames
        warped_anchor, bwd_occ_0, _ = get_warped_and_mask(self.flow_model, original_anchor_frame_tensor, original_current_frame_tensor, stylized_anchor_frame_tensor, device=self.device)
        warped_prev, bwd_occ_pre, _ = get_warped_and_mask(self.flow_model, original_prev_frame_tensor, original_current_frame_tensor, stylized_prev_frame_tensor, device=self.device)

        # Create blend masks based on occlusion
        
        blend_mask_0 = blur(F.max_pool2d(bwd_occ_0.unsqueeze(1), kernel_size=9, stride=1, padding=4))
        blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0.unsqueeze(1), 0, 1)

        blend_mask_pre = blur(F.max_pool2d(bwd_occ_pre.unsqueeze(1), kernel_size=9, stride=1, padding=4))
        blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre.unsqueeze(1), 0, 1)

        # Fuse images
        blend_results = (1 - blend_mask_pre) * warped_prev + blend_mask_pre * base_stylized_current_frame_tensor
        blend_results = (1 - blend_mask_0) * warped_anchor + blend_mask_0 * blend_results

        # Fidelity-oriented encode
        xtrg = self.fidelity_oriented_encode(blend_results.to(self.vae.dtype))

        # Calculate final fusion mask for the denoising loop
        final_occ_mask = 1 - torch.clamp((1 - bwd_occ_pre) + (1 - bwd_occ_0), 0, 1)
        final_blend_mask = blur(F.max_pool2d(final_occ_mask.unsqueeze(1), kernel_size=9, stride=1, padding=4))
        final_blend_mask = 1 - torch.clamp(final_blend_mask + final_occ_mask.unsqueeze(1), 0, 1)
        
        return xtrg, final_blend_mask

    @torch.no_grad()
    def _denoise_with_pa_fusion(
        self,
        initial_latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        xtrg: torch.Tensor,
        fusion_mask: torch.Tensor,
        mask_strength: float
    ):
        self.state.to_transfer()
        current_latent = initial_latent

        # Resize fusion mask to latent space dimensions
        fusion_mask_latent = 1.0 - F.interpolate(fusion_mask, size=current_latent.shape[-2:], mode='bilinear') * mask_strength
        fusion_mask_latent = fusion_mask_latent.to(current_latent.dtype)
        
        for t in tqdm(self.scheduler.timesteps, desc="Denoising (with PA Fusion)"):
            self.state.set_timestep(t.item())

            # Latent Inpainting / Fusion
            noise = torch.randn_like(current_latent)
            latents_ref = self.scheduler.add_noise(xtrg, noise, t)
            fused_latent = current_latent * fusion_mask_latent + latents_ref * (1 - fusion_mask_latent)
            
            # U-Net Denoising with StyleID
            noise_pred = self.unet(fused_latent, t, encoder_hidden_states=text_embeddings).sample
            
            # DDIM Step, using the fused latent as the input for this step
            current_latent = self.scheduler.step(noise_pred, t, fused_latent).prev_sample
        
        return current_latent
    
    @torch.no_grad()
    def fidelity_oriented_encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Implements a simplified fidelity-oriented image encoding from Rerender-A-Video.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        x_r = self.vae.encode(image_tensor).latent_dist.sample()
        I_r = self.vae.decode(x_r).sample
        x_rr = self.vae.encode(I_r).latent_dist.sample()
        
        compensation = x_r - x_rr
        final_latent = x_r + compensation
        return final_latent * self.vae.config.scaling_factor

    '''
    @torch.no_grad()
    def denoising_loop_with_fusion(
        self, 
        initial_latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        is_anchor_frame: bool,
        # --- 融合所需的数据 ---
        original_anchor_frame: torch.Tensor,
        original_prev_frame: torch.Tensor,
        original_current_frame: torch.Tensor,
        stylized_anchor_frame: torch.Tensor,
        stylized_prev_frame: torch.Tensor,
        # --- 其他参数 ---
        num_inference_steps: int,
        mask_strength: float
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        current_latent = initial_latent

        timesteps = self.scheduler.timesteps
        num_steps = len(timesteps)
        fusion_start_step = int(num_steps * 0.5)
        fusion_end_step = int(num_steps * 0.8)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            self.state.set_timestep(t.item())

            noise_pred_unfused = self.unet(current_latent, t, encoder_hidden_states=text_embeddings).sample

            if not is_anchor_frame and fusion_start_step <= i < fusion_end_step:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_x0_unfused = (current_latent - beta_prod_t ** (0.5) * noise_pred_unfused) / alpha_prod_t ** (0.5)
                direct_result_img = self.decode_latent(pred_x0_unfused).squeeze(0)
                
                warped_anchor, bwd_occ_0, _ = get_warped_and_mask(self.flow_model, original_anchor_frame, original_current_frame, stylized_anchor_frame, device=self.device)
                warped_prev, bwd_occ_pre, _ = get_warped_and_mask(self.flow_model, original_prev_frame, original_current_frame, stylized_prev_frame, device=self.device)

                blend_mask_0 = blur(F.max_pool2d(bwd_occ_0.unsqueeze(1), kernel_size=9, stride=1, padding=4)).squeeze()
                blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)
                
                blend_mask_pre = blur(F.max_pool2d(bwd_occ_pre.unsqueeze(1), kernel_size=9, stride=1, padding=4)).squeeze()
                blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

                blend_results = (1 - blend_mask_pre) * warped_prev + blend_mask_pre * direct_result_img
                blend_results = (1 - blend_mask_0) * warped_anchor + blend_mask_0 * blend_results
                
                xtrg = self.fidelity_oriented_encode(blend_results.to(self.vae.dtype))
                
                final_occ_mask = 1 - torch.clamp((1 - bwd_occ_pre) + (1 - bwd_occ_0), 0, 1)
                final_blend_mask = blur(F.max_pool2d(final_occ_mask.unsqueeze(1), kernel_size=9, stride=1, padding=4)).squeeze()
                final_blend_mask = 1 - torch.clamp(final_blend_mask + final_occ_mask, 0, 1)
                
                fusion_mask_latent = 1.0 - F.interpolate(final_blend_mask.unsqueeze(1), size=current_latent.shape[-2:], mode='bilinear') * mask_strength
                fusion_mask_latent = fusion_mask_latent.to(current_latent.dtype)
                noise = torch.randn_like(current_latent)
                latents_ref = self.scheduler.add_noise(xtrg, noise, t)
                
                fused_latent = current_latent * fusion_mask_latent + latents_ref * (1 - fusion_mask_latent)
                
                noise_pred = self.unet(fused_latent, t, encoder_hidden_states=text_embeddings).sample
                current_latent = self.scheduler.step(noise_pred, t, fused_latent).prev_sample
            else:
                current_latent = self.scheduler.step(noise_pred_unfused, t, current_latent).prev_sample
            
        return current_latent

    def style_transfer_video(self, content_frames: List[np.ndarray], style_image: np.ndarray, num_inference_steps: int = 50, gamma=0.75, temperature=1.5, without_init_adain=False, mask_strength: float = 0.7, output_type="pil"):
        if self.flow_model is None:
            raise ImportError("GMFlow model is not loaded. Cannot perform video style transfer.")

        # 1. SETUP
        self.update_parameters(gamma=gamma, temperature=temperature, without_init_adain=without_init_adain)
        device = self.device
        
        processed_content_frames = [normalize(frame).to(device=device, dtype=self.vae.dtype).squeeze(0) for frame in content_frames]
        text_embeddings = self.get_text_embedding()

        # 2. PRE-COMPUTATION (仅风格)
        # 使用父类的 precompute_style 方法，它会处理好风格反转并缓存特征和latents
        style_cache = self.precompute_style(style_image, num_inference_steps)
        self.state.style_features = style_cache["style_features"]
        style_latents = style_cache["style_latents"]

        # 3. 统一的逐帧生成循环
        output_frames_pil = []
        generated_frames_tensors = []

        for i in range(len(processed_content_frames)):
            is_anchor_frame = (i == 0)
            print(f"\nProcessing Frame {i} {'(Anchor Frame)' if is_anchor_frame else ''}...")
            
            current_content_tensor = processed_content_frames[i]
            content_latent = self.encode_image(current_content_tensor.unsqueeze(0))

            # --- 即时反转 (On-the-fly Inversion) ---
            print(f"  - Step 2.{i}: Inverting content frame {i}...")
            self.state.to_invert_content()
            # 使用父类的 ddim_inversion 方法，它会通过AttnProcessor自动填充 self.state.content_features
            content_latents = self.ddim_inversion(content_latent, text_embeddings)
            
            # --- 准备初始Latent ---
            print(f"  - Step 3.{i}: Preparing initial latent...")
            self.state.to_transfer() # 切换到迁移模式
            if not self.without_init_adain:
                initial_latent = (content_latents[0] - content_latents[0].mean(dim=(2,3), keepdim=True)) / (content_latents[0].std(dim=(2,3), keepdim=True) + 1e-4) * style_latents[0].std(dim=(2,3), keepdim=True) + style_latents[0].mean(dim=(2,3), keepdim=True)
            else:
                initial_latent = content_latents[0]
            
            # --- 准备融合所需的数据 ---
            original_anchor_frame = processed_content_frames[0]
            original_prev_frame = None if is_anchor_frame else processed_content_frames[i-1]
            stylized_anchor_frame = None if is_anchor_frame else generated_frames_tensors[0]
            stylized_prev_frame = None if is_anchor_frame else generated_frames_tensors[i-1]

            # --- 执行Denoising循环 ---
            print(f"  - Step 4.{i}: Denoising...")
            final_latent = self.denoising_loop_with_fusion(
                initial_latent=initial_latent,
                text_embeddings=text_embeddings,
                is_anchor_frame=is_anchor_frame,
                original_anchor_frame=original_anchor_frame,
                original_prev_frame=original_prev_frame,
                original_current_frame=current_content_tensor,
                stylized_anchor_frame=stylized_anchor_frame,
                stylized_prev_frame=stylized_prev_frame,
                num_inference_steps=num_inference_steps,
                mask_strength=mask_strength
            )
            
            # --- 解码并保存结果 ---
            with torch.no_grad():
                final_image_tensor = self.decode_latent(final_latent)
            
            final_image_pil = self.image_processor.postprocess(final_image_tensor, output_type=output_type, do_denormalize=[True])[0]
            output_frames_pil.append(final_image_pil)
            generated_frames_tensors.append(final_image_tensor.detach().clone())
            
        return {"images": output_frames_pil}
    '''
    def style_transfer_video(self, content_frames: List[np.ndarray], style_image: np.ndarray, num_inference_steps: int = 50, gamma=0.75, temperature=1.5, without_init_adain=False, mask_strength: float = 0.7, output_type="pil"):
        if self.flow_model is None: raise ImportError("GMFlow model is not loaded. Cannot perform video style transfer.")

        # --- 1. SETUP ---
        self.update_parameters(gamma=gamma, temperature=temperature, without_init_adain=without_init_adain)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        device = self.device
        
        processed_content_frames = [normalize(frame).to(device=device, dtype=self.vae.dtype).squeeze(0) for frame in content_frames]
        text_embeddings = self.get_text_embedding()

        # --- 2. PRE-COMPUTATION ---
        print("Step 1: Pre-computing style features...")
        style_cache = self.precompute_style(style_image, num_inference_steps)
        self.state.style_features = style_cache["style_features"]
        style_noisy_latent_T = style_cache["style_latents"][0] # Noisy latent z_T^s
        
        # --- Process Anchor Frame (Frame 0) ---
        print("Step 2: Processing anchor frame (Frame 0)...")
        self.state.content_features.clear()
        self.state.to_invert_content()
        content_latent_0 = self.encode_image(processed_content_frames[0].unsqueeze(0))
        content_latents_0 = self.ddim_inversion(content_latent_0, text_embeddings)
        content_latent_T_0 = content_latents_0[0]

        if not self.without_init_adain:
            initial_latent_0 = (content_latent_T_0 - content_latent_T_0.mean(dim=(2,3), keepdim=True)) / (content_latent_T_0.std(dim=(2,3), keepdim=True) + 1e-4) * style_noisy_latent_T.std(dim=(2,3), keepdim=True) + style_noisy_latent_T.mean(dim=(2,3), keepdim=True)
        else:
            initial_latent_0 = content_latent_T_0
            
        final_latent_0 = self._denoise_pure_styleid(initial_latent_0, text_embeddings)
        stylized_anchor_frame_tensor = self.decode_latent(final_latent_0)
        
        # --- Store results and initialize sliding window ---
        output_frames_pil = [self.image_processor.postprocess(stylized_anchor_frame_tensor, output_type=output_type, do_denormalize=[True])[0]]
        stylized_prev_frame_tensor = stylized_anchor_frame_tensor.detach().clone()
        
        # --- 3. FRAME-BY-FRAME GENERATION (i > 0) ---
        for i in range(1, len(processed_content_frames)):
            print(f"\nProcessing Frame {i}...")
            current_content_tensor = processed_content_frames[i]
            
            # --- 3.1: On-the-fly Inversion of current frame I_c_i ---
            self.state.content_features.clear()
            self.state.to_invert_content()
            content_latent_i = self.encode_image(current_content_tensor.unsqueeze(0))
            content_latents_i = self.ddim_inversion(content_latent_i, text_embeddings)
            content_latent_T_i = content_latents_i[0]
            
            # Initialize Latent z_T^{cs_i}
            if not self.without_init_adain:
                initial_latent_i = (content_latent_T_i - content_latent_T_i.mean(dim=(2,3), keepdim=True)) / (content_latent_T_i.std(dim=(2,3), keepdim=True) + 1e-4) * style_noisy_latent_T.std(dim=(2,3), keepdim=True) + style_noisy_latent_T.mean(dim=(2,3), keepdim=True)
            else:
                initial_latent_i = content_latent_T_i

            # --- 3.2 (Step I): Generate base result I_bar_prime_i ---
            print(f"  - Step {i}.1: Generating base stylized frame (no fusion)...")
            base_final_latent_i = self._denoise_pure_styleid(initial_latent_i.clone(), text_embeddings)
            base_stylized_frame_tensor_i = self.decode_latent(base_final_latent_i)
            
            # --- 3.3 (Step II): Build and encode fusion target xtrg ---
            print(f"  - Step {i}.2: Calculating PA Fusion target...")
            xtrg, fusion_mask = self._calculate_fusion_target(
                base_stylized_current_frame_tensor=base_stylized_frame_tensor_i,
                original_anchor_frame_tensor=processed_content_frames[0],
                original_prev_frame_tensor=processed_content_frames[i-1],
                original_current_frame_tensor=current_content_tensor,
                stylized_anchor_frame_tensor=stylized_anchor_frame_tensor,
                stylized_prev_frame_tensor=stylized_prev_frame_tensor
            )

            # --- 3.4 (Step III): Execute final denoising with PA Fusion ---
            print(f"  - Step {i}.3: Denoising with PA Fusion...")
            final_latent_i = self._denoise_with_pa_fusion(
                initial_latent=initial_latent_i.clone(),
                text_embeddings=text_embeddings,
                xtrg=xtrg,
                fusion_mask=fusion_mask,
                mask_strength=mask_strength
            )

            # --- 3.5: Decode, save, and update window ---
            final_image_tensor = self.decode_latent(final_latent_i)
            output_frames_pil.append(self.image_processor.postprocess(final_image_tensor, output_type=output_type, do_denormalize=[True])[0])
            
            stylized_prev_frame_tensor = final_image_tensor.detach().clone()
            
        ## FIX #2: Removed redundant definition of fidelity_oriented_encode from here.
        return {"images": output_frames_pil}    