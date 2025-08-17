#  styleid/styleid_video_pipeline.py

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
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer

# 导入我们自己的父类 Pipeline
from .styleid_pipeline import StyleIDPipeline, normalize, denormalize

# 导入 GMFlow
try:
    from gmflow.gmflow import GMFlow
    GMFlow_installed = True
except ImportError:
    print("Warning: GMFlow is not installed. Temporal consistency features will not be available.")
    print("Please install it using: pip install git+https://github.com/haofeixu/gmflow.git")
    GMFlow_installed = False


 

blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))

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
    image1_padded, image2_padded = padder.pad(image1[None].to(device), image2[None].to(device))
    results_dict = flow_model(image1_padded, image2_padded, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True)
    flow_pr = results_dict["flow_preds"][-1]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)
    fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(bwd_occ + (abs(image2 - warped_image1).mean(dim=1) > 255 * 0.25).float(), 0, 1).unsqueeze(0)
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ.unsqueeze(0), bwd_flow
    
# =========================================================================================
# === The Video Style Transfer Pipeline ===
# =========================================================================================

class StyleIDVideoPipeline(StyleIDPipeline):
    """
    A pipeline for video style transfer that inherits from StyleIDPipeline
    and integrates temporal consistency mechanisms from Rerender-A-Video.
    """
    
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, **kwargs):
        # 1. 调用父类 __init__ 完成所有 StyleID 的基础设置
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, **kwargs)
        
        # 2. 添加 GMFlow 模型用于视频处理
        if not GMFlow_installed:
            self.flow_model = None
            return
            
        print("Loading GMFlow model for video processing...")
        self.flow_model = GMFlow(
            feature_channels=128, num_scales=1, upsample_factor=8, num_head=1,
            attention_type="swin", ffn_dim_expansion=4, num_transformer_layers=6,
        ).to(self.device)
        self.flow_model.eval()

        try:
            checkpoint = torch.hub.load_url(
                "https://huggingface.co/Anonymous-sub/Rerender/resolve/main/models/gmflow_sintel-0c07dcb3.pth",
                map_location=self.device,
            )
            weights = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.flow_model.load_state_dict(weights, strict=False)
            print("GMFlow model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Could not load GMFlow model weights. Temporal consistency will fail. Error: {e}")
            self.flow_model = None

    @torch.no_grad()
    def fidelity_oriented_encode(self, image_tensor: torch.Tensor, low_error_threshold: float = 0.05) -> torch.Tensor:
        """
        Implements the fidelity-oriented image encoding (ε*) from Rerender-A-Video.
        This minimizes information loss from VAE encoding/decoding cycles.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        x_r = self.vae.encode(image_tensor).latent_dist.sample()
        I_r = self.vae.decode(x_r).sample
        x_rr = self.vae.encode(I_r).latent_dist.sample()
        
        compensation = x_r - x_rr
        x_prime = x_r + compensation
        I_prime = self.vae.decode(x_prime).sample
        
        error = torch.abs(image_tensor - I_prime).mean(dim=1, keepdim=True)
        M_epsilon = (error < low_error_threshold).float()
        
        final_latent = x_r + M_epsilon * compensation
        return final_latent * self.vae.config.scaling_factor

    @torch.no_grad()
    def denoising_loop_with_fusion(self, initial_latent, text_embeddings, anchor_frame_tensor, prev_frame_tensor, flow_data, num_inference_steps, mask_strength, inner_strength):
        """
        The core denoising loop that combines StyleID with Pixel-Aware Fusion.
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        current_latent = initial_latent

        # Pre-calculate warp and masks outside the loop for efficiency
        warped_anchor, _, _ = get_warped_and_mask(self.flow_model, anchor_frame_tensor, content_image_tensor=None, image3=anchor_frame_tensor, device=self.device)
        warped_prev, _, _ = get_warped_and_mask(self.flow_model, prev_frame_tensor, content_image_tensor=None, image3=prev_frame_tensor, device=self.device)

        bwd_occ_0 = flow_data["bwd_occ_0"]
        bwd_occ_pre = flow_data["bwd_occ_pre"]
        
        blend_mask_0 = blur(F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
        blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

        blend_mask_pre = blur(F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
        blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

        for t in tqdm(self.scheduler.timesteps, desc="Denoising with Fusion"):
            self.state.set_timestep(t.item())

            # --- Pixel-Aware Fusion ---
            noise_pred_unfused = self.unet(current_latent, t, encoder_hidden_states=text_embeddings).sample
            pred_x0_unfused = self.scheduler.step(noise_pred_unfused, t, current_latent).pred_original_sample
            direct_result_img = self.decode_latent(pred_x0_unfused)
            
            blend_results = (1 - blend_mask_pre) * warped_prev + blend_mask_pre * direct_result_img
            blend_results = (1 - blend_mask_0) * warped_anchor + blend_mask_0 * blend_results
            
            xtrg = self.fidelity_oriented_encode(blend_results)
            
            final_occ_mask = 1 - torch.clamp((1 - bwd_occ_pre) + (1 - bwd_occ_0), 0, 1)
            final_blend_mask = blur(F.max_pool2d(final_occ_mask, kernel_size=9, stride=1, padding=4))
            final_blend_mask = 1 - torch.clamp(final_blend_mask + final_occ_mask, 0, 1)
            
            fusion_mask_latent = 1.0-F.interpolate(final_blend_mask, size=current_latent.shape[-2:], mode='bilinear') * mask_strength
            
            noise = torch.randn_like(current_latent)
            latents_ref = self.scheduler.add_noise(xtrg, noise, t)
            
            fused_latent = current_latent * fusion_mask_latent + latents_ref * (1 - fusion_mask_latent)
            
            # --- Standard StyleID Denoising Step on the fused latent ---
            noise_pred = self.unet(fused_latent, t, encoder_hidden_states=text_embeddings).sample
            current_latent = self.scheduler.step(noise_pred, t, fused_latent).prev_sample
            
        return current_latent

    def style_transfer_video(self, content_frames: List[np.ndarray], style_image: np.ndarray, num_inference_steps: int = 50, gamma=0.75, temperature=1.5, without_init_adain=False, mask_strength: float = 0.7, inner_strength: float = 0.9, output_type="pil"):
        if self.flow_model is None:
            raise ImportError("GMFlow model is not loaded. Cannot perform video style transfer.")

        # 1. SETUP
        self.update_parameters(gamma=gamma, temperature=temperature, without_init_adain=without_init_adain)
        device = self.device
        
        processed_content_frames = [normalize(frame).to(device=device, dtype=self.vae.dtype) for frame in content_frames]
        
        text_embeddings = self.get_text_embedding()

        # 2. PRE-COMPUTATION
        print("Step 1: Pre-computing style inversion...")
        style_cache = self.precompute_style(style_image, num_inference_steps)
        style_latents = style_cache["style_latents"]
        print("Step 2: Pre-computing content inversions and optical flows...")
        content_inversions = []
        for frame_tensor in tqdm(processed_content_frames, desc="Inverting Content Frames"):
            self.state.to_invert_content()
            self.state.content_features = {}
            latents = self.encode_image(frame_tensor)
            self.ddim_inversion(latents, text_embeddings)
            content_inversions.append({"initial_latent": latents, "features": copy.deepcopy(self.state.content_features)})

        flows = {}
        for i in range(1, len(processed_content_frames)):
            _, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(self.flow_model, processed_content_frames[i-1], processed_content_frames[i], device=self.device)
            _, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(self.flow_model, processed_content_frames[0], processed_content_frames[i], device=self.device)
            flows[i] = {"bwd_flow_pre": bwd_flow_pre, "bwd_occ_pre": bwd_occ_pre, "bwd_flow_0": bwd_flow_0, "bwd_occ_0": bwd_occ_0}

        # 3. FRAME-BY-FRAME GENERATION
        output_frames_pil = []
        generated_frames_tensors = []

        # --- Generate Frame 0 (Anchor Frame without fusion) ---
        print("\nStep 3: Generating Frame 0 (Anchor Frame)...")
        self.state.content_features = content_inversions[0]["features"]
        first_frame_result = self.style_transfer(content_image=content_frames[0], style_image=style_image, num_inference_steps=num_inference_steps, gamma=gamma, temperature=temperature, without_init_adain=without_init_adain)
        
        output_frames_pil.append(first_frame_result.images[0])
        first_frame_tensor = normalize(np.array(first_frame_result.images[0])).to(device=device, dtype=self.vae.dtype)
        generated_frames_tensors.append(first_frame_tensor)
        
        # --- Generate Subsequent Frames with Fusion ---
        for i in range(1, len(processed_content_frames)):
            print(f"\nStep 4: Generating Frame {i} with Temporal Fusion...")
            
            current_content_tensor = processed_content_frames[i] 
            prev_generated_frame_tensor = generated_frames_tensors[i-1]
            
            
            self.state.style_features = style_cache["style_features"]
            self.state.content_features = content_inversions[i]["features"]
            
            # Prepare initial latent for the current frame
            self.state.to_transfer()
            content_latent = self.encode_image(current_content_tensor)
            if not self.without_init_adain:
                initial_latent = (content_latent - content_latent.mean(dim=(2,3), keepdim=True)) / (content_latent.std(dim=(2,3), keepdim=True) + 1e-4) * style_latents[0].std(dim=(2,3), keepdim=True) + style_latents[0].mean(dim=(2,3), keepdim=True)
            else:
                initial_latent = content_latent
            
            final_latent = self.denoising_loop_with_fusion(
                initial_latent, current_content_tensor, text_embeddings, first_frame_tensor, prev_generated_frame_tensor, flows[i],
                num_inference_steps, mask_strength, inner_strength
            )
            
            with torch.no_grad():
                final_image = self.decode_latent(final_latent)
            
            final_image_pil = self.image_processor.postprocess(final_image, output_type=output_type, do_denormalize=[True])[0]
            output_frames_pil.append(final_image_pil)
            generated_frames_tensors.append(final_image.detach().clone())
            
        return {"images": output_frames_pil}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # This is a helper to load the pipeline easily
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch_dtype)
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype)
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        
        return cls(vae, text_encoder, tokenizer, unet, scheduler, **kwargs)