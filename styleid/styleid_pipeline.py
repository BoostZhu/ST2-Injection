import os
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline,DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor,Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# Utility functions for image preprocessing
def normalize(image):
    image = image / 127.5 - 1
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    return image

def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return (image * 255).round().astype("uint8")

def save_image(image, filename):
    """
    Image should be in range (0, 255) and numpy array
    """
    image = Image.fromarray(image)
    image.save(filename)

# State tracking for StyleID
class StyleIDState:
    # Modes for the processor
    INVERT_STYLE = 0
    INVERT_CONTENT = 1 
    TRANSFER = 2
    
    def __init__(self):
        self.reset()
        self.style_features = {}
        self.content_features = {}
    
    def reset(self):
        self.__state = StyleIDState.INVERT_STYLE
        self.__timestep = 0
    
    def set_timestep(self, t):
        self.__timestep = t
    
    @property
    def state(self):
        return self.__state
    
    @property
    def timestep(self):
        return self.__timestep
    
    def to_invert_style(self):
        self.__state = StyleIDState.INVERT_STYLE
        
    def to_invert_content(self):
        self.__state = StyleIDState.INVERT_CONTENT
    
    def to_transfer(self):
        self.__state = StyleIDState.TRANSFER



# Custom attention processor for StyleID
class StyleIDAttnProcessor(AttnProcessor):
    def __init__(self, state: StyleIDState, layer_name, gamma=0.75, temperature=1.5):
        super().__init__()
        self.state = state
        self.layer_name = layer_name
        self.gamma = gamma
        self.temperature = temperature
        
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        t = self.state.timestep
        layer = self.layer_name
        
        # 保存原始隐藏状态用于残差连接
        residual = hidden_states

        # 处理空间正则化（如果存在）
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 处理输入维度
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 确定序列长度并准备注意力掩码
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 处理组标准化（如果存在）
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算查询向量
        query = attn.to_q(hidden_states)

        # 判断是否为自注意力（这是关键部分）
        is_self_attention = encoder_hidden_states is None
        
        # 处理键和值向量
        if is_self_attention:
            # 这是自注意力
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            # 这是交叉注意力，且需要应用标准化
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 转换维度布局以便多头注意力计算
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # 根据 StyleID 状态进行处理
        # 只在自注意力模式下应用 StyleID 技术
        if is_self_attention:
            # 获取映射的时间步（用于样本生成阶段）
            mapped_t = self.state.get_mapped_timestep(t) if hasattr(self.state, 'get_mapped_timestep') else t
            
            if self.state.state == StyleIDState.INVERT_STYLE:
                # 存储风格图像的特征
                if layer not in self.state.style_features:
                    self.state.style_features[layer] = {}
                self.state.style_features[layer][t] = (query.detach(), key.detach(), value.detach())
                    
            elif self.state.state == StyleIDState.INVERT_CONTENT:
                # 存储内容图像的特征
                if layer not in self.state.content_features:
                    self.state.content_features[layer] = {}
                self.state.content_features[layer][t] = (query.detach(), key.detach(), value.detach())
                    
            elif self.state.state == StyleIDState.TRANSFER:
                # 生成阶段：使用存储的特征进行风格迁移
                if (layer in self.state.style_features and 
                    layer in self.state.content_features and
                    mapped_t in self.state.style_features[layer] and
                    mapped_t in self.state.content_features[layer]):
                    
                    # 获取映射时间步的特征
                    q_c = self.state.content_features[layer][mapped_t][0]
                    k_s = self.state.style_features[layer][mapped_t][1]
                    v_s = self.state.style_features[layer][mapped_t][2]
                    
                    # 确保形状匹配
                    if q_c.shape[0] != query.shape[0]:
                        # 在批次维度调整大小（通常是因为CFG）
                        q_c = q_c.repeat(query.shape[0] // q_c.shape[0], 1, 1)
                    if k_s.shape[0] != key.shape[0]:
                        k_s = k_s.repeat(key.shape[0] // k_s.shape[0], 1, 1)
                    if v_s.shape[0] != value.shape[0]:
                        v_s = v_s.repeat(value.shape[0] // v_s.shape[0], 1, 1)
                    
                    # 应用风格注入公式
                    q_hat = q_c * self.gamma + query * (1 - self.gamma)
                    
                    # 替换原始的 QKV
                    query = q_hat
                    key = k_s
                    value = v_s
                    
                    # 应用温度缩放 - 重要！
                    query = query * self.temperature
        
        # 标准注意力计算
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 重塑回原始尺寸（如果输入是4D）
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 应用残差连接
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 应用输出重缩放
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# Main StyleID Pipeline
class StyleIDPipeline(StableDiffusionImg2ImgPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: Optional[Any] = None,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        
        # Initialize StyleID state
        self.state = StyleIDState()
        
        # Define the layers where we want to apply StyleID (matching the original implementation)
        self.injection_layers = [7, 8, 9, 10, 11]
        
        # Default StyleID parameters
        self.gamma = 0.75
        self.temperature = 1.5
        self.without_init_adain = False
        self.without_attn_injection = False
        
        # Setup attention processors
        self._setup_attention_processors()
        
    def _setup_attention_processors(self):
        """Initialize the attention processors for StyleID"""
        # Map layer indices to the UNet's attention blocks
        block_mapping = {
            # These mappings are based on the original StyleID implementation's mapping
            # to the diffusers UNet attention block
            7: "up_blocks.2.attentions.0",
            8: "up_blocks.2.attentions.1",
            9: "up_blocks.2.attentions.2", 
            10: "up_blocks.3.attentions.0",
            11: "up_blocks.3.attentions.1",
            12: "up_blocks.3.attentions.2",
        }
    
        # Register attention processors
        attn_processors = {}
        for name in self.unet.attn_processors.keys():
            if any(name.startswith(block_mapping[layer]) for layer in self.injection_layers):
                attn_processors[name] = StyleIDAttnProcessor(
                    self.state, name, gamma=self.gamma, temperature=self.temperature
                )
            else:
                attn_processors[name] = AttnProcessor()
                
        self.unet.set_attn_processor(attn_processors)
    
    def update_parameters(self, gamma=None, temperature=None, without_init_adain=None, without_attn_injection=None):
        """Update StyleID parameters"""
        if gamma is not None:
            self.gamma = gamma
        if temperature is not None:
            self.temperature = temperature
        if without_init_adain is not None:
            self.without_init_adain = without_init_adain
        if without_attn_injection is not None:
            self.without_attn_injection = without_attn_injection
            
        # Update processors with new parameters
        for name, processor in self.unet.attn_processors.items():
            if isinstance(processor, StyleIDAttnProcessor):
                processor.gamma = self.gamma
                processor.temperature = self.temperature
    
    def encode_image(self, image):
        """Encode image to latent space"""
        if isinstance(image, np.ndarray):
            image = normalize(image).to(device=self.device, dtype=self.vae.dtype)
            
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * 0.18215
        return latent
    
    def decode_latent(self, latent):
        """Decode latent to image"""
        with torch.no_grad():
            latent = 1 / 0.18215 * latent
            image = self.vae.decode(latent).sample
        return image
    
    def get_text_embedding(self, text=None):
        """Get text embeddings for conditioning"""
        # For StyleID, we typically use empty text conditioning
        if text is None:
            # Get empty text embedding
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            return uncond_embeddings
        else:
            # Regular text conditioning if needed
            text_input = self.tokenizer(
                [text], padding="max_length", max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            return text_embeddings
    
    def ddim_inversion(self, latent, text_embeddings=None):
        """DDIM inversion process to get noisy latents"""
        if text_embeddings is None:
            text_embeddings = self.get_text_embedding()
            
        # Prepare for inversion
        timesteps = reversed(self.scheduler.timesteps)#timesteps=[1,21,41,...,981]
        num_inference_steps = len(self.scheduler.timesteps)
        
        # Ensure scheduler uses deterministic algorithm for inversion
        self.scheduler.set_timesteps(num_inference_steps,device=self.device)
        
        # Initialize with input latent
        latents = [latent.clone()]
        
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion")):
                # Set current timestep for the state
                self.state.set_timestep(t.item())
                
                # Clone latent to avoid modifying the original
                curr_latent = latents[-1].clone()
                
                # Forward through UNet to predict noise
                model_output = self.unet(curr_latent, t, encoder_hidden_states=text_embeddings).sample
                
                # DDIM inversion step (add noise)
                # Adapted from the original StyleID implementation
                current_t=max(0, t.item() - (1000//num_inference_steps))
                next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]#x_t+1 is more noisy than x_t
                
                # Calculate noisy latent for the next step (moving forward in noise level)
                beta_t = 1 - alpha_t
                if self.scheduler.config.prediction_type == "v_prediction":
                    # v-prediction parameterization
                    pred_original_sample = alpha_t.sqrt() * curr_latent - beta_t.sqrt() * model_output
                    pred_epsilon = alpha_t.sqrt() * model_output + beta_t.sqrt() * curr_latent
                    pred_sample_direction = (1 - alpha_t_next).sqrt() * pred_epsilon
                    next_latent = alpha_t_next.sqrt() * pred_original_sample + pred_sample_direction
                else:
                    # Default to epsilon parameterization
                    next_latent = (curr_latent - (1-alpha_t).sqrt() * model_output) * (alpha_t_next/alpha_t).sqrt() + (1-alpha_t_next).sqrt() * model_output
                
                latents.append(next_latent)
                
        # Return reversed latents (from noisy to clean)
        return list(reversed(latents))
    
    def style_transfer(
        self, 
        content_image, 
        style_image,
        num_inference_steps=50,
        gamma=None,
        temperature=None,
        without_init_adain=None,
        without_attn_injection=None,
        output_type="pil",
        return_dict=True,
        guidance_scale=1.0,
        save_intermediates_dir=None
    ):
        """Main style transfer function"""
        # Update parameters if provided
        self.update_parameters(
            gamma=gamma,
            temperature=temperature,
            without_init_adain=without_init_adain,
            without_attn_injection=without_attn_injection
        )
        
        # Reset the StyleID state
        self.state.reset()
        
        device = self.device
        
        # Create save directory if needed
        if save_intermediates_dir is not None:
            os.makedirs(save_intermediates_dir, exist_ok=True)
            os.makedirs(os.path.join(save_intermediates_dir, "intermediate"), exist_ok=True)
        
        # Set number of steps for scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Initialize time step mapping
        self.state.setup_timestep_mapping(self.scheduler)
        
        # Preprocess images if they're numpy arrays
        if isinstance(content_image, np.ndarray):
            content_image = normalize(content_image).to(device=device, dtype=self.vae.dtype)
        if isinstance(style_image, np.ndarray):
            style_image = normalize(style_image).to(device=device, dtype=self.vae.dtype)
        
        # Get empty text embeddings (or use custom text if provided)
        text_embeddings = self.get_text_embedding()
        
        # Step 1: Encode images to latent space
        style_latent = self.encode_image(style_image)
        content_latent = self.encode_image(content_image)
        
        # Step 2: DDIM Inversion of style image
        self.state.to_invert_style()
        print("Inverting style image...")
        style_latents = self.ddim_inversion(style_latent, text_embeddings)
        
        if save_intermediates_dir is not None:
            # Save style inversion intermediates
            style_recon = self.decode_latent(style_latents[0])
            style_recon_np = denormalize(style_recon)[0]
            save_image(style_recon_np, os.path.join(save_intermediates_dir, "intermediate/latent_style.jpg"))
        
        # Step 3: DDIM Inversion of content image
        self.state.to_invert_content()
        print("Inverting content image...")
        content_latents = self.ddim_inversion(content_latent, text_embeddings)
        
        if save_intermediates_dir is not None:
            # Save content inversion intermediates
            content_recon = self.decode_latent(content_latents[0])
            content_recon_np = denormalize(content_recon)[0]
            save_image(content_recon_np, os.path.join(save_intermediates_dir, "intermediate/latent_content.jpg"))
        
        # Step 4: Style Transfer
        self.state.to_transfer()
        print("Transferring style...")
        
        # Initial latent processing - AdaIN if enabled
        if not self.without_init_adain:
            # Apply AdaIN (Adaptive Instance Normalization)
            latent_cs = (content_latents[-1] - content_latents[-1].mean(dim=(2, 3), keepdim=True)) / (
                content_latents[-1].std(dim=(2, 3), keepdim=True) + 1e-4
            ) * style_latents[-1].std(dim=(2, 3), keepdim=True) + style_latents[-1].mean(dim=(2, 3), keepdim=True)
        else:
            latent_cs = content_latents[-1]
        
        # DDIM sampling for style transfer
        result_latents = []
        result_images = []
        
        current_latent = latent_cs
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampling")):
            # Set current timestep
            self.state.set_timestep(t.item())
            
            # Temporarily disable attention injection if requested
            if self.without_attn_injection:
                # Save current state of processors to restore after this step
                current_processors = {k: p for k, p in self.unet.attn_processors.items()}
                # Replace StyleIDAttnProcessor with standard AttnProcessor
                for k, p in current_processors.items():
                    if isinstance(p, StyleIDAttnProcessor):
                        self.unet.attn_processors[k] = AttnProcessor()
            
            # Model prediction
            with torch.no_grad():
                noise_pred = self.unet(current_latent, t, encoder_hidden_states=text_embeddings).sample
                
                # Guidance scale can be used if text conditioning is added
                if guidance_scale > 1.0:
                    # This would require both conditional and unconditional embeddings
                    # For StyleID we typically use 1.0 (no guidance)
                    pass
                
                # DDIM step
                current_latent = self.scheduler.step(noise_pred, t, current_latent).prev_sample
                
                # Add the current prediction to our results
                result_latents.append(current_latent)
                
                # Save intermediate result
                if i % 5 == 0 or i == len(self.scheduler.timesteps) - 1:
                    with torch.no_grad():
                        image = self.decode_latent(current_latent)
                        result_images.append(image)
                
            # Restore processors if we temporarily disabled them
            if self.without_attn_injection:
                self.unet.set_attn_processor(current_processors)
                
        # Final image decoding
        with torch.no_grad():
            final_image = self.decode_latent(current_latent)
        
        # Save final result and intermediate results if requested
        if save_intermediates_dir is not None:
            final_np = denormalize(final_image)[0]
            save_image(final_np, os.path.join(save_intermediates_dir, "stylized_image.jpg"))
            
            # Save a grid of intermediate results
            if len(result_images) > 1:
                all_images = torch.cat(result_images, dim=3)  # Concatenate along width
                all_images_np = denormalize(all_images)[0]
                save_image(all_images_np, os.path.join(save_intermediates_dir, "reverse_stylized.jpg"))
        
        # Convert to output format
        if output_type == "pil":
            final_images = self.image_processor.postprocess(final_image, output_type="pil")
        elif output_type == "np":
            final_images = denormalize(final_image)
        else:
            final_images = final_image
            
        if not return_dict:
            return final_images
            
        return StableDiffusionPipelineOutput(images=final_images, nsfw_content_detected=[False] * len(final_images))
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the StyleID pipeline from a pretrained model"""
        # First load using the standard loading
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Create our StyleID pipeline
        styleid_pipe = cls(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            safety_checker=pipeline.safety_checker if hasattr(pipeline, "safety_checker") else None,
            feature_extractor=pipeline.feature_extractor if hasattr(pipeline, "feature_extractor") else None,
            requires_safety_checker=pipeline.config.requires_safety_checker if hasattr(pipeline.config, "requires_safety_checker") else False,
        )
        
        # Replace the scheduler with DDIM which is needed for inversion
        from diffusers import DDIMScheduler
        if not isinstance(pipeline.scheduler, DDIMScheduler):
            scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            styleid_pipe.scheduler = scheduler
        
        return styleid_pipe

   
# Simple demo usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnt_fn", type=str, required=True, help="Content image path")
    parser.add_argument("--sty_fn", type=str, required=True, help="Style image path")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--sd_version", type=str, default="1.5", choices=["1.5", "2.0", "2.1-base", "2.1"], help="Stable Diffusion version")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query preservation strength")
    parser.add_argument("--T", type=float, default=1.5, help="Attention temperature scaling")
    parser.add_argument("--without_init_adain", action="store_true", help="Disable initial latent AdaIN")
    parser.add_argument("--without_attn_injection", action="store_true", help="Disable attention-based style injection")
    
    args = parser.parse_args()
    
    # Map SD version to model paths
    sd_model_map = {
        "1.5": "runwayml/stable-diffusion-v1-5",
        "2.0": "stabilityai/stable-diffusion-2-base",
        "2.1-base": "stabilityai/stable-diffusion-2-1-base",
        "2.1": "stabilityai/stable-diffusion-2-1"
    }
    
    # Load content and style images
    content_image = cv2.imread(args.cnt_fn)[:, :, ::-1]  # BGR to RGB
    style_image = cv2.imread(args.sty_fn)[:, :, ::-1]    # BGR to RGB
    
    # Create StyleID pipeline
    pipe = StyleIDPipeline.from_pretrained(
        sd_model_map[args.sd_version],
        torch_dtype=torch.float16,
    ).to("cuda")
    
    # Set StyleID parameters
    pipe.update_parameters(
        gamma=args.gamma,
        temperature=args.T,
        without_init_adain=args.without_init_adain,
        without_attn_injection=args.without_attn_injection
    )
    
    # Run style transfer
    stylized_image = pipe.style_transfer(
        content_image=content_image,
        style_image=style_image,
        num_inference_steps=args.ddim_steps,
        save_intermediates_dir=args.save_dir
    )
    
    print(f"Style transfer complete. Results saved to {args.save_dir}") 