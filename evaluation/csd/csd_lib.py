# File: evaluation/csd/csd_lib.py

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from glob import glob
import os
import clip
import copy
from collections import OrderedDict
from huggingface_hub import hf_hub_download

class CSDScorer:
    """A self-contained class for calculating CSD style similarity."""
    def __init__(self, device="cuda"):
        self.device = device
        self.model = self._load_model().to(self.device).eval()
        self.preprocessor = self._get_preprocessor()
        self.style_embedding_cache = {}

    def _get_preprocessor(self):
        return transforms.Compose([
            transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    def _load_model(self):
        # --- Exact model definition to match weights ---
        class CSD_CLIP(nn.Module):
            def __init__(self, name='vit_large'):
                super(CSD_CLIP, self).__init__()
                if name == 'vit_large':
                    clipmodel, _ = clip.load("ViT-L/14", device="cpu")
                    self.backbone = clipmodel.visual
                else:
                    raise NotImplementedError('Only ViT-L is supported')
                for p in self.backbone.parameters(): p.data = p.data.float()
                self.last_layer_style = copy.deepcopy(self.backbone.proj)
                self.last_layer_content = copy.deepcopy(self.backbone.proj)
                self.backbone.proj = None
            def forward(self, input_data):
                feature = self.backbone(input_data)
                style_output = feature @ self.last_layer_style
                return nn.functional.normalize(style_output, dim=1, p=2)

        def convert_state_dict(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."): k = k.replace("module.", "")
                new_state_dict[k] = v
            return new_state_dict
        
        model = CSD_CLIP(name='vit_large')
        model_path = hf_hub_download(repo_id="tomg-group-umd/CSD-ViT-L", filename="pytorch_model.bin")
        full_checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = convert_state_dict(full_checkpoint['model_state_dict'])
        model.load_state_dict(state_dict, strict=True)
        print("INFO: CSD ViT-L model loaded successfully.")
        return model

    @torch.no_grad()
    def get_style_embedding(self, image_path: str):
        if image_path in self.style_embedding_cache:
            return self.style_embedding_cache[image_path]
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.preprocessor(image).unsqueeze(0).to(self.device)
            embedding = self.model(processed_image).squeeze().cpu().numpy()
            norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            self.style_embedding_cache[image_path] = norm_embedding
            return norm_embedding
        except Exception:
            return None

    def calculate_style_similarity_for_folder(self, generated_folder, style_reference_dir):
        try:
            folder_name = os.path.basename(generated_folder)
            style_name = folder_name.split('_stylized_')[0]
        except IndexError:
             raise ValueError(f"Could not parse style name from folder: {folder_name}")

        style_img_path_list = glob(os.path.join(style_reference_dir, f"{style_name}.*"))
        if not style_img_path_list:
             raise FileNotFoundError(f"Could not find style reference image for '{style_name}' in {style_reference_dir}")
        
        style_embedding = self.get_style_embedding(style_img_path_list[0])
        if style_embedding is None:
            raise RuntimeError(f"Failed to get embedding for style image: {style_img_path_list[0]}")

        frame_paths = sorted(glob(os.path.join(generated_folder, '*.[jp][pn]g')))
        if not frame_paths: return (0.0, 0.0)

        video_scores_cos, video_scores_l2 = [], []
        for frame_path in frame_paths:
            frame_embedding = self.get_style_embedding(frame_path)
            if frame_embedding is not None:
                cos_sim = np.dot(style_embedding, frame_embedding)
                l2_dist = np.linalg.norm(style_embedding - frame_embedding)
                video_scores_cos.append(cos_sim)
                video_scores_l2.append(l2_dist)
        
        avg_cos_sim = np.mean(video_scores_cos) if video_scores_cos else 0.0
        avg_l2_dist = np.mean(video_scores_l2) if video_scores_l2 else 0.0
        return float(avg_cos_sim), float(avg_l2_dist)