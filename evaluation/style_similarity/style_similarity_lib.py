import torch
import torchvision
from torchvision.transforms import v2 # Use the recommended v2 transforms
from PIL import Image
import os
import clip
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

# Ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="The _.split_kwargs processor argument is deprecated")

# --- Transform Function (from DINOv3 documentation) ---
def make_transform(resize_size: int = 224):
    """Creates the standard ImageNet transform for DINOv3 models."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

# --- 1. Data Loading Module (Unchanged) ---
class ImageFrameDataset(Dataset):
    """A robust dataset class for loading all image frames in a folder."""
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.image_paths = []
        supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The specified folder does not exist: {folder_path}")

        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_extensions:
                self.image_paths.append(os.path.join(folder_path, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- 2. Feature Extractor Module (Core Correction) ---
class FeatureExtractor:
    """Feature extractor, supporting DINOv2, CLIP, and a corrected DINOv3."""
    def __init__(self, model_type="dino", device="cuda"):
        self.device = device
        self.model_type = model_type.lower()
        print(f"  Loading {self.model_type.upper()} model...")
        
        if self.model_type == "dino":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
            # --- CORRECTION: Aligned with the official transform to fix the LANCZOS error ---
            # It now uses the same robust transform as DINOv3.
            self.transform = make_transform()
            self.model = self.model.to(self.device).eval()
        elif self.model_type == "clip":
            self.model, self.transform = clip.load("ViT-B/32", device=self.device)
            self.model = self.model.eval()
        elif self.model_type == "dinov3":
            pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            self.transform = make_transform()
            self.model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
            self.model = self.model.eval()
        else:
            raise ValueError("model_type must be 'dino', 'clip', or 'dinov3'")
            
        print(f"  {self.model_type.upper()} model loaded successfully.")

    @torch.no_grad()
    def extract_features_single(self, image):
        """Extracts and normalizes features from a single PIL image."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        if self.model_type == "dinov3":
            outputs = self.model(pixel_values=image_tensor.to(self.model.device))
            features = outputs.pooler_output
        else:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                if self.model_type == "dino":
                    features = self.model(image_tensor)
                else: # clip
                    features = self.model.encode_image(image_tensor)
        
        normalized_features = features / features.norm(dim=1, keepdim=True)
        return normalized_features

    @torch.no_grad()
    def extract_features_batch(self, image_batch):
        """Extracts and normalizes features from a batch of images."""
        image_batch = image_batch.to(self.device)

        if self.model_type == "dinov3":
            outputs = self.model(pixel_values=image_batch.to(self.model.device))
            features = outputs.pooler_output
        else:
             with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                if self.model_type == "dino":
                    features = self.model(image_batch)
                else: # clip
                    features = self.model.encode_image(image_batch)
        
        normalized_features = features / features.norm(dim=1, keepdim=True)
        return normalized_features

# --- 3. Style Similarity Calculation Function (Unchanged) ---
def calculate_style_similarity(frames_folder, style_image_path, feature_extractor, batch_size=32):
    if not os.path.exists(style_image_path):
        return 0.0

    style_image = Image.open(style_image_path).convert('RGB')
    style_features = feature_extractor.extract_features_single(style_image)

    dataset = ImageFrameDataset(frames_folder, transform=feature_extractor.transform)
    
    if len(dataset) == 0:
        return 0.0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    all_similarities = []
    for image_batch in dataloader:
        frame_features = feature_extractor.extract_features_batch(image_batch)
        similarities = torch.sum(frame_features * style_features.to(frame_features.device), dim=1)
        all_similarities.append(similarities)
        
    all_similarities = torch.cat(all_similarities, dim=0)
    score = all_similarities.mean().item()
    
    return score