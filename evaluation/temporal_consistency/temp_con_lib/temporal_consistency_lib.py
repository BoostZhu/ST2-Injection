import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import clip
from torch.utils.data import Dataset, DataLoader
# <<< MODIFIED: Removed AutoImageProcessor as it's no longer needed
from transformers import AutoModel

# 忽略不必要的警告
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

# --- 1. 数据加载 ---
class ImageFrameDataset(Dataset):
    """一个健壮的数据集类，用于加载文件夹中的所有图像帧。"""
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.image_paths = []
        supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
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

# --- 2. 特征提取器 (已修复 DINOv3 加载) ---
class FeatureExtractor:
    """
    特征提取器，模型在初始化时加载一次。
    新增了对 DINOv3 的支持。
    """
    def __init__(self, model_type="dino", device="cuda"):
        self.device = device
        self.model_type = model_type.lower()
        
        if self.model_type == "dino":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        # ###################### FIX IS HERE ######################
        elif self.model_type == "dino3":
            # The DINOv3 model on Hugging Face has an incomplete config,
            # so AutoImageProcessor fails. We manually create the transform instead.
            # These are the standard ImageNet parameters that DINOv3 uses.
            model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            self.model = AutoModel.from_pretrained(model_name)
            
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        # #########################################################
            
        elif self.model_type == "clip":
            self.model, self.transform = clip.load("ViT-B/32", device=self.device)
        else:
            raise ValueError("model_type must be 'dino', 'dino3', or 'clip'")
            
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def extract_features_batch(self, image_batch):
        """
        提取一个批次图像的特征并归一化。
        """
        image_batch = image_batch.to(self.device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            if self.model_type == "dino":
                features = self.model(image_batch)
            elif self.model_type == "dino3":
                features = self.model(image_batch).pooler_output
            else: # clip
                features = self.model.encode_image(image_batch)
        
        normalized_features = features / features.norm(dim=1, keepdim=True)
        return normalized_features

# --- 3. 计算函数 ---
def calculate_temporal_consistency(folder, feature_extractor, batch_size=32):
    """
    计算时间一致性的优化版本。
    """
    dataset = ImageFrameDataset(folder, transform=feature_extractor.transform)
    if len(dataset) < 2:
        return 0.0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    all_features = []
    for image_batch in dataloader:
        features = feature_extractor.extract_features_batch(image_batch)
        all_features.append(features)
        
    all_features = torch.cat(all_features, dim=0)
    
    features_t0 = all_features[:-1]
    features_t1 = all_features[1:]
    
    similarities = torch.sum(features_t0 * features_t1, dim=1)
    score = similarities.mean().item()
    
    return score