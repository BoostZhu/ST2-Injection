import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import clip
from torch.utils.data import Dataset, DataLoader

# 忽略不必要的警告
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

# --- 1. 优化数据加载 ---
class ImageFrameDataset(Dataset):
    """一个健壮的数据集类，用于加载文件夹中的所有图像帧。"""
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.image_paths = []
        # 定义支持的图片文件后缀
        supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        
        # 遍历文件夹，只添加支持的图片文件
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_extensions:
                self.image_paths.append(os.path.join(folder_path, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 打开图片并确保是RGB格式
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- 2. 优化特征提取器 ---
class FeatureExtractor:
    """
    特征提取器，模型在初始化时加载一次。
    新增了批量提取特征的方法。
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
        elif self.model_type == "clip":
            # CLIP加载时可以直接指定device
            self.model, self.transform = clip.load("ViT-B/32", device=self.device)
        else:
            raise ValueError("model_type must be either 'dino' or 'clip'")
            
        self.model = self.model.to(self.device).eval()

    @torch.no_grad() # 在方法级别禁用梯度，更安全
    def extract_features_batch(self, image_batch):
        """
        提取一个批次图像的特征并归一化。
        使用自动混合精度 (AMP) 加速。
        """
        # 确保输入在正确的设备上
        image_batch = image_batch.to(self.device)
        
        # 使用混合精度加速
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            if self.model_type == "dino":
                features = self.model(image_batch)
            else: # clip
                features = self.model.encode_image(image_batch)
        
        # 特征归一化 (L2 Norm)
        normalized_features = features / features.norm(dim=1, keepdim=True)
        return normalized_features

# --- 3. 优化计算函数 ---
def calculate_temporal_consistency(folder, feature_extractor, batch_size=32):
    """
    计算时间一致性的优化版本。
    - 接收一个已初始化的 feature_extractor 对象。
    - 使用 DataLoader 进行批量数据加载和处理。
    - 使用向量化操作计算相似度。
    """
    device = feature_extractor.device
    
    # 使用优化后的Dataset和DataLoader
    dataset = ImageFrameDataset(folder, transform=feature_extractor.transform)
    
    if len(dataset) < 2:
        # 如果图片少于2张，无法计算一致性
        return 0.0

    # num_workers > 0 会启用多进程加载，大幅提升I/O效率
    # pin_memory=True 能加速数据从CPU到GPU的传输
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    all_features = []
    for image_batch in dataloader:
        # 批量提取特征
        features = feature_extractor.extract_features_batch(image_batch)
        all_features.append(features)
        
    # 将所有批次的特征拼接成一个大张量
    all_features = torch.cat(all_features, dim=0)
    
    # --- 向量化计算 ---
    # 取出 t 和 t+1 时刻的特征向量
    features_t0 = all_features[:-1]
    features_t1 = all_features[1:]
    
    # 计算成对的余弦相似度 (点积)
    # (N-1, D) * (N-1, D) -> (N-1, D) -> (N-1)
    similarities = torch.sum(features_t0 * features_t1, dim=1)
    
    # 计算平均分并返回
    score = similarities.mean().item()
    
    return score