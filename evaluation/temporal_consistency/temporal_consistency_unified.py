import torch
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import clip

class FeatureExtractor:
    def __init__(self, model_type="dino"):
        """
        初始化特征提取器
        Args:
            model_type: 'dino' 或 'clip'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type.lower()
        
        if self.model_type == "dino":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                  std=(0.229, 0.224, 0.225))
            ])
        elif self.model_type == "clip":
            self.model, self.transform = clip.load("ViT-B/32", device=self.device)
        else:
            raise ValueError("model_type must be either 'dino' or 'clip'")
            
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features(self, image):
        """提取图像特征并归一化"""
        if self.model_type == "dino":
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image)
        else:  # clip
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image)
                
        # 特征归一化
        normalized_features = features / torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True))
        return normalized_features

def preprocess_image(image_path, model_type):
    """根据模型类型预处理图像"""
    image = Image.open(image_path).convert('RGB')
    return image

def calculate_temporal_consistency(folder, model_type="dino"):
    """
    计算时间一致性
    Args:
        folder: 图像文件夹路径
        model_type: 使用的模型类型 ('dino' 或 'clip')
    Returns:
        float: 时间一致性分数
    """
    feature_extractor = FeatureExtractor(model_type)
    file_list = sorted(glob(folder + '/*png'))
    
    if len(file_list) == 0:
        file_list = sorted(glob(folder + '/*jpg')) or sorted(glob(folder + '/*jpeg')) or sorted(glob(folder + '/*webp'))
    
    if len(file_list) == 0:
        raise ValueError(f"在{folder}中没有找到图像文件")
        
    print(f"找到{len(file_list)}个图像文件")
    normalized_feature_list = []

    # 提取每一帧的特征
    for f_path in file_list:
        image = preprocess_image(f_path, model_type)
        normalized_features = feature_extractor.extract_features(image)
        normalized_feature_list.append(normalized_features)

    # 计算时间一致性
    frame_const_list = []
    frame_const_list_sum = 0.0
    for i in range(len(normalized_feature_list) - 1):
        sim_i = torch.sum(normalized_feature_list[i] * normalized_feature_list[i + 1], dim=1)
        frame_const_list.append(sim_i)
        frame_const_list_sum += sim_i

    frame_const_list_avg = frame_const_list_sum / (len(normalized_feature_list) - 1)
    score = frame_const_list_avg.item()
    
    model_name = "DINOv2" if model_type == "dino" else "CLIP"
    print(f'{model_name}时间一致性分数: {score:.4f}')
    return score

def calculate_with_both_models(folder):
    """
    同时使用DINO和CLIP计算时间一致性
    Args:
        folder: 图像文件夹路径
    Returns:
        dict: 包含两个模型计算结果的字典
    """
    print(f"正在处理文件夹: {folder}")
    print("-" * 50)
    
    # 使用DINO计算
    dino_score = calculate_temporal_consistency(folder, "dino")
    
    # 使用CLIP计算
    clip_score = calculate_temporal_consistency(folder, "clip")
    
    # 计算平均分数
    avg_score = (dino_score + clip_score) / 2
    
    print("-" * 50)
    print(f"综合时间一致性分数: {avg_score:.4f}")
    
    return {
        "dino": dino_score,
        "clip": clip_score,
        "average": avg_score
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='计算时间一致性')
    parser.add_argument('folder_path', type=str, help='图像文件夹路径')
    parser.add_argument('--model', type=str, default='both', 
                      choices=['dino', 'clip', 'both'],
                      help='使用的模型类型 (dino, clip 或 both)')
    args = parser.parse_args()
    
    if args.model == 'both':
        calculate_with_both_models(args.folder_path)
    else:
        calculate_temporal_consistency(args.folder_path, args.model) 