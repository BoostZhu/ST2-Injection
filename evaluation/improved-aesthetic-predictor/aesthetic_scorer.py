import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import clip
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Model Definition (Copied from your script)
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# 2. A Robust Dataset for Loading Images
class AestheticDataset(Dataset):
    """Loads all valid images from a folder."""
    def __init__(self, folder_path, transform):
        super().__init__()
        self.transform = transform
        supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        self.image_paths = [
            os.path.join(folder_path, fname)
            for fname in sorted(os.listdir(folder_path))
            if os.path.splitext(fname)[1].lower() in supported_extensions
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)

# 3. An All-in-One Predictor Class
class AestheticPredictor:
    """
    Encapsulates model loading and batch prediction logic.
    """
    def __init__(self, mlp_model_path, clip_model_name="ViT-L/14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load CLIP model for feature extraction
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        # Load the aesthetic scoring MLP model
        clip_embedding_dim = self.clip_model.visual.output_dim
        self.mlp_model = MLP(clip_embedding_dim)
        try:
            state_dict = torch.load(mlp_model_path, map_location=self.device)
            self.mlp_model.load_state_dict(state_dict)
        except RuntimeError:
            # This handles cases where the model was saved as a pl.LightningModule
            self.mlp_model = MLP.load_from_checkpoint(mlp_model_path, input_size=clip_embedding_dim)

        self.mlp_model.to(self.device)
        self.mlp_model.eval()

    @torch.no_grad()
    def predict_folder_scores(self, folder_path, batch_size):
        """
        Calculates aesthetic scores for all images in a folder using batch processing.
        """
        dataset = AestheticDataset(folder_path, self.preprocess)
        if not dataset:
            print(f"Warning: No images found in {folder_path}")
            return None

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_scores = []
        for image_batch in dataloader:
            image_batch = image_batch.to(self.device)
            
            # 1. Get CLIP features in a batch
            image_features = self.clip_model.encode_image(image_batch)
            
            # 2. Normalize features ON THE GPU using PyTorch (much faster)
            # Use .float() to ensure compatibility with the MLP model
            image_features = F.normalize(image_features, p=2, dim=-1).float()
            
            # 3. Get aesthetic predictions in a batch
            predictions = self.mlp_model(image_features)
            
            # Add scores from the batch to our master list
            all_scores.extend(predictions.cpu().numpy().flatten().tolist())

        if not all_scores:
            return None
            
        return {
            "mean_score": sum(all_scores) / len(all_scores),
            "max_score": max(all_scores),
            "min_score": min(all_scores),
            "image_count": len(all_scores)
        }