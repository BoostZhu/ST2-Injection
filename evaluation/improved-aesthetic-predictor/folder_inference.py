'''import webdataset as wds

import io
import matplotlib.pyplot as plt
import os
import json
from datasets import load_dataset'''
from warnings import filterwarnings

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm
from PIL import Image
from os.path import join

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file:

# 修改图片路径为文件夹路径
folder_path = "/root/autodl-tmp/video_style_transfer/results/with_PA_fusion_2/Gondola_stylized_cat_flower"





# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("/root/autodl-tmp/video_style_transfer/evaluation/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   


# 在模型定义之后，预测之前，添加以下代码
def get_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

def predict_folder(folder_path, model, model2, preprocess, device):
    image_paths = get_image_paths(folder_path)
    scores = []
    
    print(f"Found {len(image_paths)} images in folder")
    
    for img_path in tqdm.tqdm(image_paths):
        try:
            pil_image = Image.open(img_path)
            image = preprocess(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model2.encode_image(image)
            
            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
            scores.append(prediction.item())
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return scores

# 替换原来的单图片预测代码为：
scores = predict_folder(folder_path, model, model2, preprocess, device)

if scores:
    average_score = sum(scores) / len(scores)
    print("\n评分统计:")
    print(f"图片数量: {len(scores)}")
    print(f"平均分数: {average_score:.2f}")
    print(f"最高分数: {max(scores):.2f}")
    print(f"最低分数: {min(scores):.2f}")
else:
    print("没有成功处理任何图片")


