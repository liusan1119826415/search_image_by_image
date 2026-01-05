# extract_features.py
# 使用训练后的模型提取图像特征

import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModel
import numpy as np
from tqdm import tqdm
import argparse
import json
from feat.train_trendyol import FurnitureFinetuneModel
# ----------------------------
# 配置
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints_hard/furniture_hard_epoch15.pth"  # 默认模型路径


# 图像预处理
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ----------------------------
# ArcFace Head（与训练脚本保持一致）
# ----------------------------
import math
import torch.nn.functional as F




# ----------------------------
# 特征提取函数
# ----------------------------

print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = checkpoint['arcface_head.weight'].shape[0]
# 创建模型实例
model = FurnitureFinetuneModel(
    out_dim=256,
    use_arcface=True if num_classes else False,
    num_classes=num_classes
).to(DEVICE)

# 加载模型权重

model.load_state_dict(checkpoint, strict=False)  # strict=False 允许部分匹配
model.eval()

print("Model loaded successfully!")

def extract_feature(image_path):
    """提取单张图像的特征"""
    try:
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # 添加batch维度
        
        # 提取特征
        with torch.no_grad():
            features = model(input_tensor)
            
        # 转换为numpy数组
        features_np = features.cpu().numpy().flatten()
        
        # L2归一化
        features_np = features_np / np.linalg.norm(features_np)
        
        return features_np
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_patch_features_with_hook(image_path):
    """使用hook方法提取单张图像的patch特征用于精排"""
    try:
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # 添加batch维度
        
        intermediate_output = {}
        
        def hook_fn(module, input, output):
            # 存储输入特征（在第一个transformer块之前的特征）
            intermediate_output['features'] = input[0] if isinstance(input, tuple) else input
        
        # 注册hook到第一个transformer块的归一化层
        handle = model.backbone.model.backbone.blocks[0].norm1.register_forward_hook(hook_fn)
        
        # 提取patch特征
        with torch.no_grad():
            # 前向传播
            _ = model(input_tensor)
            
            # 移除hook
            handle.remove()
        
        if 'features' in intermediate_output:
            # 转换为numpy并移除batch维度
            patch_features_np = intermediate_output['features'].squeeze(0).cpu().numpy()  # [N, hidden_dim]
            return patch_features_np
        else:
            print("Warning: Could not extract patch features with hook method")
            return None
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_patch_features_from_file(image_path):
    """从预存的npy文件中加载patch特征"""
    # 生成patch特征文件路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    patch_file_path = f"./patch_store/{base_name}_patch.npy"
    
    if os.path.exists(patch_file_path):
        try:
            patch_features = np.load(patch_file_path)
            return patch_features
        except Exception as e:
            print(f"Error loading patch features from {patch_file_path}: {e}")
            return None
    else:
        print(f"Patch features file not found: {patch_file_path}")
        return None

def extract_patch_features(image_path):
    """提取单张图像的patch特征用于精排（优先从文件加载）"""
    # 首先尝试从预存的文件加载patch特征
    patch_features = extract_patch_features_from_file(image_path)
    if patch_features is not None:
        return patch_features
    else:
        # 如果文件不存在，则实时提取
        return extract_patch_features_with_hook(image_path)

def save_patch_features(image_path, output_path):
    """提取并保存patch特征到.npy文件"""
    patch_features = extract_patch_features_with_hook(image_path)  # 使用hook方法提取并保存
    if patch_features is not None:
        np.save(output_path, patch_features)
       # print(f"✅ Patch features saved to {output_path}, shape: {patch_features.shape}")
        return patch_features
    else:
        print(f"❌ Failed to extract patch features for {image_path}")
        return None

# ----------------------------
# 主函数
# ----------------------------
def main():
    img_path = "./img/222.jpg"
    features = extract_feature(img_path)
    if features is not None:
        print(f"Extracted feature vector of shape: {features.shape}")
    
    # 提取并保存patch特征示例
    patch_output_path = "./img/222_patches.npy"
    save_patch_features(img_path, patch_output_path)

if __name__ == "__main__":
    # 添加freeze_support以支持Windows上的多进程
    from multiprocessing import freeze_support
    freeze_support()
    main()