# scripts/insert_features.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat.extract_furniture_feature import extract_feature, save_patch_features
from app.milvus_client import init_milvus, drop_milvus_collection
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import csv


def insert_images_into_milvus(img_folder, patch_dir="./patch_store", batch_size=500):
    """插入图像到Milvus（同时存储全局特征和patch特征）- 分批插入带进度条"""
    drop_milvus_collection()
    collection = init_milvus()
    
    # 创建存储目录
    os.makedirs(patch_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)
    
    print(f"📁 开始处理 {total_images} 张图片，批次大小: {batch_size}")
    
    # 分批处理
    for batch_start in tqdm(range(0, total_images, batch_size), 
                           desc="🚀 插入进度", 
                           unit="batch"):
        batch_end = min(batch_start + batch_size, total_images)
        batch_files = image_files[batch_start:batch_end]
        
        batch_vectors, batch_paths = [], []
        
        # 处理当前批次
        for filename in batch_files:
            try:
                path = os.path.abspath(os.path.join(img_folder, filename))
  
                gem_feat = extract_feature(path)  # 全局特征

                if gem_feat is not None:
                    batch_vectors.append(gem_feat.tolist())
                    batch_paths.append(path)
                    
                    # 提取并保存patch特征
                    patch_output_path = os.path.join(patch_dir, f"{os.path.splitext(filename)[0]}_patch.npy")
                    save_patch_features(path, patch_output_path)
                else:
                    print(f"❌ 提取图像特征失败: {path}")

            except Exception as e:
                print(f"❌ 处理图片 {filename} 时出错: {e}")
                continue
        
        # 插入当前批次数据
        if batch_vectors:
            collection.insert([batch_vectors, batch_paths])
            print(f"✅ 已插入批次 {batch_start//batch_size + 1}，包含 {len(batch_vectors)} 张图片")
    
    # 最后刷新
    collection.flush()
    print(f"🎉 成功插入 {total_images} 张图片，并预加载patch特征")


def insert_texts_into_milvus(img_folder, description_file, batch_size=500):
    """插入文本描述到Milvus - 分批插入带进度条"""
    # 由于当前未实现文本特征提取，暂时注释掉此功能
    print("⚠️ 文本特征提取功能暂未实现，跳过文本插入")
    return
    """
    collection = init_milvus()
    
    # 读取描述文件
    product_descriptions = {}
    with open(description_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_descriptions[row['product_id']] = row['description']
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # 从描述文件中创建反向映射：描述 -> 产品ID
    description_to_id = {desc: pid for pid, desc in product_descriptions.items()}
    
    total_images = len(image_files)
    print(f"📁 开始处理 {total_images} 张图片的文本描述，批次大小: {batch_size}")
    
    # 分批处理
    for batch_start in tqdm(range(0, total_images, batch_size), 
                           desc="🚀 插入文本进度", 
                           unit="batch"):
        batch_end = min(batch_start + batch_size, total_images)
        batch_files = image_files[batch_start:batch_end]
        
        batch_vectors, batch_paths = [], []
        
        # 处理当前批次
        for filename in batch_files:
            try:
                # 从文件名获取描述（移除扩展名）
                description = os.path.splitext(filename)[0]
                
                # 检查是否有对应的产品ID
                if description not in description_to_id:
                    print(f"⚠️ 未找到描述 '{description}' 对应的产品ID，跳过")
                    continue
                
                product_id = description_to_id[description]
                
                # 提取文本特征
                text_feat = extract_text_feature(description)
                
                if text_feat is not None:
                    batch_vectors.append(text_feat.tolist())
                    # 使用图片路径
                    path = os.path.abspath(os.path.join(img_folder, filename))
                    batch_paths.append(path)
                else:
                    print(f"❌ 提取文本特征失败: {description}")
                
            except Exception as e:
                print(f"❌ 处理图片 {filename} 的文本描述时出错: {e}")
                continue
        
        # 插入当前批次数据
        if batch_vectors:
            collection.insert([batch_vectors, batch_paths])
            print(f"✅ 已插入文本批次 {batch_start//batch_size + 1}，包含 {len(batch_vectors)} 条记录")
    
    # 最后刷新
    collection.flush()
    print(f"🎉 成功插入 {total_images} 张图片的文本描述")
    """


def insert_images_and_texts(img_folder, description_file=None, patch_dir="./patch_store", batch_size=500):
    """插入图像和文本到Milvus"""
    # 先插入图像
    insert_images_into_milvus(img_folder, patch_dir, batch_size)
    
    # 如果提供了描述文件，则插入文本
    # if description_file and os.path.exists(description_file):
    #     insert_texts_into_milvus(img_folder, description_file, batch_size)


# 使用新的图片文件夹和描述文件
insert_images_and_texts("./datasets/new_goods", "./optimized_data_descriptions.csv")