# train_hard_mining.py
import os
import random
import math
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import AutoModel

# ----------------------------
# 配置（可按需修改）
# ----------------------------
DATA_DIR = "./data"   # 你的家具数据集根目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32      # 减小batch size以适应不同商品数量(原32)
EPOCHS = 30           # 调整训练轮数(原45)
LR = 1e-4             # 微调学习率以获得更好的收敛(原1e-5)
MARGIN = 0.3          # 调整margin以获得更精细的特征区分(原0.5)
MODEL_SAVE_DIR = "./checkpoints_hard"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ArcFace参数
ARCFACE_MARGIN = 0.5  # ArcFace margin
ARCFACE_SCALE = 30.0  # ArcFace scale factor

# 解冻 backbone 最后几个 transformer block（字符串匹配）
# 例如 "blocks.9", "blocks.10", "blocks.11" — 根据你模型打印的名字调整
# 同时考虑解冻头部的一些层以适应特定任务
# 
# 推荐微调策略 ✅
# 微调方案	涉及层	参数量	适用场景	效果
# 方案A（最安全）	只微调 fc11 + bn11	中等	数据集较小（<10k 图）	👍 平衡效果和稳定性
# 方案B（增强）	微调 backbone 最后 2~3 层 + fc11 + bn11	较多	数据集较大（>10k 图）	🚀 显著提升相似度表现
# 方案C（全量）	全部解冻	极多	有数十万图像	⚠️ 需强大GPU与正则化
#
# 当前采用方案A：只微调 fc11 + bn11
#UNFREEZE_BLOCKS = ["fc11", "bn11"]  # 只微调头部层以平衡效果和稳定性
# 如果需要更强的微调效果，可以考虑方案B：
UNFREEZE_BLOCKS = ["fc11", "bn11"]  # 微调最后两层和头部层


# ----------------------------
# Dataset：返回 (anchor_img, positive_img, product_id)
# ----------------------------
class TripletFurnitureDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_images_per_product=2):
        self.root_dir = root_dir
        self.transform = transform
        # products: list of (product_path, list_of_image_paths)
        self.products = []
        
        # 遍历根目录下的所有商品文件夹（不再按类别分层）
        for product in os.listdir(root_dir):
            prod_path = os.path.join(root_dir, product)
            if not os.path.isdir(prod_path):
                continue
            imgs = [os.path.join(prod_path, f) for f in os.listdir(prod_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            # 确保每个商品至少有指定数量的图片
            if len(imgs) >= min_images_per_product:
                self.products.append((prod_path, imgs))
        
        # 产品数量
        self.num_products = len(self.products)
        if self.num_products == 0:
            raise RuntimeError(f"No product folders found under {root_dir}")
        print(f"Loaded {self.num_products} products with at least {min_images_per_product} images each")

    def __len__(self):
        return self.num_products

    def __getitem__(self, idx):
        # idx 对应一个 product（一个商品）
        prod_path, imgs = self.products[idx]
        # 随机挑两张作为 anchor + positive
        anchor_path, pos_path = random.sample(imgs, 2)
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(pos_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)

        # 返回 anchor, positive, product_id(idx)
        return anchor, positive, idx


# ----------------------------
# 多视角增强（新增）
# ----------------------------
# 基础增强
basic_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 强增强
strong_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.1),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 角度增强
angle_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 色彩增强
color_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 裁剪增强
crop_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.7, 1.3)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 多视角增强选择器
def get_random_transform():
    transforms = [basic_transform, strong_transform, angle_transform, color_transform, crop_transform]
    return random.choice(transforms)

# 修改Dataset以支持多视角增强
class MultiViewTripletFurnitureDataset(Dataset):
    def __init__(self, root_dir, min_images_per_product=2):
        self.root_dir = root_dir
        # products: list of (product_path, list_of_image_paths)
        self.products = []
        
        # 遍历根目录下的所有商品文件夹（不再按类别分层）
        for product in os.listdir(root_dir):
            prod_path = os.path.join(root_dir, product)
            if not os.path.isdir(prod_path):
                continue
            imgs = [os.path.join(prod_path, f) for f in os.listdir(prod_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            # 确保每个商品至少有指定数量的图片
            if len(imgs) >= min_images_per_product:
                self.products.append((prod_path, imgs))
        
        # 产品数量
        self.num_products = len(self.products)
        if self.num_products == 0:
            raise RuntimeError(f"No product folders found under {root_dir}")
        print(f"Loaded {self.num_products} products with at least {min_images_per_product} images each")

    def __len__(self):
        return self.num_products

    def __getitem__(self, idx):
        # idx 对应一个 product（一个商品）
        prod_path, imgs = self.products[idx]
        # 随机挑两张作为 anchor + positive
        anchor_path, pos_path = random.sample(imgs, 2)
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(pos_path).convert("RGB")

        # 应用随机选择的增强方式
        transform = get_random_transform()
        anchor = transform(anchor)
        positive = transform(positive)

        # 返回 anchor, positive, product_id(idx)
        return anchor, positive, idx


# ----------------------------
# ArcFace Head（新增）
# ----------------------------
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, margin=ARCFACE_MARGIN, scale=ARCFACE_SCALE):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input shape: (batch_size, in_features)
        # label shape: (batch_size)
        
        # 计算cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 计算sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 计算cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 将正确类别的cos(theta)替换为cos(theta + margin)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


# ----------------------------
# 模型定义：backbone 冻结大部分，只微调 head + 可选解冻最后几个 block
# 我们把 head 定义为 768->256 (如果 backbone 输出 768)
# 有些 Trendyol 版本其 last_hidden_state 可能已经是 256，脚本会自动检查
# ----------------------------
class FurnitureFinetuneModel(nn.Module):
    def __init__(self, pretrained_model_name="Trendyol/trendyol-dino-v2-ecommerce-256d", out_dim=256, unfreeze_blocks=None, use_arcface=False, num_classes=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.use_arcface = use_arcface
        # 冻结所有 backbone params 默认
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 检查并解冻指定的最后几层（如果匹配到名字）
        if unfreeze_blocks:
            matched = []
            for name, param in self.backbone.named_parameters():
                for pattern in unfreeze_blocks:
                    if pattern in name:
                        param.requires_grad = True
                        matched.append(name)
                        break
            print(f"解冻的 backbone 参数示例（匹配到 {len(matched)} 个）：")
            for n in matched[:20]:
                print("   ", n)
            if len(matched) == 0:
                print("⚠️ 未匹配到任何 backbone 参数名，请检查 UNFREEZE_BLOCKS 配置。")

        # head: map backbone 输出到 out_dim
        # 根据模型配置，Trendyol DINOv2 ecommerce模型的hidden_size是256
        self.fc = nn.Linear(256, out_dim)  # 修改为256
        self.bn = nn.BatchNorm1d(out_dim)
        # 添加dropout层以防止过拟合
        self.dropout = nn.Dropout(0.2)
        
        # ArcFace head (可选)
        if use_arcface and num_classes is not None:
            self.arcface_head = ArcFaceHead(out_dim, num_classes)
        else:
            self.arcface_head = None

    def forward_backbone(self, x):
        # 返回 backbone 的 raw embedding （尚未通过 head）
        out = self.backbone(x)
        # out 可能是 BaseModelOutput 有 last_hidden_state 或者直接 tensor
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state  # [B, N, hidden_dim]
        else:
            return out  # 有些实现直接返回 tensor

    def forward(self, x, labels=None):
        # x: [B, 3, H, W]
        with torch.set_grad_enabled(self.training):
            raw = self.forward_backbone(x)
            # raw shape: [B, N, hidden_dim] or maybe [B, hidden_dim] depending model
            if raw.dim() == 3:
                # 通常是 [B, N, C]，取全局平均或取 cls token
                feat = raw.mean(dim=1)  # -> [B, C]
            elif raw.dim() == 2:
                feat = raw  # 已经是 [B, C]
            else:
                raise RuntimeError(f"Unexpected backbone output dim: {raw.shape}")

            # 如果 backbone hidden_dim != fc.in_features（比如 256），自动适配
            if feat.shape[1] != self.fc.in_features:
                # 重新调整 fc 层以匹配 feat 的通道数（仅首次）
                in_dim = feat.shape[1]
                old_fc = self.fc
                self.fc = nn.Linear(in_dim, old_fc.out_features).to(feat.device)
                # 重置 bn（因为大小可能相同）
                self.bn = nn.BatchNorm1d(old_fc.out_features).to(feat.device)
                print(f"⚙️ 动态调整 head: fc 从 in_features={old_fc.in_features} -> {in_dim}")

            # 添加dropout层
            feat = self.dropout(feat)
            out = self.fc(feat)  # [B, out_dim]
            # BatchNorm 需要 B>1 才能正常工作（训练时），推理阶段用 eval()
            if out.shape[0] > 1:
                out = self.bn(out)
            out = nn.functional.normalize(out, dim=1)
            
            # 如果使用ArcFace并且提供了标签，则计算ArcFace输出
            if self.arcface_head is not None and labels is not None:
                arcface_out = self.arcface_head(out, labels)
                return out, arcface_out  # 返回普通embedding和ArcFace输出
                
            return out  # 只返回普通embedding


# ----------------------------
# Helper: compute pairwise cosine similarities
# ----------------------------
def pairwise_cosine_similarity(a, b):
    # a: [N, D], b: [M, D] (both assumed normalized)
    # return [N, M] similarity matrix
    return a @ b.T


# ----------------------------
# 改进的训练循环：保存每个模型并记录损失
# ----------------------------
def validate_model(model, dataloader, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2)
    
    # 检查dataloader是否为空
    if len(dataloader) == 0:
        print("Warning: Validation dataloader is empty, returning 0 loss")
        return 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            anchors, positives, prod_ids = batch
            anchors = anchors.to(device)
            positives = positives.to(device)
            prod_ids = prod_ids.to(device)
            
            # 合并为一组图片进行一次前向（2B 张）
            imgs = torch.cat([anchors, positives], dim=0)  # [2B, C, H, W]
            embeddings = model(imgs)  # [2B, D], 已归一化
            
            B = anchors.size(0)
            emb_anchor = embeddings[:B]    # [B, D]
            emb_pos = embeddings[B:2*B]    # [B, D]
            
            # 构造 candidate pool
            pool_emb = embeddings  # [2B, D]
            pool_prod_ids = torch.cat([prod_ids, prod_ids], dim=0)  # [2B]
            
            # 计算 anchor 与 pool 的相似度矩阵
            sim_matrix = pairwise_cosine_similarity(emb_anchor, pool_emb)  # [B, 2B]
            # mask out same-product entries
            prod_ids_expand = prod_ids.unsqueeze(1).expand(-1, pool_prod_ids.size(0))
            mask_same = (prod_ids_expand == pool_prod_ids.unsqueeze(0))
            sim_matrix.masked_fill_(mask_same, float("-inf"))
            
            # 找 hardest negative
            hard_neg_idx_in_pool = torch.argmax(sim_matrix, dim=1)  # [B], index in pool_emb
            hard_neg_emb = pool_emb[hard_neg_idx_in_pool]  # [B, D]
            
            # 计算 triplet loss
            loss = triplet_loss(emb_anchor, emb_pos, hard_neg_emb)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # ----------------------------
    # 训练准备
    # ----------------------------
    print("Loading dataset...")
    # 使用新的多视角增强数据集
    dataset = MultiViewTripletFurnitureDataset(DATA_DIR, min_images_per_product=2)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    # 确保验证集至少有一个样本
    if val_size == 0 and len(dataset) > 1:
        val_size = 1
        train_size = len(dataset) - val_size
    elif len(dataset) == 1:
        train_size = 1
        val_size = 0
    
    print(f"Dataset size: {len(dataset)}, Train size: {train_size}, Val size: {val_size}")
    
    if train_size > 0 and val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, val_size), shuffle=False, num_workers=4, drop_last=False)
    elif train_size > 0:
        # 如果没有足够的数据划分验证集，则全部用于训练
        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_dataloader = None
        print("Warning: Dataset too small for validation, using all data for training")
    else:
        raise RuntimeError("Dataset is empty")
    
    # 创建模型，启用ArcFace头
    model = FurnitureFinetuneModel(
        unfreeze_blocks=UNFREEZE_BLOCKS, 
        use_arcface=True, 
        num_classes=len(dataset.products)  # 使用产品的数量作为分类数
    ).to(DEVICE)
    
    # 只优化可训练参数（head + 解冻的 backbone 参数）
    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量: {sum(p.numel() for p in params_to_opt)}")
    optimizer = torch.optim.AdamW(params_to_opt, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2)  # L2 distance
    ce_loss = nn.CrossEntropyLoss()  # ArcFace损失函数
    
    # 记录损失
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # ----------------------------
    # 训练循环（batch 内 hard negative mining）
    # ----------------------------
    print("Starting training...")
    for epoch in range(1, EPOCHS+1):
        # 训练阶段
        model.train()
        total_loss = 0.0
        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        
        for batch in train_progress:
            # batch: anchors, positives, product_ids
            anchors, positives, prod_ids = batch  # anchors: [B, C, H, W]
            anchors = anchors.to(DEVICE)
            positives = positives.to(DEVICE)
            prod_ids = prod_ids.to(DEVICE)  # shape [B]
            
            # 合并为一组图片进行一次前向（2B 张）
            imgs = torch.cat([anchors, positives], dim=0)  # [2B, C, H, W]
            labels = torch.cat([prod_ids, prod_ids], dim=0)  # [2B]
            
            # 前向传播，获取普通embedding和ArcFace输出
            outputs = model(imgs, labels)  # [2B, D], [2B, num_classes]
            
            # 如果模型返回两个输出（普通embedding和ArcFace输出）
            if isinstance(outputs, tuple) and len(outputs) == 2:
                embeddings, arcface_outputs = outputs
            else:
                # 如果只返回普通embedding（可能是验证阶段）
                embeddings = outputs
                arcface_outputs = None
            
            B = anchors.size(0)
            emb_anchor = embeddings[:B]    # [B, D]
            emb_pos = embeddings[B:2*B]    # [B, D]
            
            # 构造 candidate pool：所有来自 batch 的样本（包括 anchor/pos），但 negative 必须是不同 product_id
            pool_emb = embeddings  # [2B, D]
            pool_prod_ids = torch.cat([prod_ids, prod_ids], dim=0)  # [2B]
            
            # 计算 anchor 与 pool 的相似度矩阵
            sim_matrix = pairwise_cosine_similarity(emb_anchor, pool_emb)  # [B, 2B]
            # mask out same-product entries (both anchor's own product)
            prod_ids_expand = prod_ids.unsqueeze(1).expand(-1, pool_prod_ids.size(0))
            mask_same = (prod_ids_expand == pool_prod_ids.unsqueeze(0))  # True where same product -> cannot be negative
            sim_matrix.masked_fill_(mask_same, float("-inf"))  # 将不能作为负样本的位置置 -inf
            
            # 找 hardest negative: 最大相似度（最难区分）
            hard_neg_idx_in_pool = torch.argmax(sim_matrix, dim=1)  # [B], index in pool_emb
            hard_neg_emb = pool_emb[hard_neg_idx_in_pool]  # [B, D]
            
            # 计算 triplet loss
            triplet_loss_val = triplet_loss(emb_anchor, emb_pos, hard_neg_emb)
            
            # 计算 ArcFace loss (仅在有arcface输出时计算)
            if arcface_outputs is not None:
                arcface_loss = ce_loss(arcface_outputs, labels)
                # 组合损失
                loss = triplet_loss_val + 0.1 * arcface_loss  # 权重可以根据需要调整
            else:
                loss = triplet_loss_val
            
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(params_to_opt, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 更新学习率
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        
        # 验证阶段（如果有验证集）
        val_loss = 0.0
        if val_dataloader is not None and len(val_dataloader) > 0:
            val_loss = validate_model(model, val_dataloader, DEVICE)
            print(f"Epoch {epoch} Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存基于验证损失的最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(MODEL_SAVE_DIR, f"furniture_hard_best_val.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"✅ Saved best validation model {best_ckpt_path}")
        else:
            print(f"Epoch {epoch} Train Loss: {avg_train_loss:.6f}")
            
            # 保存基于训练损失的最佳模型
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_ckpt_path = os.path.join(MODEL_SAVE_DIR, f"furniture_hard_best_train.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"✅ Saved best training model {best_ckpt_path}")
        
        # 保存每个epoch的模型
        ckpt_path = os.path.join(MODEL_SAVE_DIR, f"furniture_hard_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Saved {ckpt_path}")
        
        # 记录损失
        train_losses.append(avg_train_loss)
        if val_dataloader is not None and len(val_dataloader) > 0:
            val_losses.append(val_loss)
    
    # 训练完成后，保存损失记录
    loss_record_path = os.path.join(MODEL_SAVE_DIR, "loss_records.txt")
    with open(loss_record_path, 'w') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\n")
        for i in range(len(train_losses)):
            train_loss = train_losses[i]
            val_loss = val_losses[i] if i < len(val_losses) else "N/A"
            f.write(f"{i+1}\t{train_loss:.6f}\t{val_loss}\n")
    
    print(f"✅ Saved loss records to {loss_record_path}")
    print("Training completed!")

    # 保存最终模型用于精排模型初始化
    final_model_path = os.path.join(MODEL_SAVE_DIR, "furniture_final_for_rerank.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Saved final model for re-ranking at {final_model_path}")


if __name__ == '__main__':
    # 添加freeze_support以支持Windows上的多进程
    from multiprocessing import freeze_support
    freeze_support()
    main()