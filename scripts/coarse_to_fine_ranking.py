"""
粗排到精排的特征提取与匹配计算
实现从checkpoints_hard/furniture_hard_epoch15.pth模型中提取不同层次的特征
用于粗排和精排系统的匹配计算
"""

import torch
import os
import torch.nn.functional as F
from feat.train_trendyol import FurnitureFinetuneModel

class CoarseToFineFeatureExtractor:
    """粗排到精排特征提取器"""
    
    def __init__(self, model_path="checkpoints_hard/furniture_hard_best_val.pth"):
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        num_classes = checkpoint['arcface_head.weight'].shape[0]
        
        self.model = FurnitureFinetuneModel(
            pretrained_model_name="Trendyol/trendyol-dino-v2-ecommerce-256d",
            unfreeze_blocks=["fc11", "bn11"],
            use_arcface=True,
            num_classes=num_classes
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"✓ 加载模型: {model_path}")
        print(f"✓ 模型类别数: {num_classes}")
    
    def extract_coarse_features(self, images):
        """
        提取粗排特征（全局特征）
        
        Args:
            images: 输入图像张量 [batch_size, 3, H, W]
            
        Returns:
            global_features: 全局特征 [batch_size, 256]
        """
        with torch.no_grad():
            # 使用完整的模型前向传播获取全局特征
            global_features = self.model(images)
            return global_features
    
    def extract_fine_features_with_hook(self, images):
        """
        提取精排特征（patch-level特征）使用hook
        
        Args:
            images: 输入图像张量 [batch_size, 3, H, W]
            
        Returns:
            patch_features: patch-level特征 [batch_size, num_patches, hidden_dim]
        """
        intermediate_output = {}
        
        def hook_fn(module, input, output):
            # 存储输入特征（在第一个transformer块之前的特征）
            intermediate_output['features'] = input[0] if isinstance(input, tuple) else input
        
        # 注册hook到第一个transformer块的归一化层
        handle = self.model.backbone.model.backbone.blocks[0].norm1.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            # 前向传播
            _ = self.model(images)
            
            # 移除hook
            handle.remove()
        
        if 'features' in intermediate_output:
            return intermediate_output['features']
        else:
            return None
    
    def extract_fine_features(self, images):
        """
        提取精排特征（patch-level特征）
        
        Args:
            images: 输入图像张量 [batch_size, 3, H, W]
            
        Returns:
            patch_features: patch-level特征 [batch_size, num_patches, hidden_dim]
        """
        # 尝试使用hook方法
        patch_features = self.extract_fine_features_with_hook(images)
        
        if patch_features is not None:
            return patch_features
        else:
            # 如果hook失败，尝试直接访问backbone
            with torch.no_grad():
                backbone_output = self.model.backbone.model.backbone(images)
                return backbone_output  # 这可能已经被聚合了

class CoarseRankingMatcher:
    """粗排匹配器 - 使用全局特征"""
    
    @staticmethod
    def compute_global_similarity(query_features, candidate_features):
        """
        计算全局特征相似度（用于粗排）
        
        Args:
            query_features: 查询图像全局特征 [batch_size, 256]
            candidate_features: 候选图像全局特征 [num_candidates, 256]
            
        Returns:
            similarities: 相似度矩阵 [batch_size, num_candidates]
        """
        # 使用余弦相似度
        similarities = F.cosine_similarity(
            query_features.unsqueeze(1),  # [batch, 1, 256]
            candidate_features.unsqueeze(0),  # [1, num_candidates, 256]
            dim=-1
        )  # [batch, num_candidates]
        
        return similarities
    
    @staticmethod
    def coarse_rank(query_features, candidate_features, top_k=100):
        """
        粗排：基于全局特征进行快速筛选
        
        Args:
            query_features: 查询特征 [batch_size, 256]
            candidate_features: 候选特征 [num_candidates, 256] 
            top_k: 保留前k个候选
            
        Returns:
            top_indices: 保留的候选索引
            top_similarities: 对应的相似度分数
        """
        similarities = CoarseRankingMatcher.compute_global_similarity(query_features, candidate_features)
        
        # 对每个查询获取top-k候选
        batch_size = query_features.size(0)
        top_indices = []
        top_similarities = []
        
        for i in range(batch_size):
            sim_scores = similarities[i]  # [num_candidates]
            top_vals, top_idx = torch.topk(sim_scores, min(top_k, len(sim_scores)), largest=True)
            top_indices.append(top_idx)
            top_similarities.append(top_vals)
        
        return top_indices, top_similarities

class FineRankingMatcher:
    """精排匹配器 - 使用patch-level特征"""
    
    @staticmethod
    def compute_patch_similarity(img1_patches, img2_patches):
        """
        计算patch-level相似度矩阵
        
        Args:
            img1_patches: 图像1的patch特征 [batch_size, num_patches, hidden_dim]
            img2_patches: 图像2的patch特征 [num_candidates, num_patches, hidden_dim]
            
        Returns:
            similarity_matrix: 相似度矩阵 [batch_size, num_candidates]
        """
        batch_size, num_patches1, hidden_dim1 = img1_patches.shape
        num_candidates, num_patches2, hidden_dim2 = img2_patches.shape
        
        # 计算patch级别的相似度矩阵
        # [batch, num_patches1, hidden_dim] @ [num_candidates, hidden_dim, num_patches2]
        # 结果: [batch, num_candidates, num_patches1, num_patches2]
        patch_similarities = torch.einsum('bph,cqh->bcqp', img1_patches, img2_patches)
        
        # 对patch维度求平均，得到图像级别的相似度
        # [batch, num_candidates]
        image_similarities = patch_similarities.mean(dim=[2, 3])
        
        return image_similarities
    
    @staticmethod
    def compute_cross_attention_similarity(img1_patches, img2_patches):
        """
        使用cross-attention计算patch-level相似度
        
        Args:
            img1_patches: 图像1的patch特征 [batch_size, num_patches, hidden_dim]
            img2_patches: 图像2的patch特征 [num_candidates, num_patches, hidden_dim]
            
        Returns:
            attention_similarities: attention-based相似度 [batch_size, num_candidates]
        """
        batch_size, num_patches1, hidden_dim = img1_patches.shape
        num_candidates, num_patches2, _ = img2_patches.shape
        
        # 为每个候选图像计算cross-attention
        similarities = []
        
        for i in range(num_candidates):
            candidate_patches = img2_patches[i:i+1].expand(batch_size, -1, -1)  # [batch, num_patches2, hidden_dim]
            
            # 使用MultiheadAttention进行cross-attention
            attention_layer = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True
            )
            
            # 图像1作为query，图像2作为key和value
            attn_output, attn_weights = attention_layer(
                img1_patches,      # query
                candidate_patches, # key
                candidate_patches  # value
            )
            
            # 计算attention输出与原特征的相似度
            attn_sim = F.cosine_similarity(
                img1_patches.mean(dim=1),  # [batch, hidden_dim]
                attn_output.mean(dim=1),   # [batch, hidden_dim]
                dim=1
            )  # [batch]
            
            similarities.append(attn_sim)
        
        # [batch_size, num_candidates]
        return torch.stack(similarities, dim=1)
    
    @staticmethod
    def compute_combined_similarity(global_sim, patch_sim, global_weight=0.3, patch_weight=0.7):
        """
        组合全局和局部相似度
        
        Args:
            global_sim: 全局相似度 [batch_size, num_candidates]
            patch_sim: patch相似度 [batch_size, num_candidates] 
            global_weight: 全局相似度权重
            patch_weight: patch相似度权重
            
        Returns:
            combined_sim: 组合相似度 [batch_size, num_candidates]
        """
        combined_sim = global_weight * global_sim + patch_weight * patch_sim
        return combined_sim
    
    @staticmethod
    def fine_rank(query_patch_features, candidate_patch_features, 
                  query_global_features=None, candidate_global_features=None,
                  global_weight=0.3, patch_weight=0.7):
        """
        精排：基于patch-level特征进行精细排序
        
        Args:
            query_patch_features: 查询图像patch特征 [batch_size, num_patches, hidden_dim]
            candidate_patch_features: 候选图像patch特征 [num_candidates, num_patches, hidden_dim]
            query_global_features: 查询全局特征 [batch_size, 256]
            candidate_global_features: 候选全局特征 [num_candidates, 256]
            global_weight: 全局特征权重
            patch_weight: patch特征权重
            
        Returns:
            fine_similarities: 精排相似度分数
        """
        # 计算patch-level相似度
        patch_similarities = FineRankingMatcher.compute_patch_similarity(
            query_patch_features, candidate_patch_features
        )
        
        if query_global_features is not None and candidate_global_features is not None:
            # 计算全局相似度
            global_similarities = CoarseRankingMatcher.compute_global_similarity(
                query_global_features, candidate_global_features
            )
            
            # 组合全局和局部相似度
            combined_similarities = FineRankingMatcher.compute_combined_similarity(
                global_similarities, patch_similarities, global_weight, patch_weight
            )
            
            return combined_similarities
        else:
            return patch_similarities

def coarse_to_fine_ranking_pipeline():
    """粗排到精排的完整流程"""
    print("="*60)
    print("粗排到精排匹配计算流程")
    print("="*60)
    
    # 初始化特征提取器
    extractor = CoarseToFineFeatureExtractor()
    
    # 创建示例数据
    batch_size = 2
    num_candidates = 10
    query_images = torch.randn(batch_size, 3, 224, 224)  # 查询图像
    candidate_images = torch.randn(num_candidates, 3, 224, 224)  # 候选图像
    
    print(f"✓ 创建示例数据: 查询{batch_size}张，候选{num_candidates}张")
    
    # 1. 粗排阶段：提取全局特征
    print(f"\n1. 粗排阶段 - 提取全局特征")
    query_global_features = extractor.extract_coarse_features(query_images)
    candidate_global_features = extractor.extract_coarse_features(candidate_images)
    
    print(f"   - 查询全局特征: {query_global_features.shape}")
    print(f"   - 候选全局特征: {candidate_global_features.shape}")
    
    # 2. 粗排：快速筛选
    print(f"\n2. 粗排 - 快速筛选")
    top_k = 5  # 粗排后保留前5个候选
    top_indices, top_similarities = CoarseRankingMatcher.coarse_rank(
        query_global_features, candidate_global_features, top_k=top_k
    )
    
    print(f"   - 粗排后保留: {top_k}个候选")
    print(f"   - 第1个查询的top候选索引: {top_indices[0].tolist()}")
    print(f"   - 对应相似度: {top_similarities[0].tolist()}")
    
    # 3. 精排阶段：提取patch-level特征
    print(f"\n3. 精排阶段 - 提取patch-level特征")
    query_patch_features = extractor.extract_fine_features(query_images)
    candidate_patch_features = extractor.extract_fine_features(candidate_images)
    
    print(f"   - 查询patch特征: {query_patch_features.shape}")
    print(f"   - 候选patch特征: {candidate_patch_features.shape}")
    
    # 4. 精排：精细排序（只对粗排保留的候选进行）
    print(f"\n4. 精排 - 精细排序")
    
    # 获取粗排保留的候选patch特征
    selected_candidate_indices = top_indices[0]  # 以第1个查询为例
    selected_candidate_patches = candidate_patch_features[selected_candidate_indices]
    selected_candidate_globals = candidate_global_features[selected_candidate_indices]
    
    # 进行精排
    fine_similarities = FineRankingMatcher.fine_rank(
        query_patch_features, 
        selected_candidate_patches,
        query_global_features,
        selected_candidate_globals,
        global_weight=0.3,
        patch_weight=0.7
    )
    
    print(f"   - 精排相似度: {fine_similarities[0].tolist()}")
    
    # 5. 最终排序结果
    final_scores = fine_similarities[0]  # 第1个查询的精排分数
    _, final_ranking = torch.sort(final_scores, descending=True)
    
    print(f"\n5. 最终排序结果")
    print(f"   - 原始粗排候选索引: {selected_candidate_indices.tolist()}")
    print(f"   - 精排后排序: {final_ranking.tolist()}")
    print(f"   - 最终推荐顺序: {[selected_candidate_indices[i].item() for i in final_ranking]}")
    
    # 6. 不同匹配方法的比较
    print(f"\n6. 不同匹配方法比较")
    
    # 仅全局特征
    global_only_sim = CoarseRankingMatcher.compute_global_similarity(
        query_global_features[:1], selected_candidate_globals
    )[0]
    
    # 仅patch特征
    patch_only_sim = FineRankingMatcher.compute_patch_similarity(
        query_patch_features[:1], selected_candidate_patches
    )[0]
    
    # 组合特征
    combined_sim = fine_similarities[0]
    
    print(f"   - 全局特征相似度: {global_only_sim.tolist()}")
    print(f"   - patch特征相似度: {patch_only_sim.tolist()}")
    print(f"   - 组合特征相似度: {combined_sim.tolist()}")
    
    return {
        'query_images': query_images,
        'candidate_images': candidate_images,
        'query_global_features': query_global_features,
        'candidate_global_features': candidate_global_features,
        'query_patch_features': query_patch_features,
        'candidate_patch_features': candidate_patch_features,
        'coarse_results': (top_indices, top_similarities),
        'fine_results': fine_similarities,
        'final_ranking': final_ranking
    }

def main():
    """主函数"""
    print("粗排到精排特征提取与匹配计算")
    print("基于checkpoints_hard/furniture_hard_epoch15.pth模型")
    
    # 执行完整流程
    results = coarse_to_fine_ranking_pipeline()
    
    print(f"\n" + "="*60)
    print("总结")
    print("="*60)
    print("✓ 粗排阶段:")
    print("  - 使用全局特征进行快速筛选")
    print("  - 计算效率高，适合大规模候选集")
    print("  - 基于256维全局特征的余弦相似度")
    
    print(f"\n✓ 精排阶段:")
    print("  - 使用patch-level特征进行精细匹配")
    print("  - 特征维度: [batch, 257, 768] (256个patch + 1个cls token)")
    print("  - 支持局部特征匹配和cross-attention计算")
    
    print(f"\n✓ 特征提取:")
    print("  - 粗排: 模型最终输出的256维全局特征")
    print("  - 精排: 通过hook获取的257×768 patch特征")
    
    print(f"\n✓ 匹配策略:")
    print("  - 粗排: 全局余弦相似度")
    print("  - 精排: patch-level相似度 + 全局特征融合")
    print("  - 权重组合: 全局权重0.3 + patch权重0.7")
    
    print(f"\n✓ 性能优势:")
    print("  - 粗排快速过滤，减少精排计算量")
    print("  - 精排细粒度匹配，提高准确性")
    print("  - 全局+局部特征融合，平衡效率与精度")

if __name__ == "__main__":
    main()