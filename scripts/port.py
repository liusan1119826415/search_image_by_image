import base64
import io
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
# 导入精排模型
from feat.train_trendyol import FurnitureFinetuneModel
from feat.extract_furniture_feature import extract_feature, extract_patch_features
from app.milvus_client import search_topk, search_topk_coarse
from scripts.coarse_to_fine_ranking import CoarseToFineFeatureExtractor, FineRankingMatcher

app = Flask(__name__)
CORS(app)  # 允许跨域请求


def base64_to_image(base64_str):
    """将base64字符串转换为PIL图像"""
    image_data = base64.b64decode(base64_str.split(",")[-1])
    return Image.open(io.BytesIO(image_data))


def image_to_base64(image_path):
    """将图片转换为base64字符串"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return None
        
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# 添加图像预处理
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 全局特征缓存
global_feature_cache = {}

def load_and_cache_features(img_folder_path):
    """预加载图像特征到缓存"""
    global global_feature_cache
    print("正在预加载特征到缓存...")
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(img_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for filename in image_files:
        image_path = os.path.join(img_folder_path, filename)
        abs_path = os.path.abspath(image_path)
        
        # 提取全局特征
        global_feat = extract_feature(abs_path)
        if global_feat is not None:
            global_feature_cache[abs_path] = global_feat
        else:
            print(f"⚠️ 无法为 {abs_path} 提取特征")
    
    print(f"✅ 已预加载 {len(global_feature_cache)} 个特征到缓存")


patch_cache = {}

def get_patch_feats(path):
    if path in patch_cache:
        return patch_cache[path]
    base = os.path.basename(path).split(".")[0]
    patch_path = f"./patch_store/{base}_patch.npy"
    patch = np.load(patch_path)
    patch_cache[path] = patch
    return patch


def filter_results_by_rank_and_similarity(results, top_percentage=0.3, min_score_diff=0.1):
    """
    基于排名和分数差异过滤结果
    """
    if not results:
        return []
    
    # 按精排分数排序
    sorted_results = sorted(results, key=lambda x: x['fine_score'], reverse=True)
    
    # 计算要保留的数量（前top_percentage）
    num_to_keep = max(3, int(len(sorted_results) * top_percentage))  # 至少保留3个
    
    # 获取前N个结果
    top_results = sorted_results[:num_to_keep]
    
    # 进一步过滤：确保精排分数与第一个结果的分数差异在合理范围内
    if top_results:
        first_score = top_results[0]['fine_score']
        filtered_results = [r for r in top_results if (first_score - r['fine_score']) <= min_score_diff]
        
        # 如果过滤后结果太少，至少保留前3个
        if len(filtered_results) < 3:
            return top_results[:min(3, len(top_results))]
        else:
            return filtered_results
    
    return top_results


def coarse_to_fine_search(query_feature, query_image_path, top_k_coarse=100, top_k_final=50):
    """
    实现粗排到精排的完整流程
    """
    # 初始化特征提取器
    extractor = CoarseToFineFeatureExtractor()
    
    # 1. 加载查询图像并提取特征
    query_image = Image.open(query_image_path).convert("RGB")
    query_image_tensor = transform(query_image).unsqueeze(0)  # [1, 3, 224, 224]
    
    # 提取查询图像的全局特征和patch特征
    with torch.no_grad():
        query_global_features = extractor.extract_coarse_features(query_image_tensor)
        query_patch_features = extractor.extract_fine_features(query_image_tensor)
    
    # 2. 粗排：使用全局特征进行快速筛选
    coarse_results = search_topk_coarse(query_feature, top_k=top_k_coarse)
    
    print(f"粗排完成，找到 {len(coarse_results)} 个候选")
    
    # 3. 准备候选图像的特征（使用缓存的全局特征）
    candidate_global_features_list = []
    candidate_patch_features_list = []
    candidate_info = []
    
    for hit in coarse_results:
        candidate_path = hit.entity.get("content_path")
        if os.path.exists(candidate_path):
            # 从预存文件加载候选图像的patch特征
            candidate_patch_feat = extract_patch_features(candidate_path)
            if candidate_patch_feat is not None:
                # 从缓存中获取候选图像的全局特征，如果没有则提取并缓存
                if candidate_path in global_feature_cache:
                    candidate_global_feat = global_feature_cache[candidate_path]
                else:
                    # 如果缓存中没有，提取特征并缓存
                    candidate_global_feat = extract_feature(candidate_path)
                    if candidate_global_feat is not None:
                        global_feature_cache[candidate_path] = candidate_global_feat
                    else:
                        continue  # 跳过无法提取特征的图像
                
                # 将全局特征转换为tensor
                candidate_global_tensor = torch.from_numpy(candidate_global_feat).unsqueeze(0).float()  # 添加batch维度
                
                # 将patch特征转换为tensor
                candidate_patch_tensor = torch.from_numpy(candidate_patch_feat).unsqueeze(0).float()  # 添加batch维度
                
                candidate_global_features_list.append(candidate_global_tensor)
                candidate_patch_features_list.append(candidate_patch_tensor)
                candidate_info.append({
                    'path': candidate_path,
                    'hit': hit,
                    'coarse_score': float(hit.distance)  # 保存粗排分数用于后续分析
                })
    
    if len(candidate_global_features_list) == 0:
        return []
    
    # 堆叠候选特征
    candidate_global_features = torch.cat(candidate_global_features_list, dim=0)
    candidate_patch_features = torch.cat(candidate_patch_features_list, dim=0)
    
    # 4. 精排：使用patch-level特征进行精细排序
    fine_similarities = FineRankingMatcher.fine_rank(
        query_patch_features, 
        candidate_patch_features,
        query_global_features,
        candidate_global_features,
        global_weight=0.7,
        patch_weight=0.3
    )
    
    # 5. 获取最终排序结果
    final_scores = fine_similarities[0]  # 第一个查询的精排分数
    _, sorted_indices = torch.sort(final_scores, descending=True)
    
    # 6. 构建最终结果
    final_results = []
    for idx in sorted_indices:
        idx = idx.item()
        if idx < len(candidate_info):
            hit = candidate_info[idx]['hit']
            result = {
                'path': candidate_info[idx]['path'],
                'coarse_score': float(hit.distance),  # 粗排分数
                'fine_score': float(final_scores[idx].item()),  # 精排分数
                'hit': hit
            }
            final_results.append(result)
    
    # 7. 使用智能过滤策略
    filtered_results = filter_results_by_rank_and_similarity(final_results, top_percentage=0.4, min_score_diff=0.3)
    
    return filtered_results[:20]  # 最多返回20个结果


@app.route('/api/search', methods=['POST'])
def image_search():
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        base64_img = request.json.get('image')
        if not base64_img:
            return jsonify({"error": "No image provided"}), 400
        t0 = time.time()
        # 保存为临时文件
        # 修改为手动控制删除
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img_data = base64.b64decode(base64_img.split(",")[-1])
            tmp.write(img_data)
            tmp_path = tmp.name  # 保存路径
        print("==============================")    
        # 使用文件路径调用原函数
        vec = extract_feature(tmp_path)
        
       # print("====",vec)
        
        t1 = time.time()
        
        # 使用粗排到精排的流程
        results = coarse_to_fine_search(vec, tmp_path, top_k_coarse=100, top_k_final=50)
    
        print("====results===", results)
        t2 = time.time()
        
        # 4. 将结果中的图片路径转换为base64
        processed_results = []
        for result in results:
            path = result['path']
            coarse_score = result['coarse_score']
            fine_score = result['fine_score']
            
            if os.path.exists(path):
                base64_str = image_to_base64(path)
                processed_results.append({
                    "score": round(fine_score, 4),  # 使用精排分数
                    "coarse_score": round(coarse_score, 4),
                    "fine_score": round(fine_score, 4),
                    "image": base64_str
                })
        
        # 5. 返回结果和性能数据
        return jsonify({
            "results": processed_results,
            "metrics": {
                "feature_extraction_ms": round((t1 - t0) * 1000, 2),
                "vector_search_ms": round((t2 - t1) * 1000, 2),
                "total_ms": round((t2 - t0) * 1000, 2)
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/text_search", methods=["POST"])
def text_search():
    """添加文字搜索图片功能"""
    try:
        # 检查请求数据是否存在
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
            
        text_query = request.json.get("text")
        if not text_query:
            return jsonify({"error": "No text provided"}), 400

        t0 = time.time()

        # 当前未实现文本特征提取，暂时返回错误
        return jsonify({"error": "文本搜索功能暂未实现"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 预加载特征到缓存
    # if os.path.exists("./datasets/new_goods"):
    #     load_and_cache_features("./datasets/new_goods")
    
    app.run(host="0.0.0.0", port=5000, debug=True)