# app/milvus_client.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,utility  # 新增导入
import numpy as np
from pymilvus.exceptions import MilvusException
COLLECTION_NAME = "image_clip"
DIM = 256  # DINOv3特征维度
import time

def init_milvus():
    connections.connect(alias="default", host="127.0.0.1", port="19530")
    server_version = utility.get_server_version()
    print(f"Milvus 服务器版本: {server_version}")
    if COLLECTION_NAME not in utility.list_collections():
        # 创建新集合和索引
       # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),  # 全局特征
            FieldSchema(name="content_path", dtype=DataType.VARCHAR, max_length=500),  # 存储图像路径或文本描述

        ]
        
        schema = CollectionSchema(fields, description="Image and text retrieval with CLIP features")
        collection = Collection(COLLECTION_NAME, schema)
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
            "metric_type": "IP"
        }

        
        collection.create_index("vector", index_params)
        
        collection.load()
    else:
        # 集合已存在，检查并修复索引
        collection = Collection(COLLECTION_NAME)
        if not collection.indexes:
            print("索引不存在，重新创建...")
            index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
            "metric_type": "IP"
        }
            collection.create_index("vector", index_params)
        collection.load()
    
    return collection


def drop_milvus_collection():
    """
    安全删除指定的Milvus集合
    """
    try:
        # 连接到Milvus服务器
        connections.connect(alias="default", host="localhost", port="19530")
        
        # 检查集合是否存在
        if COLLECTION_NAME in utility.list_collections():
            print(f"正在删除集合: {COLLECTION_NAME}...")
            
            # 删除集合（同时会删除对应的索引和数据）
            utility.drop_collection(COLLECTION_NAME)
            
            # 确认删除结果
            if COLLECTION_NAME not in utility.list_collections():
                print(f"✅ 成功删除集合: {COLLECTION_NAME}")
            else:
                print(f"❌ 删除集合失败: {COLLECTION_NAME} 仍然存在")
        else:
            print(f"集合 {COLLECTION_NAME} 不存在，无需删除")
            
    except MilvusException as e:
        print(f"删除集合时发生错误: {e}")
    finally:
        # 关闭连接
        connections.disconnect("default")

def search_topk(vector, top_k=50):
    collection = init_milvus()
    results = collection.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["content_path"]
    )
    return results[0]



def search_topk_coarse(vector, top_k=40):
    """粗排：基于全局特征检索"""
    collection = init_milvus()
    results = collection.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["content_path"]
    )
    return results[0]


def search_with_global_features(vector, top_k=100):
    """搜索并返回全局特征"""
    collection = init_milvus()
    # 使用search方法，返回距离（即相似度）和实体信息
    results = collection.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["content_path"]
    )
    return results[0]


def search_with_aggregated_patches(query_global, query_agg_patch, top_k_coarse=100, top_k_final=20):
    """
    使用聚合patch特征进行两级检索
    """
    collection = init_milvus()
    
    # 1. 粗排：基于全局特征
    t_search_start = time.time()
    coarse_results = collection.search(
        data=[query_global],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k_coarse,
        output_fields=["content_path"]
    )[0]
    t_search_end = time.time()
    
    print(f"粗排完成，找到 {len(coarse_results)} 个候选，耗时: {(t_search_end-t_search_start)*1000:.2f}ms")
    
    # 2. 对查询图像的patch特征进行聚合（与存储时相同的方式）

    
    # 3. 精排：基于聚合patch特征
    reranked_results = []
    
    for i, hit in enumerate(coarse_results[:50]):  # 只处理前50个
        candidate_agg_patch = np.array(hit.entity.get("agg_patch_vector"), dtype=np.float32)
        coarse_score = float(hit.distance)
        
        # 计算聚合patch特征的相似度
        patch_similarity = float(np.dot(query_agg_patch, candidate_agg_patch))
        
        # 综合评分（可调整权重）
        if patch_similarity > 0.8:  # 聚合特征相似度高
            final_score = 0.2 * coarse_score + 0.8 * patch_similarity
        else:
            final_score = 0.7 * coarse_score + 0.3 * patch_similarity
        
        reranked_results.append({
            'hit': hit,
            'final_score': final_score,
            'coarse_score': coarse_score,
            'patch_similarity': patch_similarity,
            'content_path': hit.entity.get("content_path"),
     
        })
    
    # 按最终分数排序
    reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return reranked_results[:top_k_final]

def safe_search_with_patches(query_global, query_patch, top_k_coarse=100, top_k_final=20):
    """安全的检索函数，处理类型转换"""
    try:
        results = search_with_aggregated_patches(query_global, query_patch, top_k_coarse, top_k_final)
        
        # 确保所有数值都是Python原生类型
        for result in results:
            result['final_score'] = float(result['final_score'])
            result['coarse_score'] = float(result['coarse_score']) 
            result['patch_similarity'] = float(result['patch_similarity'])
            
        return results
    except Exception as e:
        print(f"检索失败: {e}")
        return []