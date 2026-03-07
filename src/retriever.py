"""
混合检索模块 (Hybrid Retriever)
Dense + Sparse 双路检索，RRF 融合打分
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from FlagEmbedding import BGEM3FlagModel
from pymilvus import connections, Collection


# ────────────────────── 模型加载 ──────────────────────

def load_embed_model(
    model_path: str = "/home/liuguangli/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    device: str = "cuda",
) -> BGEM3FlagModel:
    """加载BGE-M3 Embedding模型"""
    print(f"加载BGE-M3模型: {model_path}")
    model = BGEM3FlagModel(model_path, use_fp16=True, device=device)
    print("BGE-M3模型加载完成")
    return model


# ────────────────────── 查询向量化 ──────────────────────

def encode_query(model: BGEM3FlagModel, query: str) -> dict:
    """
    将用户query编码为Dense + Sparse向量
    返回: {"dense": list[float], "sparse": dict}
    """
    result = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vec = result["dense_vecs"][0].tolist()
    sparse_vec = result["lexical_weights"][0]
    if not isinstance(sparse_vec, dict):
        sparse_vec = dict(sparse_vec)

    return {"dense": dense_vec, "sparse": sparse_vec}


# ────────────────────── 双路检索 ──────────────────────

def dense_search(
    collection: Collection,
    dense_vec: list[float],
    top_k: int = 20,
) -> list[dict]:
    """Dense向量检索（语义理解）"""
    results = collection.search(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["chunk_id", "source_file", "title", "header_path", "content"],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id": hit.entity.get("chunk_id"),
            "source_file": hit.entity.get("source_file"),
            "title": hit.entity.get("title"),
            "header_path": hit.entity.get("header_path"),
            "content": hit.entity.get("content"),
            "score": hit.score,
        })
    return hits


def sparse_search(
    collection: Collection,
    sparse_vec: dict,
    top_k: int = 20,
) -> list[dict]:
    """Sparse向量检索（术语精确匹配）"""
    results = collection.search(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=top_k,
        output_fields=["chunk_id", "source_file", "title", "header_path", "content"],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id": hit.entity.get("chunk_id"),
            "source_file": hit.entity.get("source_file"),
            "title": hit.entity.get("title"),
            "header_path": hit.entity.get("header_path"),
            "content": hit.entity.get("content"),
            "score": hit.score,
        })
    return hits


# ────────────────────── RRF 融合 ──────────────────────

def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int = 60,
    top_k: int = 20,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) 融合双路检索结果

    score(d) = Σ weight_i / (k + rank_i(d))

    参数:
        dense_hits:   Dense检索结果
        sparse_hits:  Sparse检索结果
        k:            RRF超参数，默认60（鲁棒性好，无需调参）
        top_k:        最终返回的文档数
        dense_weight:  Dense路权重
        sparse_weight: Sparse路权重
    """
    score_map = {}   # chunk_id -> {"score", "doc", "dense_rank", "sparse_rank"}

    # Dense路打分
    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        rrf_score = dense_weight / (k + rank + 1)
        if cid not in score_map:
            score_map[cid] = {
                "score": 0.0,
                "doc": hit,
                "dense_rank": None,
                "sparse_rank": None,
            }
        score_map[cid]["score"] += rrf_score
        score_map[cid]["dense_rank"] = rank + 1  # 1-indexed

    # Sparse路打分
    for rank, hit in enumerate(sparse_hits):
        cid = hit["chunk_id"]
        rrf_score = sparse_weight / (k + rank + 1)
        if cid not in score_map:
            score_map[cid] = {
                "score": 0.0,
                "doc": hit,
                "dense_rank": None,
                "sparse_rank": None,
            }
        score_map[cid]["score"] += rrf_score
        score_map[cid]["sparse_rank"] = rank + 1

    # 按RRF分数降序排列
    fused = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)

    results = []
    for item in fused[:top_k]:
        doc = item["doc"].copy()
        doc["rrf_score"] = item["score"]
        doc["dense_rank"] = item["dense_rank"]    # None = 该路未命中
        doc["sparse_rank"] = item["sparse_rank"]
        results.append(doc)

    # 融合统计（用于调试和消融实验）
    dense_ids = {h["chunk_id"] for h in dense_hits}
    sparse_ids = {h["chunk_id"] for h in sparse_hits}
    overlap = dense_ids & sparse_ids
    stats = {
        "dense_only": len(dense_ids - sparse_ids),
        "sparse_only": len(sparse_ids - dense_ids),
        "overlap": len(overlap),
        "total_unique": len(dense_ids | sparse_ids),
    }

    return results, stats


# ────────────────────── 完整检索流程 ──────────────────────

def hybrid_retrieve(
    query: str,
    model: BGEM3FlagModel,
    collection: Collection,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    final_top_k: int = 20,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """
    完整的混合检索流程:
    1. Query向量化 (Dense + Sparse)
    2. 双路检索
    3. RRF融合
    4. 返回Top-K结果

    参数:
        query:         用户查询
        model:         BGE-M3模型
        collection:    Milvus Collection
        dense_top_k:   Dense路召回数量
        sparse_top_k:  Sparse路召回数量
        final_top_k:   最终返回数量（送入Reranker）
        rrf_k:         RRF超参数
        dense_weight:  Dense路权重
        sparse_weight: Sparse路权重
    """
    # 1. Query向量化
    query_vecs = encode_query(model, query)

    # 2. 双路检索
    dense_hits = dense_search(collection, query_vecs["dense"], top_k=dense_top_k)
    sparse_hits = sparse_search(collection, query_vecs["sparse"], top_k=sparse_top_k)

    print(f"  Dense召回: {len(dense_hits)} 条")
    print(f"  Sparse召回: {len(sparse_hits)} 条")

    # 3. RRF融合
    fused_results, stats = reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        k=rrf_k,
        top_k=final_top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    # 调试统计
    print(f"  RRF融合后: {len(fused_results)} 条")
    print(f"  ├─ Dense独占: {stats['dense_only']} 条")
    print(f"  ├─ Sparse独占: {stats['sparse_only']} 条")
    print(f"  ├─ 双路重叠: {stats['overlap']} 条")
    print(f"  └─ 去重总计: {stats['total_unique']} 条")

    # Top-K中双路贡献分析
    top_both = sum(1 for d in fused_results if d["dense_rank"] and d["sparse_rank"])
    top_dense_only = sum(1 for d in fused_results if d["dense_rank"] and not d["sparse_rank"])
    top_sparse_only = sum(1 for d in fused_results if not d["dense_rank"] and d["sparse_rank"])
    print(f"  Top-{len(fused_results)} 贡献: 双路命中={top_both}, Dense独占={top_dense_only}, Sparse独占={top_sparse_only}")

    return fused_results


# ────────────────────── 工具函数 ──────────────────────

def print_results(results: list[dict], show_content: bool = False):
    """打印检索结果"""
    print(f"\n{'='*60}")
    print(f"检索结果: 共 {len(results)} 条")
    print(f"{'='*60}")

    for i, doc in enumerate(results):
        # 判断来源标签
        dr = doc.get("dense_rank")
        sr = doc.get("sparse_rank")
        if dr and sr:
            source_tag = f"双路命中 (D#{dr}, S#{sr})"
        elif dr:
            source_tag = f"Dense独占 (D#{dr})"
        else:
            source_tag = f"Sparse独占 (S#{sr})"

        print(f"\n[{i+1}] RRF分数: {doc.get('rrf_score', 0):.6f}  ← {source_tag}")
        print(f"    来源: {doc['source_file']}")
        print(f"    标题: {doc['title']}")
        print(f"    路径: {doc['header_path']}")
        if show_content:
            content_preview = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            print(f"    内容: {content_preview}")

    print(f"\n{'='*60}")


# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    # 1. 连接Milvus
    print("连接Milvus...")
    connections.connect(host="localhost", port="19530")

    # 2. 加载Collection
    collection_name = "hybrid_rag_docs"
    collection = Collection(collection_name)
    collection.load()
    print(f"Collection {collection_name} 已加载，共 {collection.num_entities} 条记录")

    # 3. 加载BGE-M3
    model = load_embed_model()

    # 4. 测试检索
    test_queries = [
        "Kafka consumer max.poll.records 参数配置",
        "如何排查服务超时问题",
        "API接口鉴权机制",
    ]

    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"Query: {query}")
        print(f"{'#'*60}")

        results = hybrid_retrieve(
            query=query,
            model=model,
            collection=collection,
            final_top_k=5,
        )

        print_results(results, show_content=True)

    # 5. 断开连接
    connections.disconnect("default")
    print("\n测试完成!")

    # ---- 诊断 Sparse 检索 ----
    query = "Kafka consumer max.poll.records"
    result = model.encode([query], return_dense=True, return_sparse=True)

    sparse_vec = result["lexical_weights"][0]
    if isinstance(sparse_vec, dict):
        print(f"Sparse向量非零项数量: {len(sparse_vec)}")
        print(f"Sparse向量样例(前10): {dict(list(sparse_vec.items())[:10])}")
    else:
        print(f"Sparse向量类型异常: {type(sparse_vec)}")

    # 直接用sparse搜一下，看原始返回
    from pymilvus import Collection

    collection = Collection("hybrid_rag_docs")
    collection.load()

    raw_results = collection.search(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=5,
        output_fields=["chunk_id", "title"],
    )
    print(f"\nSparse原始返回条数: {len(raw_results[0])}")
    for hit in raw_results[0]:
        print(f"  score={hit.score:.4f}  title={hit.entity.get('title')}")