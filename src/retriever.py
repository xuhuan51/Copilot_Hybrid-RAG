"""
混合检索模块 (Hybrid Retriever)
Dense + Sparse 双路检索，RRF 融合打分
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
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


# ────────────────────── 术语抽取 ──────────────────────

def extract_terms(query: str) -> str | None:
    """
    从 query 中抽取高价值英文术语，用于 Sparse 检索

    只保留两类高价值术语:
    1. 带点号的参数名: max.poll.records, session.timeout.ms, batch.size
    2. 大写缩写/专有词: ISR, RDB, AOF, SASL, SSL

    泛词 (kafka, redis, consumer, topic, broker 等) 不走 Sparse，
    因为它们对精确匹配没有帮助，反而会引入主题相关但答案不准的噪声。
    """
    broad_terms = {
        "kafka", "redis", "mysql", "consumer", "producer", "broker",
        "topic", "cluster", "partition", "message", "server", "client",
        "config", "configuration", "data", "node", "group", "key",
        "value", "type", "set", "list", "hash", "string", "log",
        "file", "memory", "disk", "network", "security", "monitor",
        "backup", "restore", "master", "slave", "replica",
    }

    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "about", "between", "through", "and", "or",
        "but", "not", "no", "if", "then", "than", "so", "it", "its",
        "this", "that", "these", "those", "what", "which", "who",
        "how", "when", "where", "why",
    }

    terms = []

    # 1. 带点号的参数名 (最高价值)
    param_names = re.findall(r'[a-zA-Z][a-zA-Z0-9]*(?:\.[a-zA-Z][a-zA-Z0-9]*)+', query)
    terms.extend(param_names)

    # 2. 大写缩写词 (2-6个大写字母，如 ISR, RDB, AOF, SASL)
    abbreviations = re.findall(r'\b[A-Z]{2,6}\b', query)
    terms.extend(abbreviations)

    # 3. 其他英文词，但过滤停用词和泛词
    other_words = re.findall(r'[a-zA-Z]{2,}', query)
    for word in other_words:
        w_lower = word.lower()
        if w_lower not in stopwords and w_lower not in broad_terms and word not in terms:
            terms.append(word)

    if terms:
        return " ".join(terms)
    return None


# ────────────────────── 查询向量化 ──────────────────────

def encode_query(model: BGEM3FlagModel, query: str) -> dict:
    """
    将用户query编码为Dense + Sparse向量

    关键设计:
    - Dense: 用完整 query 编码（保留语义）
    - Sparse: 只用抽取出的英文术语编码（避免中文噪声）
    - 无英文术语时: Sparse 返回空字典，完全由 Dense 主导

    返回: {"dense": list[float], "sparse": dict}
    """
    dense_result = model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vec = dense_result["dense_vecs"][0].tolist()

    sparse_query = extract_terms(query)
    if sparse_query:
        sparse_result = model.encode(
            [sparse_query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        sparse_vec = sparse_result["lexical_weights"][0]
        if not isinstance(sparse_vec, dict):
            sparse_vec = dict(sparse_vec)
        print(f"  Sparse query: '{sparse_query}'")
    else:
        sparse_vec = {}
        print("  Sparse query: (无高价值术语，跳过 Sparse)")

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
) -> tuple[list[dict], dict]:
    """
    Reciprocal Rank Fusion (RRF) 融合双路检索结果

    score(d) = Σ weight_i / (k + rank_i(d))
    """
    score_map = {}

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
        score_map[cid]["dense_rank"] = rank + 1

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

    fused = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)

    results = []
    for item in fused[:top_k]:
        doc = item["doc"].copy()
        doc["rrf_score"] = item["score"]
        doc["dense_rank"] = item["dense_rank"]
        doc["sparse_rank"] = item["sparse_rank"]
        results.append(doc)

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


def _dense_only_as_rrf(dense_hits: list[dict], top_k: int, rrf_k: int) -> list[dict]:
    """把纯 Dense 结果补齐成统一字段格式，便于后续 rerank/分析。"""
    results = []
    for i, hit in enumerate(dense_hits[:top_k]):
        doc = hit.copy()
        doc["rrf_score"] = 1.0 / (rrf_k + i + 1)
        doc["dense_rank"] = i + 1
        doc["sparse_rank"] = None
        results.append(doc)
    return results


# ────────────────────── 完整检索流程 ──────────────────────

def hybrid_retrieve(
    query: str,
    model: BGEM3FlagModel,
    collection: Collection,
    intent: str | None = None,
    dense_top_k: int = 20,
    sparse_top_k: int = 5,
    final_top_k: int = 20,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 0.2,
    conceptual_dense_top_k: int = 25,
    troubleshoot_dense_top_k: int = 30,
    troubleshoot_sparse_top_k: int = 12,
    troubleshoot_sparse_weight: float = 0.15,
) -> list[dict]:
    """
    意图感知的混合检索流程

    策略分流:
    - conceptual:   纯 Dense，更大候选池，交给 reranker 挑最优
    - factual:      Dense + Sparse (有高价值术语时)
    - troubleshoot: Dense / Sparse 都扩池，再交给 reranker 做更强筛选
    """
    query_vecs = encode_query(model, query)

    _dense_top_k = dense_top_k
    _sparse_top_k = sparse_top_k
    _final_top_k = final_top_k
    _dense_weight = dense_weight
    _sparse_weight = sparse_weight
    _use_sparse = bool(query_vecs["sparse"])

    if intent == "conceptual":
        _dense_top_k = max(dense_top_k, conceptual_dense_top_k)
        _final_top_k = max(final_top_k, conceptual_dense_top_k)
        _use_sparse = False
        print("  策略: conceptual → 纯Dense + 扩大候选池")
    elif intent == "troubleshoot":
        _dense_top_k = max(dense_top_k, troubleshoot_dense_top_k)
        _sparse_top_k = max(sparse_top_k, troubleshoot_sparse_top_k)
        _final_top_k = max(final_top_k, troubleshoot_dense_top_k)
        _sparse_weight = troubleshoot_sparse_weight
        print(
            "  策略: troubleshoot → Dense 扩池"
            + (f" + Sparse 扩池(top_k={_sparse_top_k})" if _use_sparse else "")
        )
    else:
        print("  策略: factual → Dense" + (" + Sparse" if _use_sparse else ""))

    dense_hits = dense_search(collection, query_vecs["dense"], top_k=_dense_top_k)
    print(f"  Dense召回: {len(dense_hits)} 条")

    if _use_sparse:
        sparse_hits = sparse_search(collection, query_vecs["sparse"], top_k=_sparse_top_k)
        print(f"  Sparse召回: {len(sparse_hits)} 条")
    else:
        sparse_hits = []
        if intent != "conceptual":
            print("  Sparse召回: 跳过 (无高价值术语)")

    if sparse_hits:
        fused_results, stats = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            k=rrf_k,
            top_k=_final_top_k,
            dense_weight=_dense_weight,
            sparse_weight=_sparse_weight,
        )
        print(f"  RRF融合后: {len(fused_results)} 条")
        print(f"  ├─ Dense独占: {stats['dense_only']} 条")
        print(f"  ├─ Sparse独占: {stats['sparse_only']} 条")
        print(f"  ├─ 双路重叠: {stats['overlap']} 条")
        print(f"  └─ 去重总计: {stats['total_unique']} 条")
    else:
        fused_results = _dense_only_as_rrf(dense_hits, top_k=_final_top_k, rrf_k=rrf_k)
        print(f"  纯Dense模式: {len(fused_results)} 条")

    top_both = sum(1 for d in fused_results if d.get("dense_rank") and d.get("sparse_rank"))
    top_dense_only = sum(1 for d in fused_results if d.get("dense_rank") and not d.get("sparse_rank"))
    top_sparse_only = sum(1 for d in fused_results if not d.get("dense_rank") and d.get("sparse_rank"))
    print(
        f"  Top-{len(fused_results)} 贡献: "
        f"双路命中={top_both}, Dense独占={top_dense_only}, Sparse独占={top_sparse_only}"
    )

    return fused_results


# ────────────────────── 工具函数 ──────────────────────

def print_results(results: list[dict], show_content: bool = False):
    """打印检索结果"""
    print(f"\n{'='*60}")
    print(f"检索结果: 共 {len(results)} 条")
    print(f"{'='*60}")

    for i, doc in enumerate(results):
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
    print("连接Milvus...")
    connections.connect(host="localhost", port="19530")

    collection_name = "hybrid_rag_docs"
    collection = Collection(collection_name)
    collection.load()
    print(f"Collection {collection_name} 已加载，共 {collection.num_entities} 条记录")

    model = load_embed_model()

    test_queries = [
        ("Kafka consumer max.poll.records 参数配置", "factual"),
        ("Kafka consumer group 的 rebalance 机制是怎么工作的", "conceptual"),
        ("Kafka consumer 报 session timeout 异常怎么排查", "troubleshoot"),
    ]

    for query, intent in test_queries:
        print(f"\n{'#'*60}")
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print(f"{'#'*60}")

        results = hybrid_retrieve(
            query=query,
            model=model,
            collection=collection,
            intent=intent,
            final_top_k=20,
        )

        print_results(results, show_content=True)

    connections.disconnect("default")
    print("\n测试完成!")
