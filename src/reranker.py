"""
Cross-Encoder 精排模块 (Reranker)
使用 bge-reranker-v2-m3 对粗排候选进行细粒度重排序
流程: Top-20 粗排候选 -> Cross-Encoder 打分 -> Top-5 精确上下文
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from FlagEmbedding import FlagReranker


# ────────────────────── 模型加载 ──────────────────────

def load_reranker(
    model_path: str = "/home/liuguangli/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    device: str = "cuda",
    use_fp16: bool = True,
) -> FlagReranker:
    """
    加载 Cross-Encoder Reranker 模型

    注意: 如果本地有缓存，会自动使用本地模型。
    如果报错找不到模型，检查路径:
      ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/<hash>/
    并将 model_path 改为完整的 snapshot 路径。
    """
    print(f"加载Reranker模型: {model_path}")
    reranker = FlagReranker(model_path, use_fp16=use_fp16, device=device)
    print("Reranker模型加载完成")
    return reranker


# ────────────────────── 精排核心 ──────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    reranker: FlagReranker,
    top_k: int = 5,
) -> list[dict]:
    """
    Cross-Encoder 精排

    原理: 将 (query, doc) 拼接后送入 Cross-Encoder，
    模型对整个 pair 做全注意力交互，输出相关性分数。
    比双塔模型（如BGE-M3）能捕捉更细粒度的 query-doc 交互。

    参数:
        query:       用户查询
        candidates:  粗排候选文档列表 (来自 retriever 的 RRF 结果)
        reranker:    FlagReranker 模型实例
        top_k:       精排后返回的文档数
    """
    if not candidates:
        return []

    # 构造 [query, doc_content] 对
    pairs = [[query, doc["content"]] for doc in candidates]

    # Cross-Encoder 打分
    # normalize=True 将分数映射到 [0, 1]，方便设置阈值和对比
    scores = reranker.compute_score(pairs, normalize=True)

    # compute_score 在单条输入时返回 float，多条时返回 list
    if isinstance(scores, float):
        scores = [scores]

    # 将分数附加到文档上
    scored_docs = []
    for doc, score in zip(candidates, scores):
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = score
        scored_docs.append(doc_copy)

    # 按 rerank_score 降序排列
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_docs[:top_k]


# ────────────────────── 调试输出 ──────────────────────

def print_rerank_results(
    query: str,
    before: list[dict],
    after: list[dict],
    show_content: bool = False,
):
    """
    打印精排前后的对比，方便调试

    显示内容:
    - 精排前后的排名变化
    - RRF分数 vs Rerank分数
    - 排名提升/下降情况
    """
    print(f"\n{'='*70}")
    print(f"精排结果对比 | Query: {query}")
    print(f"{'='*70}")

    # 建立精排前的排名映射
    before_rank = {}
    for i, doc in enumerate(before):
        before_rank[doc["chunk_id"]] = i + 1

    print(f"\n精排后 Top-{len(after)}:")
    for i, doc in enumerate(after):
        old_rank = before_rank.get(doc["chunk_id"], "?")
        rank_change = old_rank - (i + 1) if isinstance(old_rank, int) else 0

        if rank_change > 0:
            change_str = f"↑{rank_change}"
        elif rank_change < 0:
            change_str = f"↓{abs(rank_change)}"
        else:
            change_str = "─"

        # 来源标签
        dr = doc.get("dense_rank")
        sr = doc.get("sparse_rank")
        if dr and sr:
            source_tag = f"D#{dr},S#{sr}"
        elif dr:
            source_tag = f"D#{dr}"
        else:
            source_tag = f"S#{sr}"

        print(f"\n  [{i+1}] rerank={doc['rerank_score']:.4f}  "
              f"rrf={doc.get('rrf_score', 0):.6f}  "
              f"粗排#{old_rank} {change_str}  "
              f"[{source_tag}]")
        print(f"      来源: {doc['source_file']} | {doc['title']}")

        if show_content:
            preview = doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"]
            print(f"      内容: {preview}")

    # 统计精排带来的排名变化
    promotions = 0   # 排名上升的文档数
    demotions = 0    # 排名下降的文档数（被挤出Top-K的）
    for i, doc in enumerate(after):
        old_rank = before_rank.get(doc["chunk_id"], 999)
        if old_rank > (i + 1):
            promotions += 1

    # 粗排Top-K中被精排淘汰的
    after_ids = {doc["chunk_id"] for doc in after}
    before_top_ids = {doc["chunk_id"] for doc in before[:len(after)]}
    kicked_out = before_top_ids - after_ids

    print(f"\n  排名上升: {promotions} 条 | 被淘汰: {len(kicked_out)} 条")
    print(f"{'='*70}")


# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    from pymilvus import connections, Collection
    from retriever import load_embed_model, hybrid_retrieve

    # 1. 连接Milvus
    print("连接Milvus...")
    connections.connect(host="localhost", port="19530")

    # 2. 加载Collection
    collection_name = "hybrid_rag_docs"
    collection = Collection(collection_name)
    collection.load()
    print(f"Collection {collection_name} 已加载，共 {collection.num_entities} 条记录")

    # 3. 加载模型
    embed_model = load_embed_model()
    reranker_model = load_reranker()

    # 4. 测试: 粗排 -> 精排 全流程
    test_queries = [
        "Kafka consumer max.poll.records 参数配置",
        "如何排查服务超时问题",
        "Redis 持久化 RDB 和 AOF 的区别",
    ]

    for query in test_queries:
        print(f"\n{'#'*70}")
        print(f"Query: {query}")
        print(f"{'#'*70}")

        # 粗排: Hybrid Retrieve (Top-20)
        coarse_results = hybrid_retrieve(
            query=query,
            model=embed_model,
            collection=collection,
            final_top_k=20,
        )

        # 精排: Cross-Encoder Rerank (Top-20 -> Top-5)
        fine_results = rerank(
            query=query,
            candidates=coarse_results,
            reranker=reranker_model,
            top_k=5,
        )

        # 打印精排前后对比
        print_rerank_results(
            query=query,
            before=coarse_results,
            after=fine_results,
            show_content=True,
        )

    # 5. 断开连接
    connections.disconnect("default")
    print("\n测试完成!")