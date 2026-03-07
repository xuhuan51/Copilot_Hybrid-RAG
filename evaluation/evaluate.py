"""
评测脚本 (Evaluation)
消融实验: 逐步验证每个模块的贡献
- Baseline:    仅 Dense 检索
- + Hybrid:    Dense + Sparse + RRF
- + Rerank:    Hybrid + Cross-Encoder 精排
- ★ Full Sys: 真实路由(预测意图) + Intent-aware Retrieve + Rerank

指标:
- Hit Rate@K: Top-K 中是否命中了期望的文档来源
- MRR: 第一个命中结果的排名倒数均值
"""
import os
import sys
import json
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pymilvus import connections, Collection
from retriever import (
    load_embed_model,
    encode_query,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
    hybrid_retrieve,
)
from reranker import load_reranker, rerank
from router import classify_intent


# ================= 新增：LLM 裁判模块 =================
OLLAMA_BASE_URL = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:14b"

def llm_judge(query: str, content: str) -> bool:
    """
    使用 LLM 作为裁判，判断 content 是否能回答 query
    """
    prompt = f"""你是一个客观的评测裁判。请判断提供的【参考文档】是否包含了能够回答【用户问题】的足够信息。
如果可以回答，请仅输出一个单词: YES
如果不可以回答，请仅输出一个单词: NO

用户问题: {query}
参考文档: {content}
裁判结果:"""

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": JUDGE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0, # 裁判需要极度的确定性
                    "num_predict": 5,
                },
            },
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()["response"].strip().upper()
        return "YES" in result
    except Exception as e:
        print(f"  [Judge Error] {e}")
        return False

# ────────────────────── 评测指标计算 ──────────────────────

def is_hit(query: str, results: list[dict], expected_sources: list[str]) -> bool:
    """
    新版命中判断逻辑：来源匹配 + LLM语义判定
    """
    for doc in results:
        source = doc.get("source_file", "")
        content = doc.get("content", "")

        # 1. 过滤来源：必须是你预期的那几篇官方文档中的
        source_match = any(src in source for src in expected_sources)

        if source_match:
            # 2. 如果来源对了，让大模型看看内容能不能回答问题
            if llm_judge(query, content):
                return True

    return False


def find_first_hit_rank(query: str, results: list[dict], expected_sources: list[str]) -> int:
    """
    找到第一个命中结果的排名 (1-indexed)
    """
    for i, doc in enumerate(results):
        source = doc.get("source_file", "")
        content = doc.get("content", "")

        source_match = any(src in source for src in expected_sources)
        if source_match:
            if llm_judge(query, content):
                return i + 1

    return 0


# ================= 修改 compute_metrics 以传入 query =================

def compute_metrics(results_list: list[dict], eval_data: list[dict], top_k_list: list[int] = [3, 5]) -> dict:
    metrics = {}
    details = []

    for k in top_k_list:
        hits = 0
        for results, item in zip(results_list, eval_data):
            top_k_results = results[:k]
            # 注意这里传入了 item["query"]
            if is_hit(item["query"], top_k_results, item["expected_sources"]):
                hits += 1
        metrics[f"hit_rate@{k}"] = hits / len(eval_data)

    # MRR
    rr_sum = 0
    for results, item in zip(results_list, eval_data):
        # 注意这里传入了 item["query"]
        rank = find_first_hit_rank(item["query"], results, item["expected_sources"])
        rr = 1.0 / rank if rank > 0 else 0.0
        rr_sum += rr
        details.append({
            "query": item["query"],
            "category": item["category"],
            "first_hit_rank": rank,
            "reciprocal_rank": rr,
        })

    metrics["mrr"] = rr_sum / len(eval_data)
    metrics["details"] = details

    return metrics


# ────────────────────── 工具函数 ──────────────────────

def dense_hits_to_rrf_docs(dense_hits: list[dict], top_k: int, rrf_k: int = 60) -> list[dict]:
    """把纯 Dense 结果补齐成统一字段格式。"""
    results = []
    for idx, hit in enumerate(dense_hits[:top_k]):
        doc = hit.copy()
        doc["rrf_score"] = 1.0 / (rrf_k + idx + 1)
        doc["dense_rank"] = idx + 1
        doc["sparse_rank"] = None
        results.append(doc)
    return results


# ────────────────────── 消融实验 ──────────────────────

def run_ablation(
    eval_data: list[dict],
    embed_model,
    reranker_model,
    collection: Collection,
    top_k: int = 20,
    rerank_top_k: int = 5,
    route_use_llm: bool = True,
):
    """
    运行消融实验，对比四种配置:
    1. Baseline:    仅 Dense 检索
    2. + Hybrid:    Dense + Sparse + RRF (不分意图)
    3. + Rerank:    Hybrid + Cross-Encoder 精排
    4. ★ Full Sys: 预测意图路由 + Intent-aware Retrieve + Rerank

    注意:
    - ★ Full System 现在是“真实链路”实验，不再使用数据集的 category 真值做 oracle 路由。
    - category 只用于统计预测意图和人工标签的一致性，帮助排查路由问题。
    """
    dense_only_results = []
    hybrid_results = []
    rerank_results = []
    full_system_results = []
    route_logs = []

    total = len(eval_data)

    print(f"\n开始消融实验，共 {total} 条评测数据")
    print("=" * 70)

    for i, item in enumerate(eval_data):
        query = item["query"]
        oracle_intent = item.get("category", "factual")
        predicted_intent = classify_intent(query, use_llm=route_use_llm)

        route_logs.append({
            "query": query,
            "oracle_intent": oracle_intent,
            "predicted_intent": predicted_intent,
            "match": oracle_intent == predicted_intent,
        })

        print(f"\n[{i+1}/{total}] [oracle={oracle_intent} | pred={predicted_intent}] {query}")

        # 统一先编码，供 Baseline/Hybrid/Rerank 复用
        query_vecs = encode_query(embed_model, query)

        # ── 1. Dense Only ──
        dense_hits = dense_search(collection, query_vecs["dense"], top_k=top_k)
        dense_only_results.append(dense_hits)

        # ── 2. Hybrid (不分意图，统一策略) ──
        sparse_vec = query_vecs["sparse"]
        if sparse_vec:
            sparse_hits = sparse_search(collection, sparse_vec, top_k=5)
            fused, stats = reciprocal_rank_fusion(
                dense_hits, sparse_hits, top_k=top_k,
                dense_weight=1.0, sparse_weight=0.2,
            )
        else:
            sparse_hits = []
            fused = dense_hits_to_rrf_docs(dense_hits, top_k=top_k)
            stats = {
                "dense_only": len(fused),
                "sparse_only": 0,
                "overlap": 0,
                "total_unique": len(fused),
            }
        hybrid_results.append(fused)

        print(
            f"    Hybrid: Dense={len(dense_hits)}, Sparse={len(sparse_hits)}, "
            f"Overlap={stats['overlap']}"
        )

        # ── 3. Hybrid + Rerank ──
        reranked = rerank(
            query=query,
            candidates=fused,
            reranker=reranker_model,
            top_k=rerank_top_k,
        )
        rerank_results.append(reranked)

        if reranked:
            print(
                f"    Rerank Top-1: score={reranked[0].get('rerank_score', 0):.4f} "
                f"| {reranked[0].get('source_file', '')} > {reranked[0].get('title', '')}"
            )

        # ── 4. ★ Full System (真实预测路由) ──
        full_candidates = hybrid_retrieve(
            query=query,
            model=embed_model,
            collection=collection,
            intent=predicted_intent,
            final_top_k=top_k,
        )
        full_rerank_top_k = max(rerank_top_k, 8) if predicted_intent == "troubleshoot" else rerank_top_k
        full_reranked = rerank(
            query=query,
            candidates=full_candidates,
            reranker=reranker_model,
            top_k=full_rerank_top_k,
        )
        full_system_results.append(full_reranked)

        if full_reranked:
            print(
                f"    FullSys Top-1: score={full_reranked[0].get('rerank_score', 0):.4f} "
                f"| {full_reranked[0].get('source_file', '')} > {full_reranked[0].get('title', '')}"
            )

    return dense_only_results, hybrid_results, rerank_results, full_system_results, route_logs


# ────────────────────── 结果展示 ──────────────────────

def print_ablation_report(
    dense_metrics: dict,
    hybrid_metrics: dict,
    rerank_metrics: dict,
    full_system_metrics: dict | None = None,
    route_logs: list[dict] | None = None,
):
    """打印消融实验报告。"""
    print(f"\n{'='*70}")
    print(f"{'消融实验报告':^70}")
    print(f"{'='*70}")

    print(f"\n{'方法':<28} {'Hit Rate@1':>12} {'Hit Rate@3':>12} {'MRR':>12}")
    print(f"{'-'*66}")

    rows = [
        ("Baseline (Dense Only)", dense_metrics),
        ("+ Hybrid (RRF)", hybrid_metrics),
        ("+ Rerank (CE)", rerank_metrics),
    ]
    if full_system_metrics:
        rows.append(("★ Full System (Pred Route)", full_system_metrics))

    for name, m in rows:
        print(
            f"{name:<28} "
            f"{m.get('hit_rate@1', 0):>11.1%} "
            f"{m.get('hit_rate@3', 0):>11.1%} "
            f"{m.get('mrr', 0):>11.4f}"
        )

    print(f"\n{'提升幅度 (相对 Baseline)':^70}")
    print(f"{'-'*66}")
    baseline_hr1 = dense_metrics.get("hit_rate@1", 0)
    baseline_hr3 = dense_metrics.get("hit_rate@3", 0)
    baseline_mrr = dense_metrics.get("mrr", 0)

    compare_rows = [("+ Hybrid", hybrid_metrics), ("+ Rerank", rerank_metrics)]
    if full_system_metrics:
        compare_rows.append(("★ Full System", full_system_metrics))

    for name, m in compare_rows:
        print(
            f"{name:<28} "
            f"HR@1 {m.get('hit_rate@1', 0) - baseline_hr1:>+.1%}  "
            f"HR@3 {m.get('hit_rate@3', 0) - baseline_hr3:>+.1%}  "
            f"MRR {m.get('mrr', 0) - baseline_mrr:>+.4f}"
        )

    if route_logs:
        match_count = sum(1 for x in route_logs if x["match"])
        route_acc = match_count / len(route_logs)
        print(f"\n{'路由统计':^70}")
        print(f"{'-'*66}")
        print(f"预测意图 vs 数据集标签 一致率: {route_acc:.1%} ({match_count}/{len(route_logs)})")

        mismatches = [x for x in route_logs if not x["match"]]
        if mismatches:
            print("路由不一致示例:")
            for item in mismatches[:5]:
                print(
                    f"  - [{item['oracle_intent']} -> {item['predicted_intent']}] "
                    f"{item['query']}"
                )

    print(f"\n{'='*70}")
    print(f"{'各 Query 详细命中情况':^70}")
    print(f"{'='*70}")
    print(f"\n{'Query':<52} {'Dense':>6} {'Hybrid':>7} {'Rerank':>7} {'Full':>7}")
    print(f"{'-'*84}")

    full_details = full_system_metrics["details"] if full_system_metrics else [None] * len(dense_metrics["details"])
    for d, h, r, f in zip(
        dense_metrics["details"],
        hybrid_metrics["details"],
        rerank_metrics["details"],
        full_details,
    ):
        query_short = d["query"][:49] + "..." if len(d["query"]) > 49 else d["query"]
        dr = d["first_hit_rank"] or "miss"
        hr = h["first_hit_rank"] or "miss"
        rr = r["first_hit_rank"] or "miss"
        fr = (f["first_hit_rank"] if f else 0) or "miss"
        print(f"{query_short:<52} {str(dr):>6} {str(hr):>7} {str(rr):>7} {str(fr):>7}")

    print(f"\n{'='*70}")


def save_report(
    dense_metrics: dict,
    hybrid_metrics: dict,
    rerank_metrics: dict,
    full_system_metrics: dict | None = None,
    route_logs: list[dict] | None = None,
    output_path: str = "evaluation/eval_report.json",
):
    """保存评测报告为 JSON。"""
    report = {
        "baseline_dense_only": {
            "hit_rate@1": dense_metrics.get("hit_rate@1", 0),
            "hit_rate@3": dense_metrics.get("hit_rate@3", 0),
            "mrr": dense_metrics.get("mrr", 0),
        },
        "hybrid_rrf": {
            "hit_rate@1": hybrid_metrics.get("hit_rate@1", 0),
            "hit_rate@3": hybrid_metrics.get("hit_rate@3", 0),
            "mrr": hybrid_metrics.get("mrr", 0),
        },
        "hybrid_rerank": {
            "hit_rate@1": rerank_metrics.get("hit_rate@1", 0),
            "hit_rate@3": rerank_metrics.get("hit_rate@3", 0),
            "mrr": rerank_metrics.get("mrr", 0),
        },
    }

    if full_system_metrics:
        report["full_system_pred_route"] = {
            "hit_rate@1": full_system_metrics.get("hit_rate@1", 0),
            "hit_rate@3": full_system_metrics.get("hit_rate@3", 0),
            "mrr": full_system_metrics.get("mrr", 0),
        }

    if route_logs:
        match_count = sum(1 for x in route_logs if x["match"])
        report["routing"] = {
            "accuracy_vs_dataset_category": match_count / len(route_logs),
            "details": route_logs,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n评测报告已保存: {output_path}")
    return report


# ────────────────────── 主入口 ──────────────────────

if __name__ == "__main__":
    eval_path = "eval_dataset.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"加载评测数据: {len(eval_data)} 条")

    print("连接 Milvus...")
    connections.connect(host="localhost", port="19530")
    collection = Collection("hybrid_rag_docs")
    collection.load()
    print(f"Collection 已加载，共 {collection.num_entities} 条记录")

    print("\n加载模型...")
    embed_model = load_embed_model()
    reranker_model = load_reranker()

    start = time.time()
    dense_results, hybrid_results, rerank_results, full_results, route_logs = run_ablation(
        eval_data=eval_data,
        embed_model=embed_model,
        reranker_model=reranker_model,
        collection=collection,
        top_k=20,
        rerank_top_k=5,
        route_use_llm=True,
    )
    elapsed = time.time() - start
    print(f"\n消融实验完成，耗时: {elapsed:.1f}s")

    print("\n计算评测指标...")
    dense_metrics = compute_metrics([results[:3] for results in dense_results], eval_data, top_k_list=[1, 3])
    hybrid_metrics = compute_metrics([results[:3] for results in hybrid_results], eval_data, top_k_list=[1, 3])
    rerank_metrics = compute_metrics(rerank_results, eval_data, top_k_list=[1, 3])
    full_system_metrics = compute_metrics(full_results, eval_data, top_k_list=[1, 3])

    print_ablation_report(
        dense_metrics,
        hybrid_metrics,
        rerank_metrics,
        full_system_metrics,
        route_logs,
    )
    save_report(
        dense_metrics,
        hybrid_metrics,
        rerank_metrics,
        full_system_metrics,
        route_logs,
    )

    connections.disconnect("default")
    print("\n评测完成!")
