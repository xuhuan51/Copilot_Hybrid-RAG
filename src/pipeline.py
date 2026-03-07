"""
RAG 全链路 Pipeline
串联: Query改写 → 语义路由 → 混合检索 → 精排 → 答案生成

一个函数走完从"用户提问"到"生成带引用答案"的完整流程
"""
import os
import inspect

import requests

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pymilvus import connections, Collection
from retriever import load_embed_model, hybrid_retrieve
from reranker import load_reranker, rerank
from router import classify_intent, get_prompt, build_context
from memory import ConversationMemory, rewrite_query
from generator import generate_with_sources, generate_answer


# ────────────────────── Pipeline 核心 ──────────────────────

class RAGPipeline:
    """
    企业级 Hybrid RAG 全链路 Pipeline

    流程:
    1. [Memory]    多轮对话 Query 改写
    2. [Router]    语义意图分类
    3. [Retriever] Dense + Sparse 混合检索 + RRF 融合
    4. [Reranker]  Cross-Encoder 精排
    5. [Generator] LLM 答案生成 + 引用溯源
    """

    def __init__(
        self,
        collection_name: str = "hybrid_rag_docs",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        llm_model: str = "qwen2.5:14b",
        device: str = "cuda",
    ):
        print("=" * 60)
        print("初始化 RAG Pipeline")
        print("=" * 60)

        # 1. 连接 Milvus
        print("\n[1/4] 连接 Milvus...")
        connections.connect(host=milvus_host, port=milvus_port)
        self.collection = Collection(collection_name)
        self.collection.load()
        print(f"  Collection: {collection_name} ({self.collection.num_entities} 条记录)")

        # 2. 加载 Embedding 模型
        print("\n[2/4] 加载 Embedding 模型...")
        self.embed_model = load_embed_model(device=device)

        # 3. 加载 Reranker 模型
        print("\n[3/4] 加载 Reranker 模型...")
        self.reranker_model = load_reranker(device=device)

        # 4. 初始化对话记忆
        print("\n[4/4] 初始化对话记忆...")
        self.memory = ConversationMemory(max_turns=5)

        self.llm_model = llm_model

        print(f"\n{'=' * 60}")
        print("RAG Pipeline 初始化完成!")
        print(f"{'=' * 60}\n")

    def query(
            self,
            user_query: str,
            retrieve_top_k: int = 20,
            rerank_top_k: int = 5,
            use_memory: bool = True,
            use_router: bool = True,
            use_hyde: bool = True,  # <--- HyDE 开关
            verbose: bool = True,
    ) -> dict:
        """
        完整的 RAG 问答流程 (已集成 HyDE 与动态阈值截断)
        """
        import inspect  # 确保 inspect 已导入

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"📝 用户提问: {user_query}")
            print(f"{'─' * 60}")

        # ──── Step 1: Query 改写 ────
        if use_memory:
            search_query = rewrite_query(user_query, self.memory, use_llm=True)
        else:
            search_query = user_query

        if verbose and search_query != user_query:
            print(f"\n🔄 Query 改写: {search_query}")

        # ──── 新增 Step 1.5: HyDE 拓展 ────
        if use_hyde:
            if verbose:
                print(f"\n🧠 HyDE 生成假设性回答中...")
            # 让 LLM 生成假答案并拼接，用于扩大 Dense 召回命中率
            hyde_query = self.generate_hyde_query(search_query, use_hyde=use_hyde, llm_model=self.llm_model)
            if verbose and hyde_query != search_query:
                print(f"  [HyDE 增强 Query 长度]: {len(hyde_query)}")
        else:
            hyde_query = search_query

        # ──── Step 2: 意图分类 ────
        # ⚠️ 注意：意图分类必须用原始的 search_query，防止被 HyDE 的幻觉内容干扰
        if use_router:
            intent = classify_intent(search_query, use_llm=True)
        else:
            intent = "factual"

        if verbose:
            intent_labels = {
                "factual": "📋 事实查询",
                "conceptual": "💡 原理解释",
                "troubleshoot": "🔧 故障排查",
            }
            print(f"\n🎯 意图路由: {intent_labels.get(intent, intent)}")

        # ──── Step 3: 混合检索 ────
        if verbose:
            print(f"\n🔍 混合检索中...")

        retrieve_kwargs = {
            "query": hyde_query,  # ⚠️ 这里传入增强后的 hyde_query 用于混合检索
            "model": self.embed_model,
            "collection": self.collection,
            "final_top_k": retrieve_top_k,
        }
        # 新版 retriever 已支持 intent-aware retrieval；旧版无该参数时保持兼容
        if "intent" in inspect.signature(hybrid_retrieve).parameters:
            retrieve_kwargs["intent"] = intent

        coarse_results = hybrid_retrieve(**retrieve_kwargs)

        # ──── Step 4: Cross-Encoder 精排 ────
        if verbose:
            print(f"\n⚡ Cross-Encoder 精排中...")

        effective_rerank_top_k = max(rerank_top_k, 8) if intent == "troubleshoot" else rerank_top_k
        fine_results = rerank(
            query=search_query,  # ⚠️ 精排时必须用真实的 search_query 算分，抛弃 HyDE 内容
            candidates=coarse_results,
            reranker=self.reranker_model,
            top_k=effective_rerank_top_k,
        )

        # ──── 新增 Step 4.5: 动态阈值截断 (Dynamic Cut-off) ────
        before_cutoff_len = len(fine_results)
        # 剔除排名靠后、分数骤降的噪声文档，节省 Token 并降低大模型幻觉
        fine_results = self.dynamic_cutoff(fine_results, min_score=0.15, drop_threshold=0.2)

        if verbose:
            if len(fine_results) < before_cutoff_len:
                print(f"  ✂️ 动态截断生效: 保留了 {len(fine_results)}/{before_cutoff_len} 篇高优文档")
            else:
                print(f"  ✅ 动态截断: 未触发，保留全部 {len(fine_results)} 篇文档")

            print(f"  最终精排 Top-{len(fine_results)} 结果:")
            for i, doc in enumerate(fine_results):
                print(f"    [{i + 1}] score={doc['rerank_score']:.4f} | {doc['source_file']} > {doc['title']}")

        # ──── Step 5: 组装 Prompt ────
        context = build_context(fine_results)
        prompt, intent = get_prompt(
            query=search_query,
            context=context,
            intent=intent,
        )

        # ──── Step 6: 生成答案 ────
        if verbose:
            print(f"\n🤖 生成答案中...")

        result = generate_with_sources(
            query=search_query,
            prompt=prompt,
            docs=fine_results,
            intent=intent,
            model=self.llm_model,
        )

        # ──── Step 7: 更新对话历史 ────
        if use_memory:
            self.memory.add_user_message(user_query)
            self.memory.add_assistant_message(result["answer"])

        # 组装最终结果
        output = {
            "query": user_query,
            "rewritten_query": search_query,
            "intent": intent,
            "answer": result["answer"],
            "sources": result["sources"],
            "confident": result["confident"],
            "top_rerank_score": result["top_rerank_score"] if result["top_rerank_score"] else 0,
            "retrieved_docs": fine_results,
            "retrieved_images": result.get("retrieved_images", []),  # 新增
        }

        # ──── 输出结果 ────
        if verbose:
            self._print_result(output)

        return output



    def generate_hyde_query(query: str, use_hyde: bool = True, llm_model: str = "qwen2.5:14b") -> str:
        """
        HyDE (Hypothetical Document Embeddings) 召回前置增强
        让 LLM 先生成一个假设性答案，再拼接回原 query 用于 Dense 检索。
        """
        if not use_hyde:
            return query

        hyde_prompt = f"""你是一个资深的企业级技术专家。请根据以下问题，提供一段简短的、假设性的标准答案。
    注意：不需要保证回答完全正确，只需尽可能包含该问题涉及的核心技术术语、配置参数或原理解释即可。保持简短。

    用户问题: {query}
    假设性答案:"""

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": llm_model,
                    "prompt": hyde_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # 稍微给点温度，允许发散出相关的技术词
                        "num_predict": 150,  # 限制长度，不需要长篇大论
                    },
                },
                timeout=15,
            )
            resp.raise_for_status()
            hypothetical_answer = resp.json()["response"].strip()

            # 将原问题和假设性答案拼接。这能极大丰富 Dense 向量的语义特征。
            enhanced_query = f"{query}\n{hypothetical_answer}"
            return enhanced_query
        except Exception as e:
            print(f"  [HyDE] 生成失败: {e}，回退到原始 query")
            return query

    def dynamic_cutoff(scored_docs: list[dict], min_score: float = 0.15, drop_threshold: float = 0.2) -> list[dict]:
        """
        动态阈值截断机制

        参数:
            scored_docs: 经过 Reranker 打分并降序排列的文档列表
            min_score: 绝对分数底线。低于此分数的文档直接丢弃。
            drop_threshold: 相对分数落差阈值。如果两篇相邻文档分差大于此值，丢弃后面的所有文档。
        """
        if not scored_docs:
            return []

        filtered_docs = [scored_docs[0]]  # 排名第一的肯定要保留（除非它也低于 min_score，后面统一兜底）

        for i in range(1, len(scored_docs)):
            prev_score = scored_docs[i - 1].get("rerank_score", 0)
            curr_score = scored_docs[i].get("rerank_score", 0)

            # 条件1：断崖式下跌截断
            if (prev_score - curr_score) >= drop_threshold:
                print(
                    f"  [Cut-off] 触发断崖截断: Doc {i} (score {prev_score:.3f}) -> Doc {i + 1} (score {curr_score:.3f})")
                break

            # 条件2：绝对低分截断
            if curr_score < min_score:
                print(f"  [Cut-off] 触发低分截断: Doc {i + 1} 分数 {curr_score:.3f} 低于阈值 {min_score}")
                break

            filtered_docs.append(scored_docs[i])

        # 兜底：如果第一篇的分数也低得离谱，说明这道题知识库里根本没有
        if filtered_docs and filtered_docs[0].get("rerank_score", 0) < min_score:
            print("  [Cut-off] Top-1 文档分数过低，清空所有召回结果以防幻觉。")
            return []

        return filtered_docs

    def _print_result(self, result: dict):
        """格式化输出结果"""
        print(f"\n{'═' * 60}")

        if not result["confident"]:
            print("⚠️  知识库相关度较低，答案仅供参考")
            print(f"{'─' * 60}")

        print(f"\n{result['answer']}")
        print(f"\n{'─' * 60}")
        print(result["sources"])

        images = result.get("retrieved_images", [])
        if images:
            print(f"\n🖼 命中图片:")
            for i, img in enumerate(images, 1):
                print(
                    f"  [{i}] {img['image_path']} "
                    f"| {img['source_file']} > {img.get('header_path', '')} "
                    f"(score={img.get('rerank_score', 0):.2f})"
                )

        print(f"{'═' * 60}")

    def reset_memory(self):
        """重置对话历史"""
        self.memory.clear()
        print("对话历史已清空")

    def close(self):
        """关闭连接"""
        connections.disconnect("default")
        print("Pipeline 已关闭")


# ────────────────────── 交互式测试 ──────────────────────

def interactive_mode(pipeline: RAGPipeline):
    """交互式问答模式"""
    print("\n" + "=" * 60)
    print("🚀 进入交互式问答模式")
    print("  输入问题开始问答")
    print("  输入 /clear 清空对话历史")
    print("  输入 /quit  退出")
    print("=" * 60)

    while True:
        try:
            query = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        if query == "/quit":
            break
        elif query == "/clear":
            pipeline.reset_memory()
            continue

        pipeline.query(query)


# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    # 初始化 Pipeline
    pipeline = RAGPipeline()

    # 预设测试 query
    test_queries = [
        "Kafka consumer max.poll.records 默认值是多少",
        "Redis RDB 和 AOF 持久化的区别是什么",
    ]

    for q in test_queries:
        pipeline.query(q)

    # 测试多轮对话
    print("\n\n" + "#" * 60)
    print("多轮对话测试")
    print("#" * 60)

    pipeline.reset_memory()
    pipeline.query("Kafka consumer group 是怎么工作的")
    pipeline.query("它的 rebalance 机制呢")  # 测试 query 改写

    # 进入交互模式（可选）
    # interactive_mode(pipeline)

    pipeline.close()