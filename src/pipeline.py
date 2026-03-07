"""
RAG 全链路 Pipeline
串联: Query改写 → 语义路由 → 混合检索 → 精排 → 答案生成

一个函数走完从"用户提问"到"生成带引用答案"的完整流程
"""
import os
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
        verbose: bool = True,
    ) -> dict:
        """
        完整的 RAG 问答流程

        参数:
            user_query:     用户问题
            retrieve_top_k: 混合检索召回数量
            rerank_top_k:   精排后保留数量
            use_memory:     是否启用多轮对话改写
            use_router:     是否启用语义路由
            verbose:        是否打印中间过程

        返回:
            {
                "query":           原始用户问题,
                "rewritten_query": 改写后的检索 query,
                "intent":          识别的意图类型,
                "answer":          生成的答案,
                "sources":         引用溯源信息,
                "confident":       是否置信,
                "top_rerank_score": Top-1 精排分数,
                "retrieved_docs":  精排后的文档列表,
            }
        """
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

        # ──── Step 2: 意图分类 ────
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

        coarse_results = hybrid_retrieve(
            query=search_query,
            model=self.embed_model,
            collection=self.collection,
            final_top_k=retrieve_top_k,
        )

        # ──── Step 4: Cross-Encoder 精排 ────
        if verbose:
            print(f"\n⚡ Cross-Encoder 精排中...")

        fine_results = rerank(
            query=search_query,
            candidates=coarse_results,
            reranker=self.reranker_model,
            top_k=rerank_top_k,
        )

        if verbose:
            print(f"  精排 Top-{len(fine_results)} 结果:")
            for i, doc in enumerate(fine_results):
                print(f"    [{i+1}] score={doc['rerank_score']:.4f} | {doc['source_file']} > {doc['title']}")

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
            "top_rerank_score": result["top_rerank_score"],
            "retrieved_docs": fine_results,
        }

        # ──── 输出结果 ────
        if verbose:
            self._print_result(output)

        return output

    def _print_result(self, result: dict):
        """格式化输出结果"""
        print(f"\n{'═' * 60}")

        if not result["confident"]:
            print("⚠️  知识库相关度较低，答案仅供参考")
            print(f"{'─' * 60}")

        print(f"\n{result['answer']}")
        print(f"\n{'─' * 60}")
        print(result["sources"])
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