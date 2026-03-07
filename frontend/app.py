"""
企业级研发效能 Copilot - 前端交互界面
基于 Streamlit 构建的多轮对话问答系统
"""
import sys
import os

# 将 src 目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from generator import check_confidence, LOW_CONFIDENCE_PROMPT, call_ollama_stream
from memory import rewrite_query
from pipeline import RAGPipeline
from reranker import rerank
from retriever import hybrid_retrieve
from router import classify_intent, build_context, get_prompt


import streamlit as st
import time


# ────────────────────── 页面配置 ──────────────────────

st.set_page_config(
    page_title="研发效能 Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────── 自定义样式 ──────────────────────

st.markdown("""
<style>
    /* 主标题 */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
    }

    /* 引用来源卡片 */
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .source-card .score {
        color: #667eea;
        font-weight: 600;
    }

    /* 检索统计 */
    .stats-bar {
        display: flex;
        gap: 1rem;
        padding: 0.5rem 0;
        flex-wrap: wrap;
    }
    .stat-item {
        background: #f0f2ff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #4b5563;
    }

    /* 意图标签 */
    .intent-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .intent-factual { background: #dbeafe; color: #1d4ed8; }
    .intent-conceptual { background: #fef3c7; color: #92400e; }
    .intent-troubleshoot { background: #fee2e2; color: #991b1b; }

    /* 隐藏 Streamlit 默认样式 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ────────────────────── Pipeline 初始化 ──────────────────────

@st.cache_resource
def init_pipeline():
    """初始化 RAG Pipeline（只在首次加载时执行）"""
    return RAGPipeline()


def get_pipeline():
    """获取 Pipeline 实例"""
    try:
        return init_pipeline()
    except Exception as e:
        st.error(f"Pipeline 初始化失败: {e}")
        st.info("请确认 Milvus 和 Ollama 服务已启动")
        st.stop()


# ────────────────────── 侧边栏 ──────────────────────

def render_sidebar():
    """渲染侧边栏设置"""
    with st.sidebar:
        st.markdown("## ⚙️ 检索设置")

        retrieve_top_k = st.slider(
            "粗排召回数量",
            min_value=5, max_value=50, value=20,
            help="Dense + Sparse 双路检索后 RRF 融合的候选数量"
        )

        rerank_top_k = st.slider(
            "精排保留数量",
            min_value=1, max_value=10, value=5,
            help="Cross-Encoder 精排后送入 LLM 的文档数量"
        )

        st.markdown("---")
        st.markdown("## 🔧 功能开关")

        use_memory = st.checkbox("多轮对话改写", value=True,
                                  help="开启后支持追问，自动补全上下文")
        use_router = st.checkbox("语义路由", value=True,
                                  help="根据问题意图自动选择 Prompt 模板")

        st.markdown("---")

        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            pipeline = get_pipeline()
            pipeline.reset_memory()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 系统信息")
        pipeline = get_pipeline()
        st.markdown(f"- 文档数: **{pipeline.collection.num_entities}** 条")
        st.markdown(f"- LLM: **{pipeline.llm_model}**")
        st.markdown(f"- 对话轮数: **{pipeline.memory.turn_count}**")

        return {
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_k": rerank_top_k,
            "use_memory": use_memory,
            "use_router": use_router,
        }


# ────────────────────── 结果渲染 ──────────────────────

def render_intent_tag(intent: str) -> str:
    """渲染意图标签"""
    labels = {
        "factual": ("📋 事实查询", "intent-factual"),
        "conceptual": ("💡 原理解释", "intent-conceptual"),
        "troubleshoot": ("🔧 故障排查", "intent-troubleshoot"),
    }
    label, cls = labels.get(intent, ("❓ 未知", "intent-factual"))
    return f'<span class="intent-tag {cls}">{label}</span>'


def render_sources(docs: list[dict]):
    """渲染引用来源"""
    if not docs:
        return

    with st.expander("📎 引用来源", expanded=False):
        for i, doc in enumerate(docs):
            score = doc.get("rerank_score", 0)
            source = doc.get("source_file", "未知")
            title = doc.get("title", "")
            header = doc.get("header_path", "")

            # 颜色根据分数变化
            if score >= 0.8:
                color = "#16a34a"
            elif score >= 0.5:
                color = "#d97706"
            else:
                color = "#dc2626"

            st.markdown(
                f'<div class="source-card">'
                f'<span class="score" style="color:{color}">相关度 {score:.2f}</span> | '
                f'**{source}** > {title}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 可展开查看原文
            with st.expander(f"查看资料 [{i+1}] 原文", expanded=False):
                st.text(doc.get("content", "")[:500])


def render_search_stats(result: dict):
    """渲染检索统计信息"""
    rewritten = result.get("rewritten_query", "")
    original = result.get("query", "")
    intent = result.get("intent", "")

    cols = []
    if rewritten != original:
        cols.append(f"🔄 改写: *{rewritten}*")

    intent_html = render_intent_tag(intent)

    st.markdown(
        f'<div class="stats-bar">'
        f'{intent_html}'
        f'<span class="stat-item">Top-1 相关度: {result.get("top_rerank_score", 0):.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if rewritten != original:
        st.caption(f"🔄 Query 改写: {rewritten}")


# ────────────────────── 主界面 ──────────────────────

def main():
    # 头部
    st.markdown(
        '<div class="main-header">'
        '<h1>🚀 研发效能 Copilot</h1>'
        '<p>基于 Hybrid RAG 的企业技术文档智能问答系统</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 侧边栏
    settings = render_sidebar()

    # 获取 Pipeline
    pipeline = get_pipeline()

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 如果是 assistant 消息且有额外信息
            if msg["role"] == "assistant" and "result" in msg:
                render_search_stats(msg["result"])
                render_sources(msg["result"].get("retrieved_docs", []))

    # 用户输入
    if user_input := st.chat_input("输入你的技术问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 生成回答
        with st.chat_message("assistant"):
            # ── Step 1: 检索 + 精排 ──
            status = st.status("🔍 混合检索 + 精排中...", expanded=True)
            start_time = time.time()

            with status:
                # Query 改写

                if settings["use_memory"]:
                    search_query = rewrite_query(user_input, pipeline.memory, use_llm=True)
                else:
                    search_query = user_input

                if search_query != user_input:
                    st.write(f"🔄 Query 改写: {search_query}")

                # 意图分类
                if settings["use_router"]:
                    intent = classify_intent(search_query, use_llm=True)
                else:
                    intent = "factual"
                st.write(f"🎯 意图: {intent}")

                # 混合检索
                st.write("🔍 Dense + Sparse 双路检索...")
                coarse_results = hybrid_retrieve(
                    query=search_query,
                    model=pipeline.embed_model,
                    collection=pipeline.collection,
                    final_top_k=settings["retrieve_top_k"],
                )

                # 精排
                st.write("⚡ Cross-Encoder 精排...")
                fine_results = rerank(
                    query=search_query,
                    candidates=coarse_results,
                    reranker=pipeline.reranker_model,
                    top_k=settings["rerank_top_k"],
                )

                retrieve_time = time.time() - start_time
                st.write(f"✅ 检索完成 ({retrieve_time:.1f}s)")

            status.update(label=f"检索完成 ({retrieve_time:.1f}s)", state="complete", expanded=False)

            # ── Step 2: 置信度判断 ──
            confident = check_confidence(fine_results)
            top_score = fine_results[0].get("rerank_score", 0) if fine_results else 0

            if not confident:
                st.warning("⚠️ 知识库中未找到高度相关内容，以下回答仅供参考")

            # ── Step 3: 组装 Prompt ──
            context = build_context(fine_results)
            if not confident and fine_results:
                prompt = LOW_CONFIDENCE_PROMPT.format(context=context, query=search_query)
            else:
                prompt, intent = get_prompt(query=search_query, context=context, intent=intent)

            # ── Step 4: 流式生成答案 ──
            answer_placeholder = st.empty()
            full_answer = ""

            for token in call_ollama_stream(prompt, model=pipeline.llm_model):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            # ── Step 5: 更新对话历史 ──
            if settings["use_memory"]:
                pipeline.memory.add_user_message(user_input)
                pipeline.memory.add_assistant_message(full_answer)

            # 构造结果用于显示统计和来源
            result = {
                "query": user_input,
                "rewritten_query": search_query,
                "intent": intent,
                "answer": full_answer,
                "confident": confident,
                "top_rerank_score": top_score,
                "retrieved_docs": fine_results,
            }

            # 显示检索统计和来源
            render_search_stats(result)
            render_sources(result.get("retrieved_docs", []))

        # 保存到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "result": result,
        })


if __name__ == "__main__":
    main()