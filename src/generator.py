"""
答案生成模块 (Generator)
调用 Qwen2.5 (Ollama) 生成带引用溯源的结构化答案
"""
import requests


# ────────────────────── Ollama 调用 ──────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"


def call_ollama(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    stream: bool = False,
) -> str:
    """
    调用 Ollama 本地 LLM

    参数:
        prompt:       完整的 prompt (已由 router 组装好)
        model:        模型名称
        temperature:  生成温度 (低温更准确，高温更发散)
        max_tokens:   最大生成长度
        stream:       是否流式输出
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=120,  # 生成较长回答可能需要更多时间
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def call_ollama_stream(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """
    流式调用 Ollama（用于前端逐字输出）

    Yields:
        str: 每次生成的文本片段
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if line:
            import json
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break


# ────────────────────── 答案生成 ──────────────────────

def generate_answer(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """
    生成答案（非流式）

    参数:
        prompt:  由 router.get_prompt() 组装好的完整 prompt
    """
    answer = call_ollama(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return answer


# ────────────────────── 置信度判断 ──────────────────────

def check_confidence(docs: list[dict], threshold: float = 0.15) -> bool:
    """
    根据精排分数判断检索结果是否足够可信

    如果 Top-1 的 rerank_score 低于阈值，说明知识库中
    可能没有高度相关的内容，应提示用户

    参数:
        docs:       精排后的文档列表
        threshold:  置信度阈值
    """
    if not docs:
        return False
    top_score = docs[0].get("rerank_score", 0)
    return top_score >= threshold


LOW_CONFIDENCE_PROMPT = """你是一个企业技术文档助手。

注意：当前知识库中未找到与用户问题高度相关的内容。
以下参考资料的相关性较低，请谨慎参考。

如果参考资料确实无法回答用户的问题，请诚实告知用户"当前知识库中暂未收录相关内容"，
并建议用户换一种提问方式或补充相关文档。

参考资料:
{context}

用户问题: {query}

请回答:"""


# ────────────────────── 引用溯源格式化 ──────────────────────

def format_sources(docs: list[dict]) -> str:
    """
    将引用的文档来源格式化为溯源信息

    输出格式:
    📎 引用来源:
    [1] cloudera-kafka > Consumer Configuration Properties
    [2] confluent-kafka > max.partition.fetch.bytes
    """
    if not docs:
        return ""

    lines = ["📎 引用来源:"]
    for i, doc in enumerate(docs):
        source = doc.get("source_file", "未知")
        header = doc.get("header_path", doc.get("title", ""))
        score = doc.get("rerank_score", 0)
        lines.append(f"  [{i+1}] {source} > {header}  (相关度: {score:.2f})")

    return "\n".join(lines)


def collect_retrieved_images(docs: list[dict], max_images: int = 3) -> list[dict]:
    """
    从精排结果中收集命中的图片，按文档顺序去重返回

    返回格式:
    [
        {
            "image_path": "...",
            "source_file": "...",
            "title": "...",
            "header_path": "...",
            "rerank_score": 0.91,
        }
    ]
    """
    images = []
    seen = set()

    for doc in docs:
        image_path = (doc.get("image_path") or "").strip()
        if not image_path or image_path in seen:
            continue

        seen.add(image_path)
        images.append({
            "image_path": image_path,
            "source_file": doc.get("source_file", "未知"),
            "title": doc.get("title", ""),
            "header_path": doc.get("header_path", ""),
            "rerank_score": doc.get("rerank_score", 0),
        })

        if len(images) >= max_images:
            break

    return images


# ────────────────────── 完整生成流程 ──────────────────────
def generate_with_sources(
    query: str,
    prompt: str,
    docs: list[dict],
    intent: str,
    model: str = LLM_MODEL,
    confidence_threshold: float = 0.15,
) -> dict:
    """
    完整的答案生成流程，包含置信度判断、引用溯源和图片返回
    """
    confident = check_confidence(docs, threshold=confidence_threshold)

    if not confident and docs:
        from router import build_context
        context = build_context(docs)
        prompt = LOW_CONFIDENCE_PROMPT.format(context=context, query=query)

    answer = generate_answer(prompt=prompt, model=model)
    sources = format_sources(docs)
    retrieved_images = collect_retrieved_images(docs, max_images=3)

    return {
        "answer": answer,
        "sources": sources,
        "intent": intent,
        "confident": confident,
        "top_rerank_score": docs[0].get("rerank_score", 0) if docs else 0,
        "retrieved_images": retrieved_images,   # 新增
    }

# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Generator 模块测试")
    print("=" * 60)

    # 简单测试 Ollama 连通性
    test_prompt = "用一句话解释什么是 Kafka consumer group。"

    print(f"\n测试 prompt: {test_prompt}")
    print("-" * 40)

    try:
        answer = generate_answer(test_prompt)
        print(f"回答: {answer}")
    except Exception as e:
        print(f"Ollama 调用失败: {e}")
        print("请确认 Ollama 已启动: ollama serve")

    # 测试流式输出
    print(f"\n{'─' * 40}")
    print("流式输出测试:")
    try:
        for token in call_ollama_stream("什么是 Redis 的 AOF 持久化？请简短回答。"):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"流式调用失败: {e}")

    print(f"\n{'=' * 60}")
    print("测试完成!")