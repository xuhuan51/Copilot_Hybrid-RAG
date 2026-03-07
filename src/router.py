"""
语义路由器 (Semantic Router)
根据用户问题意图自动路由到不同的 Prompt 模板
支持三种意图: 事实查询 / 原理解释 / 故障排查
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import requests


# ────────────────────── Ollama 调用 ──────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
ROUTER_MODEL = "qwen2.5:14b"   # 路由用小模型即可，快速响应


def call_ollama(prompt: str, model: str = ROUTER_MODEL) -> str:
    """调用 Ollama 本地 LLM"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,    # 路由分类需要确定性输出
                "num_predict": 20,     # 只需要返回类别标签，限制长度
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ────────────────────── 意图分类 ──────────────────────

# 三种意图类型
INTENT_FACTUAL = "factual"         # 事实查询: 参数值、配置项、API用法
INTENT_CONCEPTUAL = "conceptual"   # 原理解释: 架构原理、概念对比、设计思想
INTENT_TROUBLESHOOT = "troubleshoot"  # 故障排查: 报错处理、性能问题、异常诊断

CLASSIFICATION_PROMPT = """你是一个问题意图分类器。根据用户的问题，判断它属于以下三种类型之一：

1. factual - 事实查询：询问具体的参数值、配置项、API用法、命令语法等
2. conceptual - 原理解释：询问架构原理、概念对比、设计思想、工作机制等
3. troubleshoot - 故障排查：询问报错处理、性能问题、异常诊断、超时排查等

只回答一个单词: factual 或 conceptual 或 troubleshoot

用户问题: {query}
分类结果:"""


def classify_intent(query: str, use_llm: bool = True) -> str:
    """
    对用户 query 进行意图分类

    策略: 先用规则快速匹配，匹配不到再用 LLM 分类
    这样兼顾速度和准确率

    参数:
        query:    用户问题
        use_llm:  是否启用 LLM 分类（False 时只用规则）
    """
    # ---- 规则匹配 (快速路径) ----
    intent = _rule_based_classify(query)
    if intent:
        return intent

    # ---- LLM 分类 (兜底) ----
    if use_llm:
        return _llm_classify(query)

    # 默认走事实查询
    return INTENT_FACTUAL


def _rule_based_classify(query: str) -> str | None:
    """基于关键词规则的快速分类"""
    q = query.lower()

    # 故障排查关键词
    troubleshoot_patterns = [
        r"排查", r"报错", r"错误", r"异常", r"超时", r"失败",
        r"不工作", r"挂了", r"宕机", r"崩溃", r"告警",
        r"error", r"exception", r"timeout", r"fail",
        r"crash", r"troubleshoot", r"debug", r"fix",
        r"怎么解决", r"怎么处理", r"怎么修复",
    ]

    # 原理解释关键词
    conceptual_patterns = [
        r"原理", r"机制", r"架构", r"区别", r"对比", r"比较",
        r"为什么", r"为啥", r"如何工作", r"怎么实现",
        r"设计", r"思想", r"底层", r"本质",
        r"how does.+work", r"what is", r"difference",
        r"compare", r"vs\.?", r"versus", r"explain",
    ]

    # 事实查询关键词
    factual_patterns = [
        r"参数", r"配置", r"默认值", r"取值", r"设置",
        r"命令", r"语法", r"API", r"接口", r"用法",
        r"多少", r"几个", r"哪些", r"列表",
        r"config", r"parameter", r"default", r"value",
        r"how to set", r"how to configure",
    ]

    for pattern in troubleshoot_patterns:
        if re.search(pattern, q):
            return INTENT_TROUBLESHOOT

    for pattern in conceptual_patterns:
        if re.search(pattern, q):
            return INTENT_CONCEPTUAL

    for pattern in factual_patterns:
        if re.search(pattern, q):
            return INTENT_FACTUAL

    return None  # 规则未命中，交给 LLM


def _llm_classify(query: str) -> str:
    """使用 LLM 进行意图分类"""
    try:
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        result = call_ollama(prompt)

        # 从 LLM 输出中提取类别
        result_lower = result.lower().strip()
        if "troubleshoot" in result_lower:
            return INTENT_TROUBLESHOOT
        elif "conceptual" in result_lower:
            return INTENT_CONCEPTUAL
        elif "factual" in result_lower:
            return INTENT_FACTUAL
        else:
            print(f"  [Router] LLM 返回未知类别: {result}，默认 factual")
            return INTENT_FACTUAL
    except Exception as e:
        print(f"  [Router] LLM 分类失败: {e}，回退到 factual")
        return INTENT_FACTUAL


# ────────────────────── Prompt 模板 ──────────────────────

PROMPT_TEMPLATES = {
    INTENT_FACTUAL: """你是一个企业技术文档助手。请根据以下参考资料，准确回答用户的问题。

要求:
- 直接给出具体的参数值、配置项或用法
- 如果参考资料中有代码示例，请引用展示
- 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {query}

请回答:""",

    INTENT_CONCEPTUAL: """你是一个企业技术文档助手。请根据以下参考资料，深入解释用户询问的技术原理。

要求:
- 先给出简明的概念总结
- 再展开解释工作原理或机制
- 如果涉及多个概念的对比，请用清晰的结构说明异同
- 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {query}

请回答:""",

    INTENT_TROUBLESHOOT: """你是一个企业技术文档助手。请根据以下参考资料，帮助用户排查和解决问题。

要求:
- 先分析可能的原因
- 再给出具体的排查步骤（按优先级排序）
- 如果有相关配置建议或命令，请一并提供
- 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {query}

请回答:""",
}


def get_prompt(query: str, context: str, intent: str = None, use_llm: bool = True) -> tuple[str, str]:
    """
    根据意图选择 Prompt 模板并填充

    参数:
        query:    用户问题
        context:  检索到的参考资料（拼接好的文本）
        intent:   如果已知意图可以直接传入，跳过分类
        use_llm:  是否启用 LLM 辅助分类

    返回:
        (完整prompt, 意图类别)
    """
    if intent is None:
        intent = classify_intent(query, use_llm=use_llm)

    template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES[INTENT_FACTUAL])
    prompt = template.format(context=context, query=query)

    return prompt, intent


# ────────────────────── 上下文拼接 ──────────────────────

def build_context(docs: list[dict], max_length: int = 4000) -> str:
    """
    将精排后的文档列表拼接为 LLM 可用的上下文文本

    每个文档包含: 来源、章节路径、内容
    添加序号方便 LLM 引用溯源
    """
    context_parts = []
    current_length = 0

    for i, doc in enumerate(docs):
        part = (
            f"[资料{i+1}]\n"
            f"来源: {doc['source_file']}\n"
            f"章节: {doc.get('header_path', '')}\n"
            f"内容: {doc['content']}\n"
        )

        if current_length + len(part) > max_length:
            # 截断最后一条，尽量保留完整的前几条
            remaining = max_length - current_length
            if remaining > 100:
                part = part[:remaining] + "\n...(截断)"
                context_parts.append(part)
            break

        context_parts.append(part)
        current_length += len(part)

    return "\n---\n".join(context_parts)


# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    # 测试意图分类（仅规则，不依赖 Ollama）
    test_queries = [
        ("Kafka consumer max.poll.records 默认值是多少", INTENT_FACTUAL),
        ("Redis RDB 和 AOF 持久化的区别", INTENT_CONCEPTUAL),
        ("Kafka consumer 一直 rebalance 怎么排查", INTENT_TROUBLESHOOT),
        ("如何配置 Kafka producer 的 batch.size", INTENT_FACTUAL),
        ("为什么 Kafka 要用零拷贝技术", INTENT_CONCEPTUAL),
        ("Redis 连接超时报错怎么解决", INTENT_TROUBLESHOOT),
        ("What is the default replication factor", INTENT_FACTUAL),
        ("How does Kafka consumer group rebalancing work", INTENT_CONCEPTUAL),
    ]

    print("=" * 60)
    print("语义路由器测试 (规则模式)")
    print("=" * 60)

    correct = 0
    for query, expected in test_queries:
        result = classify_intent(query, use_llm=False)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {match} [{result:13s}] {query}")

    print(f"\n规则匹配准确率: {correct}/{len(test_queries)} = {correct/len(test_queries)*100:.0f}%")

    # 测试 LLM 分类（需要 Ollama 运行）
    print(f"\n{'=' * 60}")
    print("语义路由器测试 (LLM 模式)")
    print("=" * 60)

    llm_test_queries = [
        "Kafka 的 ISR 机制是怎么回事",
        "帮我看看这个 OutOfMemoryError 是什么问题",
        "Redis 的 maxmemory-policy 有哪些选项",
    ]

    for query in llm_test_queries:
        try:
            result = classify_intent(query, use_llm=True)
            print(f"  [{result:13s}] {query}")
        except Exception as e:
            print(f"  [LLM不可用  ] {query} -> {e}")

    print("\n测试完成!")