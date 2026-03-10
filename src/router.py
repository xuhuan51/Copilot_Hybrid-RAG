"""
语义路由器 (Semantic Router)
根据用户问题意图自动路由到不同的 Prompt 模板
支持三种主意图:
- factual: 事实查询
- conceptual: 原理解释
- troubleshoot: 故障排查

优化点:
1. 规则匹配升级为“打分制”，避免先命中先返回造成误判
2. 低置信度样本再调用 LLM，减少不必要开销
3. 返回 richer metadata: intent / confidence / source / scores
4. LLM prompt 增加边界约束，输出更稳定
5. context 拼接更稳，不在句中粗暴截断
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import requests
from dataclasses import dataclass
from typing import Optional


# ────────────────────── Ollama 调用 ──────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
ROUTER_MODEL = "qwen2.5:14b"


def call_ollama(prompt: str, model: str = ROUTER_MODEL, timeout: int = 30) -> str:
    """调用本地 Ollama 模型"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 32,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


# ────────────────────── 意图定义 ──────────────────────

INTENT_FACTUAL = "factual"
INTENT_CONCEPTUAL = "conceptual"
INTENT_TROUBLESHOOT = "troubleshoot"

VALID_INTENTS = {
    INTENT_FACTUAL,
    INTENT_CONCEPTUAL,
    INTENT_TROUBLESHOOT,
}


@dataclass
class RouteResult:
    intent: str
    confidence: float
    source: str                 # rule / llm / fallback
    scores: dict[str, float]
    matched_rules: list[str]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "source": self.source,
            "scores": self.scores,
            "matched_rules": self.matched_rules,
        }


# ────────────────────── 分类 Prompt ──────────────────────

CLASSIFICATION_PROMPT = """你是企业技术文档问答系统的意图分类器。
请把用户问题严格分类为以下三类之一，只输出一个单词，不要解释：

1. factual
- 查询具体事实、参数、默认值、配置项、API、命令、语法、字段说明
- 常见问法：默认值是多少、有哪些参数、怎么配置某个字段、命令怎么写

2. conceptual
- 查询原理、机制、架构、概念解释、设计思想、差异对比、工作流程
- 常见问法：为什么、怎么工作的、原理是什么、A 和 B 有什么区别

3. troubleshoot
- 查询报错、异常、性能问题、故障定位、排查方法、修复建议
- 常见问法：报错怎么办、为什么失败、怎么排查、怎么解决、超时/崩溃/异常

分类原则：
- 只看“用户主要诉求”
- 如果是在问报错原因、排查步骤、修复方式，优先归为 troubleshoot
- 如果是在问具体配置值、参数名、命令格式，归为 factual
- 如果是在问机制、区别、原理，归为 conceptual

示例：
Q: Kafka consumer max.poll.records 默认值是多少
A: factual

Q: Redis RDB 和 AOF 的区别是什么
A: conceptual

Q: Kafka consumer 一直 rebalance 怎么排查
A: troubleshoot

Q: OutOfMemoryError 怎么解决
A: troubleshoot

Q: Kafka ISR 机制是怎么工作的
A: conceptual

Q: max.poll.interval.ms 怎么配置
A: factual

用户问题:
{query}

请只输出:
factual 或 conceptual 或 troubleshoot
"""


# ────────────────────── 规则模式 ──────────────────────

RULE_PATTERNS: dict[str, list[tuple[str, float, str]]] = {
    INTENT_TROUBLESHOOT: [
        (r"\b(error|exception|timeout|timed out|fail(ed)?|crash|debug|fix)\b", 3.0, "en_strong_troubleshoot"),
        (r"排查|报错|错误|异常|超时|失败|挂了|宕机|崩溃|告警|修复|处理|解决", 3.0, "zh_strong_troubleshoot"),
        (r"怎么解决|怎么处理|怎么修复|怎么排查|什么问题|啥问题", 2.5, "zh_action_troubleshoot"),
        (r"rebalance|oom|outofmemory|connection refused|refused|503|502|504", 2.0, "error_token_troubleshoot"),
        (r"慢|卡住|卡顿|不稳定|连不上|不可用", 1.5, "weak_troubleshoot"),
    ],
    INTENT_CONCEPTUAL: [
        (r"原理|机制|架构|本质|设计|思想|底层|流程", 2.8, "zh_conceptual"),
        (r"区别|对比|比较|异同", 2.5, "zh_compare"),
        (r"为什么|为啥|怎么工作|如何工作|怎么实现", 2.3, "zh_why_how"),
        (r"\b(what is|how does .* work|difference|compare|vs\.?|versus|explain|architecture|mechanism)\b", 2.8, "en_conceptual"),
    ],
    INTENT_FACTUAL: [
        (r"默认值|取值|参数|配置|配置项|命令|语法|接口|字段|选项|列表", 2.8, "zh_factual"),
        (r"多少|几个|哪些|是什么值", 2.0, "zh_fact_question"),
        (r"\b(default|parameter|config|configuration|value|api|syntax|command|flag|option)\b", 2.8, "en_factual"),
        (r"怎么配置|如何配置|怎么设置|如何设置", 2.2, "zh_config_factual"),
        (r"\b(how to configure|how to set)\b", 2.2, "en_config_factual"),
    ],
}

# 一些明显的日志/报错特征，加额外权重
ERROR_HINT_PATTERNS = [
    r"[A-Za-z_]*Exception\b",
    r"[A-Za-z_]*Error\b",
    r"Traceback",
    r"stack trace",
    r"Caused by:",
]


def _normalize_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def _boost_scores_by_shape(query: str, scores: dict[str, float], matched_rules: list[str]) -> None:
    """
    根据问句形态做轻量加权，仍然保持三分类。
    """
    q = query

    # 明显日志 / 报错栈
    for pattern in ERROR_HINT_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            scores[INTENT_TROUBLESHOOT] += 3.0
            matched_rules.append(f"shape:{pattern}")
            break

    # 以“为什么/区别/原理”开头，更偏 conceptual
    if re.search(r"^(为什么|为啥|原理|机制|区别|对比)", q):
        scores[INTENT_CONCEPTUAL] += 1.2
        matched_rules.append("shape:conceptual_prefix")

    # 明确问默认值/参数/API，更偏 factual
    if re.search(r"(默认值|参数|配置项|api|命令|语法|字段)", q, flags=re.IGNORECASE):
        scores[INTENT_FACTUAL] += 1.0
        matched_rules.append("shape:factual_entity")

    # 带“怎么解决/排查/修复”，更偏 troubleshoot
    if re.search(r"(怎么解决|怎么排查|怎么修复|报错|异常|失败)", q):
        scores[INTENT_TROUBLESHOOT] += 1.2
        matched_rules.append("shape:troubleshoot_action")


def _score_rule_based(query: str) -> tuple[dict[str, float], list[str]]:
    q = _normalize_query(query)
    scores = {
        INTENT_FACTUAL: 0.0,
        INTENT_CONCEPTUAL: 0.0,
        INTENT_TROUBLESHOOT: 0.0,
    }
    matched_rules: list[str] = []

    for intent, rules in RULE_PATTERNS.items():
        for pattern, weight, rule_name in rules:
            if re.search(pattern, q, flags=re.IGNORECASE):
                scores[intent] += weight
                matched_rules.append(f"{intent}:{rule_name}")

    _boost_scores_by_shape(query, scores, matched_rules)
    return scores, matched_rules


def _pick_intent_from_scores(scores: dict[str, float]) -> tuple[str, float, float]:
    """
    返回:
        best_intent, best_score, gap_to_second
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_intent, best_score = ranked[0]
    second_score = ranked[1][1]
    gap = best_score - second_score
    return best_intent, best_score, gap


def _score_to_confidence(best_score: float, gap: float) -> float:
    """
    一个简单可解释的 confidence 映射，不追求概率意义，只用于工程阈值。
    """
    conf = 0.35 + min(best_score / 8.0, 0.4) + min(gap / 4.0, 0.25)
    return round(min(conf, 0.98), 4)


def _rule_based_classify(query: str) -> RouteResult:
    scores, matched_rules = _score_rule_based(query)
    best_intent, best_score, gap = _pick_intent_from_scores(scores)
    confidence = _score_to_confidence(best_score, gap)

    # 完全没命中时，置信度给低一点
    if best_score == 0:
        confidence = 0.20

    return RouteResult(
        intent=best_intent,
        confidence=confidence,
        source="rule",
        scores=scores,
        matched_rules=matched_rules,
    )


def _extract_llm_intent(text: str) -> Optional[str]:
    text = text.strip().lower()
    for intent in VALID_INTENTS:
        if text == intent:
            return intent

    # 容错提取
    if "troubleshoot" in text:
        return INTENT_TROUBLESHOOT
    if "conceptual" in text:
        return INTENT_CONCEPTUAL
    if "factual" in text:
        return INTENT_FACTUAL
    return None


def _fallback_by_heuristic(query: str) -> str:
    """
    当 LLM 不可用或返回异常时，用更稳的轻量启发式兜底，
    不再一律 factual。
    """
    q = _normalize_query(query)
    if re.search(r"(报错|错误|异常|失败|超时|排查|解决|修复|error|exception|timeout|fail)", q, re.IGNORECASE):
        return INTENT_TROUBLESHOOT
    if re.search(r"(原理|机制|架构|区别|对比|为什么|what is|difference|compare|how does)", q, re.IGNORECASE):
        return INTENT_CONCEPTUAL
    return INTENT_FACTUAL


def _llm_classify(query: str) -> RouteResult:
    try:
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        result = call_ollama(prompt)
        intent = _extract_llm_intent(result)

        if intent is None:
            fallback_intent = _fallback_by_heuristic(query)
            return RouteResult(
                intent=fallback_intent,
                confidence=0.42,
                source="fallback",
                scores={
                    INTENT_FACTUAL: 0.0,
                    INTENT_CONCEPTUAL: 0.0,
                    INTENT_TROUBLESHOOT: 0.0,
                },
                matched_rules=[f"llm_unknown:{result}"],
            )

        return RouteResult(
            intent=intent,
            confidence=0.78,
            source="llm",
            scores={
                INTENT_FACTUAL: 0.0,
                INTENT_CONCEPTUAL: 0.0,
                INTENT_TROUBLESHOOT: 0.0,
            },
            matched_rules=[f"llm:{intent}"],
        )

    except Exception as e:
        fallback_intent = _fallback_by_heuristic(query)
        return RouteResult(
            intent=fallback_intent,
            confidence=0.38,
            source="fallback",
            scores={
                INTENT_FACTUAL: 0.0,
                INTENT_CONCEPTUAL: 0.0,
                INTENT_TROUBLESHOOT: 0.0,
            },
            matched_rules=[f"llm_error:{type(e).__name__}"],
        )


def route_query(
    query: str,
    use_llm: bool = True,
    rule_conf_threshold: float = 0.72,
) -> RouteResult:
    """
    先规则打分。
    当规则置信度足够高时，直接返回。
    否则再用 LLM 裁决。
    """
    rule_result = _rule_based_classify(query)

    if not use_llm:
        return rule_result

    if rule_result.confidence >= rule_conf_threshold:
        return rule_result

    llm_result = _llm_classify(query)
    return llm_result


def classify_intent(query: str, use_llm: bool = True) -> str:
    """
    向下兼容旧接口：只返回 intent 字符串
    """
    return route_query(query, use_llm=use_llm).intent


# ────────────────────── Prompt 模板 ──────────────────────

PROMPT_TEMPLATES = {
    INTENT_FACTUAL: """你是一个企业技术文档助手。请根据以下参考资料，准确回答用户的问题。

要求:
- 优先直接回答具体结论，不要先大段铺垫
- 若问题涉及参数、配置项、默认值、字段、命令、API，请明确写出对应值或写法
- 如果资料中存在示例配置、命令或代码，优先提取最相关部分
- 若资料不足以确认，请明确说明“参考资料中未给出”
- 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {query}

请回答:""",

    INTENT_CONCEPTUAL: """你是一个企业技术文档助手。请根据以下参考资料，解释用户询问的技术原理。

要求:
- 先给出 1~2 句概念总结
- 再分点解释工作机制、设计动机或关键流程
- 如果涉及多个概念，请说明它们的联系与区别
- 优先基于参考资料回答，不要脱离资料自行扩写
- 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {query}

请回答:""",

    INTENT_TROUBLESHOOT: """你是一个企业技术文档助手。请根据以下参考资料，帮助用户排查和解决问题。

要求:
- 先概括最可能的原因
- 再给出排查步骤，按优先级排序
- 如果资料中有相关命令、配置建议、日志定位方法，请直接给出
- 若问题与某个参数/组件强相关，请明确指出检查项
- 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {query}

请回答:""",
}


def get_prompt(
    query: str,
    context: str,
    intent: Optional[str] = None,
    use_llm: bool = True,
) -> tuple[str, str]:
    """
    兼容旧接口:
    返回 (prompt, intent)
    """
    if intent is None:
        route_result = route_query(query, use_llm=use_llm)
        intent = route_result.intent

    template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES[INTENT_FACTUAL])
    prompt = template.format(context=context, query=query)
    return prompt, intent


def get_prompt_with_route(
    query: str,
    context: str,
    use_llm: bool = True,
) -> tuple[str, RouteResult]:
    """
    新接口:
    返回 (prompt, route_result)
    """
    route_result = route_query(query, use_llm=use_llm)
    template = PROMPT_TEMPLATES.get(route_result.intent, PROMPT_TEMPLATES[INTENT_FACTUAL])
    prompt = template.format(context=context, query=query)
    return prompt, route_result


# ────────────────────── 上下文拼接 ──────────────────────

def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _truncate_text_naturally(text: str, max_len: int) -> str:
    """
    尽量在自然边界截断，避免直接切半句。
    """
    if len(text) <= max_len:
        return text

    candidate = text[:max_len]

    # 优先在句号、换行、分号等处截断
    split_points = [
        candidate.rfind("\n\n"),
        candidate.rfind("\n"),
        candidate.rfind("。"),
        candidate.rfind(". "),
        candidate.rfind("; "),
        candidate.rfind("；"),
    ]
    cut = max(split_points)

    if cut < max_len * 0.6:
        cut = max_len

    return candidate[:cut].rstrip() + "\n...(截断)"


def build_context(docs: list[dict], max_length: int = 4000) -> str:
    """
    将文档列表拼接为上下文。
    相比旧版改进:
    - 清洗空文本
    - 不在句中粗暴截断
    - image_path 没有就不硬塞“图片: 无”
    """
    context_parts: list[str] = []
    current_length = 0

    for i, doc in enumerate(docs):
        source_file = _clean_text(doc.get("source_file", ""))
        header_path = _clean_text(doc.get("header_path", ""))
        content = _clean_text(doc.get("content", ""))
        image_path = _clean_text(doc.get("image_path", ""))
        image_caption = _clean_text(doc.get("image_caption", ""))

        meta_lines = [f"[资料{i+1}]"]
        if source_file:
            meta_lines.append(f"来源: {source_file}")
        if header_path:
            meta_lines.append(f"章节: {header_path}")
        if image_caption:
            meta_lines.append(f"图示说明: {image_caption}")
        elif image_path:
            meta_lines.append(f"图片: {image_path}")

        meta_lines.append(f"内容: {content}")
        part = "\n".join(meta_lines)

        if current_length + len(part) <= max_length:
            context_parts.append(part)
            current_length += len(part) + 5
            continue

        remaining = max_length - current_length
        if remaining < 120:
            break

        truncated = _truncate_text_naturally(part, remaining)
        context_parts.append(truncated)
        break

    return "\n---\n".join(context_parts)


# ────────────────────── 可选：给检索层的路由参数 ──────────────────────

ROUTE_RETRIEVAL_CONFIG = {
    INTENT_FACTUAL: {
        "dense_topk": 8,
        "sparse_topk": 12,
        "rerank_topk": 6,
        "context_max_length": 3200,
    },
    INTENT_CONCEPTUAL: {
        "dense_topk": 12,
        "sparse_topk": 8,
        "rerank_topk": 8,
        "context_max_length": 4200,
    },
    INTENT_TROUBLESHOOT: {
        "dense_topk": 8,
        "sparse_topk": 16,
        "rerank_topk": 8,
        "context_max_length": 3800,
    },
}


def get_retrieval_config(intent: str) -> dict:
    return ROUTE_RETRIEVAL_CONFIG.get(intent, ROUTE_RETRIEVAL_CONFIG[INTENT_FACTUAL])


# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    test_queries = [
        ("Kafka consumer max.poll.records 默认值是多少", INTENT_FACTUAL),
        ("Redis RDB 和 AOF 持久化的区别", INTENT_CONCEPTUAL),
        ("Kafka consumer 一直 rebalance 怎么排查", INTENT_TROUBLESHOOT),
        ("如何配置 Kafka producer 的 batch.size", INTENT_FACTUAL),
        ("为什么 Kafka 要用零拷贝技术", INTENT_CONCEPTUAL),
        ("Redis 连接超时报错怎么解决", INTENT_TROUBLESHOOT),
        ("What is the default replication factor", INTENT_FACTUAL),
        ("How does Kafka consumer group rebalancing work", INTENT_CONCEPTUAL),
        ("java.lang.OutOfMemoryError: Java heap space", INTENT_TROUBLESHOOT),
        ("Kafka offset 提交失败是什么问题", INTENT_TROUBLESHOOT),
        ("ISR 机制原理", INTENT_CONCEPTUAL),
    ]

    print("=" * 72)
    print("语义路由器测试 (规则优先 + 三分类)")
    print("=" * 72)

    correct = 0
    for query, expected in test_queries:
        result = route_query(query, use_llm=False)
        ok = result.intent == expected
        if ok:
            correct += 1
        mark = "✓" if ok else "✗"
        print(
            f"{mark} "
            f"[intent={result.intent:<13}] "
            f"[conf={result.confidence:.2f}] "
            f"[source={result.source:<8}] "
            f"{query}"
        )

    print(f"\n准确率: {correct}/{len(test_queries)} = {correct/len(test_queries)*100:.1f}%")

    print(f"\n{'=' * 72}")
    print("低置信度样本 LLM 模式测试")
    print("=" * 72)

    llm_test_queries = [
        "Kafka 的 ISR 机制是怎么回事",
        "帮我看看这个 OutOfMemoryError 是什么问题",
        "Redis 的 maxmemory-policy 有哪些选项",
        "Kafka consumer lag 高怎么分析",
    ]

    for query in llm_test_queries:
        result = route_query(query, use_llm=True)
        print(
            f"[intent={result.intent:<13}] "
            f"[conf={result.confidence:.2f}] "
            f"[source={result.source:<8}] "
            f"{query}"
        )

    print("\n测试完成!")