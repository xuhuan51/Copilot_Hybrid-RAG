"""
多轮对话上下文管理 (Conversation Memory)
负责对话历史维护和 Query 改写/压缩
将依赖上下文的追问改写为独立的检索 Query
"""
import requests


# ────────────────────── Ollama 调用 ──────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"


def call_ollama(prompt: str, model: str = LLM_MODEL) -> str:
    """调用 Ollama 本地 LLM"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 100,
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ────────────────────── 对话历史管理 ──────────────────────

class ConversationMemory:
    """
    管理多轮对话历史

    功能:
    - 维护对话历史 (user/assistant 交替)
    - 判断当前 query 是否需要改写
    - 将追问改写为独立的检索 query
    """

    def __init__(self, max_turns: int = 5):
        """
        参数:
            max_turns: 保留的最大对话轮数（防止上下文过长）
        """
        self.max_turns = max_turns
        self.history = []  # [{"role": "user"/"assistant", "content": "..."}]

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant_message(self, message: str):
        """添加助手回复"""
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def _trim(self):
        """保留最近 max_turns 轮对话"""
        # 每轮 = 1条user + 1条assistant = 2条消息
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history_text(self) -> str:
        """将对话历史格式化为文本"""
        if not self.history:
            return ""

        parts = []
        for msg in self.history:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")
        return "\n".join(parts)

    def clear(self):
        """清空对话历史"""
        self.history = []

    @property
    def turn_count(self) -> int:
        """当前对话轮数"""
        return len([m for m in self.history if m["role"] == "user"])


# ────────────────────── Query 改写 ──────────────────────

REWRITE_PROMPT = """你是一个 Query 改写助手。根据对话历史，将用户的最新问题改写为一个独立的、完整的检索查询。

规则:
- 如果最新问题已经是独立完整的，直接原样返回
- 如果最新问题包含代词（它、这个、那个、上面的）或省略了主语，需要结合上下文补全
- 改写后的 query 应该能脱离对话历史独立理解
- 只返回改写后的 query，不要有任何解释

对话历史:
{history}

用户最新问题: {query}

改写后的检索 query:"""


def rewrite_query(
    query: str,
    memory: ConversationMemory,
    use_llm: bool = True,
) -> str:
    """
    根据对话历史改写 query

    策略:
    - 第一轮对话 / 无历史 → 直接返回原 query
    - 有历史 + query 看起来独立 → 直接返回原 query
    - 有历史 + query 依赖上下文 → 用 LLM 改写

    参数:
        query:    用户最新问题
        memory:   对话历史
        use_llm:  是否启用 LLM 改写
    """
    # 第一轮对话，无需改写
    if memory.turn_count == 0:
        return query

    # 简单规则判断: 是否可能依赖上下文
    if not _needs_rewrite(query):
        return query

    # LLM 改写
    if use_llm:
        return _llm_rewrite(query, memory)

    return query


def _needs_rewrite(query: str) -> bool:
    """
    规则判断 query 是否需要改写

    检测是否包含:
    - 代词: 它、这个、那个、上面、前面
    - 过短且无主语的 query
    """
    # 代词和指代词
    context_indicators = [
        "它", "这个", "那个", "上面", "前面", "刚才",
        "这种", "那种", "其中", "同样", "类似",
        "还有呢", "继续", "接着说", "然后呢",
        "this", "that", "it", "these", "those",
        "the same", "also", "above",
    ]

    q_lower = query.lower()
    for indicator in context_indicators:
        if indicator in q_lower:
            return True

    # 过短的 query 可能省略了主语 (例如 "怎么配置？")
    if len(query) < 8 and "?" in query or "？" in query:
        return True

    return False


def _llm_rewrite(query: str, memory: ConversationMemory) -> str:
    """使用 LLM 改写 query"""
    try:
        history_text = memory.get_history_text()
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        rewritten = call_ollama(prompt)

        # 清理 LLM 输出（去掉可能的引号、前缀等）
        rewritten = rewritten.strip().strip('"').strip("'")

        if rewritten:
            print(f"  [Memory] Query改写: '{query}' → '{rewritten}'")
            return rewritten
        else:
            return query
    except Exception as e:
        print(f"  [Memory] 改写失败: {e}，使用原始 query")
        return query


# ────────────────────── 测试入口 ──────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("多轮对话 Query 改写测试")
    print("=" * 60)

    memory = ConversationMemory(max_turns=5)

    # 模拟多轮对话
    conversations = [
        ("Kafka consumer max.poll.records 默认值是多少", None),
        ("那它的最大值可以设多少", "需要改写"),
        ("这个参数设太大会有什么问题", "需要改写"),
        ("Redis 的持久化方式有哪些", None),
        ("它们的区别是什么", "需要改写"),
    ]

    for i, (query, expect_rewrite) in enumerate(conversations):
        print(f"\n--- 第 {i+1} 轮 ---")
        print(f"原始 query: {query}")

        # 改写
        rewritten = rewrite_query(query, memory, use_llm=True)
        if rewritten != query:
            print(f"改写结果:  {rewritten}")
        else:
            print(f"无需改写")

        # 记录对话历史
        memory.add_user_message(query)
        memory.add_assistant_message(f"[模拟回答第{i+1}轮]")

    print(f"\n当前对话轮数: {memory.turn_count}")
    print(f"\n{'=' * 60}")
    print("测试完成!")