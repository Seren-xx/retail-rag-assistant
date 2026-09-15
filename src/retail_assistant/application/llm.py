"""生成与查询理解：Qwen3-Max（DashScope 兼容模式），LCEL Chain 编排。"""

import json
import re

SYSTEM_PROMPT = (
    "你是超市智能客服助手。回答规则：\n"
    "1. 只能依据【知识资料】中的内容回答，不得使用资料之外的知识，不得编造。\n"
    "2. 资料中没有答案时，直接说明知识库暂无相关资料，建议用户联系人工客服。\n"
    "3. 涉及价格、库存、活动时间等数字，必须与资料完全一致，不得改写或估算。\n"
    "4. 回答末尾另起一行列出引用的资料编号，例如：参考资料：[kb:xxx] [product:S001]。"
)

MAX_HISTORY_MESSAGES = 6
EVIDENCE_FALLBACK_PREFIX = "根据知识库资料：\n\n"


def make_llm(settings):
    """返回 (query, context, history) -> str 的生成 Chain；未配置 Key 时返回 None。"""
    api_key = settings.dashscope_api_key or None
    if not api_key:
        return None
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    chat = ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=30,
        max_retries=1,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("user", "【知识资料】\n{context}\n\n【用户问题】{query}"),
    ])
    chain = prompt | chat | StrOutputParser()

    def generate(query: str, context: str, history: list[dict] | None = None) -> str:
        messages = []
        for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
            role = "assistant" if m.get("role") == "assistant" else "user"
            content = str(m.get("content", "")).strip()
            if content:
                messages.append((role, content))
        try:
            return chain.invoke({"query": query, "context": context, "history": messages})
        except Exception:
            # 生成层故障不拖垮问答：降级为证据直出
            return EVIDENCE_FALLBACK_PREFIX + context

    return generate


FILTER_SYSTEM_PROMPT = (
    "你是超市客服的查询理解模块。从用户查询中提取结构化检索约束，只输出一个 JSON 对象，"
    "不要输出任何其他内容。格式：\n"
    '{"brands": ["品牌名"], "category": "品类或null", "price_min": 数字或null, '
    '"price_max": 数字或null, "months": [数字]}\n'
    "category 只能是以下之一：乳制品、饮料、粮油、零食、日用清洁、个护、肉制品、生鲜、速冻。\n"
    "months 为查询中提到的月份（1-12）。查询中没有的信息填 null 或空数组，不要猜测。"
)


def _parse_filter_json(text: str) -> dict | None:
    """从模型输出中提取 JSON 对象（容忍 ```json 围栏与前后说明文字）。"""
    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def make_filter_extractor(settings):
    """返回 query -> dict | None 的约束抽取 Chain；未配置 Key 时返回 None（纯规则）。"""
    api_key = settings.dashscope_api_key or None
    if not (settings.llm_filters and api_key):
        return None
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    chat = ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=15,
        max_retries=1,
    )
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", FILTER_SYSTEM_PROMPT),
            ("user", "{query}"),
        ])
        | chat
        | StrOutputParser()
    )

    def extract(query: str) -> dict | None:
        try:
            return _parse_filter_json(chain.invoke({"query": query}))
        except Exception:
            return None

    return extract
