"""上下文组装：doc_id 去重 → 来源限额 → 字符预算打包。"""

from .. import config


def build_context(ranked_hits, max_chars=None, max_per_source=None):
    """返回 (context_text, used_docs)；缺省取 config.max_context_chars / config.max_per_source。"""
    max_chars = max_chars or config.max_context_chars
    max_per_source = max_per_source or config.max_per_source
    seen = set()
    per_source = {}
    used = []
    parts = []
    total = 0

    for doc, _score in ranked_hits:
        if doc.id in seen:
            continue
        seen.add(doc.id)

        key = str(doc.metadata.get("balance_key") or doc.metadata.get("source") or doc.id)
        if per_source.get(key, 0) >= max_per_source:
            continue

        block = f"[{doc.id}] {doc.page_content}"
        if total + len(block) > max_chars:
            continue  # 放不下的跳过，尝试后面更短的候选

        parts.append(block)
        used.append(doc)
        per_source[key] = per_source.get(key, 0) + 1
        total += len(block)

    return "\n\n".join(parts), used
