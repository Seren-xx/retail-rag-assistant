"""评测数据集：query + 人工标注 relevant_doc_ids；relevant_ids 为空表示知识库无答案（评门禁拒答）。"""

import json
import os
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    query: str
    relevant_ids: list[str] = field(default_factory=list)
    tag: str = "general"
    note: str = ""

    @property
    def is_no_answer(self) -> bool:
        return not self.relevant_ids


def load_cases(path: str) -> list[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    query=obj["query"],
                    relevant_ids=obj.get("relevant_ids", []),
                    tag=obj.get("tag", "general"),
                    note=obj.get("note", ""),
                )
            )
    return cases
