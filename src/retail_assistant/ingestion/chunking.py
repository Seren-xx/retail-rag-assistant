"""文本切分：段落优先 + 标点分句 + 块重叠（算法沿用 legacy/text_splitter.py）。"""

import re

from .. import config
from ..models import Document


class TextChunker:
    """段落优先 + 标点分句 + 块重叠的文本分割器。"""

    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.separators = config.separators

    def split_by_paragraphs(self, text: str) -> list[str]:
        """按段落分割文本。"""
        paragraphs = re.split(r"\n\n+", text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def split_by_sentences(self, text: str) -> list[str]:
        """按标点符号分句，优先中文标点。"""
        pattern = "|".join(re.escape(s) for s in self.separators if len(s) > 0)
        if not pattern:
            return [text]

        sentences = re.split(f"({pattern})", text)
        result = []
        current = ""
        for part in sentences:
            if part in self.separators:
                if current.strip():
                    result.append(current.strip() + part)
                    current = ""
            else:
                current += part
        if current.strip():
            result.append(current.strip())
        return result

    def merge_chunks_with_overlap(self, sentences: list[str]) -> list[str]:
        """将句子合并成块，保持重叠。"""
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(sentence) <= self.chunk_overlap:
                    current_chunk = sentence
                else:
                    overlap_start = max(0, len(sentence) - self.chunk_overlap)
                    current_chunk = sentence[overlap_start:] + sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """段落优先 + 标点分句 + 块重叠策略。"""
        if len(text) <= self.chunk_size:
            return [text]

        all_chunks = []
        for para in self.split_by_paragraphs(text):
            if len(para) <= self.chunk_size:
                all_chunks.append(para)
            else:
                sentences = self.split_by_sentences(para)
                all_chunks.extend(self.merge_chunks_with_overlap(sentences))

        return all_chunks

    def make_documents(
        self,
        text: str,
        doc_id: str,
        source: str,
        extra_metadata: dict | None = None,
    ) -> list[Document]:
        """切分并生成 Document：单块沿用原 doc_id，多块追加 #序号。"""
        chunks = self.split_text(text)
        docs = []
        for i, chunk in enumerate(chunks):
            cid = doc_id if len(chunks) == 1 else f"{doc_id}#{i + 1}"
            metadata = {
                "source": source,
                "kind": "knowledge",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "balance_key": source,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            docs.append(Document(id=cid, page_content=chunk, metadata=metadata))
        return docs
