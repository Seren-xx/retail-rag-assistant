"""测试专用替身：确定性 Embedder + Noop 重排 + 内存版 Redis。

仅用于单测（替代真实 BGE-M3 / CrossEncoder / Redis，保证测试快且无外部依赖）。
生产代码不引用本模块。
"""

import hashlib
import json
import re

import numpy as np


class HashingEmbedder:
    """无外部依赖的确定性 Embedder（LangChain Embeddings 接口）：token 哈希定维。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        english_words = re.findall(r"[a-zA-Z]+", text.lower())
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return english_words + chinese_chars

    def _hash_vec(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vec(text)


class NoopReranker:
    """不重排，直接截断。"""

    def rerank(self, query: str, hits: list, top_k: int | None = None):
        return hits[: top_k or len(hits)]


class FakeRedis:
    """内存版 Redis 客户端：仅实现 RedisSessionStore 用到的命令与语义。"""

    def __init__(self):
        self.data = {}
        self.ttls = {}

    def ping(self):
        return True

    def rpush(self, key, *vals):
        self.data.setdefault(key, []).extend(vals)

    def ltrim(self, key, start, end):
        lst = self.data.get(key, [])
        self.data[key] = lst[start:] if end == -1 else lst[start:end + 1]

    def lrange(self, key, start, end):
        lst = self.data.get(key, [])
        return lst[start:] if end == -1 else lst[start:end + 1]

    def expire(self, key, seconds):
        self.ttls[key] = seconds

    def delete(self, key):
        self.data.pop(key, None)

    @staticmethod
    def _json(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)
