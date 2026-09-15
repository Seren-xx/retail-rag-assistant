"""三级去重：MD5 精确 + SimHash 近似 + 向量语义（算法沿用 legacy/deduplication.py，存储注入式）。"""

import hashlib
import json
import os

import numpy as np

from .. import config


class MD5Deduplicator:
    """MD5 精确去重。"""

    def __init__(self, path):
        self.md5_path = path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.md5_path):
            open(self.md5_path, "w", encoding="utf-8").close()

    @staticmethod
    def get_md5(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, text):
        md5_hex = self.get_md5(text)
        if not os.path.exists(self.md5_path):
            return False
        with open(self.md5_path, "r", encoding="utf-8") as f:
            return md5_hex in f.read()

    def add(self, text):
        with open(self.md5_path, "a", encoding="utf-8") as f:
            f.write(self.get_md5(text) + "\n")


class SimHashDeduplicator:
    """SimHash 近似去重：64 位指纹 + 汉明距离。"""

    def __init__(self, path, threshold=None):
        self.index_path = path
        self.threshold = threshold or config.simhash_threshold
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _tokenize(text):
        # 字符 bigram：单字指纹对短文本区分度不足，会把不同文档误判为近似重复
        tokens = [text[i : i + 2] for i in range(len(text) - 1)]
        return tokens or list(text)

    def _simhash(self, text):
        hash_vector = [0] * 64
        for token in self._tokenize(text):
            token_hash = hashlib.md5(token.encode("utf-8")).hexdigest()
            # 取前 64 位作为指纹（md5 全长 128 位，直接展开会越界）
            binary_hash = bin(int(token_hash[:16], 16))[2:].zfill(64)
            for i, bit in enumerate(binary_hash):
                hash_vector[i] += 1 if bit == "1" else -1

        simhash = 0
        for i in range(64):
            if hash_vector[i] > 0:
                simhash |= 1 << i
        return simhash

    @staticmethod
    def _hamming_distance(hash1, hash2):
        return bin(hash1 ^ hash2).count("1")

    def is_duplicate(self, text):
        new_hash = self._simhash(text)
        return any(
            self._hamming_distance(new_hash, stored) <= self.threshold
            for stored in self.index.values()
        )

    def add(self, text, chunk_id):
        self.index[chunk_id] = self._simhash(text)
        self._save_index()


class VectorSemanticDeduplicator:
    """向量语义去重：余弦相似度阈值。需要传入 embedding 结果。"""

    def __init__(self, path, threshold=None):
        self.index_path = path
        self.threshold = threshold or config.vector_simhash_threshold
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        v1, v2 = np.array(vec1), np.array(vec2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm)

    def is_duplicate(self, embedding):
        return any(
            self._cosine_similarity(embedding, stored) >= self.threshold
            for stored in self.index.values()
        )

    def add(self, chunk_id, embedding):
        self.index[chunk_id] = [float(x) for x in embedding]
        self._save_index()


class DeduplicationService:
    """三级去重服务：MD5 → SimHash → 向量语义（可选）。"""

    def __init__(self, index_dir, enable_vector_dedup=None):
        os.makedirs(index_dir, exist_ok=True)
        self.md5_dedup = MD5Deduplicator(os.path.join(index_dir, config.md5_path))
        self.simhash_dedup = SimHashDeduplicator(os.path.join(index_dir, config.simhash_path))
        use_vector = config.semantic_dedup if enable_vector_dedup is None else enable_vector_dedup
        self.vector_dedup = (
            VectorSemanticDeduplicator(os.path.join(index_dir, config.vector_simhash_path))
            if use_vector
            else None
        )

    def is_duplicate(self, text, embedding=None):
        if self.md5_dedup.is_duplicate(text):
            return True
        if self.simhash_dedup.is_duplicate(text):
            return True
        if embedding is not None and self.vector_dedup is not None:
            return self.vector_dedup.is_duplicate(embedding)
        return False

    def add(self, text, chunk_id, embedding=None):
        self.md5_dedup.add(text)
        self.simhash_dedup.add(text, chunk_id)
        if embedding is not None and self.vector_dedup is not None:
            self.vector_dedup.add(chunk_id, embedding)
