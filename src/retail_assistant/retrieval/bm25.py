"""BM25 稀疏检索：中文按字、英文按词，k1=1.5、b=0.75。"""

import re
from collections import Counter

from .. import config


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_terms = []
        self.doc_freq = {}
        self.avg_doc_len = 0
        self.is_built = False

    @staticmethod
    def tokenize(text):
        """简单分词：中文按字，英文按词。"""
        english_words = re.findall(r"[a-zA-Z]+", text.lower())
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return english_words + chinese_chars

    def build(self, documents):
        self.documents = list(documents)
        self.doc_terms = [Counter(self.tokenize(d.page_content)) for d in self.documents]
        self.doc_freq = {}
        for terms in self.doc_terms:
            for term in terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        self.avg_doc_len = (
            sum(sum(c.values()) for c in self.doc_terms) / max(len(self.documents), 1)
        )
        self.is_built = True

    def _score(self, query_terms, doc_idx):
        terms = self.doc_terms[doc_idx]
        doc_len = sum(terms.values())
        score = 0.0
        n_docs = len(self.documents)

        for term in query_terms:
            tf = terms.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = max(0.0, (n_docs - df + 0.5) / (df + 0.5))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-9)
            )
            score += idf * numerator / denominator
        return score

    def search(self, query, top_k=None):
        """BM25 检索：top_k 缺省取 config.bm25_top_k，分数为 0 的文档不返回。"""
        if not self.is_built:
            return []
        top_k = top_k or config.bm25_top_k
        query_terms = self.tokenize(query)
        scored = [(d, self._score(query_terms, i)) for i, d in enumerate(self.documents)]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
