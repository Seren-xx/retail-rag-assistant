"""向量存储服务：HuggingFaceEmbeddings(BGE-M3) + langchain_chroma 持久化检索。"""

import os

from .. import config


def make_embedder():
    """BGE-M3 嵌入（langchain_huggingface 封装，输出归一化向量）。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    """Chroma 向量库（cosine 空间）：维度或文档集合变化时整库重建。"""

    def __init__(self, embedder, persist_dir):
        self.embedder = embedder
        self.persist_dir = persist_dir
        self.documents = []
        self._store = None

    def _open(self):
        """打开（或创建）langchain_chroma 向量库。"""
        if self._store is None:
            import chromadb
            from langchain_chroma import Chroma

            os.makedirs(self.persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                client.get_collection(config.collection_name)
            except Exception:
                client.create_collection(
                    config.collection_name, metadata={"hnsw:space": "cosine"}
                )
            self._store = Chroma(
                client=client,
                collection_name=config.collection_name,
                embedding_function=self.embedder,
            )
        return self._store

    def _probe_dim(self):
        return len(self.embedder.embed_query("维度探测"))

    def build(self, documents):
        """整库重建：清空 collection 后按当前 embedder 重新入库。"""
        store = self._open()
        store.delete_collection()
        self._store = None

        import chromadb
        from langchain_chroma import Chroma

        client = chromadb.PersistentClient(path=self.persist_dir)
        client.create_collection(
            config.collection_name,
            metadata={"hnsw:space": "cosine", "dim": self._probe_dim()},
        )
        self._store = Chroma(
            client=client, collection_name=config.collection_name, embedding_function=self.embedder
        )
        self.documents = list(documents)
        if documents:
            self._store.add_texts(
                texts=[d.page_content for d in documents],
                metadatas=[{**d.metadata, "doc_id": d.id} for d in documents],
                ids=[d.id for d in documents],
            )

    def search(self, query, top_k=None):
        """cosine 距离转相似度分数（1 - distance），top_k 缺省取 config.dense_top_k。"""
        if not self.documents:
            return []
        store = self._open()
        if store._collection.count() == 0:
            return []
        k = min(top_k or config.dense_top_k, len(self.documents))
        res = store.similarity_search_with_score(query, k=k)
        id2doc = {d.id: d for d in self.documents}
        out = []
        for doc, dist in res:
            # Chroma 文档 id 即 doc_id（add_texts 时 ids=d.id），metadata 仅作兜底
            canonical = id2doc.get(doc.id or doc.metadata.get("doc_id"))
            if canonical is not None:
                out.append((canonical, 1.0 - float(dist)))
        return out

    def load_or_build(self, index_dir, documents):
        """collection 维度与文档集合都对齐时直接复用，否则整库重建。"""
        store = self._open()
        meta = store._collection.metadata or {}
        doc_ids = {d.id for d in documents}
        aligned = (
            meta.get("dim") == self._probe_dim()
            and set(store.get(include=[])["ids"]) == doc_ids
            and store._collection.count() == len(doc_ids)
        )
        if aligned:
            self.documents = list(documents)
            return self
        self.build(documents)
        return self


def make_dense_index(embedder=None, index_dir=None):
    """向量库：ChromaDB 持久化（落盘 index_dir/chroma/ 子目录，缺省取 config）。"""
    return VectorStore(
        embedder or make_embedder(),
        os.path.join(index_dir or config.index_dir, config.chroma_dir),
    )
