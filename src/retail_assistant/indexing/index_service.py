"""索引服务：知识源 → 加载 → 切分 → 去重 → 文档库持久化（docstore.jsonl）。"""

import json
import os
from datetime import datetime
from pathlib import Path

from .. import config
from ..ingestion import loaders
from ..ingestion.chunking import TextChunker
from ..models import Document, doc_from_dict, doc_to_dict
from .dedup import DeduplicationService
from .manifest import Manifest


class IndexService:
    def __init__(self, settings):
        self.s = settings
        os.makedirs(settings.index_dir, exist_ok=True)
        self.chunker = TextChunker()
        self.dedup = DeduplicationService(settings.index_dir)
        self.manifest = Manifest(os.path.join(settings.index_dir, "manifest.json"))
        self._embedder = None

    def _embed_for_dedup(self, text):
        """语义去重开启时懒加载 Embedder 并返回单条向量；未开启返回 None。"""
        if self.dedup.vector_dedup is None:
            return None
        if self._embedder is None:
            from ..retrieval.dense import make_embedder

            self._embedder = make_embedder()
        return self._embedder.embed_query(text)

    @property
    def docstore_path(self):
        return os.path.join(self.s.index_dir, config.docstore_file)

    def discover_sources(self, data_dir=None):
        """扫描数据目录，返回 {相对路径: 绝对路径}。"""
        data_dir = data_dir or self.s.data_dir
        files = {}
        for root, _dirs, names in os.walk(data_dir):
            for name in sorted(names):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, data_dir).replace("\\", "/")
                if os.path.splitext(name)[1].lower() in loaders.SUPPORTED_SUFFIXES:
                    files[rel] = path
        return files

    def load_source(self, rel_path, data_dir=None):
        """加载单个知识源：csv/jsonl/json 走结构化 loader，其余格式走切分。"""
        data_dir = data_dir or self.s.data_dir
        path = os.path.join(data_dir, rel_path)
        suffix = os.path.splitext(rel_path)[1].lower()

        if suffix == ".csv":
            docs = loaders.load_csv(Path(path), id_column="product_id")
        elif suffix == ".jsonl":
            docs = loaders.load_jsonl(Path(path))
        elif suffix in (".txt", ".md", ".pdf", ".docx", ".xlsx"):
            docs = []
            for raw in loaders.load_any(path):
                docs.extend(self.chunker.make_documents(raw.page_content, raw.id, rel_path))
        else:
            docs = loaders.load_any(path)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for d in docs:
            d.metadata.setdefault("source", rel_path)
            d.metadata["updated_at"] = now
        return docs

    def sync(self, data_dir=None):
        """增量同步：manifest 判文件变化，dedup 判内容重复。"""
        data_dir = data_dir or self.s.data_dir
        files = self.discover_sources(data_dir)
        statuses = self.manifest.status(files)

        # 1. 载入现有文档库
        store = {}
        if os.path.exists(self.docstore_path):
            with open(self.docstore_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = doc_from_dict(json.loads(line))
                        store[d.id] = d

        inserted = updated = unchanged_docs = deactivated = skipped = 0

        # 2. 下线已删除文件对应的文档
        for name in [n for n, st in statuses.items() if st == "deleted"]:
            for doc_id in [i for i, d in store.items() if d.metadata.get("source") == name]:
                del store[doc_id]
                deactivated += 1
            self.manifest.remove(name)

        # 3. 处理新增 / 变化的文件；未变化的直接保留
        for name in sorted(files):
            if statuses[name] == "unchanged":
                unchanged_docs += sum(
                    1 for d in store.values() if d.metadata.get("source") == name
                )
                continue
            docs = self.load_source(name, data_dir)
            for d in docs:
                emb = self._embed_for_dedup(d.page_content)
                if self.dedup.is_duplicate(d.page_content, embedding=emb):
                    skipped += 1
                    continue
                if d.id in store:
                    updated += 1
                else:
                    inserted += 1
                store[d.id] = d
                self.dedup.add(d.page_content, d.id, embedding=emb)
            self.manifest.mark(name, files[name], len(docs))

        # 4. 持久化
        with open(self.docstore_path, "w", encoding="utf-8") as f:
            for doc_id in sorted(store):
                f.write(json.dumps(doc_to_dict(store[doc_id]), ensure_ascii=False) + "\n")

        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged_docs,
            "deactivated": deactivated,
            "skipped_duplicate": skipped,
            "active": len(store),
            "files": statuses,
        }

    def load_docs(self):
        """读取文档库（检索的唯一数据源）。"""
        docs = []
        if os.path.exists(self.docstore_path):
            with open(self.docstore_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        docs.append(doc_from_dict(json.loads(line)))
        return docs
