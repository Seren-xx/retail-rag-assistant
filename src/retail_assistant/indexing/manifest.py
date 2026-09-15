"""增量索引清单：记录知识源 hash 与处理状态，同步时未变跳过、变更重处理、删除下线。"""

import hashlib
import json
import os
from datetime import datetime


class Manifest:
    def __init__(self, path: str):
        self.path = path
        self.entries = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    @staticmethod
    def file_hash(path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def status(self, files: dict) -> dict:
        """对比磁盘文件与清单，返回 {文件名: new|changed|unchanged|deleted}。"""
        result = {}
        for name, path in files.items():
            if name not in self.entries:
                result[name] = "new"
            elif self.entries[name].get("hash") != self.file_hash(path):
                result[name] = "changed"
            else:
                result[name] = "unchanged"
        for name in self.entries:
            if name not in files:
                result[name] = "deleted"
        return result

    def mark(self, name: str, path: str, chunk_count: int, status: str = "indexed"):
        self.entries[name] = {
            "hash": self.file_hash(path),
            "chunk_count": chunk_count,
            "status": status,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save()

    def remove(self, name: str) -> bool:
        if name in self.entries:
            del self.entries[name]
            self._save()
            return True
        return False

    def stats(self) -> dict:
        return {
            "files": len(self.entries),
            "chunks": sum(e.get("chunk_count", 0) for e in self.entries.values()),
        }
