"""Manifest：文件级增量检测（new / changed / unchanged / deleted）。"""

from retail_assistant.indexing.manifest import Manifest


def test_status_lifecycle(tmp_path):
    src = tmp_path / "a.md"
    src.write_text("v1", encoding="utf-8")
    manifest = Manifest(str(tmp_path / "manifest.json"))

    assert manifest.status({"a.md": str(src)}) == {"a.md": "new"}
    manifest.mark("a.md", str(src), chunk_count=1)
    assert manifest.status({"a.md": str(src)}) == {"a.md": "unchanged"}

    src.write_text("v2", encoding="utf-8")
    assert manifest.status({"a.md": str(src)}) == {"a.md": "changed"}

    manifest.mark("a.md", str(src), chunk_count=2)
    assert manifest.status({}) == {"a.md": "deleted"}
    assert manifest.remove("a.md") is True

    entries = manifest.entries
    assert "a.md" not in entries


def test_mark_records_metadata(tmp_path):
    src = tmp_path / "b.md"
    src.write_text("内容", encoding="utf-8")
    manifest = Manifest(str(tmp_path / "manifest.json"))
    manifest.mark("b.md", str(src), chunk_count=3)
    entry = manifest.entries["b.md"]
    assert entry["chunk_count"] == 3
    assert entry["status"] == "indexed"
    assert entry["hash"]
    assert manifest.stats() == {"files": 1, "chunks": 3}
