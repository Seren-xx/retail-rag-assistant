"""IndexService：同步幂等、变更检测、删除下线。"""

from retail_assistant.indexing.index_service import IndexService


def test_first_sync_builds_docstore(settings):
    service = IndexService(settings)
    stats = service.sync()
    # 真实语料：24 商品 + 6 促销 + 8 知识文档 = 38
    assert stats["active"] == 38
    assert stats["inserted"] == 38


def test_second_sync_is_idempotent(settings):
    service = IndexService(settings)
    service.sync()
    stats = service.sync()
    assert stats["inserted"] == 0
    assert stats["updated"] == 0
    assert stats["deactivated"] == 0
    assert stats["active"] == 38
    assert all(st == "unchanged" for st in stats["files"].values())


def test_changed_file_is_reindexed(settings):
    from pathlib import Path

    service = IndexService(settings)
    service.sync()

    knowledge_dir = Path(settings.data_dir) / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "faq_extra.md").write_text(
        "# 临时FAQ\n测试增量更新内容。", encoding="utf-8"
    )
    stats = service.sync()
    assert stats["files"]["knowledge/faq_extra.md"] == "new"
    assert stats["inserted"] == 1

    # 修改后重新处理
    (knowledge_dir / "faq_extra.md").write_text(
        "# 临时FAQ\n修改后的内容。", encoding="utf-8"
    )
    stats = service.sync()
    assert stats["files"]["knowledge/faq_extra.md"] == "changed"

    # 删除后文档下线
    (knowledge_dir / "faq_extra.md").unlink()
    stats = service.sync()
    assert stats["deactivated"] == 1
    assert stats["active"] == 38


def test_load_docs_matches_docstore(settings):
    service = IndexService(settings)
    service.sync()
    docs = service.load_docs()
    assert len(docs) == 38
    ids = {d.id for d in docs}
    assert "product:S001" in ids
    assert "promo:2026-09-dairy" in ids
    assert "kb:return-policy" in ids
