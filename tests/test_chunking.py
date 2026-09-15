"""TextChunker：段落优先 + 分句 + 重叠 + Document 生成。"""

from retail_assistant.ingestion.chunking import TextChunker


def test_paragraph_split():
    c = TextChunker()
    assert c.split_by_paragraphs("第一段。\n\n第二段。") == ["第一段。", "第二段。"]


def test_sentence_split_keeps_punctuation():
    c = TextChunker()
    parts = c.split_by_sentences("你好。世界！")
    assert "你好。" in parts and "世界！" in parts


def test_short_text_not_split():
    c = TextChunker(chunk_size=500)
    assert c.split_text("短文本") == ["短文本"]


def test_long_text_split_and_metadata():
    c = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "这是一段测试文本。" * 10  # 90 字符，触发分句与合并
    docs = c.make_documents(text, doc_id="kb:x", source="knowledge/x.md")
    assert len(docs) >= 1
    for d in docs:
        assert d.metadata["source"] == "knowledge/x.md"
        assert d.metadata["balance_key"] == "knowledge/x.md"
    # 多块时 doc_id 带序号，单块沿用原 ID
    if len(docs) > 1:
        assert docs[0].id.startswith("kb:x#")
    else:
        assert docs[0].id == "kb:x"


def test_single_chunk_keeps_doc_id():
    c = TextChunker()
    docs = c.make_documents("短政策", "kb:policy", "knowledge/policy.md")
    assert [d.id for d in docs] == ["kb:policy"]
