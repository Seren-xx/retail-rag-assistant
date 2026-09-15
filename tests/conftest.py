"""测试公共夹具：临时目录 + 小语料索引。"""

import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from retail_assistant.application.session import RedisSessionStore  # noqa: E402
from retail_assistant.config import load_settings  # noqa: E402

from fakes import FakeRedis, HashingEmbedder, NoopReranker  # noqa: E402

# 测试替身嵌入维度（与生产配置解耦）
FAKE_EMBED_DIM = 256


@pytest.fixture()
def settings(tmp_path):
    """指向真实 data/ 语料，但索引/去重落盘到临时目录。

    模型组件由 qa fixture 注入测试替身，不受生产默认（BGE-M3）与环境变量影响。
    """
    return load_settings(
        data_dir=str(_ROOT / "data"),
        index_dir=str(tmp_path / "index"),
        today="2026-09-11",
        dashscope_api_key="",
    )


@pytest.fixture()
def qa(settings):
    """测试替身 QAService：Hashing Embedder + Noop 重排 + 内存 Redis，索引同步到临时目录。"""
    from retail_assistant.application.qa_service import QAService

    return QAService(
        settings=settings,
        embedder=HashingEmbedder(FAKE_EMBED_DIM),
        reranker=NoopReranker(),
        session_store=RedisSessionStore(client=FakeRedis()),
        force_sync=True,
    )


@pytest.fixture()
def knowledge_dir(tmp_path):
    """一个可复现的知识源目录（含 csv / jsonl / md）。"""
    root = tmp_path / "data"
    (root / "knowledge").mkdir(parents=True)
    (root / "products.csv").write_text(
        "product_id,name,brand,category,price,stock,unit,description,keywords\n"
        "S001,测试纯牛奶,测试牌,乳制品,10.0,5,箱,好喝的牛奶。,牛奶\n",
        encoding="utf-8",
    )
    (root / "promotions.jsonl").write_text(
        '{"id": "promo:x", "title": "促销", "category": "乳制品", "start_date": "2026-09-01", "end_date": "2026-09-30", "content": "牛奶促销"}\n',
        encoding="utf-8",
    )
    (root / "knowledge" / "policy.md").write_text(
        "# 政策\n\n支持七天无理由退货。\n\n生鲜商品除外。\n",
        encoding="utf-8",
    )
    yield root
    shutil.rmtree(root, ignore_errors=True)
