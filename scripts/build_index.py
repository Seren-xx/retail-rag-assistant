"""构建/同步知识索引：python scripts/build_index.py。

扫描 data/ 下全部受支持的知识源，走 加载→切分→去重→文档库 全流程。
manifest 增量：重复运行只处理变化文件，幂等。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from retail_assistant.config import load_settings  # noqa: E402
from retail_assistant.indexing.index_service import IndexService  # noqa: E402


def main():
    settings = load_settings()
    stats = IndexService(settings).sync()
    print("indexed docs:", stats["active"])
    print("files:", stats["files"])
    print(
        "changes:",
        {k: v for k, v in stats.items() if k not in ("files",)},
    )


if __name__ == "__main__":
    main()
