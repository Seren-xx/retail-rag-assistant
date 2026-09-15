"""全局配置：文件存储、向量库、模型、切分、去重、检索、门禁。环境变量与显式参数可覆盖。

分区沿用 legacy/config_data.py 的组织方式，常量即生效值。
"""

import os
from types import SimpleNamespace

# ============ 路径与文件存储 ============
data_dir = "./data"                 # 知识源目录
index_dir = "./index"               # 索引目录
docstore_file = "docstore.jsonl"    # 文档库文件（检索唯一数据源）
md5_path = "md5.index"              # MD5 精确去重文件（位于 index_dir 下）
simhash_path = "simhash_index.json"         # SimHash 近似去重索引
vector_simhash_path = "vector_index.json"   # 向量语义去重索引

# ============ Chroma 向量库 ============
collection_name = "dense"   # collection 名称
chroma_dir = "chroma"       # 向量库落盘子目录（位于 index_dir 下）

# ============ 模型配置 ============
embedding_model = "BAAI/bge-m3"             # 本地 sentence-transformers 嵌入
reranker_model = "BAAI/bge-reranker-v2-m3"  # CrossEncoder 重排
llm_model = "qwen3.7-flash"                     # Qwen3-Max（DashScope OpenAI 兼容模式）
llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm_temperature = 0.2
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")                    # DASHSCOPE_API_KEY；未配置时证据直出
llm_filters = True                          # LLM 约束抽取（规则兜底 + 连续失败熔断）
redis_url = "redis://localhost:6379/0"      # 会话存储

# ============ 文本分割配置 ============
# 段落优先+标点分句+块重叠策略
chunk_size = 500
chunk_overlap = 80
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""]

# ============ 去重配置 ============
semantic_dedup = False               # 第三级语义去重（开启后同步时逐块嵌入）
simhash_threshold = 3                # SimHash 汉明距离阈值
vector_simhash_threshold = 0.85      # 向量语义去重余弦阈值

# ============ 检索配置 ============
bm25_top_k = 30           # BM25 召回数
dense_top_k = 30          # 向量召回数
rrf_k = 60                # RRF 融合常数
rerank_top_k = 30         # CrossEncoder 重排候选数
final_top_k = 10          # 最终返回数
freshness_weight = 0.05   # 新鲜度加成上限
freshness_window_days = 90  # 数据过期/新鲜度窗口（对应 legacy data_expire_days）
today = ""                # 评测复现用固定日期；留空取系统日期
max_context_chars = 1800
max_per_source = 10

# ============ 质量门禁（生产模型 bge-reranker-v2-m3 校准） ============
word_score_floor = 0.35   # 词面信号 = 0.5*双路一致 + 0.5*词覆盖
rerank_rescue = 0.05      # 词面/重排不足时的补救阈值（v2-m3 分界带：无答案 ≤0.0244 / 可回答 ≥0.0631）
rerank_floor = 0.04       # 重排置信低于该值且覆盖不足 → 拒
rerank_coverage_floor = 0.60

# ============ 环境变量映射 ============
_CONFIG_KEYS = (
    "data_dir", "index_dir", "docstore_file",
    "md5_path", "simhash_path", "vector_simhash_path",
    "collection_name", "chroma_dir",
    "embedding_model", "reranker_model", "llm_model", "llm_base_url",
    "llm_temperature", "dashscope_api_key", "llm_filters", "redis_url",
    "chunk_size", "chunk_overlap", "separators",
    "semantic_dedup", "simhash_threshold", "vector_simhash_threshold",
    "bm25_top_k", "dense_top_k", "rrf_k", "rerank_top_k", "final_top_k",
    "freshness_weight", "freshness_window_days", "today",
    "max_context_chars", "max_per_source",
    "word_score_floor", "rerank_rescue", "rerank_floor", "rerank_coverage_floor",
)
_ENV_KEYS = {
    "embedding_model": "RETAIL_RAG_EMBEDDING_MODEL",
    "reranker_model": "RETAIL_RAG_RERANKER_MODEL",
    "llm_model": "RETAIL_RAG_LLM_MODEL",
    "dashscope_api_key": "DASHSCOPE_API_KEY",
    "redis_url": "REDIS_URL",
    "today": "RETAIL_RAG_TODAY",
    "data_dir": "RETAIL_RAG_DATA_DIR",
    "index_dir": "RETAIL_RAG_INDEX_DIR",
}
_ENV_BOOL_KEYS = {
    "llm_filters": "RETAIL_RAG_LLM_FILTERS",
    "semantic_dedup": "RETAIL_RAG_SEMANTIC_DEDUP",
}


def _env_bool(value, default):
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ============ 环境变量覆盖（import 时应用，模块常量即生效值） ============
for _key, _env in _ENV_KEYS.items():
    _raw = os.environ.get(_env)
    if _raw:
        globals()[_key] = _raw
for _key, _env in _ENV_BOOL_KEYS.items():
    globals()[_key] = _env_bool(os.environ.get(_env), globals()[_key])


def load_settings(**overrides):
    """模块常量 + 环境变量 + 显式覆盖 → 配置对象。"""
    values = {k: globals()[k] for k in _CONFIG_KEYS}
    for key, env in _ENV_KEYS.items():
        raw = os.environ.get(env)
        if raw:
            values[key] = raw
    for key, env in _ENV_BOOL_KEYS.items():
        values[key] = _env_bool(os.environ.get(env), values[key])
    values.update(overrides)
    return SimpleNamespace(**values)
