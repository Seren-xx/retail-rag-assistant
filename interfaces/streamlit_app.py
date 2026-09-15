"""Streamlit 客户端界面：智能问答 / 知识库管理 / 系统状态。"""

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from retail_assistant.application.qa_service import QAService  # noqa: E402
from retail_assistant.config import load_settings  # noqa: E402

st.set_page_config(page_title="超市商品智能客服", page_icon="🛒", layout="wide")
st.title("🛒 超市商品智能客服")
st.caption("混合检索 | 质量门禁 | 数值确定性回答")
st.divider()


@st.cache_resource(show_spinner="初始化检索服务...")
def get_qa():
    return QAService(settings=load_settings())


qa = get_qa()

page = st.sidebar.radio("选择功能", ["智能问答", "知识库管理", "系统状态"])
st.sidebar.metric("知识库文档数", len(qa.docs))


# ============ 智能问答 ============
if page == "智能问答":
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "您好！可以问我商品价格、促销活动、售后政策等问题。"}
        ]

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索知识库..."):
                result = qa.answer(prompt, session_id="streamlit")
            st.write(result["answer"])
            with st.expander("检索证据（decision / 质量信号）"):
                st.caption(
                    f"route={result['route']} decision={result['decision']} "
                    f"quality={result['quality']} 耗时={result['elapsed_ms']}ms"
                )
                st.json(result["evidence"])
        st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})


# ============ 知识库管理 ============
elif page == "知识库管理":
    st.subheader("📚 知识库管理")
    uploaded = st.file_uploader(
        "上传知识文档（txt / md / csv / jsonl / pdf / docx / xlsx）",
        type=["txt", "md", "csv", "jsonl", "json", "pdf", "docx", "xlsx"],
    )
    if uploaded is not None and st.button("导入并同步索引", type="primary"):
        knowledge_ext = (".txt", ".md", ".pdf", ".docx", ".xlsx")
        target = Path(qa.s.data_dir) / ("knowledge" if uploaded.name.endswith(knowledge_ext) else uploaded.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getvalue())
        stats = qa.index_service.sync()
        st.success(
            f"已导入 {uploaded.name}：新增 {stats['inserted']}，更新 {stats['updated']}，"
            f"重复跳过 {stats['skipped_duplicate']}，当前文档 {stats['active']}"
        )
        st.cache_resource.clear()


# ============ 系统状态 ============
else:
    st.subheader("📊 系统状态")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("知识库文档数", len(qa.docs))
        st.metric("索引文件数", qa.index_service.manifest.stats()["files"])
    with col2:
        st.metric("Embedding", qa.s.embedding_model)
        st.metric("Reranker", qa.s.reranker_model)
        st.metric("答案生成", qa.s.llm_model if qa.llm else "证据直出（未配置 Key）")

    st.divider()
    with st.expander("检索配置"):
        st.json(
            {
                "bm25_top_k": qa.s.bm25_top_k,
                "dense_top_k": qa.s.dense_top_k,
                "rrf_k": qa.s.rrf_k,
                "final_top_k": qa.s.final_top_k,
                "word_score_floor": qa.s.word_score_floor,
                "rerank_floor": qa.s.rerank_floor,
                "max_context_chars": qa.s.max_context_chars,
                "max_per_source": qa.s.max_per_source,
            }
        )
    if st.button("清空对话历史"):
        qa.sessions.clear("streamlit")
        st.success("已清空")
