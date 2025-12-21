import os
import re
from typing import List, Dict, Optional, Set, Tuple

import streamlit as st
from rag_pipeline import Retriever, LLMGenerator

_CITATION_RE = re.compile(
    r"\[\s*Document(?:\s+Name)?\s*:\s*(?P<doc>.*?)\s*,\s*Page(?:\s*No)?\s*:\s*(?P<page>\d+)\s*\]",
    flags=re.IGNORECASE,
)

_PAGE_NUM_RE = re.compile(r"(\d+)")

def normalize_page(value: object) -> str:
    """
    Extract first integer from strings like: 'page_6', '6', 'Page 6'
    Returns '' if no integer found.
    """
    s = str(value or "").strip().lower()
    m = _PAGE_NUM_RE.search(s)
    return m.group(1) if m else ""


def normalize_doc(value: object) -> str:
    return str(value or "").strip().lower()


def extract_citations(answer_text: str) -> Set[Tuple[str, str]]:
    cites: Set[Tuple[str, str]] = set()
    for m in _CITATION_RE.finditer(answer_text or ""):
        doc = normalize_doc(m.group("doc"))
        page = normalize_page(m.group("page"))
        if doc and page:
            cites.add((doc, page))
    return cites


def filter_docs_to_used(docs: List[Dict], answer_text: str) -> List[Dict]:
    cite_keys = extract_citations(answer_text)
    if not cite_keys:
        return []
    used = []
    for d in docs:
        doc_name = normalize_doc(d.get("document_name"))
        page_num = normalize_page(d.get("page_number"))
        if (doc_name, page_num) in cite_keys:
            used.append(d)

    used.sort(key=lambda x: float(x.get("relevance_score", 0) or 0), reverse=True)
    return used

@st.cache_resource
def init_services():
    weaviate_url=os.environ["WEAVIATE_URL"]
    weaviate_api_key=os.environ["WEAVIATE_API_KEY"]
    openai_api_key=os.environ["OPENAI_API_KEY"]
    voyage_api_key=os.environ["VOYAGE_API_KEY"]

    collection_name = "DocumentChunk"
    model_name = "gpt-4o"

    missing = [
        name
        for name, val in {
            "WEAVIATE_URL": weaviate_url,
            "WEAVIATE_API_KEY": weaviate_api_key,
            "OPENAI_API_KEY": openai_api_key,
            "VOYAGE_API_KEY": voyage_api_key,
        }.items()
        if not val
    ]

    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        st.stop()

    retriever = Retriever(
        weaviate_url=weaviate_url,
        weaviate_api_key=weaviate_api_key,
        openai_api_key=openai_api_key,
        voyage_api_key=voyage_api_key,
        collection_name=collection_name,
    )

    generator = LLMGenerator(
        openai_api_key=openai_api_key,
        model=model_name,
    )

    return retriever, generator

def render_used_sources(docs: List[Dict], show_empty: bool):
    if not docs:
        if show_empty:
            st.info("No cited sources found in the answer.")
        return

    for i, d in enumerate(docs, start=1):
        score = d.get("relevance_score")
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        title = f"{i}. {d.get('document_name','')} — {d.get('page_number','')} (score: {score_txt})"

        with st.expander(title):
            if d.get("source"):
                st.write(f"**Source:** {d['source']}")
            st.write(d.get("content", ""))


def build_multiturn_query(messages: List[Dict], max_user_turns: int) -> str:
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    return "\n".join(user_turns[-max_user_turns:]).strip()

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("RAG Chat")

retriever, generator = init_services()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_used_sources" not in st.session_state:
    st.session_state.last_used_sources = []

with st.sidebar:
    st.header("Retrieval")
    initial_limit = st.slider("Initial hybrid limit", 5, 200, 50, step=5)
    rerank_top_k = st.slider("Rerank top_k", 1, 50, 25)
    retrieval_user_turns = st.slider("Use last N user turns", 1, 8, 3)

    st.header("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
    max_tokens = st.slider("Max tokens", 128, 2000, 500, step=32)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_used_sources = []
        st.rerun()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Persistent sources container (important)
st.divider()
sources_container = st.container()

has_assistant_turn = any(m["role"] == "assistant" for m in st.session_state.messages)

with sources_container:
    st.subheader("Chunks used in the answer")
    render_used_sources(
        st.session_state.last_used_sources,
        show_empty=has_assistant_turn,
    )

# Chat input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieval_query = build_multiturn_query(
        st.session_state.messages,
        retrieval_user_turns,
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                retrieved_docs = retriever.retrieve(
                    retrieval_query,
                    initial_limit=initial_limit,
                    rerank_top_k=rerank_top_k,
                )

                answer = generator.generate_answer(
                    query=prompt,
                    context_docs=retrieved_docs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                st.session_state.last_used_sources = filter_docs_to_used(
                    retrieved_docs,
                    answer,
                )

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )
                st.session_state.last_used_sources = []

    # Force re-render of sources after update
    has_assistant_turn = True
    with sources_container:
        st.subheader("Chunks used in the answer")
        render_used_sources(
            st.session_state.last_used_sources,
            show_empty=True,
        )
