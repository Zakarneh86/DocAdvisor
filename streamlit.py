"""Streamlit UI for DocAdvisor. Run with: streamlit run streamlit.py"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

import main


APP_ROOT = Path(__file__).resolve().parent
VECTOR_ROOT = APP_ROOT / "vector_store"
INVALID_STORE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def list_stores() -> list[str]:
    if not VECTOR_ROOT.exists():
        return []
    return sorted(
        (item.name for item in VECTOR_ROOT.iterdir() if item.is_dir()),
        key=str.casefold,
    )


def validate_store_name(value: str) -> str:
    name = " ".join(value.split())
    if not name:
        raise ValueError("Enter a vector-store name.")
    if len(name) > 100:
        raise ValueError("The vector-store name must be 100 characters or fewer.")
    if name in {".", ".."} or INVALID_STORE_CHARS.search(name):
        raise ValueError('The name cannot contain < > : " / \\ | ? * or control characters.')
    if name.endswith((".", " ")):
        raise ValueError("The name cannot end with a period or space.")
    return name


def store_path(name: str) -> Path:
    root = VECTOR_ROOT.resolve()
    path = (root / validate_store_name(name)).resolve()
    if path.parent != root:
        raise ValueError("Invalid vector-store path.")
    return path


@st.cache_resource(show_spinner=False)
def openai_resources(api_key: str):
    openai_client, error, status = main.client(api_key)
    if error or openai_client is None:
        raise RuntimeError(status)
    return openai_client, main.load_embeddings(api_key)


@st.cache_resource(show_spinner="Loading ranking model…")
def ranker():
    return main.load_ranker()


def get_answer(openai_client, question: str, context: str):
    """Use main's prompt/schema while keeping main.py unchanged."""
    response = openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": main.System_prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
        temperature=0,
        response_format=main.StandardsAnswer,
    )
    return response.choices[0].message.parsed


def clear_active_store() -> None:
    st.session_state.pop("vector_store", None)
    st.session_state.pop("active_store", None)
    st.session_state.messages = []


def show_answer(answer) -> None:
    st.markdown(answer.answer)
    st.caption(f"Status: {answer.status.replace('_', ' ').title()}")
    if answer.references:
        with st.expander("References", expanded=True):
            for reference in answer.references:
                details = []
                if reference.clause:
                    details.append(f"Clause {reference.clause}")
                if reference.page is not None:
                    details.append(f"page {reference.page}")
                location = f" — {', '.join(details)}" if details else ""
                st.markdown(f"**{reference.document}**{location}")
                st.write(reference.evidence)
    if answer.notes:
        st.info(answer.notes)


st.set_page_config(page_title="Document Advisor", page_icon="📚")
st.title("Document Advisor")
st.caption("Upload standards documents, then ask questions about their content.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document store")
    api_key = st.text_input("OpenAI API key", type="password")

    stores = list_stores()
    selection = st.selectbox(
        "Existing vector stores",
        ["Create a new store", *stores],
        on_change=clear_active_store,
    )
    if selection == "Create a new store":
        store_name_input = st.text_input(
            "Vector-store name",
            placeholder="Saudi Electric Company Standard",
        )
    else:
        store_name_input = selection
        st.caption("Uploaded documents will update this store.")

    uploads = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploads,
    ):
        try:
            if not api_key:
                raise ValueError("Enter your OpenAI API key first.")
            name = validate_store_name(store_name_input)
            path = store_path(name)
            openai_client, embeddings = openai_resources(api_key)

            with st.status("Processing documents…", expanded=True) as status:
                st.write("Extracting text from uploaded PDFs…")
                documents = main.load_documents(uploads, openai_client)
                if not documents:
                    raise RuntimeError("No document text was extracted.")

                st.write("Splitting pages into searchable chunks…")
                chunks = main.split_documents(documents)
                if path.is_dir():
                    st.write(f"Updating {name}…")
                    vector_store = main.load_db(embeddings, str(path))
                    vector_store = main.update_db(vector_store, chunks)
                else:
                    st.write(f"Creating {name}…")
                    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)
                    vector_store = main.create_db(chunks, embeddings, str(path))

                st.session_state.vector_store = vector_store
                st.session_state.active_store = name
                st.session_state.messages = []
                status.update(label="Documents are ready", state="complete")
            st.success(f"Saved {len(chunks)} chunks in vector_store/{name}.")
        except Exception as exc:
            st.error(f"Could not process the documents: {exc}")

    if st.button(
        "Load selected store",
        use_container_width=True,
        disabled=selection == "Create a new store",
    ):
        try:
            if not api_key:
                raise ValueError("Enter your OpenAI API key first.")
            name = validate_store_name(selection)
            _, embeddings = openai_resources(api_key)
            st.session_state.vector_store = main.load_db(
                embeddings, str(store_path(name))
            )
            st.session_state.active_store = name
            st.session_state.messages = []
            st.success(f"Loaded {name}.")
        except Exception as exc:
            st.error(f"Could not load the vector store: {exc}")

active_store = st.session_state.get("active_store")
if active_store:
    st.caption(f"Active vector store: **{active_store}**")
else:
    st.info("Load an existing vector store or upload PDFs to create one.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message["answer"])
        else:
            st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about the documents",
    disabled=not active_store,
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if not api_key:
                raise ValueError("Enter your OpenAI API key in the sidebar.")
            openai_client, _ = openai_resources(api_key)
            with st.spinner("Searching documents and preparing an answer…"):
                retriever = main.creat_retriever(
                    st.session_state.vector_store,
                    base_retrieved=15,
                    top_retrived=5,
                    ranker_model=ranker(),
                )
                documents = retriever.invoke(question)
                context = main.format_context(documents)
                answer = get_answer(openai_client, question, context)
            show_answer(answer)
            st.session_state.messages.append(
                {"role": "assistant", "answer": answer}
            )
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")
