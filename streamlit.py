"""Streamlit UI for DocAdvisor. Run with: streamlit run streamlit.py"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
import toml

import main


APP_ROOT = Path(__file__).resolve().parent
VECTOR_ROOT = APP_ROOT / "vector_store"
INVALID_STORE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
TEMPORARY_LIBRARY = "Temporary session"


@st.cache_data(show_spinner=False)
def load_api_key() -> str:
    # Streamlit Community Cloud and standard .streamlit/secrets.toml setup.
    try:
        api_key = str(st.secrets.get("API_Keys", {}).get("openAI", "")).strip()
    except Exception:
        api_key = ""
    if api_key:
        return api_key

    # Local compatibility with the repository-root secrets.toml file.
    secrets_path = APP_ROOT / "secrets.toml"
    if not secrets_path.is_file():
        raise FileNotFoundError(
            "Configure API_Keys.openAI in Streamlit secrets or create "
            f"{secrets_path}."
        )
    config = toml.load(secrets_path)
    api_key = str(config.get("API_Keys", {}).get("openAI", "")).strip()
    if not api_key:
        raise ValueError("Missing API_Keys.openAI in Streamlit secrets.")
    return api_key


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
        raise ValueError("Enter a document-library name.")
    if len(name) > 100:
        raise ValueError("The document-library name must be 100 characters or fewer.")
    if name in {".", ".."} or INVALID_STORE_CHARS.search(name):
        raise ValueError('The name cannot contain < > : " / \\ | ? * or control characters.')
    if name.endswith((".", " ")):
        raise ValueError("The name cannot end with a period or space.")
    return name


def store_path(name: str) -> Path:
    root = VECTOR_ROOT.resolve()
    path = (root / validate_store_name(name)).resolve()
    if path.parent != root:
        raise ValueError("Invalid document-library path.")
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
    vector_store = st.session_state.get("vector_store")
    if st.session_state.get("active_is_temporary") and vector_store is not None:
        try:
            vector_store.delete_collection()
        except Exception:
            pass
    st.session_state.pop("vector_store", None)
    st.session_state.pop("active_store", None)
    st.session_state.pop("active_is_temporary", None)
    st.session_state.messages = []


def show_answer(answer) -> None:
    st.markdown(answer.answer)
    st.caption(f"Status: {answer.status.replace('_', ' ').title()}")
    if answer.references:
        with st.expander("References", expanded=True):
            for reference in answer.references:
                details = []
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

try:
    api_key = load_api_key()
except Exception as exc:
    st.error(f"OpenAI configuration error: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document library")
    st.caption("OpenAI API key loaded securely from Streamlit secrets.")

    stores = list_stores()
    actions = ["Create new library", "Temporary document session"]
    if stores:
        actions.extend(["Update existing library", "Load existing library"])
    action = st.radio(
        "Library action",
        actions,
        on_change=clear_active_store,
    )
    if not stores:
        st.caption("No existing document libraries were found.")

    if action == "Temporary document session":
        library_name = TEMPORARY_LIBRARY
        st.info(
            "Documents in this mode are kept in memory only and are not "
            "saved in the vector_store directory."
        )
    elif action == "Create new library":
        library_name = st.text_input(
            "Document-library name",
            placeholder="E.g. Employment Contract",
        )
    else:
        library_name = st.selectbox(
            "Document library",
            stores,
            on_change=clear_active_store,
        )

    if action in {
        "Create new library",
        "Update existing library",
        "Temporary document session",
    }:
        uploads = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
            key=f"uploads_{action}",
        )
        button_label = {
            "Create new library": "Create library",
            "Update existing library": "Update library",
            "Temporary document session": "Process temporary documents",
        }[action]

        if st.button(
            button_label,
            type="primary",
            use_container_width=True,
            disabled=not uploads,
        ):
            try:
                name = library_name
                path = None
                if action != "Temporary document session":
                    name = validate_store_name(library_name)
                    path = store_path(name)
                if action == "Create new library" and path.exists():
                    raise ValueError(
                        "A document library with this name already exists. "
                        "Choose Update existing library instead."
                    )
                if action == "Update existing library" and not path.is_dir():
                    raise FileNotFoundError(
                        f"The document library '{name}' no longer exists."
                    )

                openai_client, embeddings = openai_resources(api_key)
                with st.status("Processing documents…", expanded=True) as status:
                    st.write("Extracting text from uploaded PDFs…")
                    documents = main.load_documents(uploads, openai_client)
                    if not documents:
                        raise RuntimeError("No document text was extracted.")

                    st.write("Splitting pages into searchable chunks…")
                    chunks = main.split_documents(documents)
                    if action == "Temporary document session":
                        st.write("Creating temporary in-memory library…")
                        vector_store = main.create_temporary_db(chunks, embeddings)
                    elif action == "Update existing library":
                        st.write(f"Updating {name}…")
                        vector_store = main.load_db(embeddings, str(path))
                        vector_store = main.update_db(vector_store, chunks)
                    else:
                        st.write(f"Creating {name}…")
                        VECTOR_ROOT.mkdir(parents=True, exist_ok=True)
                        vector_store = main.create_db(chunks, embeddings, str(path))

                    st.session_state.vector_store = vector_store
                    st.session_state.active_store = name
                    st.session_state.active_is_temporary = (
                        action == "Temporary document session"
                    )
                    st.session_state.messages = []
                    status.update(label="Documents are ready", state="complete")

                if action == "Temporary document session":
                    st.success(
                        f"Prepared {len(chunks)} temporary chunks for this session."
                    )
                else:
                    verb = "Added" if action == "Update existing library" else "Saved"
                    st.success(f"{verb} {len(chunks)} chunks in {name}.")
            except Exception as exc:
                st.error(f"Could not process the documents: {exc}")

    elif st.button(
        "Load library",
        type="primary",
        use_container_width=True,
    ):
        try:
            name = validate_store_name(library_name)
            _, embeddings = openai_resources(api_key)
            path = store_path(name)
            if not path.is_dir():
                raise FileNotFoundError(
                    f"The document library '{name}' no longer exists."
                )
            st.session_state.vector_store = main.load_db(
                embeddings, str(path)
            )
            st.session_state.active_store = name
            st.session_state.active_is_temporary = False
            st.session_state.messages = []
            st.success(f"Loaded {name}.")
        except Exception as exc:
            st.error(f"Could not load the document library: {exc}")

active_store = st.session_state.get("active_store")
if active_store:
    st.caption(f"Active document library: **{active_store}**")
    if st.session_state.get("active_is_temporary"):
        if st.sidebar.button(
            "Clear temporary documents",
            use_container_width=True,
        ):
            clear_active_store()
            st.rerun()
else:
    st.info("Load an existing document library or upload PDFs to create one.")

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
