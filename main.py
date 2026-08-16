import pymupdf
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import base64

########## Initilizing Enviorment Variables, Clients...Etc)
# 1) OpenAI Client
def client (APIkey):
    try:
        client = OpenAI(api_key = APIkey)
        error = False
        status_text = "Client initialized successfully."
        return client, error, status_text
    except Exception as e:
        client = None
        error = True
        status_text = f"Failed to initialize OpenAI client: {e}"
        return client, error, status_text

## System Prompt
System_prompt = '''You are an assistant specialized in answering questions from standards documents.

Use only the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent requirements.
- If the context is insufficient, state that clearly.
- Preserve technical values, units, clause references, and conditions accurately.
- Base every answer on the retrieved standards context.'''

# 2) Loading Ranking Model
def load_ranker():
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return reranker_model
# 3) Loading Embeddings Model
def load_embeddings(api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    return embeddings
########## Retriever ###########
# 1) Documents Loader
## a) Text Extraction Helping Class to Set the Model Response Schema
class ExtractedPage(BaseModel):
    text: str = Field(description="All text extracted from that page")

class ExtractedDocument(BaseModel):
    document_id: str = Field(
        description = "The Document Number. Letters, numbers and scpecial char ':, -, _, \'"
    )
    document_name:str = Field(
        description = "The Document Name. Letters and numbers only"
    )
    pages: List[ExtractedPage] = Field(
        description="The pages extracted from the document"
    )

## b) Text Extraction Function to Interact with the Model
def extract_text(pages, client):
    content = [
        {
            "type": "text",
            "text": """
Extract all text from the following scanned document pages.

Requirements:
- Do not summarize.
- Preserve headings, clause numbers, values, units, and tables.
- Extract information from the main document body.
- The document number is letters, numbers and scpecial char ':, -, _, \\...etc'
- The document name is document subject. eg: 4 to 20 mA loop.
- Each image is preceded by its actual PDF page number.
- Use that exact page number in the output.
"""}]
    for page in pages:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        image_bytes = pix.tobytes("png")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        )

    response = client.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0,
        response_format=ExtractedDocument
    )
    return response.choices[0].message.parsed

## c) Full Document Text Extraction 3 pages a time
def extract_document_text(pdf, client):
  full_doc = {"document_id": None,
              "document_name": None,
              "pages": {}}
  for i in range(0, len(pdf), 3):
    pages = [pdf[x] for x in range(i, min(i + 3, len(pdf)))]
    while True:
        print(f"Extracting pages {i + 1}-{i + len(pages)}")
        result = extract_text(pages, client)
        if len(result.pages) == len(pages):
            break
        print(
              f"Extraction failed for pages "
              f"{i + 1}-{i + len(pages)}. "
              f"Expected {len(pages)} pages, "
              f"received {len(result.pages)}. Retrying..."
          )
    if i == 0:
        full_doc["document_id"] = result.document_id
        full_doc["document_name"] = result.document_name

    for x, page in enumerate(result.pages):
        page_number = i + x + 1

        full_doc["pages"][page_number] = page.text
  return full_doc

## d) Document Loader Function to Process Streamlit Uploaded Files and Return LangChain Documents
def load_documents(uploaded_files, client):
  documents = []
  for uploaded_file in uploaded_files:
      # Streamlit UploadedFile -> bytes
      pdf_bytes = uploaded_file.getvalue()
      # bytes -> PyMuPDF document
      pdf = pymupdf.open(
          stream=pdf_bytes,
          filetype="pdf")
      full_doc = extract_document_text(pdf, client)
      pdf.close()
      # Convert extracted pages into LangChain Documents
      for page_number, text in full_doc["pages"].items():
          documents.append(
              Document(
                  page_content=text,
                  metadata={
                      "source": uploaded_file.name,
                      "document_id": full_doc["document_id"],
                      "document_name": full_doc["document_name"],
                      "page": page_number
                  }
              )
          )
  return documents

# 2) Documents Splitter
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150)
    return text_splitter.split_documents(documents)

# 3) Vector Store Database
def create_db(chunks, embeddings, db_path):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path)
    return vector_store

def load_db(embeddings, db_path):
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings)
    return vector_store

def update_db(vector_store, new_chunks):
    vector_store.add_documents(documents=new_chunks)
    return vector_store

# 4) Creating the Retriver
def creat_retriever (vector_store, base_retrieved, top_retrived, ranker_model):
  base_retriever = vector_store.as_retriever(search_kwargs={"k": base_retrieved})
  compressor = CrossEncoderReranker(model=ranker_model, top_n = top_retrived)
  retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)
  return retriever

# 5) Formating the context
def format_context(documents):
    context = ""
    for doc in documents:
        context += f"""
        DOCUMENT ID: {doc.metadata["document_id"]}
        DOCUMENT NAME: {doc.metadata["document_name"]}
        PAGE: {doc.metadata["page"]}
        {doc.page_content}
        ---
        """
    return context

########## Model Interface ###########
# 1) Formating Model Answer
class StandardReference(BaseModel):
    document: str = Field(
        description="Document ID or document name"
    )

    page: int = Field(
        description="PDF page number containing the supporting information"
    )

    evidence: str = Field(
        description="Supporting text from the retrieved context"
    )

class StandardsAnswer(BaseModel):
    answer: str
    status: Literal[
        "supported",
        "partially_supported",
        "insufficient_information"
    ]
    references: List[StandardReference]
    notes: Optional[str] = None
# 2) Calling the Model
def answer_question (client, system_prompt, question, context):
    response = client.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": {system_prompt}},

            {
                "role": "user",
                "content": f'''
                            Question: {question}

                            context: {context}'''
            }],
        response_format = StandardsAnswer)
    return response.choices[0].message.parsed