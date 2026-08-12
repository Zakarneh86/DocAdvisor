import os
import tempfile
import pymupdf
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import re
import warnings
import base64
import json

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

########## Retriever ###########
# 1) Documents Loader
# a) Text Extraction Helping Class to Set the Model Response Schema
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

# b) Text Extraction Function to Interact with the Model
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

# c) Full Document Text Extraction 3 pages a time
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

# d) Document Loader Function to Process Streamlit Uploaded Files and Return LangChain Documents
def load_documents(uploaded_files):
  documents = []
  for uploaded_file in uploaded_files:
      # Streamlit UploadedFile -> bytes
      pdf_bytes = uploaded_file.getvalue()
      # bytes -> PyMuPDF document
      pdf = pymupdf.open(
          stream=pdf_bytes,
          filetype="pdf")
      full_doc = extract_document_text(pdf)
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




