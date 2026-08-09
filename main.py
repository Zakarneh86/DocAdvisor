from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import re
import warnings


with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

########## Structuring Output ##############
class StandardReference(BaseModel):
    document: str = Field(
        description="Name of the standards document"
    )

    clause: Optional[str] = Field(
        default=None,
        description="Clause, section, table, or annex reference"
    )

    page: Optional[int] = Field(
        default=None,
        description="Page number in the document"
    )

    evidence: str = Field(
        description="Supporting text from the retrieved context"
    )

class StandardsAnswer(BaseModel):
    answer: str = Field(
        description="Direct answer to the user's question"
    )

    status: Literal[
        "supported",
        "partially_supported",
        "insufficient_information"
    ] = Field(
        description="How well the retrieved context supports the answer"
    )

    references: List[StandardReference] = Field(
        default_factory=list,
        description="References supporting the answer"
    )

    notes: Optional[str] = Field(
        default=None,
        description="Important qualifications, exceptions, or limitations"
    )

