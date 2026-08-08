from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import Dict, Any
import re
import warnings


with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

print(system_prompt)