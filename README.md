# DocAdvisor

**AI-Powered Technical Standards Advisor using RAG**

DocAdvisor is an AI application designed to help engineers quickly retrieve and interpret requirements from large technical standards and specifications.

Instead of manually searching through lengthy PDF documents, users can ask engineering questions in natural language. DocAdvisor retrieves the most relevant sections of the standards, reranks them for relevance, and generates a grounded answer with references to the source document and page.

The project focuses on reducing hallucination and making AI-generated answers traceable to the original engineering documentation.

---

## Why DocAdvisor?

Engineering projects often involve hundreds or thousands of pages of technical standards.

Finding a requirement may require:

1. Identifying the correct standard
2. Searching for relevant terminology
3. Reading multiple sections
4. Comparing requirements
5. Determining whether the available documentation actually supports the answer

DocAdvisor applies Retrieval-Augmented Generation (RAG) to automate much of this process while preserving the link between the AI response and the source documentation.


## Architecture

```text
Technical Standards (PDF)
          │
          ▼
   Document Extraction
          │
          ▼
   Text Processing
          │
          ▼
       Chunking
          │
          ▼
   OpenAI Embeddings
          │
          ▼
   Chroma Vector Store
          │
          ▼
   Semantic Retrieval
          │
          ▼
 Cross-Encoder Reranking
          │
          ▼
    Relevant Context
          │
          ▼
        OpenAI LLM
          │
          ▼
   Structured Response
          │
          ▼
 Answer + Source References
```

---

## Retrieval Pipeline

DocAdvisor uses a multi-stage retrieval pipeline rather than sending the entire document directly to the language model.

### 1. Document Processing

PDF standards are extracted and converted into structured text while maintaining document and page metadata.

### 2. Embeddings

Document content is converted into vector embeddings using OpenAI embedding models.

### 3. Vector Search

Embeddings are stored in **ChromaDB** and used to retrieve candidate sections relevant to the user's question.

### 4. Cross-Encoder Reranking

Initial retrieval results are reranked using a CrossEncoder model to improve the relevance of the context provided to the LLM.

```text
User Question
      │
      ▼
Vector Search
      │
      ▼
Candidate Documents
      │
      ▼
CrossEncoder Reranker
      │
      ▼
Top Relevant Context
```

### 5. Grounded Generation

Only the highest-ranked context is passed to the language model.

The model generates a structured response containing the answer, support status, and source references.

---

## Response Philosophy

A major design goal of DocAdvisor is to distinguish between:

- **Supported** — the retrieved standards contain sufficient information to answer the question.
- **Insufficient Information** — the available documentation does not provide enough evidence for a reliable answer.

This is particularly important for engineering applications where an unsupported AI answer may be more harmful than returning no answer.

---

## Technology Stack

- Python
- OpenAI API
- LangChain
- OpenAI Embeddings
- ChromaDB
- CrossEncoder Reranking
- Pydantic
- PyMuPDF
- Streamlit

---

## Key Features

- Natural-language engineering questions
- PDF technical-standard ingestion
- Page-aware document extraction
- Vector-based semantic retrieval
- CrossEncoder reranking
- Retrieval-Augmented Generation (RAG)
- Structured LLM responses
- Source-document references
- Page-level traceability
- Detection of insufficient supporting information
- Streamlit web interface

## Project Motivation

DocAdvisor was developed from a practical engineering problem: technical decisions frequently depend on locating precise requirements scattered across large standards and specifications.

The project explores how modern LLMs can be combined with traditional information-retrieval techniques to create an engineering assistant that does not simply generate answers, but provides answers that can be traced back to engineering documentation.

---

## Future Development

Potential future development includes:

- Support for larger collections of engineering standards
- Improved document ingestion and metadata handling
- Hybrid semantic/keyword retrieval
- Retrieval evaluation and benchmarking
- Local/open-source LLM support through OpenAI-compatible APIs
- Multi-document requirement comparison
- Engineering compliance checking

---

## Author

**Mohamed Zakarneh**

Control & Automation Engineer | AI / Machine Learning

GitHub: [Zakarneh86](https://github.com/Zakarneh86)
