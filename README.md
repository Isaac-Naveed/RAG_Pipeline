# RAG Pipeline from Scratch (LangChain + ChromaDB)

A hands-on Retrieval-Augmented Generation (RAG) pipeline that lets an LLM answer questions grounded in your own PDF documents. Built step by step while learning how RAG works under the hood.

## How it works

```
PDFs → Documents → Chunks → Embeddings → Vector Store (ChromaDB)
                                                │
 User query → Query embedding → Top-k retrieval ┘
                                      │
                       Retrieved context + query → LLM → Answer
```

**Ingestion pipeline**
1. **Load** – read every PDF in `data/pdfs/` with LangChain's `PyPDFLoader`
2. **Chunk** – split pages with `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
3. **Embed** – encode chunks with `sentence-transformers` (`all-MiniLM-L6-v2`)
4. **Store** – persist vectors, text and metadata in a local ChromaDB collection

**Retrieval + generation**
1. Embed the user query with the same model
2. Fetch the top-k most similar chunks (cosine similarity) from ChromaDB
3. Pass the chunks as context to an LLM (OpenAI or Groq) to generate the answer

## Tech stack

Python · LangChain · Sentence-Transformers · ChromaDB · OpenAI API / Groq API

## Setup

```bash
git clone https://github.com/Isaac-Naveed/RAG_Pipeline.git
cd RAG_Pipeline
pip install -r requirements.txt
```

Add your PDFs to `data/pdfs/`, then set an API key for whichever LLM you want to use:

```bash
export OPENAI_API_KEY="your-key"   # or
export GROQ_API_KEY="your-key"
```

Open `rag_pipeline.ipynb` and run the cells in order.

## Project structure

```
├── rag_pipeline.ipynb   # full pipeline: ingestion, retrieval, generation
├── requirements.txt
├── data/
│   └── pdfs/            # put your PDFs here 
└── README.md
```

