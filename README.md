# Mini RAG for Bank Statements (MVP)

A minimal Retrieval-Augmented Generation (RAG) service for analyzing bank statement histories with lightweight Text-to-SQL.  
Built on top of an existing Qdrant instance.

## Scope
- Exactly **1 PDF** containing **3 full consecutive months** of transactions.
- Parser tuned for **Bank Millennium** PDFs (tabular layout: *Amount* + right column *Balance*).
- Frontend: PHP/HTML/JS (`index.php`, `proxy_upload.php`)  
- Backend: FastAPI/Python (`mrag_app.py`)

## Features
- Upload one PDF → extract transactions into SQLite.
- Index transaction snippets in Qdrant (embedding: description + amount + meta).
- Ask questions in **Polish/English**, e.g.:
  - "How much did I spend on food in July?"
  - "What are my recurring payments?"
  - "Which three merchants were the most expensive overall?"
- Answers grounded in SQLite data + up to 3 Qdrant snippets `[filename p.page]`.
- Text-to-SQL mode with safe SELECT validation. Fallback to classic RAG.

## Architecture (short)
[Frontend PHP/JS] -> /upload (FastAPI)
|
v
[PDF Parser] -> SQLite -> Qdrant
|
[Frontend] -> /ask -> [LLM: Text-to-SQL -> SQLite]
|
[Qdrant search + snippets] -> [LLM final answer]

## Requirements
- Python 3.10+
- FastAPI, Uvicorn, pdfminer.six, qdrant-client, openai, sqlparse
- Running Qdrant (default: `localhost:6333`, collection `bank_tx_1536`)
- PHP 8.x + Apache/Nginx (for frontend/proxy)
