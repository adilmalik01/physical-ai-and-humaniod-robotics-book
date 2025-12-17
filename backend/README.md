# RAG Chatbot Backend

This backend service provides the API for the RAG (Retrieval-Augmented Generation) chatbot integrated with the Physical AI & Humanoid Robotics book.

## Features

- RAG chat functionality with vector search
- Support for user-selected text as focused context
- Document processing and indexing
- Integration with Qdrant Cloud for vector storage
- OpenRouter API for LLM reasoning

## Prerequisites

- Python 3.8+
- Qdrant Cloud account
- OpenRouter API key
- (Optional) Neon Serverless Postgres account

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

- `POST /api/v1/chat` - Main chat endpoint
- `POST /api/v1/select-text` - Chat with user-selected text as context
- `POST /api/v1/process` - Process and index book documents
- `GET /api/v1/status` - Get indexing status
- `DELETE /api/v1/clear` - Clear all indexed documents
- `GET /health` - Health check

## Environment Variables

- `QDRANT_URL` - URL for Qdrant Cloud instance
- `QDRANT_API_KEY` - API key for Qdrant Cloud
- `OPENROUTER_API_KEY` - API key for OpenRouter
- `OPENROUTER_MODEL` - Model to use for generation (default: qwen/qwen2-72b-instruct)
- `QWEN_EMBEDDING_MODEL` - Model to use for embeddings (default: nlp-ai/qwen-7b-embedding)
- `QWEN_EMBEDDING_API_KEY` - API key for embedding service
- `NEON_DATABASE_URL` - Connection string for Neon Postgres
- `DEBUG` - Enable debug mode (default: false)
- `MAX_CONTEXT_LENGTH` - Maximum context length for LLM (default: 4000)
- `TOP_K` - Number of documents to retrieve for RAG (default: 5)

## Usage

1. First, process and index the book content:
   ```bash
   curl -X POST http://localhost:8000/api/v1/process
   ```

2. Then use the chat endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "What is ROS 2?"}
       ]
     }'
   ```

## Docker

To run with Docker:

```bash
docker build -t rag-chatbot-backend .
docker run -p 8000:8000 --env-file .env rag-chatbot-backend
```