"""
Script to initialize the document processing for the RAG chatbot.
This script will process all markdown files in the book_frontend/docs directory
and store them in the Qdrant vector database.
"""

import asyncio
import sys
import os

# Add the app directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.services.document_processor import document_processor


async def main():
    print("Starting document processing...")
    try:
        total_chunks = await document_processor.process_and_store_documents()
        print(f"Successfully processed and stored {total_chunks} document chunks")
        print("Document initialization complete!")
    except Exception as e:
        print(f"Error during document processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())