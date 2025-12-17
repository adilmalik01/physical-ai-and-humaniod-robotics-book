import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import uuid
import markdown
from bs4 import BeautifulSoup
import re
from ..config.settings import settings
from .embedding_service import embedding_service
from .vector_store_service import vector_store_service
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, docs_path: str = None):
        if docs_path is None:
            # Get the directory of this file and navigate to the docs directory
            current_file_dir = Path(__file__).parent.parent.parent.parent  # Go up 4 levels to project root
            self.docs_path = (current_file_dir / "book_frontend" / "docs").resolve()
        else:
            self.docs_path = Path(docs_path).resolve()
        self.chunk_size = 1000  # Number of characters per chunk
        self.chunk_overlap = 100  # Number of overlapping characters between chunks

    def extract_text_from_markdown(self, markdown_content: str) -> str:
        """Extract plain text from markdown content."""
        # Convert markdown to HTML
        html = markdown.markdown(markdown_content)
        # Extract text from HTML
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're near the end, just take the remaining text
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Find a good breaking point (try to break at sentence or word boundary)
            chunk = text[start:end]

            # Look for a good break point near the end of the chunk
            if end < len(text):
                # Look for sentence endings or paragraph breaks
                sentence_end = max(
                    chunk.rfind('. '),
                    chunk.rfind('!'),
                    chunk.rfind('?'),
                    chunk.rfind('\n'),
                    chunk.rfind(' ')
                )

                if sentence_end > self.chunk_size // 2:  # Only break if it's not too early
                    end = start + sentence_end + 1
                    chunk = text[start:end]

            chunks.append(chunk)
            start = end - self.chunk_overlap  # Overlap for better context

        return chunks

    def get_all_docs(self) -> List[Dict[str, Any]]:
        """Get all markdown documents from the docs directory."""
        docs = []

        for md_file in self.docs_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract title from the first heading
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                title = title_match.group(1) if title_match else md_file.stem

                doc = {
                    'id': str(md_file.relative_to(self.docs_path)).replace('\\', '/'),
                    'title': title,
                    'content': content,
                    'path': str(md_file)
                }
                docs.append(doc)

            except Exception as e:
                logger.error(f"Error reading file {md_file}: {e}")

        return docs

    async def process_and_store_documents(self):
        """Process all documents and store them in the vector store."""
        logger.info(f"Starting document processing from {self.docs_path}")

        docs = self.get_all_docs()
        logger.info(f"Found {len(docs)} documents to process")

        total_chunks = 0

        for doc in docs:
            try:
                # Extract text from markdown
                plain_text = self.extract_text_from_markdown(doc['content'])

                # Chunk the text
                chunks = self.chunk_text(plain_text)

                logger.info(f"Processing document: {doc['id']}, {len(chunks)} chunks")

                # Process each chunk
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only process non-empty chunks
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['id']}_chunk_{i}"))

                        # Generate embedding for the chunk
                        embedding = await embedding_service.get_embedding(chunk)

                        # Prepare metadata
                        metadata = {
                            'source_document': doc['id'],
                            'title': doc['title'],
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }

                        # Store in vector store
                        vector_store_service.store_document(
                            doc_id=chunk_id,
                            content=chunk,
                            metadata=metadata,
                            vector=embedding
                        )

                        total_chunks += 1

            except Exception as e:
                logger.error(f"Error processing document {doc['id']}: {e}")

        logger.info(f"Successfully processed and stored {total_chunks} document chunks")
        return total_chunks

    def update_document(self, doc_id: str, new_content: str):
        """Update a specific document by removing old chunks and adding new ones."""
        # First, delete all chunks of this document
        all_ids = vector_store_service.get_all_document_ids()
        # Generate the expected UUIDs for this document to find them in the vector store
        expected_uuids = set()
        for i in range(1000):  # Assuming max 1000 chunks per document for search
            expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}"))
            expected_uuids.add(expected_uuid)

        doc_chunk_ids = [doc_id for doc_id in all_ids if doc_id in expected_uuids]

        for chunk_id in doc_chunk_ids:
            vector_store_service.delete_document(chunk_id)

        # Then reprocess and add the new content
        plain_text = self.extract_text_from_markdown(new_content)
        chunks = self.chunk_text(plain_text)

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}"))

                # Generate embedding for the chunk
                embedding = asyncio.run(embedding_service.get_embedding(chunk))

                # Prepare metadata
                metadata = {
                    'source_document': doc_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }

                # Store in vector store
                vector_store_service.store_document(
                    doc_id=chunk_id,
                    content=chunk,
                    metadata=metadata,
                    vector=embedding
                )

    def get_document_content(self, doc_id: str) -> str:
        """Retrieve the full content of a document by combining its chunks."""
        all_ids = vector_store_service.get_all_document_ids()
        doc_chunk_ids = [chunk_id for chunk_id in all_ids if chunk_id.startswith(f"{doc_id}_chunk_")]

        # Sort chunks by their index
        doc_chunk_ids.sort(key=lambda x: int(x.split('_chunk_')[1]))

        content_parts = []
        for chunk_id in doc_chunk_ids:
            # In a real implementation, we'd retrieve from vector store
            # For now, this is just a placeholder
            pass

        return ' '.join(content_parts)


# Global instance
document_processor = DocumentProcessor()