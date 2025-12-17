from typing import List, Dict, Any
from .vector_store_service import vector_store_service
from .embedding_service import embedding_service
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self):
        self.top_k = settings.top_k
        self.max_context_length = settings.max_context_length

    async def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using vector search.
        """
        if top_k is None:
            top_k = self.top_k

        try:
            # Generate embedding for the query
            query_embedding = await embedding_service.get_embedding(query)

            # Search in vector store
            results = vector_store_service.search(query_embedding, top_k=top_k)

            logger.info(f"Retrieved {len(results)} context chunks for query")
            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def retrieve_context_from_selected_text(self, selected_text: str) -> List[Dict[str, Any]]:
        """
        When user provides selected text, use it directly as context instead of searching.
        """
        # Return the selected text as a single context chunk
        context_chunk = {
            "id": "selected_text",
            "content": selected_text,
            "metadata": {"source": "user_selection"},
            "score": 1.0  # Perfect relevance since user provided it
        }

        logger.info("Using user-selected text as context")
        return [context_chunk]

    def format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context chunks into a single string for the LLM.
        """
        formatted_contexts = []
        total_length = 0

        for chunk in context_chunks:
            chunk_text = chunk.get('content', '')
            chunk_metadata = chunk.get('metadata', {})

            # Create a formatted context block with metadata
            formatted_chunk = f"""
            [Source: {chunk_metadata.get('source_document', 'Unknown')} | Chunk: {chunk_metadata.get('chunk_index', 'N/A')}]
            {chunk_text}
            """

            # Check if adding this chunk would exceed the max context length
            if total_length + len(formatted_chunk) > self.max_context_length:
                logger.info(f"Context length limit reached. Using {len(formatted_contexts)} out of {len(context_chunks)} chunks.")
                break

            formatted_contexts.append(formatted_chunk)
            total_length += len(formatted_chunk)

        return "\n\n".join(formatted_contexts)

    async def get_relevant_context(self, query: str, selected_text: str = None, top_k: int = None) -> str:
        """
        Get relevant context for a query, using either vector search or user-selected text.
        """
        if selected_text and selected_text.strip():
            # Use user-selected text as context
            context_chunks = self.retrieve_context_from_selected_text(selected_text)
        else:
            # Use vector search to find relevant context
            context_chunks = await self.retrieve_context(query, top_k)

        # Format the context for the LLM
        formatted_context = self.format_context(context_chunks)
        return formatted_context


# Global instance
retrieval_service = RetrievalService()