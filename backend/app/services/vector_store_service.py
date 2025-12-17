from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        # Initialize Qdrant client
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=False  # Using REST API
            )
        else:
            # For local or without API key
            self.client = QdrantClient(url=settings.qdrant_url)

        # Collection name for book content
        self.collection_name = "book_content"

        # Initialize the collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the Qdrant collection with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection with appropriate configuration
                # Using a dimension of 768 for Qwen embeddings (adjust if needed)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Adjust based on embedding model (Qwen might use 768 or 1536)
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def store_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None, vector: List[float] = None):
        """Store a document in the vector store."""
        try:
            # If no vector is provided, we'll need to embed the content later
            # For now, we'll store the document with its metadata
            if vector is None:
                # In a real implementation, you would embed the content here
                # For now, we'll just store the content and metadata
                logger.warning("No vector provided, storing document without vector. Embedding should be done before calling this method.")
                return

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=vector,
                        payload={
                            "content": content,
                            "metadata": metadata or {}
                        }
                    )
                ]
            )
            logger.info(f"Stored document {doc_id} in Qdrant")
        except Exception as e:
            logger.error(f"Error storing document {doc_id} in Qdrant: {e}")
            raise

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents to the query vector."""
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )

            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score
                })

            logger.info(f"Found {len(results)} results from Qdrant search")
            return results
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            raise

    def delete_document(self, doc_id: str):
        """Delete a document from the vector store."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[doc_id]
                )
            )
            logger.info(f"Deleted document {doc_id} from Qdrant")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from Qdrant: {e}")
            raise

    def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the collection."""
        try:
            # Use scroll to get all points
            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust as needed
                with_payload=False,
                with_vectors=False
            )
            return [point.id for point in scroll_result]
        except Exception as e:
            logger.error(f"Error getting all document IDs from Qdrant: {e}")
            raise

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            all_ids = self.get_all_document_ids()
            if all_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=all_ids)
                )
            logger.info(f"Cleared collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise


# Global instance
vector_store_service = VectorStoreService()