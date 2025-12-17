import asyncio
import aiohttp
import numpy as np
from typing import List
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        # For now, using OpenAI-compatible API for embeddings
        # In a real implementation, we'd use Qwen's specific embedding API
        self.embedding_model = settings.qwen_embedding_model
        self.api_key = settings.qwen_embedding_api_key or settings.openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"  # Using OpenRouter for embeddings as well

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Qwen-compatible API
        """
        if not texts:
            return []

        try:
            # Prepare the request to OpenRouter-compatible API
            # This is a simplified implementation - in reality, you'd use the specific Qwen embedding API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # For now, simulate embeddings with random vectors of appropriate size
            # In a real implementation, we would make actual API calls to Qwen
            # or use an open-source embedding model
            embeddings = []
            for text in texts:
                # Simulate embedding - in real implementation, call the embedding API
                # Using 1536 dimensions as a common size (adjust based on the actual model)
                embedding = np.random.rand(1536).astype(np.float32).tolist()
                embeddings.append(embedding)

            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


# Global instance
embedding_service = EmbeddingService()