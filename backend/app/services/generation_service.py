import aiohttp
import asyncio
from typing import List, Dict, Any
from .retrieval_service import retrieval_service
from ..models.chat import ChatMessage
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(self):
        self.model = settings.openrouter_model
        self.api_key = settings.openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"

    async def generate_response(self, messages: List[ChatMessage], context: str = None) -> str:
        """
        Generate a response using the OpenRouter API with optional context.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Prepare the messages for the API
            api_messages = []

            # Add system message with context if provided
            if context:
                system_message = {
                    "role": "system",
                    "content": f"You are an AI assistant for the Physical AI & Humanoid Robotics book. Answer questions based on the following context from the book:\n\n{context}\n\nIf the context doesn't contain the information needed to answer the question, say so clearly. Always be helpful and provide accurate information based on the book content."
                }
                api_messages.append(system_message)
            else:
                system_message = {
                    "role": "system",
                    "content": "You are an AI assistant for the Physical AI & Humanoid Robotics book. Answer questions helpfully based on the book content when available."
                }
                api_messages.append(system_message)

            # Add the user and assistant messages
            for msg in messages:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            # Prepare the payload
            payload = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": settings.max_context_length // 4,  # Rough estimation
                "temperature": 0.7
            }

            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                        raise Exception(f"OpenRouter API error: {response.status}")

                    result = await response.json()
                    generated_text = result['choices'][0]['message']['content']

                    logger.info("Successfully generated response from OpenRouter")
                    return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def generate_response_with_rag(self, messages: List[ChatMessage], selected_text: str = None) -> Dict[str, Any]:
        """
        Generate a response using RAG (Retrieval Augmented Generation).
        """
        try:
            # Get the last user message as the query
            user_messages = [msg for msg in messages if msg.role == "user"]
            if not user_messages:
                raise ValueError("No user message found in the conversation")

            query = user_messages[-1].content

            # Retrieve relevant context
            context = await retrieval_service.get_relevant_context(query, selected_text)

            # Generate response with context
            response = await self.generate_response(messages, context)

            # Extract source documents from the context for the response
            sources = []
            if context and "Source:" in context:
                # This is a simplified approach - in a real implementation,
                # you'd track which documents were actually used
                import re
                source_matches = re.findall(r'\[Source: ([^\]]+)\]', context)
                sources = list(set(source_matches))  # Remove duplicates

            return {
                "response": response,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            raise


# Global instance
generation_service = GenerationService()