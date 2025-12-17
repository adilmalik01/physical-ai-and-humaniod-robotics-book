from fastapi import APIRouter, HTTPException
from typing import List
import logging
from ...models.chat import ChatRequest, ChatResponse
from ...services.generation_service import generation_service
from ...services.document_processor import document_processor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint that handles conversation with RAG.
    """
    try:
        # Generate response using RAG
        result = await generation_service.generate_response_with_rag(
            messages=chat_request.messages,
            selected_text=chat_request.selected_text
        )

        return ChatResponse(
            response=result["response"],
            sources=result["sources"]
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-text")
async def select_text_endpoint(chat_request: ChatRequest):
    """
    Endpoint specifically for queries based on user-selected text.
    This bypasses vector search and uses only the provided text as context.
    """
    try:
        # Validate that selected_text is provided
        if not chat_request.selected_text or not chat_request.selected_text.strip():
            raise HTTPException(
                status_code=400,
                detail="selected_text is required for this endpoint"
            )

        # Generate response using only the selected text as context
        result = await generation_service.generate_response_with_rag(
            messages=chat_request.messages,
            selected_text=chat_request.selected_text
        )

        return ChatResponse(
            response=result["response"],
            sources=result["sources"]
        )

    except Exception as e:
        logger.error(f"Error in select-text endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))