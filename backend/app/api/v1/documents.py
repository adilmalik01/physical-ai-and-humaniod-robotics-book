from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
from ...services.document_processor import document_processor
from ...services.vector_store_service import vector_store_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/process")
async def process_documents():
    """
    Process all documents from the book and store them in the vector database.
    """
    try:
        total_chunks = await document_processor.process_and_store_documents()
        return {
            "message": f"Successfully processed and stored {total_chunks} document chunks",
            "total_chunks_processed": total_chunks
        }
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_processing_status():
    """
    Get the status of document processing (e.g., how many documents are indexed).
    """
    try:
        all_doc_ids = vector_store_service.get_all_document_ids()
        unique_docs = set()
        for doc_id in all_doc_ids:
            # Extract the base document ID (before the chunk part)
            base_id = doc_id.split('_chunk_')[0]
            unique_docs.add(base_id)

        return {
            "total_chunks": len(all_doc_ids),
            "unique_documents": len(unique_docs),
            "documents": list(unique_docs)
        }
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

