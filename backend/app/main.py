from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1 import chat, documents

app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG chatbot integration with Physical AI & Humanoid Robotics book",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
