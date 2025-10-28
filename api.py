import asyncio
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel

from fastapi_models import (AskResponse, ChatMessage, ChatQuery, ChatResponse,
                            DocumentResponse, DocumentUploadModel,
                            DocumentUploadResponse, Query)
from models import DocumentModel
from preprocess import  DocumentParser, DocumentChunker,DocumentProcessor #parse_document
from providers import get_llm
from rag_processor import RAGPipeline 


load_dotenv(override=True)

HOST = os.environ.get("HOST", "0.0.0.0")
FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", "8000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for chat sessions (in production, use Redis or database)
# Structure: {session_id: {"messages": [...], "collection_id": "...", "created_at": "...", "last_activity": "..."}}
chat_sessions = {}
model = get_llm()
document_processor = DocumentProcessor()
rag_pipeline = RAGPipeline(model=model,db_path=os.environ.get("CHROMADB_LOCAL_DIR"))
# Session timeout configuration (in hours)
SESSION_TIMEOUT_HOURS = 24

def cleanup_inactive_sessions():
    """Clean up sessions that have been inactive for too long"""
    current_time = datetime.now()
    timeout_threshold = current_time - timedelta(hours=SESSION_TIMEOUT_HOURS)
    
    sessions_to_delete = []
    for session_id, session_data in chat_sessions.items():
        last_activity = session_data.get("last_activity", session_data.get("created_at"))
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
        
        if last_activity < timeout_threshold:
            sessions_to_delete.append(session_id)
    
    # Delete inactive sessions and their collections
    for session_id in sessions_to_delete:
        try:
            session_data = chat_sessions[session_id]
            collection_id = session_data.get("collection_id")
            
            # Delete collection if it exists
            if collection_id:
                try:
                    rag_pipeline.vector_store.delete_collection(collection_id)
                    logger.info(f"Auto-deleted collection {collection_id} for inactive session {session_id}")
                except Exception as e:
                    logger.error(f"Error auto-deleting collection {collection_id}: {e}")
                    raise Exception(f"Error auto-deleting collection {collection_id}: {e}")
            
            # Delete session
            del chat_sessions[session_id]
            logger.info(f"Auto-deleted inactive session {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            raise Exception(f"Error cleaning up session {session_id}: {e}")

def start_cleanup_scheduler():
    """Start background thread for session cleanup"""
    def cleanup_loop():
        while True:
            try:
                cleanup_inactive_sessions()
                time.sleep(3600)  # Run cleanup every hour
            except Exception as e:
                logger.error(f"Error in cleanup scheduler: {e}")
                time.sleep(3600)
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Session cleanup scheduler started")


app = FastAPI(
    title="Chatbot RAG",
    description="A simple chatbot using OpenAI. Enables asking questions and getting answers based on uploaded documents.",
    version="0.1",
)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    start_cleanup_scheduler()
    logger.info("FastAPI application started successfully")

@app.get("/")
def read_root():
    return {
        "service": "RAG Chatbot using OPENAI",
        "description": "Welcome to Chatbot RAG API",
        "status": "running",
    }
     
@app.delete("/collection/{collection_id}")
def delete_collection_endpoint(collection_id: str)->Dict[str,Any]:
    try:
        success = rag_pipeline.vector_store.delete_collection(collection_id)
        if success:
            return {"message": f"Collection {collection_id} deleted successfully", "status": "success"}
        else:
            return {"message": f"Collection {collection_id} does not exist", "status": "not_found"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}, status: failed")
    
    
@app.post("/feed")
async def upload_documents(files: List[UploadFile], filepaths: List[str] = Form(...), collection_id: str = Form(...)) -> DocumentUploadResponse:
    try:
        num_documents = 0
        num_chunks = 0
        processed_files = []
        
        for i, file in enumerate(files):
            if file.content_type != "application/pdf":
                logger.error(f"Unsupported file type: {file.content_type}")
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            content = await file.read()
            # Parse PDF using PyMuPDF
            
            current_filepath = filepaths[i] if i < len(filepaths) else file.filename
            chunks = document_processor.process_document(content, current_filepath, file.content_type)
            chunks = rag_pipeline.vector_store.convert_chunks_to_vectors(chunks)
            num_chunks += len(chunks)
            rag_pipeline.vector_store.store_document(chunks,collection_id)
            num_documents += 1
            processed_files.append(file.filename)
        return {"num_docs": num_documents,
                "num_chunks": num_chunks, 
                "status": "Success",
                "collection_id": collection_id,
                "files": [file.filename for file in files],
                "files_path": filepaths[:len(processed_files)]}
    except Exception as e:
        logger.error(f"Error uploading documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}, status: Failed")

     
@app.post("/chat")
def chat_with_history(query: ChatQuery) -> ChatResponse:
    try:
        session_id = query.session_id or str(uuid.uuid4())
        
        # Get existing session data or create new
        session_data = chat_sessions.get(session_id, {
            "messages": [],
            "collection_id": query.collection_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        })
        
        # Update collection_id if provided
        if query.collection_id:
            session_data["collection_id"] = query.collection_id
        
        # Get existing chat history
        chat_history = session_data.get("messages", [])
        
         # Process question with history
        answer, updated_history = rag_pipeline.ask_question_with_history( 
            query.collection_id, 
            query.question, 
            chat_history
        )
        
        # Update session data
        session_data["messages"] = updated_history
        session_data["last_activity"] = datetime.now()
        chat_sessions[session_id] = session_data
        
        return ChatResponse(
            question=query.question,
            answer=answer,
            session_id=session_id,
            chat_history=updated_history
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise f"fail to handle chat with history in session {session_id}: error: {str(e)}"
        
@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str)->Dict[str,Any]:
    session_data = chat_sessions.get(session_id, {})
    return {
        "session_id": session_id, 
        "messages": session_data.get("messages", []),
        "collection_id": session_data.get("collection_id"),
        "created_at": session_data.get("created_at"),
        "last_activity": session_data.get("last_activity")
    }        


@app.delete("/chat/{session_id}")
def clear_chat_history(session_id: str)->Dict[str,Any]:
    if session_id in chat_sessions:
        # Clear only messages, keep session metadata
        chat_sessions[session_id]["messages"] = []
        chat_sessions[session_id]["last_activity"] = datetime.now()
        return {"message": f"Chat history for session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}

@app.delete("/session/{session_id}")
def delete_session_and_collection(session_id: str):
    """Delete both chat session and associated collection"""
    try:
        collection_id = None
        session_cleared = False
        
        if session_id in chat_sessions:
            session_data = chat_sessions[session_id]
            collection_id = session_data.get("collection_id")
            
            # Delete the session
            del chat_sessions[session_id]
            session_cleared = True
            
            # Delete the associated collection if it exists
            if collection_id:
                try:
                    success = rag_pipeline.vector_store.delete_collection(collection_id)
                    if success:
                        logger.info(f"Collection {collection_id} deleted successfully for session {session_id}")
                    else:
                        logger.warning(f"Collection {collection_id} not found or already deleted")
                except Exception as e:
                    logger.error(f"Error deleting collection {collection_id}: {e}")
        
        return {
            "message": f"Session {session_id} and associated data cleared",
            "session_cleared": session_cleared,
            "collection_id": collection_id,
            "collection_deleted": collection_id is not None
        }
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        return {"message": f"Error deleting session: {str(e)}", "session_cleared": False}

@app.post("/cleanup/inactive")
def manual_cleanup_inactive_sessions():
    """Manually trigger cleanup of inactive sessions"""
    try:
        cleanup_inactive_sessions()
        return {"message": "Inactive sessions cleanup completed", "status": "success"}
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}", exc_info=True)
        return {"message": f"Error during cleanup: {str(e)}", "status": "error"}

@app.get("/sessions")
def list_active_sessions():
    """List all active sessions"""
    try:
        sessions_info = []
        for session_id, session_data in chat_sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "collection_id": session_data.get("collection_id"),
                "created_at": session_data.get("created_at"),
                "last_activity": session_data.get("last_activity"),
                "message_count": len(session_data.get("messages", []))
            })
        return {"sessions": sessions_info, "total": len(sessions_info)}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        return {"sessions": [], "total": 0, "error": str(e)}
        
        
if __name__ == "__main__":
    # Start the cleanup scheduler
    start_cleanup_scheduler()
    uvicorn.run(app, host=HOST, port=FASTAPI_PORT,)