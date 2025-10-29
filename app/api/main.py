"""
FastAPI application for RAG (Retrieval-Augmented Generation) chatbot.

This module provides a REST API for document upload, chat functionality with history,
and session management. It uses ChromaDB for vector storage and supports PDF document
processing with chunking and embedding generation.

Key Features:
- Document upload and processing (PDF only)
- Chat with conversation history
- Session management with automatic cleanup
- Vector database operations
- RESTful API endpoints

Author: RAG Development Team
Version: 0.1
"""


import logging
import os
import sys
from pathlib import Path
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile

PARENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PARENT_DIR))

from app.api.models import (
    ChatQuery,
    ChatResponse,
    DocumentResponse,
    DocumentUploadModel,
    DocumentUploadResponse,
    Query,
)

from app.rag.preprocess import DocumentProcessor
from app.rag.providers import get_llm
from app.rag.rag_processor import RAGPipeline

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration from environment variables
HOST = os.environ.get("HOST", "0.0.0.0")  # Server host address
FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", "8000"))  # Server port number

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for chat sessions (in production, use Redis or database)
# Structure: {session_id: {"messages": [...], "collection_id": "...", "created_at": "...", "last_activity": "..."}}
chat_sessions = {}

# Initialize core components
model = get_llm()  # Language model for text generation
document_processor = DocumentProcessor()  # Document parsing and chunking
rag_pipeline = RAGPipeline(model=model, db_path=os.environ.get("CHROMADB_LOCAL_DIR"))  # RAG processing pipeline

# Session timeout configuration (in hours)
SESSION_TIMEOUT_HOURS = 24


def cleanup_inactive_sessions():
    """
    Clean up sessions that have been inactive for too long.
    
    This function identifies sessions that haven't been active for more than
    SESSION_TIMEOUT_HOURS and removes them along with their associated vector
    collections to free up resources.
    
    Process:
    1. Calculate timeout threshold based on SESSION_TIMEOUT_HOURS
    2. Identify inactive sessions by comparing last_activity timestamps
    3. Delete associated vector collections from ChromaDB
    4. Remove session data from memory
    
    Raises:
        Exception: If there's an error during collection or session deletion
    
    Note:
        This function is called automatically by the cleanup scheduler
        and can also be triggered manually via the /cleanup/inactive endpoint.
    """
    current_time = datetime.now()
    timeout_threshold = current_time - timedelta(hours=SESSION_TIMEOUT_HOURS)

    # Identify sessions to delete based on inactivity
    sessions_to_delete = []
    for session_id, session_data in chat_sessions.items():
        last_activity = session_data.get("last_activity", session_data.get("created_at"))
        # Handle both string and datetime objects for timestamp
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)

        if last_activity < timeout_threshold:
            sessions_to_delete.append(session_id)

    # Delete inactive sessions and their collections
    for session_id in sessions_to_delete:
        try:
            session_data = chat_sessions[session_id]
            collection_id = session_data.get("collection_id")

            # Delete associated vector collection if it exists
            if collection_id:
                try:
                    rag_pipeline.vector_store.delete_collection(collection_id)
                    logger.info(f"Auto-deleted collection {collection_id} for inactive session {session_id}")
                except Exception as e:
                    logger.error(f"Error auto-deleting collection {collection_id}: {e}")
                    raise Exception(f"Error auto-deleting collection {collection_id}: {e}")

            # Remove session from memory
            del chat_sessions[session_id]
            logger.info(f"Auto-deleted inactive session {session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            raise Exception(f"Error cleaning up session {session_id}: {e}")


def start_cleanup_scheduler():
    """
    Start background thread for automatic session cleanup.
    
    This function creates a daemon thread that runs continuously in the background,
    periodically cleaning up inactive sessions. The cleanup runs every hour to
    prevent memory leaks and free up vector database resources.
    
    The cleanup scheduler:
    - Runs every 3600 seconds (1 hour)
    - Is a daemon thread (dies when main process exits)
    - Handles errors gracefully and continues running
    - Logs all cleanup activities
    
    Note:
        This function is called during FastAPI startup and should not be
        called multiple times as it creates additional threads.
    """

    def cleanup_loop():
        """
        Main cleanup loop that runs in the background thread.
        
        This inner function contains the actual cleanup logic that runs
        continuously. It calls cleanup_inactive_sessions() every hour and
        handles any exceptions that might occur during cleanup.
        """
        while True:
            try:
                cleanup_inactive_sessions()
                time.sleep(3600)  # Run cleanup every hour (3600 seconds)
            except Exception as e:
                logger.error(f"Error in cleanup scheduler: {e}")
                # Continue running even if cleanup fails
                time.sleep(3600)

    # Create and start the cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Session cleanup scheduler started")


# Initialize FastAPI application
app = FastAPI(
    title="Chatbot RAG",
    description="A simple chatbot using XAI. Enables asking questions and getting answers based on uploaded documents.",
    version="0.1",
)


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.
    
    This function is called automatically when the FastAPI application starts.
    It initializes background services and performs any necessary setup tasks.
    
    Current startup tasks:
    - Start the session cleanup scheduler
    - Log successful application startup
    
    Note:
        This is an async function that runs during FastAPI's startup sequence.
        Any blocking operations should be avoided here.
    """
    start_cleanup_scheduler()
    logger.info("FastAPI application started successfully")


@app.get("/")
def read_root()->Dict[str, str]:
    """
    Root endpoint for API health check and basic information.
    
    This endpoint provides basic information about the API service and
    can be used to verify that the API is running and accessible.
    
    Returns:
        Dict[str, str]: A dictionary containing:
            - service: Name of the service
            - description: Brief description of the API
            - status: Current status of the service
    
    Example:
        GET / HTTP/1.1
        
        Response:
        {
            "service": "RAG Chatbot using OPENAI",
            "description": "Welcome to Chatbot RAG API",
            "status": "running"
        }
    """
    return {
        "service": "RAG Chatbot using OPENAI",
        "description": "Welcome to Chatbot RAG API",
        "status": "running",
    }


@app.delete("/collection/{collection_id}")
def delete_collection_endpoint(collection_id: str) -> Dict[str, Any]:
    """
    Delete a specific vector collection from the database.
    
    This endpoint allows clients to delete a vector collection by its ID.
    This is useful for cleaning up collections that are no longer needed
    or for managing storage resources.
    
    Args:
        collection_id (str): The unique identifier of the collection to delete
    
    Returns:
        Dict[str, Any]: Response containing:
            - message: Descriptive message about the operation result
            - status: Either "success" or "not_found"
    
    Raises:
        HTTPException: If there's an error during deletion (status 500)
    
    Example:
        DELETE /collection/123e4567-e89b-12d3-a456-426614174000
        
        Success Response:
        {
            "message": "Collection 123e4567-e89b-12d3-a456-426614174000 deleted successfully",
            "status": "success"
        }
    """
    try:
        success = rag_pipeline.vector_store.delete_collection(collection_id)
        if success:
            return {
                "message": f"Collection {collection_id} deleted successfully",
                "status": "success",
            }
        else:
            return {
                "message": f"Collection {collection_id} does not exist",
                "status": "not_found",
            }
    except Exception as e:
        logger.error(f"Error deleting collection: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}, status: failed",
        )


@app.post("/feed")
async def upload_documents(
    files: List[UploadFile],
    filepaths: List[str] = Form(...),
    collection_id: str = Form(...),
) -> DocumentUploadResponse:
    """
    Upload and process PDF documents for RAG system.
    
    This endpoint accepts multiple PDF files, processes them into chunks,
    generates embeddings, and stores them in the specified vector collection.
    The documents are parsed, chunked, and converted to vectors for similarity search.
    
    Args:
        files (List[UploadFile]): List of uploaded PDF files
        filepaths (List[str]): List of file paths corresponding to the uploaded files
        collection_id (str): Target collection ID for storing the processed documents
    
    Returns:
        DocumentUploadResponse: Response containing:
            - num_docs: Number of documents processed
            - num_chunks: Total number of chunks created
            - status: Processing status ("Success" or error)
            - collection_id: The collection where documents were stored
            - files: List of processed filenames
            - files_path: List of file paths
    
    Raises:
        HTTPException: If file type is not PDF (status 400) or processing fails (status 500)
    
    Process Flow:
        1. Validate file types (PDF only)
        2. Read file content
        3. Process document into chunks using DocumentProcessor
        4. Convert chunks to vectors using vector store
        5. Store vectors in specified collection
        6. Return processing statistics
    
    Example:
        POST /feed
        Content-Type: multipart/form-data
        
        Form data:
        - files: [file1.pdf, file2.pdf]
        - filepaths: ["/path/to/file1.pdf", "/path/to/file2.pdf"]
        - collection_id: "123e4567-e89b-12d3-a456-426614174000"
    """
    try:
        num_documents = 0
        num_chunks = 0
        processed_files = []

        # Process each uploaded file
        for i, file in enumerate(files):
            # Validate file type - only PDF files are supported
            if file.content_type != "application/pdf":
                logger.error(f"Unsupported file type: {file.content_type}")
                raise HTTPException(status_code=400, detail="Only PDF files are supported")

            # Read file content
            content = await file.read()
            
            # Get filepath (use provided path or fallback to filename)
            current_filepath = filepaths[i] if i < len(filepaths) else file.filename
            
            # Process document: parse, chunk, and convert to vectors
            chunks = document_processor.process_document(content, current_filepath, file.content_type)
            chunks = rag_pipeline.vector_store.convert_chunks_to_vectors(chunks)
            
            # Store vectors in the specified collection
            num_chunks += len(chunks)
            rag_pipeline.vector_store.store_document(chunks, collection_id)
            num_documents += 1
            processed_files.append(file.filename)
            
        return {
            "num_docs": num_documents,
            "num_chunks": num_chunks,
            "status": "Success",
            "collection_id": collection_id,
            "files": [file.filename for file in files],
            "files_path": filepaths[: len(processed_files)],
        }
    except Exception as e:
        logger.error(f"Error uploading documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}, status: Failed",
        )


@app.post("/chat")
def chat_with_history(query: ChatQuery) -> ChatResponse:
    """
    Process a chat query with conversation history.
    
    This endpoint handles chat interactions by processing user questions
    through the RAG pipeline while maintaining conversation context.
    It supports both new sessions and continuing existing conversations.
    
    Args:
        query (ChatQuery): Chat request containing:
            - question: The user's question
            - collection_id: Vector collection to search in
            - session_id: Optional session ID for conversation continuity
    
    Returns:
        ChatResponse: Response containing:
            - question: The original question
            - answer: Generated answer from RAG pipeline
            - session_id: Session ID (existing or newly created)
            - chat_history: Updated conversation history
    
    Raises:
        Exception: If chat processing fails
    
    Process Flow:
        1. Generate or retrieve session ID
        2. Get/create session data with chat history
        3. Update collection_id if provided
        4. Process question through RAG pipeline with history
        5. Update session with new conversation state
        6. Return response with answer and updated history
    
    Session Management:
        - New sessions are created automatically if session_id is not provided
        - Session data includes messages, collection_id, timestamps
        - Last activity is updated on each interaction
        - Sessions are automatically cleaned up after SESSION_TIMEOUT_HOURS
    
    Example:
        POST /chat
        {
            "question": "What is the main topic of the document?",
            "collection_id": "123e4567-e89b-12d3-a456-426614174000",
            "session_id": "optional-session-id"
        }
    """
    try:
        # Generate session ID if not provided
        session_id = query.session_id or str(uuid.uuid4())

        # Get existing session data or create new session
        session_data = chat_sessions.get(
            session_id,
            {
                "messages": [],
                "collection_id": query.collection_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
            },
        )

        # Update collection_id if provided in the query
        if query.collection_id:
            session_data["collection_id"] = query.collection_id

        # Get existing chat history from session
        chat_history = session_data.get("messages", [])

        # Process question through RAG pipeline with conversation history
        answer, updated_history = rag_pipeline.ask_question_with_history(
            query.collection_id, query.question, chat_history
        )

        # Update session data with new conversation state
        session_data["messages"] = updated_history
        session_data["last_activity"] = datetime.now()
        chat_sessions[session_id] = session_data

        return ChatResponse(
            question=query.question,
            answer=answer,
            session_id=session_id,
            chat_history=updated_history,
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise f"fail to handle chat with history in session {session_id}: error: {str(e)}"


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str) -> Dict[str, Any]:
    """
    Retrieve chat history for a specific session.
    
    This endpoint allows clients to fetch the complete conversation history
    for a given session, including metadata like creation time and last activity.
    
    Args:
        session_id (str): The unique identifier of the session
    
    Returns:
        Dict[str, Any]: Session data containing:
            - session_id: The requested session ID
            - messages: List of chat messages in the conversation
            - collection_id: Associated vector collection ID
            - created_at: Session creation timestamp
            - last_activity: Last activity timestamp
    
    Note:
        If the session doesn't exist, returns empty values for all fields
        except session_id.
    
    Example:
        GET /chat/123e4567-e89b-12d3-a456-426614174000/history
        
        Response:
        {
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": "2023-..."},
                {"role": "assistant", "content": "Hi there!", "timestamp": "2023-..."}
            ],
            "collection_id": "collection-123",
            "created_at": "2023-...",
            "last_activity": "2023-..."
        }
    """
    session_data = chat_sessions.get(session_id, {})
    return {
        "session_id": session_id,
        "messages": session_data.get("messages", []),
        "collection_id": session_data.get("collection_id"),
        "created_at": session_data.get("created_at"),
        "last_activity": session_data.get("last_activity"),
    }


@app.delete("/chat/{session_id}")
def clear_chat_history(session_id: str) -> Dict[str, Any]:
    """
    Clear chat history for a specific session.
    
    This endpoint removes all chat messages from a session while preserving
    the session metadata (collection_id, timestamps, etc.). This is useful
    for starting a fresh conversation within the same session context.
    
    Args:
        session_id (str): The unique identifier of the session to clear
    
    Returns:
        Dict[str, Any]: Response message indicating success or failure:
            - message: Descriptive message about the operation result
    
    Note:
        - Only clears messages, session metadata is preserved
        - Updates last_activity timestamp to current time
        - Session remains active and can continue with new messages
    
    Example:
        DELETE /chat/123e4567-e89b-12d3-a456-426614174000
        
        Success Response:
        {"message": "Chat history for session 123e4567-e89b-12d3-a456-426614174000 cleared"}
        
        Not Found Response:
        {"message": "Session 123e4567-e89b-12d3-a456-426614174000 not found"}
    """
    if session_id in chat_sessions:
        # Clear only messages, keep session metadata
        chat_sessions[session_id]["messages"] = []
        chat_sessions[session_id]["last_activity"] = datetime.now()
        return {"message": f"Chat history for session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}


@app.delete("/session/{session_id}")
def delete_session_and_collection(session_id: str):
    """
    Delete both chat session and associated vector collection.
    
    This endpoint performs a complete cleanup of a session by removing both
    the chat session from memory and the associated vector collection from
    the database. This is useful for completely removing all traces of a
    conversation and its associated data.
    
    Args:
        session_id (str): The unique identifier of the session to delete
    
    Returns:
        Dict[str, Any]: Response containing:
            - message: Descriptive message about the operation result
            - session_cleared: Boolean indicating if session was cleared
            - collection_id: ID of the collection that was deleted (if any)
            - collection_deleted: Boolean indicating if collection was deleted
    
    Process Flow:
        1. Check if session exists in memory
        2. Extract collection_id from session data
        3. Delete session from memory
        4. Delete associated vector collection from database
        5. Log the results of both operations
    
    Error Handling:
        - Continues with session deletion even if collection deletion fails
        - Logs all errors for debugging
        - Returns detailed status information
    
    Example:
        DELETE /session/123e4567-e89b-12d3-a456-426614174000
        
        Response:
        {
            "message": "Session 123e4567-e89b-12d3-a456-426614174000 and associated data cleared",
            "session_cleared": true,
            "collection_id": "collection-123",
            "collection_deleted": true
        }
    """
    try:
        collection_id = None
        session_cleared = False

        if session_id in chat_sessions:
            session_data = chat_sessions[session_id]
            collection_id = session_data.get("collection_id")

            # Delete the session from memory
            del chat_sessions[session_id]
            session_cleared = True

            # Delete the associated vector collection if it exists
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
            "collection_deleted": collection_id is not None,
        }
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        return {
            "message": f"Error deleting session: {str(e)}",
            "session_cleared": False,
        }


@app.post("/cleanup/inactive")
def manual_cleanup_inactive_sessions():
    """
    Manually trigger cleanup of inactive sessions.
    
    This endpoint allows manual triggering of the session cleanup process,
    which normally runs automatically every hour. This is useful for:
    - Immediate cleanup when needed
    - Testing the cleanup functionality
    - Administrative maintenance
    
    Returns:
        Dict[str, str]: Response containing:
            - message: Descriptive message about the cleanup result
            - status: Either "success" or "error"
    
    Note:
        This performs the same cleanup as the automatic scheduler,
        removing sessions inactive for more than SESSION_TIMEOUT_HOURS.
    
    Example:
        POST /cleanup/inactive
        
        Success Response:
        {
            "message": "Inactive sessions cleanup completed",
            "status": "success"
        }
        
        Error Response:
        {
            "message": "Error during cleanup: [error details]",
            "status": "error"
        }
    """
    try:
        cleanup_inactive_sessions()
        return {"message": "Inactive sessions cleanup completed", "status": "success"}
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}", exc_info=True)
        return {"message": f"Error during cleanup: {str(e)}", "status": "error"}


@app.get("/sessions")
def list_active_sessions():
    """
    List all currently active sessions.
    
    This endpoint provides information about all active chat sessions,
    including their metadata and statistics. This is useful for:
    - Monitoring active sessions
    - Administrative oversight
    - Debugging session-related issues
    
    Returns:
        Dict[str, Any]: Response containing:
            - sessions: List of session information dictionaries
            - total: Total number of active sessions
            - error: Error message if listing failed (optional)
    
    Session Information includes:
        - session_id: Unique session identifier
        - collection_id: Associated vector collection ID
        - created_at: Session creation timestamp
        - last_activity: Last activity timestamp
        - message_count: Number of messages in the conversation
    
    Example:
        GET /sessions
        
        Response:
        {
            "sessions": [
                {
                    "session_id": "123e4567-e89b-12d3-a456-426614174000",
                    "collection_id": "collection-123",
                    "created_at": "2023-...",
                    "last_activity": "2023-...",
                    "message_count": 5
                }
            ],
            "total": 1
        }
    """
    try:
        sessions_info = []
        for session_id, session_data in chat_sessions.items():
            sessions_info.append(
                {
                    "session_id": session_id,
                    "collection_id": session_data.get("collection_id"),
                    "created_at": session_data.get("created_at"),
                    "last_activity": session_data.get("last_activity"),
                    "message_count": len(session_data.get("messages", [])),
                }
            )
        return {"sessions": sessions_info, "total": len(sessions_info)}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        return {"sessions": [], "total": 0, "error": str(e)}


if __name__ == "__main__":
    """
    Main entry point for running the FastAPI application.
    
    This block is executed when the script is run directly (not imported).
    It starts the cleanup scheduler and runs the FastAPI server using uvicorn.
    
    Configuration:
        - HOST: Server host address (from environment or default "0.0.0.0")
        - FASTAPI_PORT: Server port (from environment or default 8000)
    
    Note:
        The cleanup scheduler is started here for direct execution,
        but it's also started in the startup event for production deployment.
    """
    # Start the cleanup scheduler
    start_cleanup_scheduler()
    
    # Run the FastAPI application with uvicorn
    uvicorn.run(
        app,
        host=HOST,
        port=FASTAPI_PORT,
    )
