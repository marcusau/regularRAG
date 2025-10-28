"""
FastAPI Pydantic models for the RAG system API endpoints.

This module defines all the request and response models used by the FastAPI endpoints
for document processing, querying, and chat functionality in the RAG system.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel

from models import DBRetrieveModel


class Query(BaseModel):
    """
    Request model for document query operations.
    
    Used when users want to search for relevant documents in a specific collection.
    
    Attributes:
        question (str): The user's search query/question
        collection_id (str): Unique identifier for the document collection to search in
    """
    question: str
    collection_id: str


class DocumentResponse(BaseModel):
    """
    Response model for document retrieval operations.
    
    Contains the results of a document search query along with metadata about the search.
    
    Attributes:
        documents (List[DBRetrieveModel]): List of retrieved document chunks with metadata
        total (int): Total number of documents found
        query (str): The original search query that was processed
        error (str, optional): Error message if the operation failed, None if successful
    """
    documents: List[DBRetrieveModel]
    total: int
    query: str
    error: str = None


class DocumentUploadModel(BaseModel):
    """
    Request model for document upload operations.
    
    Used when users want to upload and process new documents into the system.
    
    Attributes:
        files (List[str]): List of file paths or file names to be uploaded and processed
    """
    files: List[str]


class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload operations.
    
    Provides feedback about the document processing results including statistics
    and any errors that occurred during processing.
    
    Attributes:
        files (List[str]): List of files that were processed
        num_docs (int): Number of documents successfully processed
        num_chunks (int): Total number of text chunks created from the documents
        status (str): Processing status (e.g., "success", "failed", "partial")
        collection_id (str, optional): Unique identifier for the created collection
        error (str, optional): Error message if processing failed, None if successful
    """
    files: List[str]
    num_docs: int
    num_chunks: int
    status: str
    collection_id: str = None
    error: str = None


class AskResponse(BaseModel):
    """
    Response model for simple question-answering operations.
    
    Used for direct Q&A without chat history context.
    
    Attributes:
        question (str): The user's original question
        answer (str): The system's generated answer based on retrieved documents
        error (str, optional): Error message if the operation failed, None if successful
    """
    question: str
    answer: str
    error: str = None


class ChatMessage(BaseModel):
    """
    Model representing a single message in a chat conversation.
    
    Each message has a role (user or assistant), content, and timestamp for
    maintaining conversation context and history.
    
    Attributes:
        role (Literal["user", "assistant"]): The sender of the message - either "user" or "assistant"
        content (str): The actual text content of the message
        timestamp (datetime): When the message was created/sent
    """
    role: Literal["user", "assistant"]  # "user" or "assistant"
    content: str
    timestamp: datetime


class ChatHistory(BaseModel):
    """
    Model for storing and managing chat conversation history.
    
    Groups messages by session ID to maintain conversation context across
    multiple interactions.
    
    Attributes:
        session_id (str): Unique identifier for the chat session
        messages (List[ChatMessage]): Chronologically ordered list of messages in the conversation
    """
    session_id: str
    messages: List[ChatMessage]


class ChatQuery(BaseModel):
    """
    Request model for chat-based question-answering operations.
    
    Supports both new conversations and continuing existing ones with chat history.
    
    Attributes:
        question (str): The user's current question
        collection_id (str): Unique identifier for the document collection to search in
        session_id (Optional[str]): Existing session ID to continue conversation, None for new sessions
        chat_history (Optional[List[ChatMessage]]): Previous conversation history for context
    """
    question: str
    collection_id: str
    session_id: Optional[str] = None  # For new sessions
    chat_history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    """
    Response model for chat-based question-answering operations.
    
    Returns the answer along with updated chat history and session information.
    
    Attributes:
        question (str): The user's original question
        answer (str): The system's generated answer based on retrieved documents and chat context
        session_id (str): Unique identifier for the chat session
        chat_history (List[ChatMessage]): Updated conversation history including the new exchange
        error (str, optional): Error message if the operation failed, None if successful
    """
    question: str
    answer: str
    session_id: str
    chat_history: List[ChatMessage]
    error: str = None
