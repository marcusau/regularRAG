from pydantic import BaseModel
from typing import List, Literal, Optional
from models import DBRetrieveModel
from datetime import datetime

class Query(BaseModel):
    question: str
    collection_id: str
 
class DocumentResponse(BaseModel):
    documents: List[DBRetrieveModel]
    total: int
    query: str
    error: str = None

class DocumentUploadModel(BaseModel):
    files: List[str]

class DocumentUploadResponse(BaseModel):
    files: List[str]
    num_docs: int
    num_chunks : int
    status: str    
    collection_id: str = None
    error: str = None

class AskResponse(BaseModel):
    question: str
    answer: str
    error: str = None
    
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]# "user" or "assistant"
    content: str
    timestamp: datetime

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    
class ChatQuery(BaseModel):
    question: str
    collection_id: str
    session_id: Optional[str] = None  # For new sessions
    chat_history: Optional[List[ChatMessage]] = None
    
class ChatResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    chat_history: List[ChatMessage]
    error: str = None