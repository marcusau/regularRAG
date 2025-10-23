from pydantic import BaseModel
from typing import List
from models import DBSearchModel


class Query(BaseModel):
    question: str
    collection_id: str
 

class DocumentResponse(BaseModel):
    documents: List[DBSearchModel]
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