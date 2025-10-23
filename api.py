import os
import uvicorn
from fastapi import FastAPI, UploadFile,Form

from pydantic import BaseModel
from typing import List,Dict
import logging
import pymupdf
from markdownify import markdownify

from fastapi_models import DocumentResponse,DocumentUploadResponse,AskResponse,Query,DocumentUploadModel
from models import DocumentModel
from providers import get_llm
from preprocess  import parse_document, chunk_document
from vector_store import convert_chunks_to_vector, store_document, get_db_collection, retrieve_document, delete_collection
from rag_processor import ask_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = get_llm()


app = FastAPI(
    title="Chatbot RAG",
    description="A simple chatbot using OpenAI. Enables asking questions and getting answers based on uploaded documents.",
    version="0.1",
)

@app.get("/")
def read_root():
    return {
        "service": "RAG Chatbot using OPENAI",
        "description": "Welcome to Chatbot RAG API",
        "status": "running",
    }
    
@app.post("/documents")
def search_documents(query: Query)-> DocumentResponse:
    try:
        question = query.question
        collection_id = query.collection_id
        documents = retrieve_document(collection_id,question)
        return {"documents": documents, "total": len(documents), "query": question }
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        raise f"error: {str(e)},  total: {0}, query: {question }"    
    
    
@app.post("/feed")
async def upload_documents(files: List[UploadFile], filepaths: List[str] = Form(...), collection_id: str = Form(...)) -> DocumentUploadResponse:
    try:
        num_documents = 0
        num_chunks = 0
        processed_files = []
        
        for i, file in enumerate(files):
            if file.content_type != "application/pdf":
                logger.error(f"Unsupported file type: {file.content_type}")
                raise ValueError("Only PDF files are supported")
            
            content = await file.read()
            # Parse PDF using PyMuPDF
            
            current_filepath = filepaths[i] if i < len(filepaths) else file.filename
            document = parse_document(content, current_filepath, filetype=file.content_type)
            chunks = chunk_document(document)
            # for chunk in chunks:
            #     print(f"chunk: {chunk.metadata}\n")
            chunks = convert_chunks_to_vector(chunks)
            num_chunks += len(chunks)
            store_document(chunks,collection_id)
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
        raise f"error: {str(e)}, num_docs: {0}, num_chunks: {0}, status: Failed"

@app.post("/ask")
def ask(query: Query) -> AskResponse:
    try:
        question = query.question
        collection_id = query.collection_id
        answer = ask_question(model,collection_id,question)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"Error asking question: {e}", exc_info=True)
        raise f"error: {str(e)}, question: {question}"

@app.delete("/collection/{collection_id}")
def delete_collection_endpoint(collection_id: str):
    try:
        success = delete_collection(collection_id)
        if success:
            return {"message": f"Collection {collection_id} deleted successfully", "status": "success"}
        else:
            return {"message": f"Collection {collection_id} does not exist", "status": "not_found"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}", exc_info=True)
         raise f"error: {str(e)}, status: failed"

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000,)