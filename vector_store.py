
import logging
import os
import uuid
from typing import List, Union, Optional, Callable, Dict, Any

from dotenv import load_dotenv

import chromadb
from models import ChunkModel, DBRetrieveModel
from preprocess import DocumentProcessor, DocumentParser, DocumentChunker # parse_document
from providers import get_embedding

load_dotenv(override=True)

CHROMADB_LOCAL_DIR = os.environ.get("CHROMADB_LOCAL_DIR")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

document_parser = DocumentParser()
# 1 convert_chunks_to_vector
# 2. add_vectors_to_db
class VectorStore:
    def __init__(self, chromadb_path: Optional[str] = None, embed_func: Optional[Callable] = None) -> None:
        self.chromadb_path: str = chromadb_path or os.environ.get("CHROMADB_LOCAL_DIR")
        if not self.chromadb_path:
            raise ValueError("CHROMADB_LOCAL_DIR environment variable must be set or chromadb_path must be provided")
        
        self.client: chromadb.PersistentClient = chromadb.PersistentClient(path=self.chromadb_path)
        self.embedding_func: Callable = embed_func or get_embedding
        
    def _get_collection(self, collection_name: str) -> Any:
        if not isinstance(collection_name, str):
            self.logger.error(f"collection_name must be a string")
            raise TypeError(f"collection_name must be a string")
        
        if len(collection_name) == 0:
            self.logger.error(f"collection_name cannot be empty")
            raise Exception(f"collection_name cannot be empty")
        
        return self.client.get_or_create_collection(name=collection_name)
        
    def convert_chunks_to_vectors(self, chunks: List[ChunkModel]) -> List[ChunkModel]:
        self._validate_chunks(chunks)
        chunks_text = [chunk.page_content for chunk in chunks]
        logger.info(f"converting chunks into embedding")
        try:
            vectors = self.embedding_func(chunks_text)
            logger.info(f"finished embedding conversion")
        except Exception as e:
            logger.error(f"Cannot convert Chunks into vector with error: {e}")
            raise Exception(f"Cannot convert Chunks into vector with error: {e}")
        
        for vector, chunk in zip(vectors, chunks):
            chunk.vector = vector
            
        return chunks
    
    def store_document(self, chunks: List[ChunkModel], collection_name: str) -> None:
        self._validate_chunks(chunks)
        collection = self._get_collection(collection_name=collection_name)
        collection.add(
            embeddings=[chunk.vector for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            documents=[chunk.page_content for chunk in chunks],
            ids=[chunk.metadata['index'] for chunk in chunks],
        )
        logger.info(f"finished adding vector to chromadb")
        
    def retrieve_document(self, collection_name: str, query: Union[str, List[str]], topk: int = 10) -> List[DBRetrieveModel]:
        self._validate_query(query)
        logger.info(f"convert query into embedding")
        query_emb = self.embedding_func(query)
        
        collection = self._get_collection(collection_name)
        
        results = collection.query(query_embeddings=query_emb, n_results=topk)

        if results is None or len(results) == 0:
            logger.error(f"No result searched from vector DB with query: {query if isinstance(query, str) else ','.join(query)}")
            raise Exception(f"No result searched from vector DB with query: {query if isinstance(query, str) else ','.join(query)}")

        retrived_docs = results['documents']
        retrived_metadata = results['metadatas']
        retrived_distances = results['distances']
        search_items = []
        for q_idx, q in enumerate(query if isinstance(query, list) else [query]):
            for idx, (text, meta, dist) in enumerate(zip(retrived_docs[q_idx], retrived_metadata[q_idx], retrived_distances[q_idx])):
                search_item = {"id": idx, "text": text, "metadata": meta, "distance": dist}
                db_retrieve_result = DBRetrieveModel(**search_item)
                search_items.append(db_retrieve_result)
        return search_items
        
    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection {collection_id} deleted successfully")
            return True
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.error(f"Collection {collection_name} does not exist or already deleted for deletion")
                return False
            else:
                logger.error(f"Error deleting collection {collection_name}: {e}")
                raise Exception(f"Error deleting collection {collection_name}: {e}")

    def _validate_chunks(self, chunks: List[ChunkModel]) -> None:
        if not isinstance(chunks, list):
            logger.error(f"Chunk to vector conversion must start with input as list of ChunkModel")
            raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
        
        if len(chunks) == 0:
            logger.error(f"Chunk to vector conversion does not accept empty list")
            raise Exception(f"Chunk to vector conversion does not accept empty list")
        
        if not all([isinstance(chunk, ChunkModel) for chunk in chunks]):
            logger.error(f"Chunk to vector conversion must start with input as list of ChunkModel")
            raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    def _validate_query(self, query: Union[str, List[str]]) -> None:
        if not isinstance(query, str) and not isinstance(query, list) or not all(isinstance(item, str) for item in query):
            raise TypeError(f"query must be a string or a list of strings")
    
    # =============================================================================
    # ADDITIONAL COLLECTION MANAGEMENT METHODS
    # =============================================================================
    
    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            raise Exception(f"Error listing collections: {e}")
    
    def collection_exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(name=collection_name)
            return True
        except Exception:
            return False
    
if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    collection_id=str(uuid.uuid4())  
    print(f"collection_id:{collection_id}")
    # docs = parse_document(filepath)
    # chunks = chunk_document(docs)
    # chunks = convert_vector(chunks)
    # store_document(chunks,collection_id)