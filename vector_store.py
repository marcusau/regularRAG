
import logging
import os
import uuid
from typing import List, Union

from dotenv import load_dotenv

import chromadb
from models import ChunkModel, DBRetrieveModel
from preprocess import chunk_document, parse_document
from providers import get_embedding

load_dotenv(override=True)

CHROMADB_LOCAL_DIR = os.environ.get("CHROMADB_LOCAL_DIR")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 1 convert_chunks_to_vector
# 2. add_vectors_to_db


def convert_vector(chunks:List[ChunkModel])->List[ChunkModel]:
    if not isinstance(chunks,list) :
        logger.error(f"Chunk to vector conversion must start with input as list of ChunkModel")
        raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    if len(chunks) == 0 :
        logger.error(f"Chunk to vector conversion does not accept empty list")
        raise Exception(f"Chunk to vector conversion does not accept empty list")
    
    if not all([isinstance(chunk,ChunkModel) for chunk in chunks]):
        logger.error(f"Chunk to vector conversion must start with input as list of ChunkModel")
        raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    chunks_text = [chunk.page_content for chunk in chunks]
    logger.info(f"converting chunks into embedding")
    try:
        vectors = get_embedding(chunks_text)
        logger.info(f"finished embedding conversion")
    except Exception as e:
        logger.error(f"Cannot convert Chunks into vector with error: {e}")
        raise Exception(f"Cannot convert Chunks into vector with error: {e}")
    
    for vector,chunk in zip(vectors,chunks):
        chunk.vector = vector
        
    return chunks

    
def get_db_collection(collection_name:str):
    if not isinstance(collection_name,str):
        logger.error(f"collection_name must be a string")
        raise TypeError(f"collection_name must be a string")
    if len(collection_name) == 0:
        logger.error(f"collection_name cannot be empty")
        raise Exception(f"collection_name cannot be empty")
    
    client = chromadb.PersistentClient(path=CHROMADB_LOCAL_DIR)
    return client.get_or_create_collection(name=collection_name)

def store_document(chunks: List[ChunkModel],collection_name:str)->None:
    
    if not isinstance(chunks,list):
        logger.error(f"store_document function only accepts input as list of ChunkModel")
        raise TypeError(f"store_document function only accepts input as list of ChunkModel")
    
    if len(chunks) == 0:
        logger.error(f"store_document function cannot accept empty list")
        raise Exception(f"store_document function cannot accept empty list")
    
    if not any([isinstance(chunk,ChunkModel) for chunk in chunks]):
        logger.error(f"store_document function only accepts input as list of ChunkModel")
        raise TypeError(f"store_document function only accepts input as list of ChunkModel")    
    
    logger.info(f"add vectors to chromadb with collection : {collection_name}")
    collection = get_db_collection(collection_name)
    if collection is None:
        logger.error(f"no such collection :{collection_name} inside the chromadb")
        raise Exception(f"no such collection :{collection_name} inside the chromadb")
    
    collection.add(
                embeddings=[chunk.vector for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                documents=[chunk.page_content for chunk in chunks],
            ids=[chunk.metadata['index'] for chunk in chunks],)
    logger.info(f"finished adding vector to chromadb")


def retrieve_document(collection_id:str,query:Union[str,List[str]],topk:int=10)->List[DBRetrieveModel]:
    
    if not isinstance(query,str) and not isinstance(query,list) or not all(isinstance(item,str) for item in query):
        raise TypeError(f"query must be a string or a list of strings")
    
    print(f"convert query into embedding")
    query_emb = get_embedding(query)
    
    print(f"start collection")
    collection = get_db_collection(collection_id)
    if collection is None:
        raise Exception(f"no such collection :{collection_id} inside the chromadb")
    
    print(f"fetch topk result from collection")
    results = collection.query(query_embeddings=query_emb, n_results=topk, )

    if results is None or len(results)==0:
        logger.error(f"No result searched from vector DB with query: {query if isinstance(query,str) else ",".join(query)}")
        raise Exception(f"No result searched from vector DB with query: {query if isinstance(query,str) else ",".join(query)}")

    retrived_docs=results['documents']
    retrived_metadata=results['metadatas']
    retrived_distances=results['distances']
    search_items=[]
    for q_idx, q in enumerate(query if isinstance(query,list) else [query]):
        for idx,(text,meta,dist) in enumerate(zip(retrived_docs[q_idx],retrived_metadata[q_idx],retrived_distances[q_idx])):
            search_item = {"id":idx,"text":text,"metadata":meta,"distance":dist}
            db_retrieve_result = DBRetrieveModel(**search_item)
            search_items.append(db_retrieve_result)
    return search_items

def delete_collection(collection_id: str) -> bool:
    """Delete a collection from ChromaDB"""
    try:
        client = chromadb.PersistentClient(path=CHROMADB_LOCAL_DIR)
        
        # Check if collection exists before trying to delete
        try:
            collection = client.get_collection(name=collection_id)
            client.delete_collection(name=collection_id)
            logger.info(f"Collection {collection_id} deleted successfully")
            return True
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.error(f"Collection {collection_id} does not exist or already deleted for deletion")
                return False
            else:
                logger.error(f"Error deleting collection {collection_id}: {e}")
                raise Exception(f"Error deleting collection {collection_id}: {e}")
    except Exception as e:
            logger.error(f"Error deleting collection {collection_id}: {e}")
            raise Exception(f"Error deleting collection {collection_id}: {e}")
          


if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    collection_id=str(uuid.uuid4())  
    print(f"collection_id:{collection_id}")
    docs = parse_document(filepath)
    chunks = chunk_document(docs)
    chunks = convert_vector(chunks)
    store_document(chunks,collection_id)