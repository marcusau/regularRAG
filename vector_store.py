
import os
import uuid
import chromadb
from typing import List,Union
from dotenv import load_dotenv

from preprocess import parse_document, chunk_document
from models import ChunkModel,DBSearchModel
from providers import get_embedding

load_dotenv(override=True)

CHROMADB_LOCAL_DIR = os.environ.get("CHROMADB_LOCAL_DIR")

# 1 convert_chunks_to_vector
# 2. add_vectors_to_db


def convert_chunks_to_vector(chunks:List[ChunkModel])->List[ChunkModel]:
    if not isinstance(chunks,list) :
        raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    if len(chunks) == 0 :
        raise Exception(f"Chunk to vector conversion does not accept empty list")
    
    if not all([isinstance(chunk,ChunkModel) for chunk in chunks]):
        raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    chunks_text = [chunk.page_content for chunk in chunks]
    try:
        print(f"converting chunks into embedding")
        vectors = get_embedding(chunks_text)
        print(f"finished embedding conversion")
    except Exception as e:
        raise f"Cannot convert Chunks into vector with error: {e}"
    
    for vector,chunk in zip(vectors,chunks):
        chunk.vector = vector
        
    return chunks

def store_document(chunks: List[ChunkModel],collection_name:str)->None:
    
    client = chromadb.PersistentClient(path=CHROMADB_LOCAL_DIR)
    
    collection = client.get_or_create_collection(name=collection_name)
    
    if not isinstance(chunks,list):
        raise TypeError(f"store_document function only accepts input as list of ChunkModel")
    
    if len(chunks) == 0:
        raise Exception(f"store_document function cannot accept empty list")
    
    if not any([isinstance(chunk,ChunkModel) for chunk in chunks]):
         raise TypeError(f"store_document function only accepts input as list of ChunkModel")    
    
    print(f"add vectors to chromadb with collection : {collection_name}")
    collection.add(
                embeddings=[chunk.vector for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                documents=[chunk.page_content for chunk in chunks],
            ids=[chunk.metadata['index'] for chunk in chunks],)
    print(f"finished adding vector to chromadb")
    
def get_db_collection(collection_name:str):
    client = chromadb.PersistentClient(path=CHROMADB_LOCAL_DIR)
    return client.get_or_create_collection(name=collection_name)

def retrieve_document(collection_id:str,query:Union[str,List[str]],topk:int=10)->List[DBSearchModel]:
    
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
        raise Exception(f"No result searched from vector DB with query: {query}")

    retrived_docs=results['documents']
    retrived_metadata=results['metadatas']
    retrived_distances=results['distances']
    search_items=[]
    for q_idx, q in enumerate(query if isinstance(query,list) else [query]):
        for idx,(text,meta,dist) in enumerate(zip(retrived_docs[q_idx],retrived_metadata[q_idx],retrived_distances[q_idx])):
            search_item = {"id":idx,"text":text,"metadata":meta,"distance":dist}
            search_items.append(DBSearchModel(**search_item))
    return search_items

def delete_collection(collection_id: str) -> bool:
    """Delete a collection from ChromaDB"""
    try:
        client = chromadb.PersistentClient(path=CHROMADB_LOCAL_DIR)
        
        # Check if collection exists before trying to delete
        try:
            collection = client.get_collection(name=collection_id)
            client.delete_collection(name=collection_id)
            print(f"Collection {collection_id} deleted successfully")
            return True
        except Exception as e:
            if "does not exist" in str(e).lower():
                print(f"Collection {collection_id} does not exist")
                return False
            else:
                raise e
    except Exception as e:
        print(f"Error deleting collection {collection_id}: {e}")
        return False


if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    collection_id=str(uuid.uuid4())  
    print(f"collection_id:{collection_id}")
    docs = parse_document(filepath)
    chunks = chunk_document(docs)
    chunks = convert_chunks_to_vector(chunks)
    store_document(chunks,collection_id)