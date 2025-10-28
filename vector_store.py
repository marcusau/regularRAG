"""Vector store utilities backed by ChromaDB.

This module exposes a `VectorStore` class that handles:
- Converting text chunks into embeddings using a pluggable embedding function
- Persisting embeddings/documents/metadata into a Chroma persistent collection
- Query-time retrieval using embedding similarity
- Basic collection management helpers

The implementation assumes a local persistent ChromaDB path is available in
the `CHROMADB_LOCAL_DIR` environment variable unless a path is provided.
"""

import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv

import chromadb
from models import ChunkModel, DBRetrieveModel
from preprocess import DocumentParser  # parse_document
from providers import get_embedding

load_dotenv()

CHROMADB_LOCAL_DIR = os.environ.get("CHROMADB_LOCAL_DIR")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

document_parser = DocumentParser()


# 1 convert_chunks_to_vector
# 2. add_vectors_to_db
class VectorStore:
    """High-level wrapper around a persistent Chroma collection.

    Responsibilities:
    - Manage a Chroma persistent client rooted at a local directory
    - Provide helpers to embed text chunks and store/query them
    - Offer minimal collection management helpers

    Parameters
    ----------
    chromadb_path: Optional[str]
        Filesystem path to the Chroma persistent directory. If not provided,
        `CHROMADB_LOCAL_DIR` is used.
    embed_func: Optional[Callable]
        A callable that takes a list[str] and returns a list[vector] embeddings.
        If omitted, defaults to `providers.get_embedding`.
    """
    def __init__(self, chromadb_path: Optional[str] = None, embed_func: Optional[Callable] = None) -> None:
        """Initialize the persistent client and embedding function.

        Raises
        ------
        ValueError
            If neither `chromadb_path` nor `CHROMADB_LOCAL_DIR` is available.
        """
        self.chromadb_path: str = chromadb_path or os.environ.get("CHROMADB_LOCAL_DIR")
        if not self.chromadb_path:
            raise ValueError("CHROMADB_LOCAL_DIR environment variable must be set or chromadb_path must be provided")

        self.client: chromadb.PersistentClient = chromadb.PersistentClient(path=self.chromadb_path)
        self.embedding_func: Callable = embed_func or get_embedding

    def _get_collection(self, collection_name: str) -> Any:
        """Fetch or create a Chroma collection by name.

        Parameters
        ----------
        collection_name: str
            Identifier for the collection.

        Returns
        -------
        Any
            The Chroma collection instance.

        Raises
        ------
        TypeError
            If `collection_name` is not a string.
        Exception
            If `collection_name` is empty.
        """
        if not isinstance(collection_name, str):
            self.logger.error(f"collection_name must be a string")
            raise TypeError(f"collection_name must be a string")

        if len(collection_name) == 0:
            self.logger.error(f"collection_name cannot be empty")
            raise Exception(f"collection_name cannot be empty")

        return self.client.get_or_create_collection(name=collection_name)

    def convert_chunks_to_vectors(self, chunks: List[ChunkModel]) -> List[ChunkModel]:
        """Embed each chunk's text and attach the resulting vector to the chunk.

        Parameters
        ----------
        chunks: list[ChunkModel]
            Chunk objects with `page_content` populated.

        Returns
        -------
        list[ChunkModel]
            The same chunks with `vector` set.

        Raises
        ------
        TypeError, Exception
            If validation fails or embedding conversion errors.
        """
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
        """Persist chunk embeddings, documents, and metadata into a collection.

        Notes
        -----
        - Expects `chunks` to already have `vector` populated (use
          `convert_chunks_to_vectors` beforehand).
        - Uses `chunk.metadata["index"]` as the persistent id.
        """
        self._validate_chunks(chunks)
        collection = self._get_collection(collection_name=collection_name)
        collection.add(
            embeddings=[chunk.vector for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            documents=[chunk.page_content for chunk in chunks],
            ids=[chunk.metadata["index"] for chunk in chunks],
        )
        logger.info(f"finished adding vector to chromadb")

    def retrieve_document(
        self, collection_name: str, query: Union[str, List[str]], topk: int = 10
    ) -> List[DBRetrieveModel]:
        """Embed the input query and retrieve the top-k most similar documents.

        Parameters
        ----------
        collection_name: str
            Name of the Chroma collection to query.
        query: str | list[str]
            Query text or a list of query texts.
        topk: int
            Number of results to return per query.

        Returns
        -------
        list[DBRetrieveModel]
            Flattened list of retrieved items across all queries, containing
            text, metadata, and distance.

        Raises
        ------
        TypeError
            If `query` type is invalid.
        Exception
            If no results are returned from Chroma.
        """
        self._validate_query(query)
        logger.info(f"convert query into embedding")
        query_emb = self.embedding_func(query)

        collection = self._get_collection(collection_name)

        results = collection.query(query_embeddings=query_emb, n_results=topk)

        if results is None or len(results) == 0:
            logger.error(
                f"No result searched from vector DB with query: {query if isinstance(query, str) else ','.join(query)}"
            )
            raise Exception(
                f"No result searched from vector DB with query: {query if isinstance(query, str) else ','.join(query)}"
            )

        retrived_docs = results["documents"]
        retrived_metadata = results["metadatas"]
        retrived_distances = results["distances"]
        search_items = []
        for q_idx, q in enumerate(query if isinstance(query, list) else [query]):
            for idx, (text, meta, dist) in enumerate(
                zip(retrived_docs[q_idx], retrived_metadata[q_idx], retrived_distances[q_idx])
            ):
                search_item = {"id": idx, "text": text, "metadata": meta, "distance": dist}
                db_retrieve_result = DBRetrieveModel(**search_item)
                search_items.append(db_retrieve_result)
        return search_items

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection if it exists.

        Parameters
        ----------
        collection_name: str
            Name of the collection to delete.

        Returns
        -------
        bool
            True if deletion succeeded, False if the collection did not exist.

        Raises
        ------
        Exception
            For unexpected errors from the Chroma client.
        """
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
        """Ensure chunks is a non-empty list of `ChunkModel`.

        Raises
        ------
        TypeError
            If `chunks` is not a list or contains non-`ChunkModel` elements.
        Exception
            If the list is empty.
        """
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
        """Validate that a query is a string or list of strings."""
        if (
            not isinstance(query, str)
            and not isinstance(query, list)
            or not all(isinstance(item, str) for item in query)
        ):
            raise TypeError(f"query must be a string or a list of strings")

    # =============================================================================
    # ADDITIONAL COLLECTION MANAGEMENT METHODS
    # =============================================================================

    def list_collections(self) -> List[str]:
        """List existing Chroma collections by name."""
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            raise Exception(f"Error listing collections: {e}")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists by attempting to fetch it."""
        try:
            self.client.get_collection(name=collection_name)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    collection_id = str(uuid.uuid4())
    print(f"collection_id:{collection_id}")
    # docs = parse_document(filepath)
    # chunks = chunk_document(docs)
    # chunks = convert_vector(chunks)
    # store_document(chunks,collection_id)
