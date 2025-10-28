"""
Core data models for the RAG system.

This module defines the fundamental data structures used throughout the RAG system,
including document models, chunk models, retrieval results, and reranking components.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator


class DocumentModel(BaseModel):
    """
    Model representing a processed document with metadata.
    
    This model captures essential information about a document that has been
    processed by the RAG system, including file details, processing status,
    and content information.
    
    Attributes:
        file (str): Name of the document file
        extension (str): File format/extension (e.g., 'pdf', 'txt', 'docx')
        file_path (str): Full path to the document file
        status (str): Processing status of the document (e.g., 'processed', 'failed')
        length (int): Length of the document content in characters
        content (str): The actual text content extracted from the document
    """
    file: str = Field(description="File name of the document")
    extension: str = Field(description="Format of the document")
    file_path: str = Field(description="File path of the document")
    status: str = Field(description="Status of the document")
    length: int = Field(description="Length of the document")
    content: str = Field(description="Content of the document")


class ChunkModel(Document):
    """
    Extended Document class that inherits from LangChain Document and adds vector storage.
    
    This model represents a text chunk (segment of a document) that has been processed
    for use in the RAG system. It extends LangChain's Document class to include
    vector embeddings for similarity search and retrieval.
    
    Attributes:
        vector (Optional[List[float]]): Vector embedding for the document chunk, used for similarity search
        page_content (str): The text content of the chunk (inherited from Document)
        metadata (Dict[str, Any]): Additional metadata about the chunk (inherited from Document)
    """

    vector: Optional[List[float]] = Field(default_factory=list, description="Vector embedding for the document chunk")

    def __init__(
        self,
        page_content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize a ChunkModel with content, vector, and metadata.
        
        Args:
            page_content (str): The text content of the chunk
            vector (Optional[List[float]]): Vector embedding for the chunk. Defaults to empty list.
            metadata (Optional[Dict[str, Any]]): Additional metadata. Defaults to empty dict.
            **kwargs: Additional arguments passed to the parent Document class
        """
        super().__init__(page_content=page_content, metadata=metadata or {}, **kwargs)
        self.vector = vector or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary including the vector.
        
        Returns:
            Dict[str, Any]: Dictionary representation including vector, content, and metadata
        """
        doc_dict = super().to_dict()
        doc_dict["vector"] = self.vector
        return doc_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkModel":
        """
        Create a ChunkModel from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing chunk data including vector
            
        Returns:
            ChunkModel: New ChunkModel instance created from the dictionary data
        """
        vector = data.pop("vector", [])
        doc = super().from_dict(data)
        doc.vector = vector
        return doc


class DBRetrieveModel(BaseModel):
    """
    Model representing a document chunk retrieved from the vector database.
    
    This model contains the results of a similarity search operation, including
    the retrieved chunk content, metadata, and similarity score.
    
    Attributes:
        id (int): Unique identifier/index of the chunk in the database
        text (str): The text content of the retrieved chunk
        metadata (Dict[str, Any]): Additional metadata associated with the chunk
        distance (float): Similarity distance score (lower = more similar)
    """
    id: int = Field(description="index of the chunk")
    text: str = Field(description="content of the chunk")
    metadata: Dict[str, Any] = Field(description="metadata of the chunk")
    distance: float = Field(description="distance of the chunk")


class ReRankerLabel(BaseModel):
    """
    Model representing a single chunk evaluation from the reranking process.
    
    This model captures the reranking system's evaluation of a document chunk,
    including the reasoning process and relevance score.
    
    Attributes:
        chunk_id (int): Unique identifier of the text chunk being evaluated
        chain_of_thought (str): The reasoning process used to evaluate relevance
        relevancy (int): Relevance score from 0 to 10, where 10 is most relevant
        text (str): The actual text content of the chunk being evaluated
    """
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(description="The reasoning process used to evaluate the relevance")
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )
    text: str = Field(description="The text of the chunk")


class RerankerModel(BaseModel):
    """
    Model containing the complete reranking results for a set of document chunks.
    
    This model holds the results of the reranking process, including all evaluated
    chunks with their relevance scores. The labels are automatically sorted by
    relevance score in descending order (most relevant first).
    
    Attributes:
        labels (list[ReRankerLabel]): List of labeled and ranked chunks, sorted by relevance
        
    Note:
        The field_validator automatically sorts labels by relevancy score in descending order
    """
    labels: list[ReRankerLabel] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[ReRankerLabel]) -> list[ReRankerLabel]:
        """
        Validate and sort the reranking labels by relevance score.
        
        This validator ensures that the labels are automatically sorted by
        relevancy score in descending order (most relevant first).
        
        Args:
            v (list[ReRankerLabel]): List of reranking labels to validate and sort
            
        Returns:
            list[ReRankerLabel]: Sorted list of labels with highest relevance first
        """
        return sorted(v, key=lambda x: x.relevancy, reverse=True)
