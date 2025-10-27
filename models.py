from pydantic import BaseModel, Field, field_validator
from typing import List,Optional,Any,Dict
from langchain_core.documents import Document

class DocumentModel(BaseModel):
    file: str = Field(description="File name of the document")
    extension: str = Field(description="Format of the document")
    file_path: str = Field(description="File path of the document")
    status: str = Field(description="Status of the document")
    length: int = Field(description="Length of the document")
    content: str = Field(description="Content of the document")


class ChunkModel(Document):
    """
    Extended Document class that inherits from LangChain Document
    and adds a vector attribute for storing embeddings.
    """
    vector: Optional[List[float]] = Field(default_factory=list, description="Vector embedding for the document chunk")
    
    def __init__(
        self,
        page_content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            page_content=page_content,
            metadata=metadata or {},
            **kwargs
        )
        self.vector = vector or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary including the vector."""
        doc_dict = super().to_dict()
        doc_dict['vector'] = self.vector
        return doc_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkModel':
        """Create a ChunkModel from a dictionary."""
        vector = data.pop('vector', [])
        doc = super().from_dict(data)
        doc.vector = vector
        return doc

class DBRetrieveModel(BaseModel):
    id: int = Field(description="index of the chunk")
    text: str = Field(description="content of the chunk")
    metadata: Dict[str,Any] = Field(description="metadata of the chunk")
    distance: float = Field(description="distance of the chunk")


class ReRankerLabel(BaseModel):
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(
        description="The reasoning process used to evaluate the relevance"
    )
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )
    text: str = Field(description="The text of the chunk")

class RerankerModel(BaseModel):
    labels: list[ReRankerLabel] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[ReRankerLabel]) -> list[ReRankerLabel]:
        return sorted(v, key=lambda x: x.relevancy, reverse=True)