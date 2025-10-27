import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union,Optional
from dataclasses import dataclass
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdownify import markdownify

from models import ChunkModel, DocumentModel
from utils import timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
class DocumentParser:
    
    def _validate_input(self,content:bytes,filepath:str,filetype:str)->None:
        if not isinstance(content, bytes):
            raise TypeError(f"{content} is not byte data") 
        if not isinstance(filepath, str):
            raise TypeError(f"{filepath} is not string. Only string is accept")
        if not isinstance(filetype, str):
            raise TypeError(f"{filetype} is not string. Only string is accept")


    # Add content validation
    def _validate_content(self, content: bytes) -> None:
        if not content or len(content) == 0:
            raise ValueError("Content cannot be empty")
    
    @timer
    def parse_document(self,content: bytes, filepath:str, filetype:str)->DocumentModel:
        self._validate_input(content=content,filepath=filepath,filetype=filetype)
        self._validate_content(content)
        try:
            parsed_content = self._fetch_content(content)
         # Convert to Markdown
            markdown_docs = self._convert_to_markdown(parsed_content)
        
            doc_info={
                "file": filepath.split("/")[-1],
                "extension": filetype,
                "file_path": filepath,
                "status": "Success",
                "length": len(markdown_docs),
                "content": markdown_docs
        }
            return DocumentModel(**doc_info)
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise Exception(f"Error parsing document: {e}")
        
    def _fetch_content(self,content:bytes)->str:
        try:
            parse_doc = ""
            with pymupdf.open(stream=content, filetype="pdf") as doc:
                for page in doc:
                    parse_doc += page.get_text() 
            return parse_doc
        except Exception as e:
            logger.error(f"Error loading content: {e}")
            raise Exception(f"Error loading content: {e}")

    def _convert_to_markdown(self,parsed_content:str)->str:
        if parsed_content.strip() in ["",None]:
            raise Exception(f"parse content cannot be empty")
        try:
            return markdownify(parsed_content)
        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            raise Exception(f"Error converting to markdown: {e}")


class DocumentChunker:
    def __init__(self,chunk_size:int=1000,chunk_overlap:int=100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,)
    
    def _validate_input(self,doc: DocumentModel)->None:
        if not isinstance(doc, DocumentModel):
            raise TypeError(f"{doc} is not DocumentModel. Only DocumentModel is accept")
        if not isinstance(doc.content, str):
            raise TypeError(f"{doc.content} is not string. Only string is accept")

    @timer
    def chunk_document(self,doc:DocumentModel)-> List[ChunkModel]:
        self._validate_input(doc)
        try:
            split_texts = self.text_splitter.split_text(doc.content)
            chunks = []
            for idx, split_text in enumerate(split_texts):
                file_path_str = str(doc.file.replace("/", "_").replace(".", "_").strip()) 
                doc_idx = "_".join([file_path_str, str(idx)])
                chunk = ChunkModel(page_content=split_text,
                                   metadata={
                                        "index": doc_idx,
                                        "source": doc.file_path,
                                        "create_date": date.today().strftime("%Y-%m-%d"),  },)
                chunks.append(chunk)
            logger.info(f"numbers of text split: {len(chunks)}")
            return chunks
        except Exception as e:
            raise Exception(f"Error when split documents into chunks - {e}")

@dataclass
class ProcessingConfig:
    """Configuration for document processing pipeline."""
    chunk_size: int = 1000
    chunk_overlap: int = 100  
    
class DocumentProcessor:
    
    def __init__(self,config:Optional[ProcessingConfig]= None,
                 parser: Optional["DocumentParser"]= None,
                 chunker : Optional["DocumentChunker"]= None
                 ):
        self.config = config or ProcessingConfig()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or DocumentChunker(
                                    chunk_size=self.config.chunk_size,
                                    chunk_overlap=self.config.chunk_overlap)
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.config.chunk_size}, "
                   f"chunk_overlap={self.config.chunk_overlap}")
    
    @timer
    def process_document(self, 
                        content: bytes, 
                        filepath: str, 
                        filetype: str) -> List[ChunkModel]:
        try:
            logger.info(f"Starting processing of document: {filepath}")
            
            # Step 1: Parse document
            document = self.parser.parse_document(content, filepath, filetype)
            logger.info(f"Document parsed successfully. Length: {document.length} characters")
            
            # Step 2: Chunk document
            chunks = self.chunker.chunk_document(document)
            logger.info(f"Document chunked successfully. Number of chunks: {len(chunks)}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {filepath}: {e}")
            raise Exception(f"Failed to process document {filepath}: {e}")
        
    def process_multiple_documents(self,  documents: List[tuple]) -> List[List[ChunkModel]]:

        results = []
        successful = 0
        failed = 0
        
        for content, filepath, filetype in documents:
            try:
                chunks = self.process_document(content, filepath, filetype)
                results.append(chunks)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                results.append([])  
                failed += 1
                
        logger.info(f"Processing completed. {successful} documents processed successfully, {failed} documents failed")
        return results
    
    def get_processing_stats(self, chunks: List[ChunkModel]) -> dict:
       
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0, "avg_chunk_size": 0}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "sources": list(set(chunk.metadata.get("source", "") for chunk in chunks))
        }
    
    
if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    
    #docs = parse_document(filepath)
    #chunks = chunk_document(docs)
    
