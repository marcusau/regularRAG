import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdownify import markdownify

from models import ChunkModel, DocumentModel
from utils import timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentParser:
    """
    A class for parsing various document formats and converting them to markdown.
    
    This parser handles PDF documents using PyMuPDF and converts the extracted
    text content to markdown format for further processing in the RAG pipeline.
    """
    
    def _validate_input(self, content: bytes, filepath: str, filetype: str) -> None:
        """
        Validate input parameters for document parsing.
        
        Args:
            content (bytes): Raw document content as byte data
            filepath (str): Path to the document file
            filetype (str): Type/extension of the document file
            
        Raises:
            TypeError: If any parameter is not of the expected type
        """
        if not isinstance(content, bytes):
            raise TypeError(f"{content} is not byte data")
        if not isinstance(filepath, str):
            raise TypeError(f"{filepath} is not string. Only string is accept")
        if not isinstance(filetype, str):
            raise TypeError(f"{filetype} is not string. Only string is accept")

    def _validate_content(self, content: bytes) -> None:
        """
        Validate that the document content is not empty.
        
        Args:
            content (bytes): Raw document content to validate
            
        Raises:
            ValueError: If content is empty or None
        """
        if not content or len(content) == 0:
            raise ValueError("Content cannot be empty")

    @timer
    def parse_document(self, content: bytes, filepath: str, filetype: str) -> DocumentModel:
        """
        Parse a document from raw bytes and convert to DocumentModel.
        
        This method extracts text content from PDF documents and converts it
        to markdown format, then wraps it in a DocumentModel for further processing.
        
        Args:
            content (bytes): Raw document content as byte data
            filepath (str): Path to the document file
            filetype (str): Type/extension of the document file
            
        Returns:
            DocumentModel: Parsed document with metadata and markdown content
            
        Raises:
            Exception: If document parsing fails
        """
        self._validate_input(content=content, filepath=filepath, filetype=filetype)
        self._validate_content(content)
        try:
            parsed_content = self._fetch_content(content)
            # Convert to Markdown
            markdown_docs = self._convert_to_markdown(parsed_content)

            doc_info = {
                "file": filepath.split("/")[-1],
                "extension": filetype,
                "file_path": filepath,
                "status": "Success",
                "length": len(markdown_docs),
                "content": markdown_docs,
            }
            return DocumentModel(**doc_info)
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise Exception(f"Error parsing document: {e}")

    def _fetch_content(self, content: bytes) -> str:
        """
        Extract text content from PDF document using PyMuPDF.
        
        This method opens a PDF document from byte content and extracts
        all text from each page, concatenating them into a single string.
        
        Args:
            content (bytes): Raw PDF document content
            
        Returns:
            str: Extracted text content from all pages
            
        Raises:
            Exception: If PDF parsing fails
        """
        try:
            parse_doc = ""
            with pymupdf.open(stream=content, filetype="pdf") as doc:
                for page in doc:
                    parse_doc += page.get_text()
            return parse_doc
        except Exception as e:
            logger.error(f"Error loading content: {e}")
            raise Exception(f"Error loading content: {e}")

    def _convert_to_markdown(self, parsed_content: str) -> str:
        """
        Convert extracted text content to markdown format.
        
        This method takes the raw text extracted from PDF and converts it
        to markdown format using the markdownify library for better
        structure and readability in the RAG pipeline.
        
        Args:
            parsed_content (str): Raw text content extracted from PDF
            
        Returns:
            str: Content converted to markdown format
            
        Raises:
            Exception: If content is empty or markdown conversion fails
        """
        if parsed_content.strip() in ["", None]:
            raise Exception(f"parse content cannot be empty")
        try:
            return markdownify(parsed_content)
        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            raise Exception(f"Error converting to markdown: {e}")


class DocumentChunker:
    """
    A class for splitting documents into smaller chunks for vector storage.
    
    This chunker uses RecursiveCharacterTextSplitter to split documents into
    overlapping chunks of specified size, which are then used for embedding
    generation and vector storage in the RAG pipeline.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the DocumentChunker with specified chunk parameters.
        
        Args:
            chunk_size (int): Maximum size of each text chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Raises:
            ValueError: If chunk_size is not positive, chunk_overlap is negative,
                       or chunk_overlap is greater than or equal to chunk_size
        """
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
            is_separator_regex=False,
        )

    def _validate_input(self, doc: DocumentModel) -> None:
        """
        Validate input document for chunking.
        
        Args:
            doc (DocumentModel): Document to validate
            
        Raises:
            TypeError: If doc is not a DocumentModel or content is not a string
        """
        if not isinstance(doc, DocumentModel):
            raise TypeError(f"{doc} is not DocumentModel. Only DocumentModel is accept")
        if not isinstance(doc.content, str):
            raise TypeError(f"{doc.content} is not string. Only string is accept")

    @timer
    def chunk_document(self, doc: DocumentModel) -> List[ChunkModel]:
        """
        Split a document into overlapping chunks for vector storage.
        
        This method takes a parsed document and splits it into smaller chunks
        with specified overlap, creating ChunkModel objects with metadata
        for each chunk including source file, creation date, and unique index.
        
        Args:
            doc (DocumentModel): Parsed document to chunk
            
        Returns:
            List[ChunkModel]: List of chunk objects with metadata
            
        Raises:
            Exception: If document chunking fails
        """
        self._validate_input(doc)
        try:
            split_texts = self.text_splitter.split_text(doc.content)
            chunks = []
            for idx, split_text in enumerate(split_texts):
                file_path_str = str(doc.file.replace("/", "_").replace(".", "_").strip())
                doc_idx = "_".join([file_path_str, str(idx)])
                chunk = ChunkModel(
                    page_content=split_text,
                    metadata={
                        "index": doc_idx,
                        "source": doc.file_path,
                        "create_date": date.today().strftime("%Y-%m-%d"),
                    },
                )
                chunks.append(chunk)
            logger.info(f"numbers of text split: {len(chunks)}")
            return chunks
        except Exception as e:
            raise Exception(f"Error when split documents into chunks - {e}")


@dataclass
class ProcessingConfig:
    """
    Configuration class for document processing pipeline.
    
    This dataclass holds configuration parameters for the document processing
    pipeline, including chunk size and overlap settings used by the DocumentChunker.
    """
    
    chunk_size: int = 1000  # Maximum size of each text chunk in characters
    chunk_overlap: int = 100  # Number of characters to overlap between chunks


class DocumentProcessor:
    """
    Main orchestrator class for document processing pipeline.
    
    This class coordinates the entire document processing workflow, from parsing
    raw document content to creating chunks ready for vector storage. It combines
    DocumentParser and DocumentChunker to provide a complete processing pipeline.
    """
    
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        parser: Optional["DocumentParser"] = None,
        chunker: Optional["DocumentChunker"] = None,
    ):
        """
        Initialize the DocumentProcessor with optional components.
        
        Args:
            config (Optional[ProcessingConfig]): Configuration for processing pipeline
            parser (Optional[DocumentParser]): Document parser instance
            chunker (Optional[DocumentChunker]): Document chunker instance
        """
        self.config = config or ProcessingConfig()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or DocumentChunker(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )

        logger.info(
            f"DocumentProcessor initialized with chunk_size={self.config.chunk_size}, "
            f"chunk_overlap={self.config.chunk_overlap}"
        )

    @timer
    def process_document(self, content: bytes, filepath: str, filetype: str) -> List[ChunkModel]:
        """
        Process a single document through the complete pipeline.
        
        This method orchestrates the full document processing workflow:
        1. Parse the document from raw bytes to DocumentModel
        2. Split the document into chunks using DocumentChunker
        3. Return list of ChunkModel objects ready for vector storage
        
        Args:
            content (bytes): Raw document content as byte data
            filepath (str): Path to the document file
            filetype (str): Type/extension of the document file
            
        Returns:
            List[ChunkModel]: List of processed chunks with metadata
            
        Raises:
            Exception: If document processing fails at any stage
        """
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

    def process_multiple_documents(self, documents: List[tuple]) -> List[List[ChunkModel]]:
        """
        Process multiple documents in batch through the processing pipeline.
        
        This method processes a list of documents (as tuples of content, filepath, filetype)
        and returns a list of chunk lists, one for each document. Failed documents
        are logged but don't stop the processing of other documents.
        
        Args:
            documents (List[tuple]): List of tuples containing (content, filepath, filetype)
            
        Returns:
            List[List[ChunkModel]]: List of chunk lists, one per document
        """
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
        """
        Generate statistics about processed chunks.
        
        This method calculates various statistics about the processed chunks
        including total count, character count, average size, and source files.
        
        Args:
            chunks (List[ChunkModel]): List of processed chunks
            
        Returns:
            dict: Dictionary containing processing statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0, "avg_chunk_size": 0}

        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "sources": list(set(chunk.metadata.get("source", "") for chunk in chunks)),
        }


if __name__ == "__main__":
    """
    Main execution block for testing document processing pipeline.
    
    This section can be used to test the document processing functionality
    with sample PDF files. Currently contains commented-out example usage.
    """
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"

    # Example usage (commented out):
    # docs = parse_document(filepath)
    # chunks = chunk_document(docs)
