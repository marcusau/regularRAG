import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union,List
from datetime import date
import logging
import pymupdf
from markdownify import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import timer
from models import DocumentModel,ChunkModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_content(content:bytes)->str:
    if not isinstance(content, bytes):
        raise TypeError(f"{content} is not bytes. Only bytes is accept")
    try:
        parse_doc = ""
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                parse_doc += page.get_text() 
        return parse_doc
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        raise Exception(f"Error loading content: {e}")
    
def convert_to_markdown(parsed_content:str)->str:
    if not isinstance(parsed_content, str):
        raise TypeError(f"{parsed_content} is not string. Only string is accept")
    try:
        return markdownify(parsed_content)
    except Exception as e:
        logger.error(f"Error converting to markdown: {e}")
        raise Exception(f"Error converting to markdown: {e}")

@timer
def parse_document(content: bytes,filepath: str,filetype: str) -> DocumentModel:
    if not isinstance(content, bytes):
        raise FileNotFoundError(f"{filepath} does not exist") 
    if not isinstance(filepath, str):
        raise TypeError(f"{filepath} is not string. Only string is accept")
    if not isinstance(filetype, str):
        raise TypeError(f"{filetype} is not string. Only string is accept")
    try:
        parsed_content = fetch_content(content)
        # Convert to Markdown
        markdown_docs = convert_to_markdown(parsed_content)
        
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
    

@timer
def chunk_document(doc:DocumentModel,chunk_size:int=1000,chunk_overlap:int=100)-> List[ChunkModel]:
    
    if not isinstance(doc, DocumentModel):
        raise TypeError(f"{doc} is not DocumentModel. Only DocumentModel is accept")
    if not isinstance(doc.content, str):
        raise TypeError(f"{doc.content} is not string. Only string is accept")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        # separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " ", ""]
    )
    try:
        split_texts = text_splitter.split_text(doc.content)

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
        raise f"Error when split documents into chunks - {e}"
    
if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    docs = parse_document(filepath)
    chunks = chunk_document(docs)
    
