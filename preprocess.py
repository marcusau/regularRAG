import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union,List
from datetime import date

import pymupdf4llm,pymupdf
from markdownify import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import timer
from models import DocumentModel,ChunkModel


# @timer
# def parse_document(filepath: Union[str, Path]) -> DocumentModel:
#     if isinstance(filepath, str):
#         filepath = Path(filepath)
#     if not filepath.is_file():
#         raise FileNotFoundError(f"{filepath} does not exist")

#     try:
#         md_text = pymupdf4llm.to_markdown(filepath, show_progress=True)
#         # Get document info
#         doc_info = {
#             "file": filepath.name,
#             "extension": filepath.suffix,
#             "file_path": str(filepath),
#             "status": "Success",
#             "length": len(md_text),
#             "content": md_text,
#         }
#         return DocumentModel(**doc_info)
#     except Exception as e:
#         raise f"Error converting document to markdown: {e}"

@timer
def parse_document(content: bytes,filepath: str,filetype: str) -> DocumentModel:
    if not isinstance(content, bytes):
        raise FileNotFoundError(f"{filepath} does not exist")

    try:
        parse_doc= ""
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                parse_doc += page.get_text() 
        ##print(f"parse_doc: {parse_doc}")
        # Convert to Markdown
        markdown_docs = markdownify(parse_doc)
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
        raise f"Error converting document to markdown: {e}"
    

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
        print(f"numbers of text split: {len(chunks)}")

        return chunks
    except Exception as e:
        raise f"Error when split documents into chunks - {e}"
    
if __name__ == "__main__":
    filepath = "data/20130208-etf-performance-and-perspectives.pdf"
    docs = parse_document(filepath)
    chunks = chunk_document(docs)
    
