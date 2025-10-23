import os,sys
import chromadb

from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime,date
from typing import List,Union,Dict,Any

from langchain_core.prompts import PromptTemplate
from langchain_xai import ChatXAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from providers import get_llm, get_embedding
from models import RerankerModel,DBSearchModel
from vector_store import retrieve_document
from prompts import Prompts



# 1 get llm
# 2 db_search_func
# 3 rerank_func
# 4 rag_generator_func

prompts = Prompts()
model = get_llm()

def rerank_document(model,query:str,chunks: List[DBSearchModel]) -> RerankerModel:
    
    if not isinstance(chunks,list):
        raise TypeError("re-ranking function only accepts input type as list of DBSearchModel")
    
    if len(chunks) == 0:
        raise Exception("No VectorDB search item is provided")
    
    if not any([isinstance(chunk,DBSearchModel) for chunk in chunks]):
        raise TypeError("re-ranking function only accepts input type as list of DBSearchModel")
   
    # Format the chunks for the prompt
    chunks_text = ""
    for chunk in chunks:
        chunks_text += f'<chunk id="{chunk.id}">\n    {chunk.text}\n</chunk>\n'
    
    # Create the full prompt
    system_prompt=SystemMessage(content=prompts.rerank.system )
    human_prompt_template = PromptTemplate.from_template(template=prompts.rerank.user)
    human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
    human_prompt=HumanMessage(content=human_prompt_template)
    
    structured_model = model.with_structured_output(RerankerModel)
    try:
        print(f"doing reranking with LLM")
        resp=structured_model.invoke([system_prompt,human_prompt])
        print(f"finished reranking")
        return resp
    except Exception as e:
        raise f"Fail to generate re-ranking response from model as error :{e}"
    


def rag_generator(model,query: str, chunks: RerankerModel,) -> str:
    
    if not isinstance(chunks, RerankerModel):
        raise ValueError("chunks must be a RerankedResponsesModel")
    
    if not isinstance(query,str) or query.strip() in [None,""]:
        raise ValueError("query must be string and not empty string")
    
    chunks_text = ""
    for label in chunks.labels:
        chunks_text += f"Chunk id:{label.chunk_id}\nRelevancy: {label.relevancy}):\nReasoning:{label.chain_of_thought}\n Text: {label.text}\n\n\n"
    # Create the full prompt
    system_prompt=SystemMessage(content=prompts.generator.system)
    
    human_prompt_template = PromptTemplate.from_template(template=prompts.generator.user)
    human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
    human_prompt=HumanMessage(content=human_prompt_template)
    
    try:
        print(f"RAG generating answer from LLM")
        resp = model.invoke([system_prompt,human_prompt])
        print(f"RAG finished answer")
        return resp
    except Exception as e:
        raise f"Fail to generate response from RAG Generator model with error : {e}"
    
def ask_question(model, collection_id:str,query: str) -> str:
    db_chunks = retrieve_document(collection_id,query)
    rerank_docs = rerank_document(model,query,db_chunks)
    answer = rag_generator(model,query,rerank_docs)
    return answer.content


if __name__ == "__main__":
    collection_id = "7524460c-fe56-426f-be50-17653efa8649"
    
    query = "what is the ETF performance?"
    
    print(ask_question(model,collection_id,query))
    
    # search_results = retrieve_document(collection_id,query)
    
    # rerank_results = rerank_document(model,query,search_results)
    
    # answer = rag_generator(model,query,rerank_results)
    
    # print(f"answer:\n {answer}")