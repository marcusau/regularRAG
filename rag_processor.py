import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_xai import ChatXAI
from tqdm import tqdm

import chromadb
from fastapi_models import ChatHistory, ChatMessage, ChatQuery, ChatResponse
from models import DBRetrieveModel, RerankerModel
from prompts import Prompts
from providers import get_embedding, get_llm
from vector_store import retrieve_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 1 get llm
# 2 db_search_func
# 3 rerank_func
# 4 rag_generator_func

prompts = Prompts()
model = get_llm()

def rerank_document(model,query:str,chunks: List[DBRetrieveModel]) -> RerankerModel:
    
    if not isinstance(model,ChatXAI) or model is None:
        logger.error(f"model must be a ChatXAI model")
        raise ValueError("model must be a ChatXAI model")
    
    if not isinstance(query,str) or query.strip() in [None,""]:
        logger.error(f"query must be string and not empty string")
        raise ValueError("query must be string and not empty string")
    
    if not isinstance(chunks,list):
        logger.error(f"re-ranking function only accepts input type as list of DBRetrieveModel")
        raise TypeError("re-ranking function only accepts input type as list of DBRetrieveModel")
    
    if len(chunks) == 0:
        logger.error(f"No VectorDB search item is provided")
        raise Exception("No VectorDB search item is provided")
    
    if not any([isinstance(chunk,DBRetrieveModel) for chunk in chunks]):
        logger.error("re-ranking function only accepts input type as list of DBRetrieveModel")
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
    prompt = [system_prompt,human_prompt]
    
    structured_model = model.with_structured_output(RerankerModel)
    try:
        logger.info(f"doing reranking with LLM")
        reranker_results = structured_model.invoke(prompt)
        logger.info(f"finished reranking")
        return reranker_results
    except Exception as e:
        logger.error(f"Fail to generate re-ranking response from model as error :{e}")
        raise f"Fail to generate re-ranking response from model as error :{e}"
    


def rag_generator(model,query: str, chunks: RerankerModel,):
    
    if not isinstance(model,ChatXAI) or model is None:
        logger.error(f"model must be a ChatXAI model")
        raise ValueError("model must be a ChatXAI model")
    
    if not isinstance(chunks, RerankerModel) or chunks is None:
        logger.error(f"chunks must be a RerankedResponsesModel")
        raise ValueError("chunks must be a RerankedResponsesModel")
    
    if not isinstance(query,str) or query.strip() in [None,""]:
        logger.error(f"query must be string and not empty string")
        raise ValueError("query must be string and not empty string")
    
    # Format the chunks for the prompt
    chunks_text = ""
    for label in chunks.labels:
        chunks_text += f"Chunk id:{label.chunk_id}\nRelevancy: {label.relevancy}):\nReasoning:{label.chain_of_thought}\n Text: {label.text}\n\n\n"
    # Create the full prompt
    system_prompt=SystemMessage(content=prompts.generator.system)
    human_prompt_template = PromptTemplate.from_template(template=prompts.generator.user)
    human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
    human_prompt=HumanMessage(content=human_prompt_template)
    prompt = [system_prompt,human_prompt]
    
    try:
        logger.info(f"RAG generating answer from LLM")
        resp = model.invoke(prompt )
        logger.info(f"RAG finished answer")
        return resp
    except Exception as e:
        logger.error(f"Fail to generate response from RAG Generator model with error : {e}")
        raise Exception(f"Fail to generate response from RAG Generator model with error : {e}")
    

def ask_question_with_history(model, collection_id: str, query: str, chat_history: list = None) -> tuple[str, list]:
    """
    Enhanced ask_question function that considers chat history
    """
    if not isinstance(model,ChatXAI) or model is None:
        logger.error(f"model must be a ChatXAI model")
        raise ValueError("model must be a ChatXAI model")
    
    if not isinstance(collection_id,str) or collection_id.strip() in [None,""]:
        logger.error(f"collection_id must be string and not empty string")
        raise ValueError("collection_id must be string and not empty string")
    
    if not isinstance(query,str) or query.strip() in [None,""]:
        logger.error(f"query must be string and not empty string")
        raise ValueError("query must be string and not empty string")
    
    if not isinstance(chat_history,list) :
        logger.error(f"chat_history must be list")
        raise ValueError("chat_history must be list")

    # Format chat history for context
    history_context = ""
    if chat_history:
        for msg in chat_history[-5:]:  # Use last 5 messages for context
            history_context += f"{msg['role']}: {msg['content']}\n"
    
    # Enhance the query with chat history context
    enhanced_query = f"""
    Previous conversation context:
    {history_context}
    
    Current question: {query}
    """
    
    # Get documents using enhanced query
    db_chunks = retrieve_document(collection_id, query)
    rerank_docs = rerank_document(model, query, db_chunks)
    answer = rag_generator(model, enhanced_query, rerank_docs)
    
    # Update chat history
    updated_history = chat_history.copy() if chat_history else []
    updated_history.append({"role": "user", "content": query, "timestamp": datetime.now()})
    updated_history.append({"role": "assistant", "content": answer.content, "timestamp": datetime.now()})
    
    return answer.content, updated_history

if __name__ == "__main__":
    collection_id = "7524460c-fe56-426f-be50-17653efa8649"
    
    query = "what is the ETF performance?"
    
    print(ask_question(model,collection_id,query))
    
    # search_results = retrieve_document(collection_id,query)
    
    # rerank_results = rerank_document(model,query,search_results)
    
    # answer = rag_generator(model,query,rerank_results)
    
    # print(f"answer:\n {answer}")