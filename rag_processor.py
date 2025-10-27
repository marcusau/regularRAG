import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Union,Tuple
from abc import ABC, abstractmethod

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
from vector_store import VectorStore


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 1 get llm
# 2 db_search_func
# 3 rerank_func
# 4 rag_generator_func

#prompts = Prompts()
model = get_llm()
vector_store = VectorStore(os.environ.get("CHROMADB_LOCAL_DIR"))


class BaseProcessor(ABC):
    def __init__(self, model: ChatXAI, prompts:Prompts = Prompts()):
        self.model = model
        self.prompts = prompts
        # ==================== Common Validation Methods ====================
    def _validate_model(self) -> None:
        if not isinstance(self.model, ChatXAI) or self.model is None:
            logger.error("model must be a ChatXAI model")
            raise ValueError("model must be a ChatXAI model")
    
    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str) or query.strip() in [None, ""]:
            logger.error("query must be string and not empty string")
            raise ValueError("query must be string and not empty string")
        
    # ==================== Abstract Methods (Must be implemented by subclasses) ====================
    @abstractmethod
    def validate_chunks(self, chunks: Any) -> None:
        pass
    
    @abstractmethod
    def format_chunks_to_text(self, chunks: Any) -> str:
        pass

    @abstractmethod
    def get_prompts(self) -> Tuple[str, str]:
        pass
    
    @abstractmethod
    def get_process_name(self) -> str:
        pass
    
    @abstractmethod
    def invoke_model(self, prompt: List) -> Any:  
        pass
    
     # ==================== Template Method (Concrete Implementation) ====================
    
    def process(self, query: str, chunks: Any) -> Any:
        # Step 1: Input Validation
        self._validate_model()
        self._validate_query(query)
        self.validate_chunks(chunks)
    
        # Step 2: Format chunks into text
        chunks_text = self.format_chunks_to_text(chunks)
        
        # Step 3: Formulate the prompt
        system_prompt_text, user_prompt_template = self.get_prompts()
        
        system_prompt = SystemMessage(content=system_prompt_text)
        user_prompt = PromptTemplate.from_template(template=user_prompt_template)
        user_prompt_formatted = user_prompt.format(query=query, chunks_text=chunks_text)
        human_message = HumanMessage(content=user_prompt_formatted)
        
        prompt = [system_prompt, human_message]
        
                # Step 4: Model Inference
        try:
            process_name = self.get_process_name()
            logger.info(f"Starting {process_name}")
            result = self.invoke_model(prompt)
            logger.info(f"Finished {process_name}")
            return result
        except Exception as e:
            process_name = self.get_process_name()
            error_msg = f"Failed to generate response from {process_name} with error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        
class RerankerProcessor(BaseProcessor):
    
    def __init__(self, model: ChatXAI, prompts:Prompts = Prompts()):
        super().__init__(model=model, prompts=prompts)
    
    def validate_chunks(self, chunks: Any) -> None:
        # Check if chunks is a list
        if not isinstance(chunks, list):
            logger.error("re-ranking function only accepts input type as list of DBRetrieveModel")
            raise TypeError("re-ranking function only accepts input type as list of DBRetrieveModel")
        
        # Check if chunks list is not empty
        if len(chunks) == 0:
            logger.error("No VectorDB search item is provided")
            raise Exception("No VectorDB search item is provided")
        
        # Check if all items in chunks are DBRetrieveModel instances
        if not all(isinstance(chunk, DBRetrieveModel) for chunk in chunks):
            logger.error("re-ranking function only accepts input type as list of DBRetrieveModel")
            raise TypeError("re-ranking function only accepts input type as list of DBRetrieveModel")
    
    def format_chunks_to_text(self, chunks: List[DBRetrieveModel]) -> str:
        chunks_text = ""
        for chunk in chunks:
            chunks_text += f'<chunk id="{chunk.id}">\n    {chunk.text}\n</chunk>\n'
        return chunks_text
    
    def get_prompts(self) -> Tuple[str, str]:
        return (self.prompts.rerank.system, self.prompts.rerank.user)
    
    def get_process_name(self) -> str:
        return "reranking with LLM"
    
    def invoke_model(self, prompt: List) -> RerankerModel:
        structured_model = self.model.with_structured_output(RerankerModel)
        reranker_results = structured_model.invoke(prompt)
        return reranker_results
    
    def rerank(self, query: str, chunks: List[DBRetrieveModel]) -> RerankerModel:
        return self.process(query, chunks)
    
class GeneratorProcessor(BaseProcessor):
    def __init__(self, model: ChatXAI, prompts:Prompts = Prompts()):
        super().__init__(model=model, prompts=prompts)
    
    def validate_chunks(self, chunks: Any) -> None:
        if not isinstance(chunks, RerankerModel) or chunks is None:
            logger.error("chunks must be a RerankedResponsesModel")
            raise ValueError("chunks must be a RerankedResponsesModel")
    
    def format_chunks_to_text(self, chunks: RerankerModel) -> str:
        chunks_text = ""
        for label in chunks.labels:
            chunks_text += f"Chunk id:{label.chunk_id}\nRelevancy: {label.relevancy}):\nReasoning:{label.chain_of_thought}\n Text: {label.text}\n\n\n"
        return chunks_text
    
    def get_prompts(self) -> Tuple[str, str]:
        return (self.prompts.generator.system, self.prompts.generator.user)
    
    def get_process_name(self) -> str:
        return "RAG generating answer from LLM"
    
    def invoke_model(self, prompt: List) -> AIMessage:
        generator_results = self.model.invoke(prompt)
        return generator_results
    
    def generate_answer(self, query: str, chunks: RerankerModel) -> AIMessage:
        return self.process(query, chunks)
    

class RAGPipeline:
    def __init__(self, model: ChatXAI, prompts: Prompts=Prompts(),db_path: str=os.environ.get("CHROMADB_LOCAL_DIR")):
        self.model = model 
        self.prompts = prompts
        self.reranker_processor = RerankerProcessor(model=self.model, prompts=self.prompts)
        self.generator_processor = GeneratorProcessor(model=self.model, prompts=self.prompts)
        self.vector_store = VectorStore(chromadb_path=db_path)
        
    def retrieve_document(self,collection_id: str,query: str) -> List[DBRetrieveModel]:
        db_chunks = self.vector_store.retrieve_document(collection_id, query)
        return db_chunks
                                                   
    def rerank_document(self,query: str,chunks: List[DBRetrieveModel]) -> RerankerModel:
        return self.reranker_processor.rerank(query,chunks)
    
    def generate_answer(self,query: str,chunks: RerankerModel) -> str:
        return self.generator_processor.generate_answer(query,chunks)
    
    
    def _validate_collection_id(self,collection_id: str) -> None:
        if not isinstance(collection_id,str) or collection_id.strip() in [None,""]:
            logger.error(f"collection_id must be string and not empty string")
            raise ValueError("collection_id must be string and not empty string")
    
    def _validate_chat_history(self,chat_history: list) -> None:
        if not isinstance(chat_history,list) :
            logger.error(f"chat_history must be list")
            raise ValueError("chat_history must be list")
    
    def _format_chat_history(self, query: str, chat_history: List[ChatMessage]) -> str:
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
        return enhanced_query
    

    def ask_question_with_history(self,collection_id: str, query: str, chat_history: list = None) -> tuple[ str, list]:
        self._validate_collection_id(collection_id)
        self._validate_chat_history(chat_history)
        
        enhanced_query = self._format_chat_history(query, chat_history)
        db_chunks = self.retrieve_document(collection_id, query)
        rerank_docs = self.rerank_document(enhanced_query, db_chunks)
        answer = self.generate_answer(enhanced_query, rerank_docs)
        
        # Update chat history
        updated_history = chat_history.copy() if chat_history else []
        updated_history.append({"role": "user", "content": query, "timestamp": datetime.now()})
        updated_history.append({"role": "assistant", "content": answer.content, "timestamp": datetime.now()})
        
        return answer.content, updated_history

    

# def rerank_document(model,query:str,chunks: List[DBRetrieveModel]) -> RerankerModel:
    
#     if not isinstance(model,ChatXAI) or model is None:
#         logger.error(f"model must be a ChatXAI model")
#         raise ValueError("model must be a ChatXAI model")
    
#     if not isinstance(query,str) or query.strip() in [None,""]:
#         logger.error(f"query must be string and not empty string")
#         raise ValueError("query must be string and not empty string")
    
#     if not isinstance(chunks,list):
#         logger.error(f"re-ranking function only accepts input type as list of DBRetrieveModel")
#         raise TypeError("re-ranking function only accepts input type as list of DBRetrieveModel")
    
#     if len(chunks) == 0:
#         logger.error(f"No VectorDB search item is provided")
#         raise Exception("No VectorDB search item is provided")
    
#     if not any([isinstance(chunk,DBRetrieveModel) for chunk in chunks]):
#         logger.error("re-ranking function only accepts input type as list of DBRetrieveModel")
#         raise TypeError("re-ranking function only accepts input type as list of DBSearchModel")
   
#     # Format the chunks for the prompt
#     chunks_text = ""
#     for chunk in chunks:
#         chunks_text += f'<chunk id="{chunk.id}">\n    {chunk.text}\n</chunk>\n'
    
#     # Create the full prompt
#     system_prompt=SystemMessage(content=prompts.rerank.system )
#     human_prompt_template = PromptTemplate.from_template(template=prompts.rerank.user)
#     human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
#     human_prompt=HumanMessage(content=human_prompt_template)
#     prompt = [system_prompt,human_prompt]
    
#     structured_model = model.with_structured_output(RerankerModel)
#     try:
#         logger.info(f"doing reranking with LLM")
#         reranker_results = structured_model.invoke(prompt)
#         logger.info(f"finished reranking")
#         return reranker_results
#     except Exception as e:
#         logger.error(f"Fail to generate re-ranking response from model as error :{e}")
#         raise f"Fail to generate re-ranking response from model as error :{e}"
    


# def rag_generator(model,query: str, chunks: RerankerModel,):
    
#     if not isinstance(model,ChatXAI) or model is None:
#         logger.error(f"model must be a ChatXAI model")
#         raise ValueError("model must be a ChatXAI model")
    
#     if not isinstance(chunks, RerankerModel) or chunks is None:
#         logger.error(f"chunks must be a RerankedResponsesModel")
#         raise ValueError("chunks must be a RerankedResponsesModel")
    
#     if not isinstance(query,str) or query.strip() in [None,""]:
#         logger.error(f"query must be string and not empty string")
#         raise ValueError("query must be string and not empty string")
    
    # Format the chunks for the prompt
    # chunks_text = ""
    # for label in chunks.labels:
    #     chunks_text += f"Chunk id:{label.chunk_id}\nRelevancy: {label.relevancy}):\nReasoning:{label.chain_of_thought}\n Text: {label.text}\n\n\n"
    # # Create the full prompt
    # system_prompt=SystemMessage(content=prompts.generator.system)
    # human_prompt_template = PromptTemplate.from_template(template=prompts.generator.user)
    # human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
    # human_prompt=HumanMessage(content=human_prompt_template)
    # prompt = [system_prompt,human_prompt]
    
    # try:
    #     logger.info(f"RAG generating answer from LLM")
    #     resp = model.invoke(prompt )
    #     logger.info(f"RAG finished answer")
    #     return resp
    # except Exception as e:
    #     logger.error(f"Fail to generate response from RAG Generator model with error : {e}")
    #     raise Exception(f"Fail to generate response from RAG Generator model with error : {e}")
    

# def ask_question_with_history(model, collection_id: str, query: str, chat_history: list = None) -> tuple[str, list]:
#     """
#     Enhanced ask_question function that considers chat history
#     """
#     if not isinstance(model,ChatXAI) or model is None:
#         logger.error(f"model must be a ChatXAI model")
#         raise ValueError("model must be a ChatXAI model")
    
#     if not isinstance(collection_id,str) or collection_id.strip() in [None,""]:
#         logger.error(f"collection_id must be string and not empty string")
#         raise ValueError("collection_id must be string and not empty string")
    
#     if not isinstance(query,str) or query.strip() in [None,""]:
#         logger.error(f"query must be string and not empty string")
#         raise ValueError("query must be string and not empty string")
    
#     if not isinstance(chat_history,list) :
#         logger.error(f"chat_history must be list")
#         raise ValueError("chat_history must be list")

#     # Format chat history for context
#     history_context = ""
#     if chat_history:
#         for msg in chat_history[-5:]:  # Use last 5 messages for context
#             history_context += f"{msg['role']}: {msg['content']}\n"
    
#     # Enhance the query with chat history context
#     enhanced_query = f"""
#     Previous conversation context:
#     {history_context}
    
#     Current question: {query}
#     """
    
#     # Get documents using enhanced query
#     db_chunks = vector_store.retrieve_document(collection_id, query)
#     rerank_docs = rerank_document(model, query, db_chunks)
#     answer = rag_generator(model, enhanced_query, rerank_docs)
    
#     # Update chat history
#     updated_history = chat_history.copy() if chat_history else []
#     updated_history.append({"role": "user", "content": query, "timestamp": datetime.now()})
#     updated_history.append({"role": "assistant", "content": answer.content, "timestamp": datetime.now()})
    
    # return answer.content, updated_history

if __name__ == "__main__":
    collection_id = "7524460c-fe56-426f-be50-17653efa8649"
    
    query = "what is the ETF performance?"
    
    #print(ask_question(model,collection_id,query))
    
    # search_results = retrieve_document(collection_id,query)
    
    # rerank_results = rerank_document(model,query,search_results)
    
    # answer = rag_generator(model,query,rerank_results)
    
    # print(f"answer:\n {answer}")