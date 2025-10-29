import logging
import os
import sys

from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_xai import ChatXAI

PARENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PARENT_DIR))

from app.rag.models import DBRetrieveModel, RerankerModel
from app.api.models import ChatMessage
from app.rag.prompts import Prompts
from app.rag.providers import get_llm
from app.rag.vector_store import VectorStore

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 1 get llm
# 2 db_search_func
# 3 rerank_func
# 4 rag_generator_func

# prompts = Prompts()
model = get_llm()
vector_store = VectorStore(os.environ.get("CHROMADB_LOCAL_DIR"))


class BaseProcessor(ABC):
    def __init__(self, model: ChatXAI, prompts: Prompts = Prompts()):
        """Base template for RAG processors.

        Provides common validation, prompt formatting, and a template
        `process` method that concrete processors specialize by
        implementing abstract methods.

        Args:
            model: Chat model used for inference.
            prompts: Collection of system and user prompts.
        """
        self.model = model
        self.prompts = prompts
        # ==================== Common Validation Methods ====================

    def _validate_model(self) -> None:
        """Ensure `self.model` is a valid `ChatXAI` instance."""
        if not isinstance(self.model, ChatXAI) or self.model is None:
            logger.error("model must be a ChatXAI model")
            raise ValueError("model must be a ChatXAI model")

    def _validate_query(self, query: str) -> None:
        """Validate the user query is a non-empty string."""
        if not isinstance(query, str) or query.strip() in [None, ""]:
            logger.error("query must be string and not empty string")
            raise ValueError("query must be string and not empty string")

    # ==================== Abstract Methods (Must be implemented by subclasses) ====================
    @abstractmethod
    def validate_chunks(self, chunks: Any) -> None:
        """Validate the retrieved chunks type/shape required by the processor."""
        pass

    @abstractmethod
    def format_chunks_to_text(self, chunks: Any) -> str:
        """Convert chunks into a textual representation for prompting."""
        pass

    @abstractmethod
    def get_prompts(self) -> Tuple[str, str]:
        """Return a tuple of (system_prompt_text, user_prompt_template)."""
        pass

    @abstractmethod
    def get_process_name(self) -> str:
        """Return a friendly name for logging the processor step."""
        pass

    @abstractmethod
    def invoke_model(self, prompt: List) -> Any:
        """Run model inference with the prepared prompt and return raw output."""
        pass

    # ==================== Template Method (Concrete Implementation) ====================

    def process(self, query: str, chunks: Any) -> Any:
        """Execute the end-to-end processor flow.

        Steps:
        1) Validate inputs; 2) Format chunks; 3) Build prompt; 4) Invoke model.

        Args:
            query: User query string.
            chunks: Retrieved or preprocessed chunks to condition the model.

        Returns:
            The model output specific to the processor implementation.
        """
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
    def __init__(self, model: ChatXAI, prompts: Prompts = Prompts()):
        """Processor that re-ranks retrieved chunks using the LLM."""
        super().__init__(model=model, prompts=prompts)

    def validate_chunks(self, chunks: Any) -> None:
        """Validate input is a non-empty list of `DBRetrieveModel`."""
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
        """Format retrieved chunks into XML-like blocks for prompting."""
        chunks_text = ""
        for chunk in chunks:
            chunks_text += f'<chunk id="{chunk.id}">\n    {chunk.text}\n</chunk>\n'
        return chunks_text

    def get_prompts(self) -> Tuple[str, str]:
        """Return system and user prompts for the re-ranking step."""
        return (self.prompts.rerank.system, self.prompts.rerank.user)

    def get_process_name(self) -> str:
        """Human-readable process name for logs."""
        return "reranking with LLM"

    def invoke_model(self, prompt: List) -> RerankerModel:
        """Invoke the LLM with structured output to obtain ranking labels."""
        structured_model = self.model.with_structured_output(RerankerModel)
        reranker_results = structured_model.invoke(prompt)
        return reranker_results

    def rerank(self, query: str, chunks: List[DBRetrieveModel]) -> RerankerModel:
        """Public API: perform re-ranking for a query and list of chunks."""
        return self.process(query, chunks)


class GeneratorProcessor(BaseProcessor):
    def __init__(self, model: ChatXAI, prompts: Prompts = Prompts()):
        """Processor that generates final answers using re-ranked chunks."""
        super().__init__(model=model, prompts=prompts)

    def validate_chunks(self, chunks: Any) -> None:
        """Ensure `chunks` is an instance of `RerankerModel`."""
        if not isinstance(chunks, RerankerModel) or chunks is None:
            logger.error("chunks must be a RerankedResponsesModel")
            raise ValueError("chunks must be a RerankedResponsesModel")

    def format_chunks_to_text(self, chunks: RerankerModel) -> str:
        """Convert re-ranked labels into a readable context string."""
        chunks_text = ""
        for label in chunks.labels:
            chunks_text += f"Chunk id:{label.chunk_id}\nRelevancy: {label.relevancy}):\nReasoning:{label.chain_of_thought}\n Text: {label.text}\n\n\n"
        return chunks_text

    def get_prompts(self) -> Tuple[str, str]:
        """Return system and user prompts for the generation step."""
        return (self.prompts.generator.system, self.prompts.generator.user)

    def get_process_name(self) -> str:
        """Human-readable process name for logs."""
        return "RAG generating answer from LLM"

    def invoke_model(self, prompt: List) -> AIMessage:
        """Call the chat model to generate the final answer message."""
        generator_results = self.model.invoke(prompt)
        return generator_results

    def generate_answer(self, query: str, chunks: RerankerModel) -> AIMessage:
        """Public API: generate an answer given a query and re-ranked chunks."""
        return self.process(query, chunks)


class RAGPipeline:
    def __init__(
        self,
        model: ChatXAI,
        prompts: Prompts = Prompts(),
        db_path: str = os.environ.get("CHROMADB_LOCAL_DIR"),
    ):
        """High-level RAG pipeline wiring retrieval, re-ranking, and generation.

        Args:
            model: Chat model used for both re-ranking and generation.
            prompts: Prompt bundle for processors.
            db_path: Path to local ChromaDB directory.
        """
        self.model = model
        self.prompts = prompts
        self.reranker_processor = RerankerProcessor(model=self.model, prompts=self.prompts)
        self.generator_processor = GeneratorProcessor(model=self.model, prompts=self.prompts)
        self.vector_store = VectorStore(chromadb_path=db_path)

    def retrieve_document(self, collection_id: str, query: str) -> List[DBRetrieveModel]:
        """Retrieve candidate chunks from the vector store for a collection and query."""
        db_chunks = self.vector_store.retrieve_document(collection_id, query)
        return db_chunks

    def rerank_document(self, query: str, chunks: List[DBRetrieveModel]) -> RerankerModel:
        """Run the re-ranking processor on retrieved chunks."""
        return self.reranker_processor.rerank(query, chunks)

    def generate_answer(self, query: str, chunks: RerankerModel) -> str:
        """Generate the final answer using the generation processor."""
        return self.generator_processor.generate_answer(query, chunks)

    def _validate_collection_id(self, collection_id: str) -> None:
        """Validate that a non-empty `collection_id` string is provided."""
        if not isinstance(collection_id, str) or collection_id.strip() in [None, ""]:
            logger.error(f"collection_id must be string and not empty string")
            raise ValueError("collection_id must be string and not empty string")

    def _validate_chat_history(self, chat_history: list) -> None:
        """Ensure chat history is a list (or None is handled upstream)."""
        if not isinstance(chat_history, list):
            logger.error(f"chat_history must be list")
            raise ValueError("chat_history must be list")

    def _format_chat_history(self, query: str, chat_history: List[ChatMessage]) -> str:
        """Combine recent chat turns with the current query for better context."""
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

    def ask_question_with_history(self, collection_id: str, query: str, chat_history: list = None) -> tuple[str, list]:
        """End-to-end helper that retrieves, re-ranks, generates, and updates history.

        Args:
            collection_id: Vector collection identifier.
            query: User question text.
            chat_history: Prior conversation turns to provide context.

        Returns:
            A tuple of (answer_text, updated_chat_history).
        """
        self._validate_collection_id(collection_id)
        self._validate_chat_history(chat_history)

        enhanced_query = self._format_chat_history(query, chat_history)
        db_chunks = self.retrieve_document(collection_id, query)
        rerank_docs = self.rerank_document(enhanced_query, db_chunks)
        answer = self.generate_answer(enhanced_query, rerank_docs)

        # Update chat history
        updated_history = chat_history.copy() if chat_history else []
        updated_history.append({"role": "user", "content": query, "timestamp": datetime.now()})
        updated_history.append(
            {
                "role": "assistant",
                "content": answer.content,
                "timestamp": datetime.now(),
            }
        )

        return answer.content, updated_history

if __name__ == "__main__":
    collection_id = "7524460c-fe56-426f-be50-17653efa8649"

    query = "what is the ETF performance?"
