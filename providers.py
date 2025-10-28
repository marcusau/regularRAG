"""
Provider module for LLM and embedding services in the RAG system.

This module provides centralized access to language models and embedding services,
handling configuration, initialization, and performance monitoring for external AI services.
"""

import os
from typing import List, Union

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_xai import ChatXAI

from utils import timer

# Load environment variables from .env file
load_dotenv()

# Environment variable configuration for AI services
os.environ["XAI_API_KEY"] = os.environ.get("XAI_API_KEY")  # XAI API key for language model access
XAI_MODEL = os.environ.get("XAI_MODEL")  # XAI model name to use for text generation
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL")  # Ollama model for embeddings
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama server URL


class EmbeddingFunction:
    """
    Custom embedding function wrapper for ChromaDB using Ollama embeddings.
    
    This class provides a callable interface that ChromaDB can use to generate
    embeddings for documents. It wraps the Ollama embeddings service and includes
    performance monitoring through the @timer decorator.
    
    Attributes:
        embeddings_obj: The underlying Ollama embeddings object for generating vectors
    """

    def __init__(self, embeddings_obj):
        """
        Initialize the embedding function with an Ollama embeddings object.
        
        Args:
            embeddings_obj: An instance of OllamaEmbeddings configured for the desired model
        """
        self.embeddings_obj = embeddings_obj

    @timer
    def __call__(self, input):
        """
        Generate embeddings for the given input text.
        
        This method is called by ChromaDB when it needs to embed documents or queries.
        It handles both single strings and lists of strings, ensuring consistent
        processing format.
        
        Args:
            input (Union[str, List[str]]): Text to embed - can be a single string or list of strings
            
        Returns:
            List[List[float]]: List of embedding vectors, where each vector is a list of floats
            
        Note:
            The @timer decorator automatically logs the execution time for performance monitoring
        """
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.embeddings_obj.embed_documents(
            input,
        )


def get_embedding(text: Union[str, List[str]]) -> List[float]:
    """
    Generate embeddings for the given text using Ollama embeddings service.
    
    This is a convenience function that creates an EmbeddingFunction instance
    and generates embeddings for the provided text. It handles both single
    strings and lists of strings.
    
    Args:
        text (Union[str, List[str]]): Text to embed - can be a single string or list of strings
        
    Returns:
        List[float]: Embedding vector as a list of floats
        
    Raises:
        TypeError: If the input text is not a string or list of strings
        
    Example:
        >>> embedding = get_embedding("Hello world")
        >>> embeddings = get_embedding(["Hello", "World"])
    """
    if isinstance(text, str) or isinstance(text, list) or all(isinstance(item, str) for item in text):
        embedding_func = EmbeddingFunction(OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL))
        return embedding_func(text)
    else:
        raise TypeError(f"text must be a string or a list of strings")


def get_llm(temperature: float = 0, max_token: int = None, timeout: int = None, max_retries: int = 2) -> ChatXAI:
    """
    Create and configure a ChatXAI language model instance.
    
    This function provides a centralized way to create language model instances
    with consistent configuration. It uses the XAI service for text generation.
    
    Args:
        temperature (float, optional): Controls randomness in generation. 
                                      Lower values (0) make output more deterministic,
                                      higher values (1) make it more creative. Defaults to 0.
        max_token (int, optional): Maximum number of tokens to generate in the response.
                                  If None, uses model's default limit.
        timeout (int, optional): Request timeout in seconds. If None, uses default timeout.
        max_retries (int, optional): Maximum number of retry attempts for failed requests.
                                    Defaults to 2.
    
    Returns:
        ChatXAI: Configured ChatXAI instance ready for text generation
        
    Example:
        >>> llm = get_llm(temperature=0.7, max_token=1000)
        >>> response = llm.invoke("What is machine learning?")
    """
    return ChatXAI(
        model=XAI_MODEL, temperature=temperature, max_tokens=max_token, timeout=timeout, max_retries=max_retries
    )


if __name__ == "__main__":
    # Test the LLM functionality when running this module directly
    llm = get_llm()
    print(llm.invoke("hi"))
