import os
from typing import List,Union
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_xai import ChatXAI

# Add the project root to Python path


from utils import timer

# Load environment variables from .env file
load_dotenv()
os.environ["XAI_API_KEY"] = os.environ.get("XAI_API_KEY")
XAI_MODEL = os.environ.get("XAI_MODEL")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL")

class EmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, embeddings_obj):
        self. embeddings_obj =  embeddings_obj
    @timer
    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.embeddings_obj.embed_documents(input,)

def get_embedding(text:Union[str,List[str]])->List[float]:
    if isinstance(text,str) or isinstance(text,list) or all(isinstance(item,str) for item in text):
        embedding_func = EmbeddingFunction(OllamaEmbeddings( model=OLLAMA_EMBEDDING_MODEL ))
        return embedding_func(text)
    else:
        raise TypeError(f"text must be a string or a list of strings")
    
def get_llm(temperature:float=0,max_token:int=None,timeout:int=None,max_retries:int=2)->ChatXAI:
    model= ChatXAI(
    model=XAI_MODEL,
    temperature=temperature,
    max_tokens=max_token,
    timeout=timeout,
    max_retries=max_retries,
    # other params..
    )
    return model

if __name__ == "__main__":
    llm = get_llm()
    print(llm.invoke("hi"))