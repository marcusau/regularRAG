"""
Prompt management module for the RAG system.

This module handles loading, organizing, and providing access to system and user prompts
used by different components of the RAG system, including reranking and text generation.
"""

import functools
import os
from dataclasses import dataclass

from utils import read_txtfile

# Directory structure for organizing prompts
prompt_master_folder = "prompts"  # Root directory for all prompt files
reranker_master_folder = os.path.join(prompt_master_folder, "rerank")  # Reranker-specific prompts
generator_master_folder = os.path.join(prompt_master_folder, "generator")  # Generator-specific prompts

# File paths for reranker prompts
reranker_system_path = os.path.join(reranker_master_folder, "system.txt")  # System prompt for reranking
reranker_user_path = os.path.join(reranker_master_folder, "user.txt")  # User prompt template for reranking

# Load reranker prompts from files
rerank_system_prompt = read_txtfile(reranker_system_path)
rerank_user_prompt = read_txtfile(reranker_user_path)

# File paths for generator prompts
generator_system_path = os.path.join(generator_master_folder, "system.txt")  # System prompt for text generation
generator_user_path = os.path.join(generator_master_folder, "user.txt")  # User prompt template for generation

# Load generator prompts from files
generator_system_prompt = read_txtfile(generator_system_path)
generator_user_prompt = read_txtfile(generator_user_path)


@dataclass
class ReRankPrompt:
    """
    Data class containing prompts for the reranking component.
    
    This class holds both system and user prompts used by the reranking system
    to evaluate and score the relevance of retrieved document chunks.
    
    Attributes:
        system (str): System prompt that defines the reranking task and instructions
        user (str): User prompt template that will be filled with specific content
    """
    system: str = rerank_system_prompt
    user: str = rerank_user_prompt


@dataclass
class GeneratorPrompt:
    """
    Data class containing prompts for the text generation component.
    
    This class holds both system and user prompts used by the text generation
    system to create responses based on retrieved documents and user queries.
    
    Attributes:
        system (str): System prompt that defines the generation task and behavior
        user (str): User prompt template that will be filled with query and context
    """
    system: str = generator_system_prompt
    user: str = generator_user_prompt


@dataclass
class Prompts:
    """
    Main prompt management class that provides access to all system prompts.
    
    This class serves as a centralized access point for all prompts used throughout
    the RAG system. It organizes prompts by component type for easy retrieval.
    
    Attributes:
        rerank (ReRankPrompt): Prompts for the reranking component
        generator (GeneratorPrompt): Prompts for the text generation component
    """
    rerank = ReRankPrompt()
    generator = GeneratorPrompt()


if __name__ == "__main__":
    # Test the prompt loading functionality when running this module directly
    prompts = Prompts()
    print(prompts.generator.system)
