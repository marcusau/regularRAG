import os
import functools
from dataclasses import dataclass

from utils import read_txtfile


prompt_master_folder = "prompts"
reranker_master_folder = os.path.join(prompt_master_folder,"rerank")
generator_master_folder = os.path.join(prompt_master_folder,"generator")

reranker_system_path = os.path.join(reranker_master_folder,"system.txt")
reranker_user_path = os.path.join(reranker_master_folder,"user.txt")

rerank_system_prompt = read_txtfile(reranker_system_path)
rerank_user_prompt = read_txtfile(reranker_user_path)

generator_system_path = os.path.join(generator_master_folder,"system.txt")
generator_user_path = os.path.join(generator_master_folder,"user.txt")

generator_system_prompt = read_txtfile(generator_system_path)
generator_user_prompt = read_txtfile(generator_user_path)



@dataclass
class ReRankPrompt:
    system: str = rerank_system_prompt
    user: str =rerank_user_prompt 
    
@dataclass
class GeneratorPrompt:
    system: str = generator_system_prompt
    user: str =generator_user_prompt 
    
@dataclass
class Prompts:
    rerank = ReRankPrompt()
    generator = GeneratorPrompt()

if __name__ == "__main__":
    prompts = Prompts()
    print(prompts.generator.system)