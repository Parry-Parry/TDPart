from typing import Any, List, Optional
from fastchat.model import get_conversation_template
from ftfy import fix_text
import re 

class DocumentFormatter:
    def __init__(self, max_length : int, **kwargs) -> None:
        self.max_length = max_length
    
    def __call__(self, texts : List[str]) -> str:
        texts = [text[:self.max_length] for text in texts]
        return '\n'.join([f'[{i}] {text}' for i, text in enumerate(texts)])

def replace_number(s):
    return re.sub(r"\[(\d+)\]", r"(\1)", s)

class RankPrompt:
    SYSTEM_MESSAGE = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
    PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
    POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."
    MAX_TOKENS = 4096
    def __init__(self, 
                 model : str,
                 tokenizer,
                 max_length : int = 300,
                 rankllm : bool = False) -> None:
        self.rankllm = rankllm
        self.model = model
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def get_num_tokens(self, text : str) -> int:
        return len(self.tokenizer.encode(text))
    
    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        output_token_estimate = (
            len(
                self.tokenizer.encode(
                    " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                )
            )
            - 1
        )
        return output_token_estimate

    def __call__(self, query, texts, num, **kwargs) -> str:
        max_length = self.max_length
        while True:
            conv = get_conversation_template(self.model)
            if self.rankllm:
                conv.set_system_message(self.SYSTEM_MESSAGE)
            prefix = self.PRE.format(query=query, num=num)
            rank = 0
            input_context = f"{prefix}\n"
            for text in texts:
                rank += 1
                content = " ".join(text.split()[:int(max_length)])
                input_context += f"[{rank}] {replace_number(content)}\n"

            input_context += self.POST.format(query=query, num=num)
            conv.append_message(conv.roles[0], input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.MAX_TOKENS - self.num_output_tokens(num):
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.MAX_TOKENS + self.num_output_tokens(num))
                    // ((num) * 4),
                )
        return prompt