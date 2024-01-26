from typing import Dict, List, Any, Tuple, Union
from transformers.generation import GenerationConfig
from ftfy import fix_text
import time
import tiktoken
from .prompt import replace_number
import openai
import re
import torch

# Based on https://github.com/castorini/rank_llm/blob/main/src/rank_llm/

class GPTRanker:
    def __init__(self, 
                 model: str, 
                 context_size: int, 
                 key,) -> None:

        self.model = model
        self.context_size = context_size

        openai.api_key = key

    def parse_output(self, output : str, length : int) -> List[int]:
        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < length}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(length) if i not in output] # backfill missing passages
        return order
    
    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        if self.model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"]:
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif self.model in ["gpt-4-0314", "gpt-4"]:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message, tokens_per_name = 0, 0

        try:
            encoding = tiktoken.get_encoding(self.model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        else:
            num_tokens += len(encoding.encode(prompt))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _get_prefix_for_rank_gpt_prompt(
        self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_suffix_for_rank_gpt_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    def create_rank_gpt_prompt(
        self, query, texts, num
    ) -> Tuple[List[Dict[str, str]], int]:
        max_length = 300
        while True:
            messages = self._get_prefix_for_rank_gpt_prompt(query, num)
            rank = 0
            for text in texts:
                rank += 1
                content = text.strip()
                content = fix_text(content)
                content = " ".join(content.split()[: int(max_length)])
                messages.append(
                    {"role": "user", "content": f"[{rank}] {replace_number(content)}"}
                )
                messages.append(
                    {"role": "assistant", "content": f"Received passage [{rank}]."}
                )
            messages.append(
                {
                    "role": "user",
                    "content": self._get_suffix_for_rank_gpt_prompt(query, num),
                }
            )
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(num):
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens(num))
                    // ((num) * 4),
                )
        return messages, self.get_num_tokens(messages)

    def _call_completion(
        self,
        *args,
        return_text=False,
        reduce_length=False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    *args, **kwargs, timeout=30
                )
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "The response was filtered" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                time.sleep(0.1)
        if return_text:
            completion = (
                completion["choices"][0]["message"]["content"]
            )
        return completion
    
    def __call__(self, text : str, window_len : int):
        if isinstance(text, str): text = [text]
        inputs = self._tokenizer(text)
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}

        gen_cfg = GenerationConfig.from_model_config(self._model.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(window_len)
        gen_cfg.min_new_tokens = self.num_output_tokens(window_len)
        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False

        output_ids = self._model.generate(**inputs, generation_config=gen_cfg)
        if self._model.config.is_encoder_decoder: output_ids = output_ids[0]
        else: output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self._tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
        return self.parse_output(outputs, window_len)