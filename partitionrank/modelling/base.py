import logging
from typing import List, Any, Optional
import torch
from fastchat.model import load_model
from transformers.generation import GenerationConfig
import re

# Based on https://github.com/castorini/rank_llm/blob/main/src/rank_llm/

class LLMRanker:
    def __init__(self, checkpoint : Any, n_gpu : Optional[int] = None, device : Any = None) -> None:
        if not device:
            device = 'cuda' if n_gpu else 'cpu'
        self.device = device
        self._model, self._tokenizer = load_model(checkpoint, device=device, num_gpus=n_gpu)

    def parse_output(self, output : str, length : int) -> List[int]:
        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < length}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(length) if i not in output] # backfill missing passages
        return order
    
    def num_output_tokens(self, current_window_size : int) -> int:
        output_token_estimate = (
            len(
                self._tokenizer.encode(
                    " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                )
            )
            - 1
        )
        return output_token_estimate
    
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