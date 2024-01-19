from typing import List, Any, Optional
import torch
from lightchain import chainable
from fastchat.model import load_model
from transformers.generation import GenerationConfig
import re

@chainable
class LLMRanker:
    def __init__(self, checkpoint : Any, max_new_tokens : int = 200, n_gpu : Optional[int] = None) -> None:
        device = 'cuda' if n_gpu else 'cpu'
        self.device = device
        self.model, self.tokenizer = load_model(checkpoint, device=device, n_gpu=n_gpu)
        
        config = GenerationConfig.from_model_config(self.model.config)
        config.max_new_tokens = max_new_tokens
        config.min_length = 1
        config.do_sample = False
        self.generation_config = config

    def parse_output(self, output : str, length : int) -> List[int]:
        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < length}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(length) if i not in output] # backfill missing passages
        return order
    
    def logic(self, texts : List[str], window_len : List[int]):
        if isinstance(texts, str): texts, window_len = [texts], [window_len]
        inputs = self.tokenizer(texts)
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        texts = self.model.generate(**inputs, generation_config=self.generation_config)
        if self._llm.config.is_encoder_decoder: output_ids = output_ids[0]
        else: output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self._tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
        return [*map(lambda x, y : self.parse_output(x, y), zip(outputs, window_len))]