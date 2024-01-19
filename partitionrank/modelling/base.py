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

    def parse_output(self, output : str) -> List[int]:
        all_num = re.findall(r'\[(\d+)\]', output)
        return [*map(int, list(set(all_num)))]
    
    def logic(self, texts : List[str]):
        if isinstance(texts, str): texts = [texts]
        inputs = self.tokenizer(texts)
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        texts = self.model.generate(**inputs, generation_config=self.generation_config)
        if self._llm.config.is_encoder_decoder: output_ids = output_ids[0]
        else: output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = self._tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
        return [*map(self.parse_output, outputs)]