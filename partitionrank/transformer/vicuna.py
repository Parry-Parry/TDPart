from typing import Optional, Union, List
import numpy as np
from partitionrank.transformer import ListWiseTransformer
from partitionrank.modelling.prompt import RankPrompt
from partitionrank.modelling.base import LLMRanker
import torch

class RankVicuna(ListWiseTransformer):

    CHECKPOINT = 'castorini/rank_vicuna_7b_v1'
    MAX_LENGTH = 300

    def __init__(self, 
                 device : Union[str, int] = 'cuda', 
                 n_gpu : Optional[int] = 1, 
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = LLMRanker(checkpoint=self.CHECKPOINT, device=device, n_gpu=n_gpu)
        self.prompt = RankPrompt(model=self.CHECKPOINT, tokenizer=self.model._tokenizer, max_length=self.MAX_LENGTH, rankllm=False)
    
    def score(self, query : str, doc_text : List[str], window_len : int, **kwargs):
        self.current_query.inferences += 1
        prompt = self.prompt(query=query, texts=doc_text, num=window_len)
        order = self.model(text=prompt, window_len=window_len)
        return np.array(order)