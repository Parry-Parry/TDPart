from typing import Optional, Union, List
import numpy as np
from partitionrank.transformer import ListWiseTransformer
from partitionrank.modelling.prompt import RankPrompt
from partitionrank.modelling.base import LLMRanker
import torch

class RankVicuna(ListWiseTransformer):

    PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}."
    POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."
    CHECKPOINT = 'castorini/rank_vicuna_7b_v1'
    MAX_LENGTH = 200

    def __init__(self, 
                 device : Union[str, int] = 'cuda', 
                 n_gpu : Optional[int] = None, 
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.prompt = RankPrompt(components=[self.PRE, '{documents}', self.POST], doc_formatter=True, model=self.CHECKPOINT, max_length=self.MAX_LENGTH)
        self.model = LLMRanker(checkpoint=self.CHECKPOINT, device=device, n_gpu=n_gpu)
    
    def score(self, query : str, doc_text : List[str], window_len : int, **kwargs):
        self.current_query.inferences += 1
        prompt = self.prompt(query=query, texts=doc_text.tolist())
        order = self.model(prompt=prompt, num=window_len)
        return order