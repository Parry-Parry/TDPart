from typing import Optional, Union, List
from partitionrank.transformer import ListWiseTransformer
from partitionrank.modelling.gpt import GPTRanker
import torch
import os

class RankGPT(ListWiseTransformer):
    CHECKPOINT = "gpt-3.5-turbo-0125"
    PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}."
    POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."
    MAX_LENGTH = 200

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = GPTRanker(model=self.CHECKPOINT, context_size=4096, key=os.getenv('OPENAI_API_KEY'))
    
    def score(self, query : str, doc_text : List[str], window_len : int, **kwargs):
        self.current_query.inferences += 1
        order = self.model(query=query, texts=doc_text.tolist(), num=window_len)
        return order