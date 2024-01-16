from typing import Optional, Union
from partitionrank.transformer import ListWiseTransformer
from partitionrank.modelling.prompt import RankPrompt
from partitionrank.modelling.base import LLMRanker
import pandas as pd 

class RankVicuna(ListWiseTransformer):

    PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}."
    POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."
    CHECKPOINT = ''
    MAX_LENGTH = 200
    CONTEXT_LENGTH = 4096

    def __init__(self, device : Union[str, int] = 'cuda', n_gpu : Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.prompt = RankPrompt([self.PRE, '{documents}', self.POST], doc_formatter=True, model=self.CHECKPOINT, max_length=self.MAX_LENGTH)
        self.model = LLMRanker(checkpoint=self.CHECKPOINT, device=device, n_gpu=n_gpu, fast_chat=True)
        self.chain = self.prompt >> self.model
    
    def score(self, subset : pd.DataFrame, start : int):
        query = subset['query'].iloc[0]
        texts = current['text'].tolist()
        order = self.chain(query=query, texts=texts, num=len(texts))
        
        current = current.iloc[order].reset_index(drop=True)
        current['score'] = [1/(start+i) for i in range(len(current))]
        return current