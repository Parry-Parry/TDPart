PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}."
POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."

import pyterrier as pt 
if not pt.started(): pt.init()
from abc import ABC

class ListWiseTransformer(pt.Transformer, ABC):
    def __init__(self, partition_type : str, stride : int = 10, window : int = 20) -> None:
        super().__init__()
        self.partition_type = partition_type
        self.stride = stride
        self.window = window
    
    def sliding_window():
        raise NotImplementedError

    def pivot():
        raise NotImplementedError