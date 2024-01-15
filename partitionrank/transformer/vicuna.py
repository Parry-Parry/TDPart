from pandas import DataFrame
from partitionrank.transformer import ListWiseTransformer

class RankVicuna(ListWiseTransformer):

    PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}."
    POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."
    MODEL_TYPE = 'vicuna'
    MAX_LENGTH = 200
    CONTEXT_LENGTH = 4096

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)