from pandas import DataFrame
from partitionrank.transformer import ListWiseTransformer

class RankZephyr(ListWiseTransformer):

    PRE = ''
    POST = ''
    MODEL_TYPE = 'zephyr'
    MAX_LENGTH = 200
    CONTEXT_LENGTH = 4096

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        pass 