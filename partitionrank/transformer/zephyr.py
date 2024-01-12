from pandas import DataFrame
import pyterrier as pt 
if not pt.started(): pt.init()
from partitionrank.modelling.base import LLMRanker
from partitionrank.modelling.prompt import RankPrompt

class RankZephyr(pt.Transformer):

    PRE = ''
    POST = ''
    MODEL_TYPE = 'zephyr'
    MAX_LENGTH = 200
    CONTEXT_LENGTH = 4096

    def __init__(self, checkpoint : str, n_gpu : int) -> None:
        self.chain = RankPrompt(self.PRE, self.POST, self.MODEL_TYPE, self.MAX_LENGTH, self.CONTEXT_LENGTH) >> LLMRanker(checkpoint, n_gpu=n_gpu)

    def sliding_window()

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        pass 