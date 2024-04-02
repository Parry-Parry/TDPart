import pyterrier as pt 
if not pt.started(): pt.init()
from . import ListWiseTransformer
import numpy as np

class SortOracleTransformer(ListWiseTransformer):
    def __init__(self, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 mode = 'sliding', 
                 max_iters : int = 100) -> None:
        super().__init__(window_size=window_size, stride=stride, buffer=buffer, mode=mode, max_iters=max_iters)
    
    def score(self, qid : str, doc_text, **kwargs):
        self.current_query.inferences += 1
        doc_text = doc_text.tolist()
        order = sorted(range(len(doc_text)), key=lambda k: doc_text[k], reverse=True)
        return np.array(order)