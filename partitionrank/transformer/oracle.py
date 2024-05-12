from collections import defaultdict
import pandas as pd
import pyterrier as pt 
import numpy as np
if not pt.started(): pt.init()
from . import ListWiseTransformer

class OracleTransformer(ListWiseTransformer):
    def __init__(self, 
                 qrels : pd.DataFrame, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 mode = 'sliding', 
                 max_iters : int = 100,
                 **kwargs) -> None:
        super().__init__(window_size=window_size, stride=stride, buffer=buffer, mode=mode, max_iters=max_iters)
        self.qrels = qrels
    
    def score(self, qid : str, doc_idx, **kwargs):
        self.current_query.inferences += 1
        doc_idx = doc_idx.tolist()
        q_rels = self.qrels[self.qrels['query_id'] == qid].set_index('doc_id').relevance.to_dict()
        q_rels = defaultdict(lambda: 0, q_rels)

        doc_rel = [q_rels[i] for i in doc_idx]
        order = sorted(range(len(doc_rel)), key=lambda k: doc_rel[k], reverse=True)
        return order