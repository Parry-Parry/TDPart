from typing import List
import numpy as np
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from tqdm.auto import tqdm
from .lit_t5 import _iter_windows

class OracleTransformer(pt.Transformer):
    def __init__(self, qrels : pd.DataFrame, window_size=20, stride=10, budget=1, depth : int = None, mode = 'sliding') -> None:
        super().__init__()
        self.qrels = qrels

        self.window_size = window_size  
        self.stride = stride
        self.max_budget = budget if not depth else (depth / stride) - 1
        self.budget = budget

        self.process = {
            'sliding': self.sliding_window,
            'pivot': self.pivot
        }[mode]
    
    def score(self, qid, doc_idx):
        q_rels = self.qrels[self.qrels['qid'] == qid].set_index('docno').relevance.to_dict()
        doc_rel = [q_rels[i] for i in doc_idx]
        order = sorted(range(len(doc_rel)), key=lambda k: doc_rel[k], reverse=True)
        return order
    
    def _reset_budget(self):
        self.budget = self.max_budget

    def _pivot(self, query : str, doc_idx : List[str], doc_texts : List[str], allocation : int = 0):
        end_idx = min(self.window_size, len(doc_texts))
        window_len = end_idx

        '''
        -- GET PIVOT --

        * sort top-k docs using model
        * take pivot as last element in sorted list
        '''

        order = self.score(query, doc_texts[:end_idx], 0, end_idx, window_len)
        orig_idxs = np.arange(end_idx)
        doc_idx[orig_idxs] = doc_idx[order]
        doc_texts[orig_idxs] = doc_texts[order]
        pivot_id = doc_idx[order[-1]]
        pivot_text = doc_texts[order[-1]]

        candidate_texts = doc_texts[:window_len-1].to_list() # add top-k-1 docs to candidates
        candidate_idxs = doc_idx[:window_len-1].to_list()
        filler_idx, filler_texts = [], [] # store for backfilling
        doc_idx = doc_idx[end_idx:] # pop processed docs
        doc_texts = doc_texts[end_idx:]

        '''
        -- GET CANDIDATES --

        * get next partition
        * sort partition using model
        * find pivot in sorted partition
        * add candidates to list
        '''

        for _ in range(allocation - 1):
            if not doc_texts: break
            end_idx = min(self.window_size - 1, len(doc_texts))
            window_len = end_idx + 1
            _texts = [pivot_text] + doc_texts[:end_idx].to_list() # get next partition
            _idx = [pivot_id] + doc_idx[:end_idx].to_list()
            doc_idx = doc_idx[end_idx:] # pop processed docs
            doc_texts = doc_texts[end_idx:]

            order = self.score(query, _texts, 0, end_idx, window_len) 
            _idx = np.array(_idx)[order]
            _texts = np.array(_texts)[order]

            id = _idx.index(pivot_id) # find pivot
            candidate_idxs.extend(_idx[:id]) # add candidates
            candidate_texts.extend(_texts[:id])
            filler_idx.extend(_idx[id+1:])
            filler_texts.extend(_texts[id+1:])
        
        return candidate_idxs, candidate_texts, filler_idx, filler_texts
        

    def pivot(self, query : str, query_results : pd.DataFrame):
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()

        first_allocation = self.budget // 2
        candidate_idxs, candidate_texts, filler_idx, filler_texts = self._pivot(query, doc_idx, doc_texts)
        self.budget -= first_allocation
        core_idxs, core_texts, sub_filler_idx, sub_filler_texts = self._pivot(query, candidate_idxs, candidate_texts, self.budget)

        sub_filler_idx.extend(filler_idx)
        sub_filler_texts.extend(filler_texts)

        core_idxs.extend(sub_filler_idx)
        core_texts.extend(sub_filler_texts)

        return core_idxs, core_texts

    
    def sliding_window(self, qid : str, query_results : pd.DataFrame):
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        for start_idx, end_idx, _ in _iter_windows(len(query_results), self.window_size, self.stride):
            order = self.score(qid, doc_texts[start_idx:end_idx].to_list())
            new_idxs = start_idx + np.array(order)
            orig_idxs = np.arange(start_idx, end_idx)
            doc_idx[orig_idxs] = doc_idx[new_idxs]
            doc_texts[orig_idxs] = doc_texts[new_idxs]
        return doc_idx, doc_texts
    
    def transform(self, inp):
        res = {
            'qid': [],
            'docno': [],
            'rank': [],
        }
        for qid, query_results in tqdm(inp.groupby('qid'), unit='q'):
            self._reset_budget()
            doc_idx = self.process(qid, query_results)
            res['qid'].extend([qid] * len(doc_idx))
            res['docno'].extend(doc_idx)
            res['rank'].extend(list(range(len(doc_idx))))
        res = pd.DataFrame(res)
        res['score'] = -res['rank'].astype(float)
        return res