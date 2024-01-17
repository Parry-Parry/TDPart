from collections import defaultdict
from typing import List
import numpy as np
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from tqdm.auto import tqdm
from .lit_t5 import _iter_windows, _split

class OracleTransformer(pt.Transformer):
    def __init__(self, qrels : pd.DataFrame, window_size=20, stride=10, buffer : int = 20, mode = 'sliding') -> None:
        super().__init__()
        self.qrels = qrels

        self.window_size = window_size  
        self.stride = stride
        self.buffer = buffer

        self.process = {
            'sliding': self.sliding_window,
            'pivot': self.pivot
        }[mode]
    
    def score(self, qid, doc_idx):
        q_rels = self.qrels[self.qrels['qid'] == qid].set_index('docno').relevance.to_dict()
        q_rels = defaultdict(lambda: 0, q_rels)
        
        doc_rel = [q_rels[i] for i in doc_idx]
        order = sorted(range(len(doc_rel)), key=lambda k: doc_rel[k], reverse=True)
        return order

    def _pivot(self, query : str, doc_idx : List[str], doc_texts : List[str]):
        '''
        l : current left partition being scored
        r : current right partition being the remainder of the array
        c : current candidates
        b : current backfill
        p : current pivot
        '''
        l_text, l_idx = doc_texts[:self.window_size], doc_idx[:self.window_size]
        r_text, r_idx = doc_texts[self.window_size:], doc_idx[self.window_size:]

        order = self.score(query, l_text, 0, self.window_size, self.window_size)
        orig_idxs = np.arange(self.window_size)
        l_text[orig_idxs], l_idx[orig_idxs] = l_text[order], l_idx[order]

        p_id, p_text = doc_idx[order[9]], doc_texts[order[9]]
        c_text = np.concatenate([l_text[:9],l_text[10:self.window_size]])
        c_idx = np.concatenate([l_idx[:9],l_idx[10:self.window_size]])

        b_text, b_idx = [], []
        sub_window_size = self.window_size - 1

        cutoff = len(r_text) % sub_window_size
        r_text, r_idx = r_text[:-cutoff], r_idx[:-cutoff]

        while len(c_text) < self.buffer and len(r_text) > 0:
            l_text, r_text = _split(r_text, sub_window_size)
            l_idx, r_idx = _split(r_idx, sub_window_size)

            # prepend pivot to left partition
            l_text = np.concatenate([[p_text], l_text])
            l_idx = np.concatenate([[p_id], l_idx])

            order = self.score(query, l_text, 0, self.window_size, self.window_size)
            orig_idxs = np.arange(self.window_size)
            l_text[orig_idxs], l_idx[orig_idxs] = l_text[order], l_idx[order]

            # find index of pivot id
            p_idx = np.where(l_idx == p_id)[0][0]
            # add left of pivot to candidates and right of pivot to backfill
            c_text = np.concatenate([c_text, l_text[:p_idx]])
            c_idx = np.concatenate([c_idx, l_idx[:p_idx]])
            b_text = np.concatenate([b_text, l_text[p_idx+1:]])
            b_idx = np.concatenate([b_idx, l_idx[p_idx+1:]])
        
        return c_idx[:self.buffer], c_text[:self.buffer], b_idx, b_text
        

    def pivot(self, query : str, query_results : pd.DataFrame):
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()

        c_idx, c_text, f_idx, f_text = self._pivot(query, doc_idx, doc_texts)
        c_idx, c_text, b_idx, b_text = self._pivot(query, c_idx, c_text)

        c_idx = np.concatenate([c_idx, b_idx, f_idx])
        c_text = np.concatenate([c_text, b_text, f_text])

        return c_idx, c_text

    
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