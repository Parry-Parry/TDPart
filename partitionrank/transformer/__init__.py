from dataclasses import dataclass, field
import logging
from typing import List
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import numpy as np
from numpy import concatenate as concat
import torch

def _iter_windows(n, window_size, stride):
    # TODO: validate window_size and stride
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride), unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len

def _split(l, i):
    return l[:i], l[i:]

@dataclass
class QueryLog:
    qid : str
    inferences : int = 0
    in_tokens : int = 0
    out_tokens : int = 0

@dataclass
class MainLog:
    queries : List[QueryLog] = field(default_factory=list) 

    @property
    def inferences(self):
        return sum([i.inferences for i in self.queries])

    @property
    def in_tokens(self):
        return sum([i.in_tokens for i in self.queries])
    
    @property
    def out_tokens(self):
        return sum([i.out_tokens for i in self.queries])

class ListWiseTransformer(pt.Transformer, ABC):

    def __init__(self, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 cutoff : int = 10, 
                 mode='sliding', 
                 max_iters : int = 100) -> None:
        super().__init__()

        self.window_size = window_size
        self.stride = stride
        self.buffer = buffer
        self.cutoff = cutoff - 1
        self.max_iters = max_iters

        assert cutoff < window_size, "cutoff must be less than window_size"

        self.log = MainLog()
        self.current_query = None

        self.process = {
            'sliding': self.sliding_window,
            'pivot': self.pivot
        }[mode]
    
    @abstractmethod
    def score(self, *args, **kwargs):
        raise NotImplementedError
    
    def _pivot(self, qid : str, query : str, doc_idx : List[str], doc_texts : List[str]):
        '''
        l : current left partition being scored
        r : current right partition being the remainder of the array
        c : current candidates
        b : current backfill
        p : current pivot
        '''
        logging.info(f"Processing query {qid} with {len(doc_idx)} documents")
        l_idx, l_text = doc_idx[:self.window_size], doc_texts[:self.window_size]
        r_idx, r_text = doc_idx[self.window_size:], doc_texts[self.window_size:]

        kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': l_text,
            'doc_idx': l_idx,
            'start_idx': 0,
            'end_idx': len(l_text), # initial sort may be less than window size
            'window_len': len(l_text)
        }

        order = np.array(self.score(**kwargs))
        orig_idxs = np.arange(len(l_text))
        l_idx[orig_idxs], l_text[orig_idxs],  = l_idx[order], l_text[order]
        logging.info(f"Initial sort complete for query {qid}, len: {len(l_text)}")
        if len(l_text) < self.window_size: 
            logging.info('Breaking out')
            breakpoint()
            return l_idx, l_text, r_idx, r_text, True # breakout as only single sort is required
        p_id, p_text = l_idx[self.cutoff], l_text[self.cutoff]

        c_idx, c_text = l_idx[:self.cutoff], l_text[:self.cutoff] # create initial < p
        b_idx, b_text = l_idx[self.cutoff+1:], l_text[self.cutoff+1:] # create initial > p
        breakpoint()
        sub_window_size = self.window_size - 1 # account for addition of p

        while len(c_text) <= self.buffer and len(r_text) >= sub_window_size:
            l_text, r_text = _split(r_text, sub_window_size)
            l_idx, r_idx = _split(r_idx, sub_window_size)

            # prepend pivot to left partition
            l_text = concat([[p_text], l_text])
            l_idx = concat([[p_id], l_idx])

            kwargs = {
                'qid': qid,
                'query': query,
                'doc_text': l_text,
                'doc_idx': l_idx,
                'start_idx': 0,
                'end_idx': self.window_size,
                'window_len': self.window_size
            }

            order = np.array(self.score(**kwargs))
            orig_idxs = np.arange(self.window_size)
            l_idx[orig_idxs], l_text[orig_idxs],  = l_idx[order], l_text[order]
            breakpoint()
            p_idx = np.where(l_idx == p_id)[0][0] # find index of pivot id
            # add left of pivot to candidates and right of pivot to backfill
            c_text = concat([c_text, l_text[:p_idx]])
            c_idx = concat([c_idx, l_idx[:p_idx]])
            b_text = concat([b_text, l_text[p_idx+1:]])
            b_idx = concat([b_idx, l_idx[p_idx+1:]])
        
        # we have found no candidates better than p
        if len(c_text) == self.cutoff - 1: return concat([c_idx, [p_id]]), concat([c_text, [p_text]]), b_idx, b_text, True 
        # we have found candidates better than p
        return c_idx, c_text, concat([[p_id], b_idx]), concat([[p_text], b_text]), False
    
    def pivot(self, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        self.current_query = QueryLog(qid=qid)
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()

        indicator = False
        num_iters = 0
        c_idx, c_text = doc_idx, doc_texts
        b_idx, b_text = [], []

        while not indicator and num_iters < self.max_iters:
            num_iters += 1
            c_idx, c_text, _idx, _text, indicator = self._pivot(qid, query, c_idx, c_text)
            b_idx = concat([b_idx, _idx])
            b_text = concat([b_text, _text])
        if num_iters == self.max_iters:
            print(f"WARNING: max_iters reached for query {qid}")

        self.log.queries.append(self.current_query)

        return concat([c_idx, b_idx]), concat([c_text, b_text])
    
    def sliding_window(self, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        self.current_query = QueryLog(qid=qid)
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        for start_idx, end_idx, window_len in _iter_windows(len(query_results), self.window_size, self.stride):
            kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': doc_texts[start_idx:end_idx],
            'doc_idx': doc_idx[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_len': window_len
            }
            order = np.array(self.score(**kwargs))
            new_idxs = start_idx + order
            orig_idxs = np.arange(start_idx, end_idx)
            doc_idx[orig_idxs] = doc_idx[new_idxs]
            doc_texts[orig_idxs] = doc_texts[new_idxs]
        self.log.queries.append(self.current_query)
        return doc_idx, doc_texts
    
    def single_window(self, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        self.current_query = QueryLog(qid=qid)
        query_results = query_results.sort_values('score', ascending=False).iloc[:self.window_size]
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        
        kwargs = {
        'qid': qid,
        'query': query,
        'doc_text': doc_texts,
        'doc_idx': doc_idx,
        'start_idx': 0,
        'end_idx': len(doc_texts),
        'window_len': len(doc_texts)
        }
        order = np.array(self.score(**kwargs))
        orig_idxs = np.arange(0, len(doc_texts))
        doc_idx[orig_idxs] = doc_idx[order]
        doc_texts[orig_idxs] = doc_texts[order]
        self.log.queries.append(self.current_query)
        return doc_idx, doc_texts

    def transform(self, inp : pd.DataFrame):
        res = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'rank': [],
        }
        with torch.no_grad():
            for (qid, query), query_results in tqdm(inp.groupby(['qid', 'query']), unit='q'):
                doc_idx, doc_texts = self.process(query, query_results)
                res['qid'].extend([qid] * len(doc_idx))
                res['query'].extend([query] * len(doc_idx))
                res['docno'].extend(doc_idx)
                res['text'].extend(doc_texts)
                res['rank'].extend(list(range(len(doc_idx))))
        res = pd.DataFrame(res)
        res['score'] = -res['rank'].astype(float)
        return res