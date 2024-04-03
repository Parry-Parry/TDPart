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
    
class RankedList(object):
    def __init__(self, doc_texts=None, doc_idx=None) -> None:
        self.doc_texts = doc_texts if doc_texts is not None else np.array([])
        self.doc_idx = doc_idx if doc_idx is not None else np.array([])
    
    def __len__(self):
        return len(self.doc_idx)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RankedList(self.doc_idx[key], self.doc_texts[key])
        elif isinstance(key, int):
            return self.doc_idx[key], self.doc_texts[key]
        elif isinstance(key, (list, np.ndarray)):
            return RankedList([self.doc_idx[i] for i in key], [self.doc_texts[i] for i in key])
        else:
            raise TypeError("Invalid key type. Please use int, slice, list, or numpy array.")

    def __setitem__(self, key, value):
        # THIS IS INCORRECT
        if isinstance(key, int):
            self.doc_idx[key], self.doc_texts[key] = value
        elif isinstance(key, (list, np.ndarray)):
            if isinstance(value, RankedList):
                if len(key) != len(value):
                    raise ValueError("Assigning RankedList requires the same length as the key.")
                for i, idx in enumerate(key):
                    self.doc_idx[idx], self.doc_texts[idx] = value.doc_idx[i], value.doc_texts[i]
            else:
                for i, idx in enumerate(key):
                    self.doc_idx[idx], self.doc_texts[idx] = value[i]
        else:
            raise TypeError("Invalid key type. Please use int, list, or numpy array.")

    def __add__(self, other):
        if not isinstance(other, RankedList):
            raise TypeError("Unsupported operand type(s) for +: 'RankedList' and '{}'".format(type(other)))
        return RankedList(concat(self.doc_idx, other.doc_idx), concat(self.doc_texts, other.doc_texts))

class ListWiseTransformer(pt.Transformer, ABC):

    def __init__(self, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 cutoff : int = 10, 
                 n_child : int = 3,
                 mode='sliding', 
                 max_iters : int = 100,
                 depth : int = 100,
                 verbose : bool = False,
                 **kwargs) -> None:
        super().__init__()

        self.window_size = window_size
        self.stride = stride
        self.buffer = buffer
        self.cutoff = cutoff - 1
        self.num_child = n_child
        self.max_iters = max_iters
        self.depth = depth
        self.verbose = verbose

        assert cutoff < window_size, "cutoff must be less than window_size"

        self.log = MainLog()
        self.current_query = None

        self.process = {
            'sliding': self.sliding_window,
            'pivot': self.pivot,
            'single': self.single_window
        }[mode]

    
    @abstractmethod
    def score(self, *args, **kwargs):
        raise NotImplementedError
    
    def _pivot(self, qid : str, query : str, ranking : RankedList):
        '''
        l : current left partition being scored
        r : current right partition being the remainder of the array
        c : current candidates
        b : current backfill
        p : current pivot
        '''
        logging.info(f"Processing query {qid} with {len(ranking)} documents")
        l = ranking[:self.window_size]
        r = ranking[self.window_size:]

        kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': l.doc_texts,
            'doc_idx': l.doc_idx,
            'start_idx': 0,
            'end_idx': len(l), # initial sort may be less than window size
            'window_len': len(l)
        }

        order = np.array(self.score(**kwargs))
        orig_idxs = np.arange(len(l))
        l.doc_idx[orig_idxs], l.doc_texts[orig_idxs],  = l.doc_idx[order], l.doc_texts[order]
        logging.info(f"Initial sort complete for query {qid}, len: {len(l)}")
        if len(l) < self.window_size: 
            logging.info('Breaking out')
            return l, r, True # breakout as only single sort is required
        p = l[self.cutoff]
        c = l[:self.cutoff]
        b = l[self.cutoff+1:]
        sub_window_size = self.window_size - 1 # account for addition of p

        while len(c) < self.buffer and len(r) > 0:
            l, r = _split(r, sub_window_size)
            l = p + l

            kwargs = {
                'qid': qid,
                'query': query,
                'doc_text': l.doc_texts.tolist(),
                'doc_idx': l.doc_idx.tolist(),
                'start_idx': 0,
                'end_idx': len(l),
                'window_len': len(l)
            }

            order = np.array(self.score(**kwargs))
            orig_idxs = np.arange(len(l))
            l[orig_idxs] = l[order]

            p_idx = np.where(l.doc_idx == p.doc_idx[0])[0][0] # find index of pivot id
            # add left of pivot to candidates and right of pivot to backfill
            c = c + l[:p_idx]
            b = b + l[p_idx+1:]
        
        # we have found no candidates better than p
        if len(c) == self.cutoff - 1: 
            top = c + p 
            bottom = b + r
            return top, bottom, True 
        # we have found candidates better than p

        # split c by budget b
        c, ac = _split(c, self.buffer)
        ac = ac + p + b + r

        return c, ac, False
    
    def pivot(self, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        self.current_query = QueryLog(qid=qid)
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()

        indicator = False
        num_iters = 0
        c = RankedList(doc_texts, doc_idx)
        b = RankedList()

        while not indicator and num_iters < self.max_iters:
            num_iters += 1
            c, _b, indicator = self._pivot(qid, query, c)
            b = b + _b
        if num_iters == self.max_iters:
            print(f"WARNING: max_iters reached for query {qid}")

        self.log.queries.append(self.current_query)
        out = c + b
        return out.doc_idx, out.doc_texts
    
    def sliding_window(self, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        self.current_query = QueryLog(qid=qid)
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_texts, doc_idx)
        for start_idx, end_idx, window_len in _iter_windows(len(query_results), self.window_size, self.stride):
            kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': ranking[start_idx:end_idx].doc_texts.tolist(),
            'doc_idx': ranking[start_idx:end_idx].doc_idx.tolist(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_len': window_len
            }
            order = np.array(self.score(**kwargs))
            new_idxs = start_idx + order
            orig_idxs = np.arange(start_idx, end_idx)
            ranking[orig_idxs] = ranking[new_idxs]
        self.log.queries.append(self.current_query)
        return ranking.doc_idx, ranking.doc_texts
    
    def single_window(self, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        self.current_query = QueryLog(qid=qid)
        query_results = query_results.sort_values('score', ascending=False)
        candidates = query_results.iloc[:self.window_size]
        rest = query_results.iloc[self.window_size:]
        doc_idx = candidates['docno'].to_numpy()
        doc_texts = candidates['text'].to_numpy()
        rest_idx = rest['docno'].to_numpy()
        rest_texts = rest['text'].to_numpy()
        
        kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': doc_texts.tolist(),
            'doc_idx': doc_idx.tolist(),
            'start_idx': 0,
            'end_idx': len(doc_texts),
            'window_len': len(doc_texts)
        }
        order = np.array(self.score(**kwargs))
        orig_idxs = np.arange(0, len(doc_texts))
        doc_idx[orig_idxs] = doc_idx[order]
        doc_texts[orig_idxs] = doc_texts[order]
        self.log.queries.append(self.current_query)

        return concat([doc_idx, rest_idx]), concat([doc_texts, rest_texts])
    
    # from https://github.com/ielab/llm-rankers/blob/main/llmrankers/setwise.py

    def _heapify(self, query, ranking, n, i):
        # Find largest among root and children
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        li_comp = self.score(**{
            'query': query['query'].iloc[0],
            'doc_text': [ranking.doc_texts[i], ranking.doc_texts[l]],
            'start_idx': 0,
            'end_idx': 1,
            'window_len': 2
        })
        rl_comp = self.score(**{
            'query': query['query'].iloc[0],
            'doc_text': [ranking.doc_texts[r], ranking.doc_texts[largest]],
            'start_idx': 0,
            'end_idx': 1,
            'window_len': 2
        })
        if l < n and li_comp == 0: largest = l
        if r < n and rl_comp == 0: largest = r

        # If root is not largest, swap with largest and continue heapifying
        if largest != i:
            ranking[i], ranking[largest] = ranking[largest], ranking[i]
            self._heapify(query, ranking, n, largest)

    def _setwise(self, query : str, query_results : pd.DataFrame):
        self.current_query = QueryLog(qid=qid)
        qid = query_results['qid'].iloc[0]
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_texts, doc_idx)
        n = len(query_results)
        ranked = 0
        # Build max heap
        for i in range(n // 2, -1, -1):
            self._heapify(query, ranking, n, i)
        for i in range(n - 1, 0, -1):
            # Swap
            ranking[i], ranking[0] = ranking[0], ranking[i]
            ranked += 1
            if ranked == self.k:
                break
            # Heapify root element
            self._heapify(query, ranking, i, 0)     
        self.log.queries.append(self.current_query)
        return ranking.doc_idx, ranking.doc_texts  

    def transform(self, inp : pd.DataFrame):
        res = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'rank': [],
        }
        progress = not self.verbose
        for (qid, query), query_results in tqdm(inp.groupby(['qid', 'query']), unit='q', disable=progress):
            query_results.sort_values('score', ascending=False, inplace=True)
            with torch.no_grad():
                doc_idx, doc_texts = self.process(query, query_results.iloc[:self.depth])
            res['qid'].extend([qid] * len(doc_idx))
            res['query'].extend([query] * len(doc_idx))
            res['docno'].extend(doc_idx)
            res['text'].extend(doc_texts)
            res['rank'].extend(list(range(len(doc_idx))))
        res = pd.DataFrame(res)
        res['score'] = -res['rank'].astype(float)
        return res