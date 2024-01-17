from collections import namedtuple
from dataclasses import dataclass
from typing import List
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from abc import ABC, abstractmethod


class QueryLog(namedtuple):
    qid : str
    inferences : int = 0
    in_tokens : int = 0
    out_tokens : int = 0

@dataclass
class MainLog:
    queries : List[QueryLog] = []

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

    PRE = ''
    POST = ''
    MODEL_TYPE = ''
    MAX_LENGTH = 200
    CONTEXT_LENGTH = 4096

    def __init__(self, 
                 partition_type : str,  
                 stride : int = 10, 
                 window : int = 20, 
                 depth : int = 100, 
                 shuffle : bool = False,
                 mode : str = 'sliding') -> None:
        super().__init__()

        self.partition_type = partition_type
        self.stride = stride
        self.window = window
        self.depth = depth
        self.shuffle = shuffle

        self.process = {
            'sliding': self.sliding_window,
            'pivot': self.pivot
        }[mode]
    
    @abstractmethod
    def score(self, subset : pd.DataFrame, start : int):
        raise NotImplementedError
    
    def sliding_window(self, subset : pd.DataFrame):
        in_token, out_token = 0, 0
        subset = subset.sort_values('score', ascending=False).reset_index(drop=True).iloc[:self.depth] # get top k
        if self.shuffle: subset = subset.sample(frac=1).reset_index(drop=True) # shuffle
        subset['score'] = [1/(i+1) for i in range(len(subset))] # set initial score

        end = self.depth # set initial end
        start = end - self.window # set initial start

        while start >= 0: # while start is greater than 0
            start = max(0, start) 
            end = min(self.depth, end)
            current = self.score(subset.iloc[start:end].copy(), start)
            subset.iloc[start:end] = current
            end -= self.stride
            start -= self.stride

        return subset, in_token, out_token             

    def pivot(self, subset : pd.DataFrame):
        in_token, out_token = 0, 0
        subset = subset.sort_values('score', ascending=False).reset_index(drop=True).iloc[:self.depth]
        if self.shuffle: subset = subset.sample(frac=1).reset_index(drop=True)
        subset['score'] = [1/(i+1) for i in range(len(subset))]

        start = 0
        end = start + self.stride 

        # sort initial to get pivot point 

        current = subset.iloc[start:end].copy()
        scores = self.score(current, start)
        current = current.iloc[scores[::-1]].reset_index(drop=True)
        current['score'] = [1/(start+i) for i in range(len(current))]

        candidates = current.copy()
        pivot = candidates.iloc[-1]

        start += self.stride
        end += self.stride

        while end <= len(subset):
            start = max(0, start)
            end = min(len(subset), end)

            current = subset.iloc[start:end-1].copy()
            current.append(pivot)
            scores = self.score(current, start)
            current = current.iloc[scores[::-1]].reset_index(drop=True)
            # get index of pivot doc id
            pivot_index = subset[subset['docid'] == pivot['docid']].index[0]
            more_rel = current.iloc[:pivot_index]
            more_rel.append(pivot)
            candidates = pd.concat([candidates.iloc[:-1], more_rel], ignore_index=True)

    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        in_token, out_token = 0, 0
        out = []
        for _, subset in topics_or_res.groupby('qid'):
            _subset, _in_token, _out_token = self.process(subset)
            in_token += _in_token
            out_token += _out_token
            out.append(_subset)
        return pd.concat(out, ignore_index=True)