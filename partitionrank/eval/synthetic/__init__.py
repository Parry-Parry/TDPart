import copy
from enum import Enum
import pandas as pd


MARCO = 'msmarco-passage/train/triples-small'
IR_DATASETS = ['msmarco-passage/trec-dl-2019/judged', 'msmarco-passage/trec-dl-2020/judged']

class Order(Enum):
    RANDOM = 0
    ASC = 1
    DESC = 2

def get_sample(qrels, qid, num_items : int = 20, order = Order.RANDOM, ratio : int = 1, cutoff : int = 2):
    ratio = int(ratio * num_items)
    qrels = qrels[qrels['query_id'] == str(qid)]
    non_relevant = qrels[qrels['relevance'] < cutoff].copy()
    relevant = qrels[qrels['relevance'] >= cutoff].copy()
    non_relevant = non_relevant.sample(n=num_items-ratio, replace=False)
    if len(relevant) < ratio: raise ValueError("Not enough relevant documents")
    relevant = relevant.sample(n=ratio, replace=False)
    qrels = pd.concat([non_relevant, relevant]).reset_index(drop=True)
    if order == Order.ASC:
        qrels = qrels.sort_values(by=['relevance'])
    elif order == Order.DESC:
        qrels = qrels.sort_values(by=['relevance'], ascending=False)
    qrels = qrels.rename(columns={'doc_id': 'docno', 'query_id': 'qid'}).reset_index(drop=True)
    qrels['score'] = [i for i in range(num_items)]
    return qrels

class Generator(object):
    def __init__(self, qrels, num_items : int = 20, cutoff : int = 2, ratios : float = [0.2, 0.4, 0.6, 0.8]):
        self.qrels = qrels
        self.qids = self.qrels['query_id'].unique()
        self.num_items = num_items
        self.cutoff = cutoff
        self.ratios = ratios

        default = {qid : None for qid in self.qrels['query_id'].unique()}
        self.current, self.rel, self.nrel = copy(default), copy(default), copy(default)
        self.new_sample()
    
    def new_sample(self):
        for qid in self.qids:
            self.rel[qid] = self.qrels[(self.qrels['query_id'] == qid) & (self.qrels['relevance'] >= self.cutoff)].copy().sample(n=self.num_items, replace=False)
            self.nrel[qid] = self.qrels[(self.qrels['query_id'] == qid) & (self.qrels['relevance'] < self.cutoff)].copy().sample(n=self.num_items, replace=False)
    
    def get_samples(self, qid):
        for ratio in self.ratios:
            self.current[qid] = pd.concat([self.rel[qid].sample(n=int(ratio*self.num_items), replace=False), self.nrel[qid].sample(n=int((1-ratio)*self.num_items), replace=False)]).reset_index(drop=True)
            yield self.current[qid]
