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
    def __init__(self, qrels, num_items=20, cutoff=2, ratios=[0.2, 0.4, 0.6, 0.8]):
        self.qrels = qrels
        self.qids = self.qrels['query_id'].unique()
        self.num_items = num_items
        self.cutoff = cutoff
        self.ratios = ratios
        self.prev = 0.

        default = {qid : None for qid in self.qrels['query_id'].unique()}
        self.current, self.rel, self.nrel = copy.deepcopy(default), copy.deepcopy(default), copy.deepcopy(default)
        self.new_sample()

    def new_sample(self):
        for qid in self.qids:
            _qrels = self.qrels[self.qrels['query_id'] == qid]
            self.nrel[qid] = _qrels[_qrels['relevance'] < self.cutoff].copy().sample(n=self.num_items-1, replace=False)
            self.rel[qid] = _qrels[_qrels['relevance'] >= self.cutoff].copy().sample(n=self.num_items-1, replace=False)

    def get_samples(self, qid):
        for ratio in self.ratios:
            if self.current[qid] is None:
                self.current[qid] = (self.rel[qid].sample(n=int(ratio*self.num_items), replace=False), self.nrel[qid].sample(n=int((1-ratio)*self.num_items), replace=False))
                next_samples = pd.concat([*self.current[qid]])
            else:
                curr_rel, curr_nrel = self.current[qid]
                num_rel_to_replace = int((ratio-self.prev) * self.num_items)
                num_nrel_to_replace = self.num_items - num_rel_to_replace
                # Drop some of the current non-relevant documents
                curr_nrel = curr_nrel.iloc[num_nrel_to_replace:]
                # Add new relevant documents
                new_rel = self.rel[qid].iloc[:num_rel_to_replace]
                self.rel[qid] = new_rel
                next_samples = pd.concat([curr_nrel, new_rel])
                self.current[qid] = (new_rel, next_samples)
            self.prev = ratio
            yield next_samples, ratio

def sort_df(df, order):
    if order == Order.ASC:
        df = df.sort_values(by=['relevance'])
    elif order == Order.DESC:
        df = df.sort_values(by=['relevance'], ascending=False)
    else:
        df = df.sample(frac=1)
    df['score'] = [-(i+1) for i in range(len(df))]
    return df