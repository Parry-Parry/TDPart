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

        default = {qid: None for qid in self.qrels['query_id'].unique()} # defaults before first population
        self.current, self.rel_ptr, self.nrel_ptr = copy.deepcopy(default), copy.deepcopy(default), copy.deepcopy(default)
        self.new_sample()

    def new_sample(self):
        for qid in self.qids: # Get a new sample for each query before each iteration
            _qrels = self.qrels[self.qrels['query_id'] == qid]
            self.nrel_ptr[qid] = _qrels[_qrels['relevance'] < self.cutoff].sample(n=self.num_items-1, replace=False)
            self.rel_ptr[qid] = _qrels[_qrels['relevance'] >= self.cutoff].sample(n=self.num_items-1, replace=False)

    def get_samples(self, qid):
        for ratio in self.ratios: # iterate over the ratios
            if self.current[qid] is None: # first ratio
                # Initialize pointers in the first iteration
                rel_end = int(ratio * self.num_items) # num_relevant is (0, ratio*num_items) 
                nrel_end = int((1-ratio)*self.num_items) # num_non_relevant is (0, (1-ratio)*num_items)
                self.current[qid] = (rel_end, nrel_end, ratio) # store the current state
            else:
                rel_end, nrel_end, prev = self.current[qid] # get the previous state
                new_rel = int((ratio - prev) * self.num_items) # get the new number of relevant documents
                rel_end += new_rel # update the number of relevant documents
                nrel_end -= new_rel # update the number of non-relevant documents

                self.current[qid] = (rel_end, nrel_end, ratio) # store the current state

            print(f'qid: {qid}, rel_end: {rel_end}, nrel_end: {nrel_end}')
            print(self.rel_ptr[qid].iloc[:rel_end])
            print(self.nrel_ptr[qid].iloc[:nrel_end])
            next_samples = pd.concat([
                self.rel_ptr[qid].iloc[:rel_end], # get the relevant documents
                self.nrel_ptr[qid].iloc[:nrel_end] # get the non-relevant documents
            ])
            yield next_samples, ratio  # Yield the samples 



def sort_df(df, order):
    if order == Order.ASC:
        df = df.sort_values(by=['relevance'])
    elif order == Order.DESC:
        df = df.sort_values(by=['relevance'], ascending=False)
    else:
        df = df.sample(frac=1)
    df['score'] = [-(i+1) for i in range(len(df))]
    return df