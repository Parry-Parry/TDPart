from typing import List
import ir_datasets as irds 
import pandas as pd
from enum import Enum
from ir_measures import *
from ir_measures import evaluator

MARCO = 'irds:msmarco-passage/train/triples-small'

class Order(Enum):
    RANDOM = 0
    ASC = 1
    DESC = 2

def sample(qrels, qid, num_items : int = 20, order = Order.RANDOM, ratio : float = 0.5):
    qrels = qrels[qrels['query_id'] == qid].copy()
    qrels = qrels.sample(n=num_items)
    if order == Order.ASC:
        qrels = qrels.sort_values(by=['relevance'])
    elif order == Order.DESC:
        qrels = qrels.sort_values(by=['relevance'], ascending=False)
    return qrels.rename(columns={'doc_id': 'docno', 'query_id': 'qid'})

def create_synthetic(datasets : List[str], order : int, window_len : int, n_samples : int = 10):
    eval = evaluator([NDCG(cutoff=10), NDCG(cutoff=5), NDCG(cutoff=1)], pd.DataFrame(irds.load(MARCO).qrels_iter()))
    order = Order(order)
    datasets = {}
    for dataset in datasets:
        datasets[dataset] = irds.load(dataset)
    all_qrels = pd.concat([pd.DataFrame(ds.qrels_iter()) for ds in datasets.values()])
    all_queries = pd.concat([pd.DataFrame(ds.queries_iter()) for ds in datasets.values()]).set_index('query_id').text.to_dict()
    all_docs = pd.DataFrame(irds.load(MARCO).docs_iter()).set_index('doc_id').text.to_dict()

    output = {
        'qid': [],
        'iter': [],
        'ratio': [],
        'ndcg@10' : [],
        'ndcg@5' : [],
        'ndcg@1' : [],
    }

    for qid, qrel in all_qrels.groupby('qid'):
        query = all_queries[qid]
        for i in range(n_samples):
            window = sample(qrel, qid, window_len, order)
            window['text'] = window['docno'].apply(lambda x: all_docs[x])
            window['query'] = query

            for ratio


        

