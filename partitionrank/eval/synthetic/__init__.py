from enum import Enum
import pandas as pd


MARCO = 'msmarco-passage/train/triples-small'
IR_DATASETS = ['msmarco-passage/trec-dl-2019/judged', 'msmarco-passage/trec-dl-2020/judged']

class Order(Enum):
    RANDOM = 0
    ASC = 1
    DESC = 2

RATIOS = {
        5 : [x * 0.1 for x in range(2, 10, 2)],
        10 : [x * 0.1 for x in range(1, 10, 1)],
        20 : [x * 0.01 for x in range(5, 100, 5)],
    }

def get_sample(qrels, qid, num_items : int = 20, order = Order.RANDOM, ratio : int = 1):
    ratio = int(ratio * num_items)
    qrels = qrels[qrels['query_id'] == str(qid)]
    non_relevant = qrels[qrels['relevance'].isin([0, 1])].copy()
    relevant = qrels[qrels['relevance'].isin([2, 3])].copy()
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