from typing import List
import ir_datasets as irds 
import pandas as pd
from enum import Enum
from .. import LOAD_FUNCS
from ir_measures import *
from ir_measures import evaluator
from os.path import join
from fire import Fire

MARCO = 'irds:msmarco-passage/train/triples-small'

class Order(Enum):
    RANDOM = 0
    ASC = 1
    DESC = 2

def sample(qrels, qid, num_items : int = 20, order = Order.RANDOM, ratio : int = 1):
    ratio = int(ratio * num_items)
    qrels = qrels[qrels['query_id'] == qid]
    non_relevant = qrels[qrels['relevance'].isin([0, 1])].copy()
    relevant = qrels[qrels['relevance'].isin([2, 3])].copy()
    non_relevant = non_relevant.sample(frac=num_items-ratio, replace=False)
    if len(relevant) < ratio: raise ValueError("Not enough relevant documents")
    relevant = relevant.sample(frac=ratio, replace=False)
    if order == Order.ASC:
        qrels = qrels.sort_values(by=['relevance'])
    elif order == Order.DESC:
        qrels = qrels.sort_values(by=['relevance'], ascending=False)
    qrels = qrels.rename(columns={'doc_id': 'docno', 'query_id': 'qid'})
    qrels['score'] = [i for i in range(num_items)]
    return qrels

def create_synthetic(out_path : str, datasets : List[str], order : int, window_len : int, n_samples : int = 10, model = None):
    model = LOAD_FUNCS[model](mode='single', window_size=window_len)
    eval = evaluator([nDCG@10, nDCG@5, nDCG@1], pd.DataFrame(irds.load(MARCO).qrels_iter()))
    order = Order(order)
    datasets = {}
    for dataset in datasets:
        datasets[dataset] = irds.load(dataset)
    all_qrels = pd.concat([pd.DataFrame(ds.qrels_iter()) for ds in datasets.values()])
    all_queries = pd.concat([pd.DataFrame(ds.queries_iter()) for ds in datasets.values()]).set_index('query_id').text.to_dict()
    all_docs = pd.DataFrame(irds.load(MARCO).docs_iter()).set_index('doc_id').text.to_dict()

    # filter queries by whether or not they have at least 19 relevant documents

    all_queries = {qid: query for qid, query in all_queries.items() if len(all_qrels[(all_qrels['query_id'] == qid) & (all_qrels['relevance'].isin([2, 3]))]) >= 19}

    output = {
        'qid': [],
        'iter': [],
        'ratio': [],
        'nDCG@10' : [],
        'nDCG@5' : [],
        'nDCG@1' : [],
    }

    for qid, qrel in all_qrels.groupby('qid'):
        query = all_queries[qid]
        for i in range(n_samples):
            for ratio in range(0.05, 1.0, 0.05):
                sample = sample(qrel, qid, window_len, order, ratio)
                sample['text'] = sample['docno'].apply(lambda x: all_docs[str(x)])
                sample['query'] = query
                rez = model.transform(sample)
                metrics = eval.calc_aggregate(rez)
                metrics = {str(k) : v for k, v in metrics.items()}

                output['qid'].append(qid)
                output['iter'].append(i)
                output['ratio'].append(ratio)
                output['nDCG@10'].append(metrics['nDCG@10'])
                output['nDCG@5'].append(metrics['nDCG@5'])
                output['nDCG@1'].append(metrics['nDCG@1'])
    
    out_name = f"{model}.{order.name}.{window_len}.tsv.gz"

    pd.DataFrame(output).to_csv(join(out_path, out_name), sep='\t', index=False)

if __name__ == '__main__':
    Fire(create_synthetic)