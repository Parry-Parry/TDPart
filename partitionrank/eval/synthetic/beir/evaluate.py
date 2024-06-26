import pandas as pd 
import pyterrier as pt
if not pt.started(): pt.init()
import ir_datasets as irds 
from tqdm.auto import tqdm
from fire import Fire
import torch
from ir_measures import *
from ir_measures import evaluator
from typing import Any
from os.path import join
from partitionrank.eval import LOAD_FUNCS
from partitionrank.eval.synthetic import Order
import os

def evaluate(in_path : str, out_path : str, model : Any, dataset : str, pt_dataset : str, mode='single', cutoff=2, batch_size=8, skip : bool = False):
    if skip:
        if os.path.exists(join(out_path, f"{model}.{mode}.tsv.gz")):
            return
    corpus = irds.load(dataset)
    all_qrels = pd.DataFrame(corpus.qrels_iter())
    eval = evaluator([nDCG@10, nDCG@5, nDCG@1], all_qrels)
    _model = None

    output = {
        'iter': [],
        'ratio': [],
        'window_len': [],
        'order': [],
        'before_nDCG@10' : [],
        'before_nDCG@5' : [],
        'before_nDCG@1' : [],
        'after_nDCG@10' : [],
        'after_nDCG@5' : [],
        'after_nDCG@1' : []
    }
    
    progress = tqdm(total=3*5*3*4)
    for window_len in [5, 10, 20]:
        if _model is not None:
            del _model
            torch.cuda.empty_cache()
        _model = LOAD_FUNCS[model](dataset=pt.get_dataset(pt_dataset), mode=mode, window_size=window_len, cutoff=window_len-1, batch_size=batch_size)
        for i in range(5):
            for order in range(3):
                _order = Order(order)
                for ratio in [0.2, 0.4, 0.6, 0.8]:
                    _ratio = str(ratio).replace('.', '_')
                    input_name = f"{_order.name}.{_ratio}.{window_len}.{i}.tsv.gz"
                    sample = pd.read_csv(join(in_path, input_name), sep='\t', dtype={'doc_id': str, 'query_id': str})[['query_id', 'doc_id', 'text', 'query', 'score']]
                    old_metrics = eval.calc_aggregate(sample)
                    old_metrics = {str(k) : v for k, v in old_metrics.items()}
                    rez = _model.transform(sample.rename(columns={'doc_id': 'docno', 'query_id': 'qid'}))
                    metrics = eval.calc_aggregate(rez.rename(columns={'docno': 'doc_id', 'qid': 'query_id'}))
                    metrics = {str(k) : v for k, v in metrics.items()}
                    output['iter'].append(i)
                    output['ratio'].append(ratio)
                    output['window_len'].append(window_len)
                    output['order'].append(_order.name)
                    output['before_nDCG@10'].append(old_metrics['nDCG@10'])
                    output['before_nDCG@5'].append(old_metrics['nDCG@5'])
                    output['before_nDCG@1'].append(old_metrics['nDCG@1'])
                    output['after_nDCG@10'].append(metrics['nDCG@10'])
                    output['after_nDCG@5'].append(metrics['nDCG@5'])
                    output['after_nDCG@1'].append(metrics['nDCG@1'])
                    progress.update(1)
    
    out_name = f"{model}.{mode}.tsv.gz"
    pd.DataFrame(output).to_csv(join(out_path, out_name), sep='\t', index=False)

if __name__ == '__main__':
    Fire(evaluate)
