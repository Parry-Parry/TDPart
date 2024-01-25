import pandas as pd 
import pyterrier as pt
if not pt.started(): pt.init()
import ir_datasets as irds 
from tqdm.auto import tqdm
from fire import Fire
from ir_measures import *
from ir_measures import evaluator
from typing import Any
from os.path import join
from partitionrank.eval import LOAD_FUNCS
from partitionrank.eval.synthetic import RATIOS, Order, IR_DATASETS

def evaluate(in_path : str, out_path : str, model : Any):
    datasets = {}
    for dataset in IR_DATASETS:
        datasets[dataset] = irds.load(dataset)
    all_qrels = pd.concat([pd.DataFrame(ds.qrels_iter()) for ds in datasets.values()])
    eval = evaluator([nDCG@10, nDCG@5, nDCG@1], all_qrels)

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

    

    for window_len in [5, 10, 20]:
        _model = LOAD_FUNCS[model](dataset=pt.get_dataset('irds:msmarco-passage'), mode='single', window_size=window_len, cutoff=window_len-1)
        progress = tqdm(total=10*3*3*len(RATIOS[window_len]))
        for i in range(10):
            for order in range(3):
                _order = Order(order)
                for ratio in RATIOS[window_len]:
                    out_frame = output.copy()
                    _ratio = str(ratio).replace('.', '_')
                    input_name = f"{_order.name}.{_ratio}.{window_len}.{i}.tsv.gz"
                    sample = pd.read_csv(join(in_path, input_name), sep='\t')
                    print(sample.head())
                    old_metrics = eval.calc_aggregate(sample.rename(columns={'docno': 'doc_id', 'qid': 'query_id'}))
                    old_metrics = {str(k) : v for k, v in old_metrics.items()}
                    rez = _model.transform(sample)
                    metrics = eval.calc_aggregate(rez.rename(columns={'docno': 'doc_id', 'qid': 'query_id'}))
                    metrics = {str(k) : v for k, v in metrics.items()}
                    out_frame['iter'].append(i)
                    out_frame['ratio'].append(ratio)
                    out_frame['window_len'].append(window_len)
                    out_frame['order'].append(_order.name)
                    out_frame['before_nDCG@10'].append(old_metrics['nDCG@10'])
                    out_frame['before_nDCG@5'].append(old_metrics['nDCG@5'])
                    out_frame['before_nDCG@1'].append(old_metrics['nDCG@1'])
                    out_frame['after_nDCG@10'].append(metrics['nDCG@10'])
                    out_frame['after_nDCG@5'].append(metrics['nDCG@5'])
                    out_frame['after_nDCG@1'].append(metrics['nDCG@1'])
                    progress.update(1)
    
    out_name = f"{model}.{i}.tsv.gz"
    pd.DataFrame(output).to_csv(join(out_path, out_name), sep='\t', index=False)

if __name__ == '__main__':
    Fire(evaluate)
