from fire import Fire 
import os 
from os.path import join
from ir_measures import evaluator, read_trec_run 
from ir_measures import *
import ir_datasets as irds
import pandas as pd

def main(eval :str, run_dir : str, out_dir : str):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(run_dir) if os.path.isfile(join(run_dir, f))]
    ds = irds.load(eval)
    qrels = ds.qrels_iter()
    metrics = [AP(rel=2), NDCG(cutoff=10), R(rel=2)@100, P(rel=2, cutoff=10), RR(rel=2), RR(rel=2, cutoff=10)]
    evaluate = evaluator(metrics, qrels)
    df = []
    for file in files:
        if file.endswith(".gz"):
            name = file.strip('.gz')
            run = read_trec_run(join(run_dir, file))
            res = evaluate.calc_aggregate(run)
            res = {str(k) : v for k, v in res.items()}
            res['name'] = name 
            df.append(res)
    
    df = pd.DataFrame.from_records(df)
    df.to_csv(join(out_dir, 'metrics.tsv'), sep='\t', index=False)

    return "Success!"

if __name__ == '__main__':
    Fire(main)