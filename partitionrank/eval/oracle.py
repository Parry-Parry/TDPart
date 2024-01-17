import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from partitionrank.transformer.oracle import OracleTransformer
from fire import Fire
import ir_datasets as irds
from os.path import join
from json import dump

def score_oracle(qrels : str, topics_or_res : str, output_path : str, window_size : int = 20, stride : int = 10, mode : str = 'sliding', buffer : int = 20, max_iters : int = 100, **kwargs):
    topics_or_res = read_results(topics_or_res)
    ds = irds.load(qrels)
    qrels = pd.DataFrame(ds.qrels_iter())
    out_file = join(output_path, f"oracle.{mode}.{buffer}.{window_size}.{stride}.tsv.gz")
    log_file = join(output_path, f"oracle.{mode}.{buffer}.{window_size}.{stride}.log")

    model = OracleTransformer(qrels, mode=mode, window_size=window_size, buffer=buffer, stride=stride, max_iters=max_iters)
    res = model.transform(topics_or_res)

    with open(log_file, 'w') as f:
        dump(model.log.__dict__, f, default=lambda obj: obj.__dict__)

    write_results(res, out_file)

if __name__ == '__main__':
    Fire(score_oracle)