import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from fire import Fire
from os.path import join
from json import dump
import ir_datasets as irds
import pandas as pd
import logging
from . import LOAD_FUNCS
import os

def score_model(dataset : str, 
                 qrels : str, 
                 model_type : str,
                 topics_or_res : str, 
                 output_path : str, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 mode : str = 'sliding', 
                 buffer : int = 20, 
                 max_iters : int = 100,
                 n_gpu : int = 1,
                 depth : int = 100,
                 **kwargs):
    topics_or_res = read_results(topics_or_res)
    ds = irds.load(qrels)
    dataset = pt.get_dataset(dataset)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()
    topics_or_res['query'] = topics_or_res['qid'].apply(lambda x: queries[str(x)])
    del queries
    out_file = join(output_path, f"{model_type}.{mode}.{buffer}.{window_size}.{stride}.{depth}.tsv.gz")
    if os.path.exists(out_file): 
        logging.info(f"Skipping {model_type}.{mode}.{buffer}.{window_size}.{stride}.{depth}, already exists")
        return 
    log_file = join(output_path, f"{model_type}.{mode}.{buffer}.{window_size}.{stride}.{depth}.log")
    logging.info(f"Loading {model_type} model")
    pipe = LOAD_FUNCS[model_type](dataset, qrels=pd.DataFrame(ds.qrels_iter()), n_gpu=n_gpu, mode=mode, window_size=window_size, buffer=buffer, stride=stride, max_iters=max_iters, depth=depth, **kwargs)

    res = pt.model.add_ranks(pipe.transform(topics_or_res))

    # write model.log to log_file as a dict json dump

    with open(log_file, 'w') as f:
        dump(pipe.models[-1].log.__dict__, f, default=lambda obj: obj.__dict__)

    write_results(res, out_file)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(score_model)