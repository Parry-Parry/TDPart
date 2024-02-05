import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from partitionrank.transformer.lit_t5 import LiT5
from fire import Fire
from os.path import join
from json import dump
import ir_datasets as irds
import pandas as pd
import logging
import os

def score_lit_t5(dataset : str, qrels : str, topics_or_res : str, output_path : str, checkpoint : str, window_size : int = 20, stride : int = 10, mode : str = 'sliding', buffer : int = 20, max_iters : int = 100, depth : int = 100, **kwargs):
    topics_or_res = read_results(topics_or_res)
    dataset = pt.get_dataset(dataset)
    ds = irds.load(qrels)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()
    topics_or_res['query'] = topics_or_res['qid'].apply(lambda x: queries[str(x)])
    del queries
    out_file = join(output_path, f"lit5.{mode}.{buffer}.{window_size}.{stride}.{depth}.tsv.gz")
    if os.path.exists(out_file): 
        logging.info(f"Skipping lit5.{mode}.{buffer}.{window_size}.{stride}.{depth}, already exists")
        return
    log_file = join(output_path, f"lit5.{mode}.{buffer}.{window_size}.{stride}.log")
    logging.info("Running with the following parameters:")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Qrels: {qrels}")
    logging.info(f"Topics: {topics_or_res}")
    logging.info(f"Output: {out_file}")
    logging.info(f"Mode: {mode}")
    logging.info(f"Buffer: {buffer}")
    logging.info("Loading LiT5 model")
    model = LiT5(model_path=checkpoint, mode=mode, window_size=window_size, buffer=buffer, stride=stride, max_iters=max_iters, depth=depth)
    pipe = pt.text.get_text(dataset, "text") >> model

    res = pipe.transform(topics_or_res)

    # write model.log to log_file as a dict json dump

    with open(log_file, 'w') as f:
        dump(model.log.__dict__, f, default=lambda obj: obj.__dict__)

    write_results(res, out_file)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(score_lit_t5)