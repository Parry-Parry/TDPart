import numpy as np
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from partitionrank.transformer.sort_oracle import SortOracleTransformer
from fire import Fire
import logging

def score_oracle(window_size : int = 20, stride : int = 10, mode : str = 'sliding', buffer : int = 20, max_iters : int = 100, **kwargs):

    logging.info("Loading Oracle model")

    model = SortOracleTransformer(mode=mode, window_size=window_size, buffer=buffer, stride=stride, max_iters=max_iters)
    pipe = model

    # construct a fake set of 3 queries with 100 documents each where each document is a random integer between 0 and 100

    records = []
    for i in range(3):
        for j in range(100):
            records.append({'qid': i, 'query' : str(i), 'docno': j, 'text': np.random.randint(0, 1000)})
    topics_or_res = pd.DataFrame.from_records(records)   
    res = pipe.transform(topics_or_res)
    print(res.head(20))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(score_oracle)