import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from utility import load_yaml
from partitionrank.transformer.oracle import OracleTransformer
from fire import Fire
import ir_datasets as irds

def score_oracle(config : str):
    config = load_yaml(config)
    topics_or_res = read_results(config['topics_or_res'])
    output_path = config['output_path']
    mode = config['mode']
    qrels = config['qrels']
    ds = irds.load(qrels)
    qrels = pd.DataFrame(ds.qrels_iter())

    window_size = config['window_size']
    buffer = config.pop('buffer', 20)
    stride = config['stride']
    mode = config['mode']
    
    model = OracleTransformer(qrels, mode=mode, window_size=window_size, buffer=buffer, stride=stride)
    res = model.transform(topics_or_res)

    write_results(res, output_path)

if __name__ == '__main__':
    Fire(score_oracle)