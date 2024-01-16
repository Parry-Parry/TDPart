import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import write_results
from utility import load_yaml
from partitionrank.eval.first_stage.load import LOAD_FUNCS
import ir_datasets as irds
import pandas as pd
from fire import Fire

def precompute(config : str):
    config = load_yaml(config)
    first_stage = config.pop(['first_stage'])
    second_stage = config.pop(['second_stage'], None)
    eval_set = config.pop(['eval_set'])
    out_file = config.pop(['out_file'])
    topk = config.pop(['topk'], 100)

    model = LOAD_FUNCS[first_stage['model']](**first_stage['kwargs']) % topk

    if second_stage:
        text_ref = pt.get_dataset(config.pop(['text_ref']))
        second_stage = LOAD_FUNCS[second_stage['model']](**second_stage['kwargs'])
        model = model >> pt.text.get_text(text_ref, 'text') >> second_stage
    
    eval_set = irds.load(eval_set)
    topics = pd.DataFrame(eval_set.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    res = model.transform(topics)
    write_results(res, out_file)

if __name__ == '__main__':
    Fire(precompute)