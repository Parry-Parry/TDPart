import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from utility import load_yaml
from partitionrank.transformer.lit_t5 import LitT5
from fire import Fire

def score_lit_t5(config : str):
    config = load_yaml(config)
    dataset = pt.get_dataset(config['dataset'])
    topics_or_res = read_results(config['topics_or_res'])
    output_path = config['output_path']

    window_size = config['window_size']
    passes = config['passes']
    stride = config['stride']
    mode = config['mode']
    
    model = LitT5(model_path=config['checkpoint'], mode=mode, window_size=window_size, passes=passes, stride=stride)
    model = pt.text.get_text(dataset, "text") >> model

    res = model.transform(topics_or_res)

    write_results(res, output_path)

if __name__ == '__main__':
    Fire(score_lit_t5)