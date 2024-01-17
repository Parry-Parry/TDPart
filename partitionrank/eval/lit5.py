import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from partitionrank.transformer.lit_t5 import LiT5
from fire import Fire
from os.path import join

def score_lit_t5(dataset : str, topics_or_res : str, output_path : str, window_size : str, stride : str, mode : str, checkpoint : str, buffer : int = 20):
    topics_or_res = read_results(topics_or_res)
    out_file = join(output_path, f"lit5.{mode}.{buffer}.{window_size}.{stride}.tsv.gz")
    
    model = LiT5(model_path=checkpoint, mode=mode, window_size=window_size, buffer=buffer, stride=stride)
    model = pt.text.get_text(dataset, "text") >> model

    res = model.transform(topics_or_res)

    write_results(res, out_file)

if __name__ == '__main__':
    Fire(score_lit_t5)