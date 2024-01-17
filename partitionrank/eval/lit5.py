import pyterrier as pt 
if not pt.started(): pt.init()
from pyterrier.io import read_results, write_results
from partitionrank.transformer.lit_t5 import LiT5
from fire import Fire
from os.path import join
from json import dump

def score_lit_t5(dataset : str, topics_or_res : str, output_path : str, checkpoint : str, window_size : int = 20, stride : int = 10, mode : str = 'sliding', buffer : int = 20, max_iters : int = 100, **kwargs):
    topics_or_res = read_results(topics_or_res)
    out_file = join(output_path, f"lit5.{mode}.{buffer}.{window_size}.{stride}.tsv.gz")
    log_file = join(output_path, f"lit5.{mode}.{buffer}.{window_size}.{stride}.log")
    model = LiT5(model_path=checkpoint, mode=mode, window_size=window_size, buffer=buffer, stride=stride, max_iters=max_iters)
    model = pt.text.get_text(dataset, "text") >> model

    res = model.transform(topics_or_res)

    # write model.log to log_file as a dict json dump

    with open(log_file, 'w') as f:
        dump(model.log.__dict__, f, default=lambda obj: obj.__dict__)

    write_results(res, out_file)

if __name__ == '__main__':
    Fire(score_lit_t5)