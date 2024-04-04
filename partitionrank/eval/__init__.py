import pyterrier as pt 
if not pt.started(): pt.init()
import pandas as pd
from typing import Any

def load_rankgpt(dataset : Any, **model_kwargs):
    from partitionrank.transformer.rank_gpt import RankGPT
    return pt.text.get_text(dataset, "text") >> RankGPT(**model_kwargs)

def load_rankvicuna(dataset : Any, **model_kwargs):
    from partitionrank.transformer.vicuna import RankVicuna
    return pt.text.get_text(dataset, "text") >> RankVicuna(**model_kwargs)

def load_rankzephyr(dataset : Any, **model_kwargs):
    from partitionrank.transformer.zephyr import RankZephyr
    return pt.text.get_text(dataset, "text") >> RankZephyr(**model_kwargs)

def load_lit5(dataset : Any, **model_kwargs):
    from partitionrank.transformer.lit_t5 import LiT5
    return pt.text.get_text(dataset, "text") >> LiT5(**model_kwargs)

def load_pairt5(dataset : Any, **model_kwargs):
    from partitionrank.transformer.pair_t5 import PairT5
    return pt.text.get_text(dataset, "text") >> PairT5(**model_kwargs)

def load_oracle(dataset : Any, **model_kwargs):
    qrels = model_kwargs.pop('qrels')
    from partitionrank.transformer.oracle import OracleTransformer
    return pt.text.get_text(dataset, "text") >> OracleTransformer(qrels, **model_kwargs)

LOAD_FUNCS = {
    'gpt': load_rankgpt,
    'vicuna': load_rankvicuna,
    'zephyr': load_rankzephyr,
    'lit5': load_lit5,
    'pairt5': load_pairt5,
}