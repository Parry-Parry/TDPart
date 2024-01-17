from typing import List
import torch
import numpy as np
import re
from transformers import T5Tokenizer
from pyterrier_t5.modeling_fid import FiD
from . import ListWiseTransformer

class LiT5(ListWiseTransformer):
    template = "Search Query: {q} Passage: [{i}] {d} Relevance Ranking: "
    def __init__(self, model_path='castorini/LiT5-Distill-large', batch_size=16, verbose=True, bfloat16=None, window_size : int = 20, stride : int = 10, buffer : int = 20, mode='sliding', max_iters : int = 100):
        super().__init__(window_size=window_size, stride=stride, buffer=buffer, mode=mode, max_iters=max_iters)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, return_dict=False, legacy=False, use_fast=True)
        self.model = FiD.from_pretrained(model_path, from_flax=False).cuda().eval()
        self.model.encoder.config.n_passages = window_size
        self.model.encoder.config.batch_size = batch_size
        if bfloat16 is None:
            try:
                self.model = self.model.bfloat16()
                bfloat16 = True
            except:
                bfloat16 = False
        elif bfloat16:
            self.model = self.model.bfloat16()
        self.bfloat16 = bfloat16

    def score(self, query : str, doc_text : List[str], start_idx : int, end_idx : int, window_len : int, **kwargs):
        self.current_query.inferences += 1
        passages = [self.template.format(q=query, i=i+1, d=text) for i, text in enumerate(doc_text + ["" for _ in range(end_idx - start_idx, self.window_size)])]
        inputs = self.tokenizer.batch_encode_plus(passages, return_tensors="pt", padding='max_length', max_length=150, truncation=True)
        # get number of tokens in batch
        self.current_query.in_tokens += inputs['input_idx'].view(-1).shape[0]
        outputs = self.model.generate(
            input_idx=inputs['input_idx'].cuda().reshape(1, -1),
            attention_mask=inputs['attention_mask'].cuda().reshape(1, -1),
            max_length=100,
            do_sample=False,
        )
        self.current_query.out_tokens += len(outputs[0])
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < window_len}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(window_len) if i not in output] # backfill missing passages
        return np.array(order)