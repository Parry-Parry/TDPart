from typing import List
import torch
import pandas as pd
import numpy as np
import re
from transformers import T5Tokenizer
from pyterrier_t5.modeling_fid import FiD
from . import QueryLog, MainLog
import pyterrier as pt
from tqdm.auto import tqdm

def _iter_windows(n, window_size, stride):
    # TODO: validate window_size and stride
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride), unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len

def _split(l, i):
    return l[:i], l[i:]

class LiT5(pt.Transformer):
    template = "Search Query: {q} Passage: [{i}] {d} Relevance Ranking: "
    def __init__(self, model_path='castorini/LiT5-Distill-large', batch_size=16, verbose=True, bfloat16=None, window_size : int = 20, stride : int = 10, buffer : int = 20, mode='sliding'):
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
        self.window_size = window_size
        self.stride = stride
        self.buffer = buffer

        self.log = MainLog()
        self.current_query = None

        self.process = {
            'sliding': self.sliding_window,
            'pivot': self.pivot
        }[mode]

    def score(self, query : str, doc_texts : List[str], start_idx : int, end_idx : int, window_len : int):
        self.current_query.inferences += 1
        passages = [self.template.format(q=query, i=i+1, d=text) for i, text in enumerate(doc_texts + ["" for _ in range(end_idx - start_idx, self.window_size)])]
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

    def _pivot(self, query : str, doc_idx : List[str], doc_texts : List[str]):
        '''
        l : current left partition being scored
        r : current right partition being the remainder of the array
        c : current candidates
        b : current backfill
        p : current pivot
        '''
        l_text, l_idx = doc_texts[:self.window_size], doc_idx[:self.window_size]
        r_text, r_idx = doc_texts[self.window_size:], doc_idx[self.window_size:]

        order = self.score(query, l_text, 0, self.window_size, self.window_size)
        orig_idxs = np.arange(self.window_size)
        l_text[orig_idxs], l_idx[orig_idxs] = l_text[order], l_idx[order]

        if len(l_text) == self.window_size: return l_idx, l_text, r_idx, r_text

        p_id, p_text = doc_idx[order[9]], doc_texts[order[9]]
        c_text = np.concatenate([l_text[:9],l_text[10:]])
        c_idx = np.concatenate([l_idx[:9],l_idx[10:]])

        b_text, b_idx = [], []
        sub_window_size = self.window_size - 1

        cutoff = len(r_text) % sub_window_size
        r_text, r_idx = r_text[:-cutoff], r_idx[:-cutoff]

        while len(c_text) < self.buffer and len(r_text) > 0:
            l_text, r_text = _split(r_text, sub_window_size)
            l_idx, r_idx = _split(r_idx, sub_window_size)

            # prepend pivot to left partition
            l_text = np.concatenate([[p_text], l_text])
            l_idx = np.concatenate([[p_id], l_idx])

            order = self.score(query, l_text, 0, self.window_size, self.window_size)
            orig_idxs = np.arange(self.window_size)
            l_text[orig_idxs], l_idx[orig_idxs] = l_text[order], l_idx[order]

            p_idx = np.where(l_idx == p_id)[0][0] # find index of pivot id
            # add left of pivot to candidates and right of pivot to backfill
            c_text = np.concatenate([c_text, l_text[:p_idx]])
            c_idx = np.concatenate([c_idx, l_idx[:p_idx]])
            b_text = np.concatenate([b_text, l_text[p_idx+1:]])
            b_idx = np.concatenate([b_idx, l_idx[p_idx+1:]])
        
        return c_idx[:self.buffer], c_text[:self.buffer], b_idx, b_text
        

    def pivot(self, query : str, query_results : pd.DataFrame):
        self.current_query = QueryLog(qid=query_results['qid'].iloc[0])
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()

        c_idx, c_text, f_idx, f_text = self._pivot(query, doc_idx, doc_texts)
        c_idx, c_text, b_idx, b_text = self._pivot(query, c_idx, c_text)

        c_idx = np.concatenate([c_idx, b_idx, f_idx])
        c_text = np.concatenate([c_text, b_text, f_text])

        self.log.queries.append(self.current_query)

        return c_idx, c_text
    
    def sliding_window(self, query : str, query_results : pd.DataFrame):
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        for start_idx, end_idx, window_len in _iter_windows(len(query_results), self.window_size, self.stride):
            order = self.score(query, doc_texts[start_idx:end_idx].to_list(), start_idx, end_idx, window_len)
            new_idxs = start_idx + np.array(order)
            orig_idxs = np.arange(start_idx, end_idx)
            doc_idx[orig_idxs] = doc_idx[new_idxs]
            doc_texts[orig_idxs] = doc_texts[new_idxs]
        return doc_idx, doc_texts

    def transform(self, inp : pd.DataFrame):
        res = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'rank': [],
        }
        with torch.no_grad():
            for (qid, query), query_results in tqdm(inp.groupby(['qid', 'query']), unit='q'):
                self._reset_budget()
                doc_idx, doc_texts = self.process(query, query_results)
                res['qid'].extend([qid] * len(doc_idx))
                res['query'].extend([query] * len(doc_idx))
                res['docno'].extend(doc_idx)
                res['text'].extend(doc_texts)
                res['rank'].extend(list(range(len(doc_idx))))
        res = pd.DataFrame(res)
        res['score'] = -res['rank'].astype(float)
        return res
    
'''
def _pivot2(self, query : str, doc_idx : List[str], doc_texts : List[str]):
        end_idx = min(self.window_size, len(doc_texts))
        window_len = end_idx


        -- GET PIVOT --

        * sort top-k docs using model
        * take pivot as last element in sorted list


        order = self.score(query, doc_texts[:end_idx], 0, end_idx, window_len)
        orig_idxs = np.arange(end_idx)
        doc_idx[orig_idxs] = doc_idx[order]
        doc_texts[orig_idxs] = doc_texts[order]
        pivot_id = doc_idx[order[9]] # get the 10th doc as pivot
        pivot_text = doc_texts[order[9]]

        candidate_texts = doc_texts[:window_len-1].to_list() # add top-k-1 docs to candidates
        candidate_idxs = doc_idx[:window_len-1].to_list()
        filler_idx, filler_texts = [], [] # store for backfilling
        doc_idx = doc_idx[end_idx:] # pop processed docs
        doc_texts = doc_texts[end_idx:]


        -- GET CANDIDATES --

        * get next partition
        * sort partition using model
        * find pivot in sorted partition
        * add candidates to list


        while doc_texts and len(candidate_texts) <= self.buffer:
            if not doc_texts: break
            end_idx = min(self.window_size - 1, len(doc_texts))
            window_len = end_idx + 1
            _texts = [pivot_text] + doc_texts[:end_idx].to_list() # get next partition
            _idx = [pivot_id] + doc_idx[:end_idx].to_list()
            doc_idx = doc_idx[end_idx:] # pop processed docs
            doc_texts = doc_texts[end_idx:]

            order = self.score(query, _texts, 0, end_idx, window_len) 
            _idx = np.array(_idx)[order]
            _texts = np.array(_texts)[order]

            id = _idx.index(pivot_id) # find pivot
            candidate_idxs.extend(_idx[:id]) # add candidates
            candidate_texts.extend(_texts[:id])
            filler_idx.extend(_idx[id+1:])
            filler_texts.extend(_texts[id+1:])
        
        return candidate_idxs, candidate_texts, filler_idx, filler_texts
'''