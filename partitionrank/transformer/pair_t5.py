from itertools import combinations
from typing import List

import numpy as np
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from . import ListWiseTransformer
from tqdm import tqdm
from more_itertools import chunked

def create_pairs(num : int):
    array = []
    for i in range(num):
        for j in range(num):
            array.append((i, j))
    return array

class PairT5(ListWiseTransformer):
    template = """Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""

    def __init__(self, 
                 model_path : str = 'google/flan-t5-xl', 
                 batch_size : int = 16, 
                 verbose : bool = True, 
                 bfloat16 : bool = None, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 mode='sliding', 
                 max_iters : int = 100,
                 max_length : int = 300,
                 **kwargs):
        super().__init__(window_size=window_size, stride=stride, buffer=buffer, mode=mode, max_iters=max_iters, verbose=verbose, **kwargs)
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, return_dict=False, legacy=False, use_fast=True)
        self.tokenizer.max_len = 4096
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).cuda().eval()
        if bfloat16 is None:
            try:
                self.model = self.model.bfloat16()
                bfloat16 = True
            except:
                bfloat16 = False
        elif bfloat16:
            self.model = self.model.bfloat16()
        self.bfloat16 = bfloat16
        self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.model.device)
        self.decoder_input_ids = self.decoder_input_ids.repeat(self.batch_size, 1)
        self.A, self.B = self.tokenizer.encode("A")[0], self.tokenizer.encode("B")[0]
        self.max_length = max_length

    def _strip(self, doc_text : list):
        # split by space and truncate to max_length
        return [" ".join(doc.split()[:self.max_length]) for doc in doc_text]

    def score(self, query : str, doc_text : list, **kwargs):
        idx = create_pairs(len(doc_text))
        score_matrix = np.zeros((len(doc_text), len(doc_text)))
        doc_text = self._strip(doc_text)
        for batch in tqdm(chunked(idx, self.batch_size), unit='batch'):
            self.current_query.inferences += len(batch)
            prompts = [self.template.format(query=query, doc1=doc_text[i], doc2=doc_text[j]) for i, j in batch]
            inputs = self.tokenizer(prompts, truncation=True, return_tensors="pt").input_ids.to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, decoder_input_ids=self.decoder_input_ids, max_new_tokens=2)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k in range(0, len(outputs)):
                i, j = batch[k]
                output1 = outputs[i]
                if output1 == "Passage A":
                    score_matrix[i, j] = 1
                    score_matrix[j, i] = 0
                elif output1 == "Passage B":
                    score_matrix[i, j] = 0
                    score_matrix[j, i] = 1
                else:  # conflict
                    score_matrix[i, j] = 0.5
                    score_matrix[j, i] = 0.5
        
        for i in range(len(doc_text)):
            for j in range(len(doc_text)):
                if score_matrix[i, j] > 0.5 and score_matrix[j, i] < 0.5: score_matrix[i, j], score_matrix[j, i] = 1., 0.
                elif score_matrix[i, j] < 0.5 and score_matrix[j, i] > 0.5: score_matrix[i, j], score_matrix[j, i] = 0., 1.
                else: score_matrix[i, j], score_matrix[j, i] = 0.5, 0.5
        
        scores = score_matrix.sum(axis=1)
        return np.argsort(scores)[::-1]
    

class PairT5(ListWiseTransformer):
    template = """Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""

    def __init__(self, 
                 model_path : str = 'google/flan-t5-xl', 
                 batch_size : int = 16, 
                 verbose : bool = True, 
                 bfloat16 : bool = None, 
                 window_size : int = 20, 
                 stride : int = 10, 
                 buffer : int = 20, 
                 mode='sliding', 
                 max_iters : int = 100,
                 max_length : int = 300,
                 **kwargs):
        super().__init__(window_size=window_size, stride=stride, buffer=buffer, mode=mode, max_iters=max_iters, verbose=verbose, **kwargs)
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, return_dict=False, legacy=False, use_fast=True)
        self.tokenizer.max_len = 4096
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).cuda().eval()
        if bfloat16 is None:
            try:
                self.model = self.model.bfloat16()
                bfloat16 = True
            except:
                bfloat16 = False
        elif bfloat16:
            self.model = self.model.bfloat16()
        self.bfloat16 = bfloat16
        self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.model.device)
        self.decoder_input_ids = self.decoder_input_ids.repeat(self.batch_size, 1)
        self.A, self.B = self.tokenizer.encode("A")[0], self.tokenizer.encode("B")[0]
        self.max_length = max_length

    def _strip(self, doc_text : list):
        # split by space and truncate to max_length
        return [" ".join(doc.split()[:self.max_length]) for doc in doc_text]

    def _compare(self, query: str, docs: List):
        doc1, doc2 = docs[0], docs[1]
        input_texts = [self.template.format(query=query, doc1=doc1, doc2=doc2),
                       self.template.format(query=query, doc1=doc2, doc2=doc1)]
       
        input_ids = self.tokenizer(input_texts,
                                    padding='longest',
                                    return_tensors="pt").input_ids.to(self.model.device)
        
        output_ids = self.model.generate(input_ids,
                                        decoder_input_ids=self.decoder_input_ids,
                                        max_new_tokens=2)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output

    def score(self, query : str, doc_text : list, **kwargs):
        idx = create_pairs(len(doc_text))
        score_matrix = np.zeros((len(doc_text), len(doc_text)))
        doc_text = self._strip(doc_text)

        for batch in chunked(idx, self.batch_size):
            self.current_query.inferences += len(batch)
            prompts = [self.template.format(query=query, doc1=doc_text[i], doc2=doc_text[j]) for i, j in batch]
            inputs = self.tokenizer(prompts, max_length=512, padding='longest', return_tensors="pt").input_ids.to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, decoder_input_ids=self.decoder_input_ids
                                                  if self.decoder_input_ids.shape[0] == len(inputs)
                                                  else self.decoder_input_ids[:len(inputs), :], max_new_tokens=2)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k in range(0, len(outputs)):
                i, j = batch[k]
                output1 = outputs[i]
                if output1 == "Passage A":
                    score_matrix[i, j] = 1
                    score_matrix[j, i] = 0
                elif output1 == "Passage B":
                    score_matrix[i, j] = 0
                    score_matrix[j, i] = 1
                else:  # conflict
                    score_matrix[i, j] = 0.5
                    score_matrix[j, i] = 0.5
        
        for i in range(len(doc_text)):
            for j in range(len(doc_text)):
                if score_matrix[i, j] > 0.5 and score_matrix[j, i] < 0.5: score_matrix[i, j], score_matrix[j, i] = 1., 0.
                elif score_matrix[i, j] < 0.5 and score_matrix[j, i] > 0.5: score_matrix[i, j], score_matrix[j, i] = 0., 1.
                else: score_matrix[i, j], score_matrix[j, i] = 0.5, 0.5
        
        scores = score_matrix.sum(axis=1)
        return np.argsort(scores)[::-1]