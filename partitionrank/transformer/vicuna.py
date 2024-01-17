import re
from typing import Optional, Union, List

import numpy as np
from partitionrank.transformer import ListWiseTransformer
from partitionrank.modelling.prompt import RankPrompt
from partitionrank.modelling.base import LLMRanker
from transformers import GenerationConfig
import torch

class RankVicuna(ListWiseTransformer):

    PRE = "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}."
    POST = "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."
    CHECKPOINT = 'castorini/rank_vicuna_7b_v1'
    MAX_LENGTH = 200

    def __init__(self, device : Union[str, int] = 'cuda', n_gpu : Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.prompt = RankPrompt([self.PRE, '{documents}', self.POST], doc_formatter=True, model=self.CHECKPOINT, max_length=self.MAX_LENGTH)
        self.model = LLMRanker(checkpoint=self.CHECKPOINT, device=device, n_gpu=n_gpu, fast_chat=True)
        self.chain = self.prompt >> self.model
    
    def score(self, query : str, doc_text : List[str], window_len : int, **kwargs):
        self.current_query.inferences += 1
        order = self.chain(query=query, texts=doc_text.tolist(), num=window_len)
        inputs = {k: torch.tensor(v).cuda() for k, v in inputs.items()}

        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_length = 1
        gen_cfg.do_sample = False
        output_ids = self._llm.generate(**inputs, generation_config=gen_cfg)

        if self._llm.config.is_encoder_decoder: output_ids = output_ids[0]
        else: output_ids = output_ids[0][len(inputs["input_ids"][0]) :]

        self.current_query.out_tokens += len(output_ids)
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        output = re.sub(r'[^0-9]', ' ', outputs) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < window_len}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(window_len) if i not in output] # backfill missing passages
        return np.array(order)