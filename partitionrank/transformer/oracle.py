import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from tqdm.auto import tqdm

class OracleTransformer(pt.Transformer):
    def __init__(self, qrels : pd.DataFrame) -> None:
        super().__init__()
        self.qrels = qrels
    
    def process(self, qid : str, query_results : pd.DataFrame):
        q_qrels = self.qrels[self.qrels['qid'] == qid].set_index('docno').relevance.to_dict()
        doc_idx = query_results['docno'].tolist()
        doc_rel = [q_qrels[i] for i in doc_idx]

        # sort by relevance
        order = sorted(range(len(doc_rel)), key=lambda k: doc_rel[k], reverse=True)
        doc_idx = [doc_idx[i] for i in order]

        return doc_idx
    
    def transform(self, inp):
        res = {
            'qid': [],
            'docno': [],
            'rank': [],
        }
        for qid, query_results in tqdm(inp.groupby('qid'), unit='q'):
            doc_idx = self.process(qid, query_results)
            res['qid'].extend([qid] * len(doc_idx))
            res['docno'].extend(doc_idx)
            res['rank'].extend(list(range(len(doc_idx))))
        res = pd.DataFrame(res)
        res['score'] = -res['rank'].astype(float)
        return res