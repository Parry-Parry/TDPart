import ir_datasets as irds 
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from ir_measures import *
from os.path import join
from fire import Fire

from .. import Order, get_sample

def create_synthetic(dataset : str, out_path : str, n_samples : int = 10):
    corpus = irds.load(dataset)
    all_qrels = pd.DataFrame(corpus.qrels_iter())
    all_queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
    all_docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()

    # filter queries by whether or not they have at least 19 relevant documents

    all_queries = {qid: query for qid, query in all_queries.items() if len(all_qrels[(all_qrels['query_id'] == qid) & (all_qrels['relevance'].isin([2, 3]))]) >= 19}
    all_queries = {qid: query for qid, query in all_queries.items() if len(all_qrels[(all_qrels['query_id'] == qid) & (all_qrels['relevance'].isin([0,1]))]) >= 19}

    print(f"Number of queries: {len(all_queries)}")

    for i in range(n_samples):
        for order in range(3):
            _order = Order(order)
            for window_len in [5, 10, 20]:
                for ratio in [0.2, 0.4, 0.6, 0.8]:
                    _ratio = str(ratio).replace('.', '_')
                    output_name = f"{_order.name}.{_ratio}.{window_len}.{i}.tsv.gz"
                    frame = []
                    for qid, query in all_queries.items():
                        sample = get_sample(all_qrels, qid, window_len, order, ratio)
                        sample['text'] = sample['docno'].apply(lambda x: all_docs[str(x)])
                        sample['query'] = query
                        frame.append(sample)
                    frame = pd.concat(frame)
                    frame.to_csv(join(out_path, output_name), sep='\t', index=False)

if __name__ == '__main__':
    Fire(create_synthetic)