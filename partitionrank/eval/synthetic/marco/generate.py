import ir_datasets as irds 
import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
from ir_measures import *
from os.path import join
from fire import Fire

from .. import MARCO, Order, IR_DATASETS, Generator, sort_df

def create_synthetic(out_path : str, n_samples : int = 10):
    marco = irds.load(MARCO)
    datasets = {}
    for dataset in IR_DATASETS:
        datasets[dataset] = irds.load(dataset)
    all_qrels = pd.concat([pd.DataFrame(ds.qrels_iter()) for ds in datasets.values()])
    all_queries = pd.concat([pd.DataFrame(ds.queries_iter()) for ds in datasets.values()]).set_index('query_id').text.to_dict()
    all_docs = pd.DataFrame(marco.docs_iter()).set_index('doc_id').text.to_dict()

    # filter queries by whether or not they have at least 19 relevant documents

    all_queries = {qid: query for qid, query in all_queries.items() if len(all_qrels[(all_qrels['query_id'] == qid) & (all_qrels['relevance'].isin([2, 3]))]) >= 19}
    all_queries = {qid: query for qid, query in all_queries.items() if len(all_qrels[(all_qrels['query_id'] == qid) & (all_qrels['relevance'].isin([0,1]))]) >= 19}

    all_qrels = all_qrels[all_qrels['query_id'].isin(all_queries.keys())]

    print(f"Number of queries: {len(all_queries)}")

    for window_len in [5, 10, 20]:
        generator = Generator(all_qrels, window_len)
        for i in range(n_samples):
            generator.new_sample()
            df = []
            for qid in all_queries.keys():
                for sample, ratio in generator.get_samples(qid):
                    sample['text'] = sample['doc_id'].apply(lambda x: all_docs[str(x)])
                    sample['query'] = all_queries[qid]
                    for order in range(3):
                        _order = Order(order)
                        sample = sort_df(sample, _order)
                        sample['order'] = _order.name
                        sample['ratio'] = ratio
                        df.append(sample)
            df = pd.concat(df)
            # split up by ratio and order and dump each to file
            for ratio in [0.2, 0.4, 0.6, 0.8]:
                for order in range(3):
                    _order = Order(order)
                    _ratio = str(ratio).replace('.', '_')
                    output_name = f"{_order.name}.{_ratio}.{window_len}.{i}.tsv.gz"
                    df[(df['order'] == _order.name) & (df['ratio'] == ratio)].to_csv(join(out_path, output_name), sep='\t', index=False)
    
    return "Done"

if __name__ == '__main__':
    Fire(create_synthetic)