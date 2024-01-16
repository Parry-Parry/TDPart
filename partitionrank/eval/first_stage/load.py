
def load_monot5(checkpoint : str ='castorini/monot5-base-msmarco', batch_size : int = 64, **kwargs):
    from pyterrier_t5 import MonoT5ReRanker 

    return MonoT5ReRanker(model=checkpoint, batch_size=batch_size)

def load_bi_encoder(checkpoint : str ='sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size : int = 64, **kwargs):
    from transformers import AutoModel, AutoTokenizer
    from pyterrier_dr import HgfBiEncoder, BiScorer

    model = AutoModel.from_pretrained(checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device)
    return BiScorer(backbone, batch_size=batch_size)

def load_electra(checkpoint : str ='crystina-z/monoELECTRA_LCE_nneg31', batch_size : int = 64, **kwargs):
    from pyterrier_dr import ElectraScorer

    return ElectraScorer(model_name=checkpoint, batch_size=batch_size)

def load_splade(checkpoint : str = 'naver/splade-cocondenser-ensembledistil', batch_size : int = 128, index : str = 'msmarco_passage', **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyt_splade import SpladeFactory
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index, num_threads=4).quantized()
    splade = SpladeFactory(checkpoint)
    return splade.query_encoder(batch_size=batch_size) >> index.quantized()

def load_bm25(index : str = 'msmarco_passage', **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()

    index = pt.IndexFactory.of(pt.get_dataset(index).get_index('terrier_stemmed'), memory=True)
    return pt.BatchRetrieve(index, wmodel='BM25')

LOAD_FUNCS = {
    'monot5': load_monot5,
    'bi_encoder': load_bi_encoder,
    'electra': load_electra,
    'splade': load_splade,
    'bm25': load_bm25
}