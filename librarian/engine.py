import hashlib
import json
import re
import time
import os
from collections import OrderedDict
from pathlib import Path

try:
    import pyarrow
except ImportError:
    pass
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from .corpus import DATA
from .generation import generate, configuration
os.environ.setdefault('HF_HOME',str(DATA/'cache/huggingface'))
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDER = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
STOP = set('a an the is are was were i you how do does to in of for and or can with python what when which my it'.split())

def tokens(text):
    return re.findall(r'[a-z_][a-z_0-9]*', text.lower())

def fingerprint():
    return hashlib.sha256((DATA/'chunks.jsonl').read_bytes()).hexdigest()

def build_index():
    import torch
    torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS','4')))
    chunks = [json.loads(line) for line in (DATA/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    model = SentenceTransformer(EMBEDDER)
    vectors = model.encode([c['title']+'\n'+c['section']+'\n'+c['text'] for c in chunks], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    folder=DATA/'index'; folder.mkdir(exist_ok=True)
    np.save(folder/'vectors.npy',vectors)
    (folder/'manifest.json').write_text(json.dumps(dict(corpus_sha256=fingerprint(), embedder=EMBEDDER, chunks=len(chunks)),indent=2),encoding='utf-8')

class Engine:
    def __init__(self):
        import torch
        torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS','4')))
        self.chunks = [json.loads(line) for line in (DATA/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        manifest=json.loads((DATA/'index/manifest.json').read_text())
        if manifest['corpus_sha256'] != fingerprint() or manifest['embedder'] != EMBEDDER:
            raise ValueError('Index does not match corpus/model. Run python -m librarian.engine')
        self.vectors=np.load(DATA/'index/vectors.npy')
        self.bm25=BM25Okapi([tokens(c['title']+' '+c['section']+' '+c['text']) for c in self.chunks])
        self.embedder=SentenceTransformer(EMBEDDER)
        self.reranker=CrossEncoder(RERANKER)
        self.cache=OrderedDict()

    def search(self, question, mode='hybrid_rerank', k=5, candidates=24, page=None):
        start=time.perf_counter()
        eligible=np.array([i for i,c in enumerate(self.chunks) if not page or c['page']==page],dtype=int)
        if not len(eligible): return [], (time.perf_counter()-start)*1000
        sparse=self.bm25.get_scores(tokens(question))
        sparse_order=eligible[np.argsort(-sparse[eligible])[:candidates]]
        dense = None
        if mode != 'keyword':
            query=self.embedder.encode(question, normalize_embeddings=True)
            dense=self.vectors @ query
            dense_order=eligible[np.argsort(-dense[eligible])[:candidates]]
        if mode=='keyword': order=list(sparse_order)
        elif mode=='vector': order=list(dense_order)
        else:
            fusion={}
            for ranking in (sparse_order,dense_order):
                for rank,index in enumerate(ranking): fusion[int(index)]=fusion.get(int(index),0)+1/(60+rank+1)
            order=sorted(fusion,key=fusion.get,reverse=True)[:candidates]
        scores={}
        if mode=='hybrid_rerank' and order:
            values=self.reranker.predict([(question,self.chunks[i]['text']) for i in order], batch_size=24)
            scores=dict(zip(order,map(float,values)))
            order=sorted(order,key=scores.get,reverse=True)
        hits=[{**self.chunks[int(i)], 'rerank_score':scores.get(i), 'vector_score':float(dense[i]) if dense is not None else None, 'keyword_score':float(sparse[i])} for i in order[:k]]
        return hits,(time.perf_counter()-start)*1000

    def ask(self, question, profile='optimized', page=None, use_cache=True):
        start=time.perf_counter(); provider,model=configuration()
        key=(question.strip(),profile,page,provider,model,os.getenv('RERANK_THRESHOLD','0'))
        if use_cache and key in self.cache:
            result=dict(self.cache[key]); result.update(cache_hit=True,latency_ms=(time.perf_counter()-start)*1000,cost_usd=0.0,usage={})
            self.cache.move_to_end(key); return result
        hits,search_ms=self.search(question,candidates=24 if profile=='baseline' else 8,k=3,page=page)
        # A conservative gate, calibrated separately by eval/calibrate.py. Never use RRF as confidence.
        threshold=float(__import__('os').environ.get('RERANK_THRESHOLD','0'))
        abstained=not hits or hits[0]['rerank_score'] < threshold
        selected=[] if abstained else hits
        # The no-key generator returns verbatim evidence, avoiding unsupported paraphrases.
        answer="I don't know. I couldn't find sufficiently relevant evidence in this Python 3.12 corpus."
        if selected:
            answer='\n\n'.join(f'[{i}] '+hit['text'] for i,hit in enumerate(selected,1))
        generated=generate(question,selected) if selected else None
        if generated:
            answer=generated.text
            abstained=generated.abstained
            selected=[] if abstained else selected
        result=dict(answer=answer, answer_type='extractive', abstained=abstained, citations=[dict(number=i,id=h['id'],title=h['title'],section=h['section'],url=h['url'],text=h['text']) for i,h in enumerate(selected,1)], search_ms=search_ms,latency_ms=(time.perf_counter()-start)*1000,cache_hit=False,profile=profile,cost_usd=0.0,cost_note='Local inference; excludes hardware, electricity and hosting.',version='3.12')
        result.update(answer_type=generated.kind if generated else provider,model=generated.model if generated else None,
                      usage=generated.usage if generated else {},cost_usd=generated.cost_usd if generated else 0.0,
                      cost_note='Groq free allowance assumed when GROQ_FREE_TIER=true; excludes local retrieval hardware and hosting.',
                      generation_ms=(time.perf_counter()-start)*1000-search_ms)
        if generated and not abstained:
            result['citations']=[c for c in result['citations'] if c['number'] in generated.citation_numbers]
        if use_cache:
            self.cache[key]=result
            while len(self.cache)>128: self.cache.popitem(last=False)
        return result

if __name__=='__main__': build_index()
