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
from .corpus import DATA
from .generation import generate, configuration
from .retrieval import Retriever, RetrievalConfig, tokens
from .reranking import CrossEncoderReranker, CandidateBudgets, select_evidence
os.environ.setdefault('HF_HOME',str(DATA/'cache/huggingface'))
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDER = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
STOP = set('a an the is are was were i you how do does to in of for and or can with python what when which my it'.split())

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
        self.embedder=SentenceTransformer(EMBEDDER)
        self.retriever=Retriever(self.chunks,self.vectors,self.embedder,RetrievalConfig(
            bm25_weight=float(os.getenv('BM25_WEIGHT','1')),
            vector_weight=float(os.getenv('VECTOR_WEIGHT','1')),
            rrf_k=int(os.getenv('RRF_K','60'))))
        self.reranker=CrossEncoder(RERANKER)
        self.reranking=CrossEncoderReranker(self.reranker,int(os.getenv('RERANK_BATCH_SIZE','24')))
        self.budgets=CandidateBudgets(int(os.getenv('BASELINE_CANDIDATES','24')),int(os.getenv('OPTIMIZED_CANDIDATES','8')))
        self.cache=OrderedDict()

    def search(self, question, mode='hybrid_rerank', k=5, candidates=24, page=None):
        start=time.perf_counter()
        if k<1:raise ValueError('Result count must be positive.')
        retrieval_mode='hybrid' if mode=='hybrid_rerank' else mode
        hits=self.retriever.retrieve(question,mode=retrieval_mode,candidates=candidates,page=page)
        if mode=='hybrid_rerank' and hits:
            hits=self.reranking.rank(question,hits)
        return hits[:k],(time.perf_counter()-start)*1000

    def ask(self, question, profile='optimized', page=None, use_cache=True):
        start=time.perf_counter(); provider,model=configuration()
        key=(question.strip(),profile,page,provider,model,os.getenv('RERANK_THRESHOLD','0'),os.getenv('RERANK_SCORE_GAP',''))
        if use_cache and key in self.cache:
            result=dict(self.cache[key]); result.update(cache_hit=True,latency_ms=(time.perf_counter()-start)*1000,cost_usd=0.0,usage={})
            self.cache.move_to_end(key); return result
        hits,search_ms=self.search(question,candidates=self.budgets.for_profile(profile),k=3,page=page)
        # A conservative gate, calibrated separately by eval/calibrate.py. Never use RRF as confidence.
        threshold=float(__import__('os').environ.get('RERANK_THRESHOLD','0'))
        gap=os.getenv('RERANK_SCORE_GAP','')
        selected=select_evidence(hits,threshold,float(gap) if gap else None)
        abstained=not selected
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
            for citation in result['citations']:
                citation['supporting_quotes']=generated.supporting_quotes.get(citation['number'],[])
        if use_cache:
            self.cache[key]=result
            while len(self.cache)>128: self.cache.popitem(last=False)
        return result

if __name__=='__main__': build_index()
