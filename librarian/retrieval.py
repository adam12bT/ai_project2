"""Configurable lexical/dense retrieval and weighted reciprocal rank fusion."""
from dataclasses import dataclass
import math
import re
import numpy as np
from rank_bm25 import BM25Okapi

def tokens(text):
    return re.findall(r'[a-z_][a-z_0-9]*', text.lower())

@dataclass(frozen=True)
class RetrievalConfig:
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    rrf_k: int = 60

    def __post_init__(self):
        weights=(self.bm25_weight,self.vector_weight)
        if any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
            raise ValueError('Fusion weights must be finite, nonnegative and not both zero.')
        if self.rrf_k<1:raise ValueError('RRF constant must be positive.')

def reciprocal_rank_fusion(rankings, weights, constant=60):
    if len(rankings)!=len(weights):raise ValueError('Every ranking requires a weight.')
    scores={}
    for ranking,weight in zip(rankings,weights):
        if weight==0:continue
        seen=set()
        for rank,index in enumerate(ranking,1):
            index=int(index)
            if index in seen:continue
            seen.add(index)
            scores[index]=scores.get(index,0.0)+weight/(constant+rank)
    return sorted(scores,key=lambda index:(-scores[index],index)),scores

class Retriever:
    def __init__(self,chunks,vector_store,embedder,config=None):
        self.chunks=chunks;self.vector_store=vector_store;self.embedder=embedder
        self.row_by_id={chunk['id']:i for i,chunk in enumerate(chunks)}
        self.config=config or RetrievalConfig()
        self.keyword_index=BM25Okapi([tokens(c['title']+' '+c['section']+' '+c['text']) for c in chunks])

    def retrieve(self,question,mode='hybrid',candidates=24,page=None):
        if mode not in ('keyword','vector','hybrid'):raise ValueError('Unknown retrieval mode.')
        if candidates<1:raise ValueError('Candidate count must be positive.')
        eligible=np.array([i for i,c in enumerate(self.chunks) if not page or c['page']==page],dtype=int)
        if not len(eligible):return []
        sparse=self.keyword_index.get_scores(tokens(question))
        lexical=eligible[np.argsort(-sparse[eligible],kind='stable')[:candidates]]
        dense={};fusion={}
        if mode!='keyword':
            query=self.embedder.encode(question,normalize_embeddings=True)
            matches=self.vector_store.search(query,min(candidates,len(eligible)),page=page)
            dense={self.row_by_id[chunk_id]:score for chunk_id,score in matches}
            semantic=list(dense)
        if mode=='keyword':order=list(lexical)
        elif mode=='vector':order=list(semantic)
        else:
            order,fusion=reciprocal_rank_fusion([lexical,semantic],[self.config.bm25_weight,self.config.vector_weight],self.config.rrf_k)
        return [{**self.chunks[int(i)],'keyword_score':float(sparse[i]),
                 'vector_score':dense.get(int(i)),
                 'fusion_score':fusion.get(int(i)),'rerank_score':None} for i in order[:candidates]]
