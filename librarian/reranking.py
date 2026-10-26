"""Cross-encoder ranking, candidate budgets, and passage-level evidence gates."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CandidateBudgets:
    baseline: int = 24
    optimized: int = 8

    def __post_init__(self):
        if self.baseline<1 or self.optimized<1:raise ValueError('Candidate budgets must be positive.')

    def for_profile(self,profile):
        if profile not in ('baseline','optimized'):raise ValueError('Unknown profile.')
        return getattr(self,profile)

class CrossEncoderReranker:
    def __init__(self,model,batch_size=24):
        if batch_size<1:raise ValueError('Reranker batch size must be positive.')
        self.model=model;self.batch_size=batch_size

    def rank(self,question,hits):
        if not hits:return []
        scores=self.model.predict([(question,hit['text']) for hit in hits],batch_size=self.batch_size)
        if len(scores)!=len(hits):raise ValueError('Reranker returned an unexpected score count.')
        ranked=[{**hit,'rerank_score':float(score)} for hit,score in zip(hits,scores)]
        return sorted(ranked,key=lambda hit:hit['rerank_score'],reverse=True)

def select_evidence(hits,threshold=0.0,score_gap=None):
    """Gate every passage, rather than accepting all after a good first hit.

    Cross-encoder scores are logits, not calibrated probabilities. Score-gap
    filtering is optional and must be chosen using development evidence.
    """
    if not math.isfinite(threshold):raise ValueError('Threshold must be finite.')
    if score_gap is not None and (not math.isfinite(score_gap) or score_gap<0):
        raise ValueError('Score gap must be finite and nonnegative.')
    eligible=[h for h in hits if h.get('rerank_score') is not None
              and math.isfinite(h['rerank_score']) and h['rerank_score']>=threshold]
    if not eligible or score_gap is None:return eligible
    best=max(h['rerank_score'] for h in eligible)
    return [h for h in eligible if best-h['rerank_score']<=score_gap]
