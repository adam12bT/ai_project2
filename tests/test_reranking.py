import pytest
from librarian.reranking import CandidateBudgets,CrossEncoderReranker,select_evidence

def test_ranking_preserves_inputs_and_reorders_by_cross_encoder():
    class Model:
        def predict(self,pairs,batch_size):
            assert pairs[0]==('question','one') and batch_size==2
            return [1.2,-3,2.5]
    hits=[dict(text=text,rerank_score=None) for text in ['one','two','three']]
    ranked=CrossEncoderReranker(Model(),2).rank('question',hits)
    assert [h['text'] for h in ranked]==['three','one','two']
    assert hits[0]['rerank_score'] is None

def test_low_quality_tail_is_not_accepted_after_a_good_first_hit():
    hits=[dict(text='supported',rerank_score=5),dict(text='unrelated',rerank_score=-2)]
    assert [h['text'] for h in select_evidence(hits)]==['supported']

def test_gap_can_exclude_weak_but_positive_passages():
    hits=[dict(rerank_score=5),dict(rerank_score=1)]
    assert len(select_evidence(hits,score_gap=2))==1
    assert select_evidence(hits,threshold=6)==[]

def test_candidate_profiles_are_explicit():
    budgets=CandidateBudgets(24,6)
    assert budgets.for_profile('optimized')==6
    with pytest.raises(ValueError):budgets.for_profile('unknown')

def test_nonfinite_scores_cannot_become_evidence():
    assert select_evidence([dict(rerank_score=float('nan'))])==[]
