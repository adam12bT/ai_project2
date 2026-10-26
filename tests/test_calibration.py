from eval.calibrate import recommend_threshold

def test_gate_separates_supported_evidence_and_negatives():
    result=recommend_threshold([dict(score=4,relevant=True),dict(score=3,relevant=True),dict(score=-2,relevant=False),dict(score=-3,relevant=False)])
    assert result['balanced_accuracy_proxy']==1
    assert -2<result['threshold']<=3

def test_missing_evidence_is_rejected():
    result=recommend_threshold([dict(score=None,relevant=False),dict(score=4,relevant=True)])
    assert result['balanced_accuracy_proxy']==1
