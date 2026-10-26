import numpy as np
import pytest
from librarian.retrieval import Retriever,RetrievalConfig,reciprocal_rank_fusion

class Encoder:
    def encode(self,*args,**kwargs):return np.array([1.,0.])

def fixture():
    chunks=[dict(id=str(i),title='Docs',section='API',page=page,text=text) for i,(page,text) in enumerate([
        ('library/csv','CSV table rows reader'),('library/pathlib','filesystem directory paths'),('library/json','JSON decoder mappings')])]
    return chunks,np.array([[0.,1.],[1.,0.],[0.,1.]])

def test_weights_can_change_the_selected_candidate():
    chunks,vectors=fixture()
    lexical=Retriever(chunks,vectors,Encoder(),RetrievalConfig(1,0))
    semantic=Retriever(chunks,vectors,Encoder(),RetrievalConfig(0,1))
    assert lexical.retrieve('CSV',candidates=1)[0]['id']=='0'
    assert semantic.retrieve('CSV',candidates=1)[0]['id']=='1'

def test_filter_applies_to_both_retrievers_before_ranking():
    chunks,vectors=fixture();retriever=Retriever(chunks,vectors,Encoder())
    assert retriever.retrieve('CSV',page='library/pathlib')[0]['id']=='1'
    assert retriever.retrieve('CSV',page='missing')==[]

def test_shared_hits_accumulate_fusion_scores():
    order,scores=reciprocal_rank_fusion([[2,1],[1,3]],[1,1])
    assert order[0]==1 and scores[1]>scores[2]

@pytest.mark.parametrize('weights',[(0,0),(-1,1),(float('nan'),1)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):RetrievalConfig(*weights)
