import numpy as np
import pytest
from librarian.retrieval import Retriever,RetrievalConfig,reciprocal_rank_fusion
from librarian.vector_store import ChromaVectorStore

class Encoder:
    def encode(self,*args,**kwargs):return np.array([1.,0.])

def fixture(tmp_path):
    chunks=[dict(id=str(i),title='Docs',section='API',page=page,text=text) for i,(page,text) in enumerate([
        ('library/csv','CSV table rows reader'),('library/pathlib','filesystem directory paths'),('library/json','JSON decoder mappings')])]
    store=ChromaVectorStore(tmp_path/'chroma')
    store.build(chunks,np.array([[0.,1.],[1.,0.],[0.,1.]]),'test-corpus','test-embedder')
    return chunks,store

def test_weights_can_change_the_selected_candidate(tmp_path):
    chunks,vectors=fixture(tmp_path)
    lexical=Retriever(chunks,vectors,Encoder(),RetrievalConfig(1,0))
    semantic=Retriever(chunks,vectors,Encoder(),RetrievalConfig(0,1))
    assert lexical.retrieve('CSV',candidates=1)[0]['id']=='0'
    assert semantic.retrieve('CSV',candidates=1)[0]['id']=='1'

def test_filter_applies_to_both_retrievers_before_ranking(tmp_path):
    chunks,vectors=fixture(tmp_path);retriever=Retriever(chunks,vectors,Encoder())
    assert retriever.retrieve('CSV',page='library/pathlib')[0]['id']=='1'
    assert retriever.retrieve('CSV',page='missing')==[]

def test_shared_hits_accumulate_fusion_scores():
    order,scores=reciprocal_rank_fusion([[2,1],[1,3]],[1,1])
    assert order[0]==1 and scores[1]>scores[2]

@pytest.mark.parametrize('weights',[(0,0),(-1,1),(float('nan'),1)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):RetrievalConfig(*weights)


def test_chroma_persists_and_rejects_mismatched_corpus(tmp_path):
    chunks,store=fixture(tmp_path)
    reopened=ChromaVectorStore(tmp_path/'chroma')
    reopened.open('test-corpus','test-embedder',len(chunks))
    assert reopened.search(np.array([1.,0.]),1)[0][0]=='1'
    assert reopened.collection.get(ids=['1'])['documents']==['filesystem directory paths']
    with pytest.raises(ValueError,match='does not match'):
        reopened.open('different-corpus','test-embedder',len(chunks))


def test_chroma_rebuild_removes_deleted_chunks(tmp_path):
    chunks,store=fixture(tmp_path)
    store.build(chunks[:1],np.array([[0.,1.]]),'new-corpus','test-embedder')
    assert store.collection.count()==1
    assert store.search(np.array([1.,0.]),1)[0][0]=='0'
