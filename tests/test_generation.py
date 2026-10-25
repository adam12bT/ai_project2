import pytest
from librarian import generation as g

HITS=[dict(title='CSV',section='DictReader',text='DictReader maps each row to a dict.')]

class Response:
    status_code=200
    ok=True
    headers={}
    def json(self):
        return {'choices':[{'message':{'content':'{"claims":[{"text":"Use DictReader.","sources":[1]}],"abstained":false}'}}],
                'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}

def test_groq_request_and_usage(monkeypatch):
    monkeypatch.setenv('GENERATOR','groq'); monkeypatch.setenv('GROQ_API_KEY','test-only-key')
    monkeypatch.setenv('GROQ_MODEL','openai/gpt-oss-20b')
    monkeypatch.setenv('GROQ_FREE_TIER','true')
    def post(url,**kwargs):
        assert url=='https://api.groq.com/openai/v1/chat/completions'
        assert kwargs['json']['response_format']['type']=='json_schema'
        assert 'DictReader' in kwargs['json']['messages'][1]['content']
        return Response()
    monkeypatch.setattr(g.requests,'post',post)
    answer=g.generate('Read CSV?',HITS)
    assert answer.kind=='groq' and answer.citation_numbers==[1]
    assert answer.usage['total_tokens']==110 and answer.cost_usd==0

@pytest.mark.parametrize('payload',[
    {'answer':'Use a dict [9].','abstained':False},
    {'answer':'Use a dict.','abstained':False},
    {'answer':'Use a dict [1].\n\nUnsupported factual paragraph.','abstained':False},
    {'answer':'Use a dict [1].','abstained':'false'},
])
def test_invalid_citations_rejected(payload):
    with pytest.raises(g.GenerationError):g.validate_payload(payload,1)

def test_abstention():
    _,refs,abstained=g.validate_payload({'answer':'unknown','abstained':True},1)
    assert abstained and refs==[]

def test_structured_claims_attach_sources():
    text,refs,abstained=g.validate_payload({'claims':[{'text':'Use DictReader.','sources':[1]}],'abstained':False},1)
    assert text=='Use DictReader. [1]' and refs==[1] and not abstained

def test_structured_claims_reject_unknown_source():
    with pytest.raises(g.GenerationError):
        g.validate_payload({'claims':[{'text':'Use DictReader.','sources':[2]}],'abstained':False},1)

def test_rate_limit_does_not_expose_credentials(monkeypatch):
    monkeypatch.setattr(g.time,'sleep',lambda seconds:None)
    monkeypatch.setenv('GENERATOR','groq');monkeypatch.setenv('GROQ_API_KEY','secret-test-value')
    r=Response();r.status_code=429;r.ok=False
    monkeypatch.setattr(g.requests,'post',lambda *a,**kw:r)
    with pytest.raises(g.GenerationError,match='rate limit') as error:g.generate('Read CSV?',HITS)
    assert 'secret-test-value' not in str(error.value)

def test_missing_key(monkeypatch):
    monkeypatch.setenv('GENERATOR','groq');monkeypatch.delenv('GROQ_API_KEY',raising=False)
    with pytest.raises(g.GenerationError,match='GROQ_API_KEY'):g.generate('Read CSV?',HITS)

def test_extractive_does_not_call_api(monkeypatch):
    monkeypatch.setenv('GENERATOR','extractive')
    monkeypatch.setattr(g.requests,'post',lambda *a,**kw:pytest.fail('Unexpected network call'))
    assert g.generate('Read CSV?',HITS).kind=='extractive'
