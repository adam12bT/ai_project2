import pytest
from librarian import generation as g

HITS=[dict(title='CSV',section='DictReader',text='DictReader maps each row to a dict.')]

class Response:
    status_code=200
    ok=True
    headers={}
    def json(self):
        return {'choices':[{'message':{'content':'{"claims":[{"text":"Use DictReader.","sources":[1],"quotes":[{"source":1,"quote_id":1}]}],"abstained":false}'}}],
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
    assert answer.supporting_quotes[1]==['DictReader maps each row to a dict.']

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

def test_fabricated_quotation_rejected():
    payload={'claims':[{'text':'A claim','sources':[1],'quotes':[{'source':1,'quote':'This invented evidence does not exist.'}]}],'abstained':False}
    with pytest.raises(g.GenerationError,match='not found'):g.validate_payload(payload,HITS)

def test_every_cited_passage_requires_its_own_quotation():
    payload={'claims':[{'text':'A claim','sources':[1,2],'quotes':[{'source':1,'quote':HITS[0]['text']}]}],'abstained':False}
    with pytest.raises(g.GenerationError,match='Every cited source'):g.validate_payload(payload,HITS+HITS)

def test_quotes_allow_harmless_whitespace_differences():
    payload={'claims':[{'text':'A claim','sources':[1],'quotes':[{'source':1,'quote':'DictReader maps\neach row to a dict.'}]}],'abstained':False}
    assert g.validate_payload(payload,HITS)[1]==[1]

def test_production_evidence_does_not_accept_legacy_unquoted_answers():
    with pytest.raises(g.GenerationError,match='structured claims'):
        g.validate_payload({'answer':'Use DictReader [1].','abstained':False},HITS)


def test_openai_request_and_unknown_cost(monkeypatch):
    monkeypatch.setenv('GENERATOR','openai')
    monkeypatch.setenv('OPENAI_API_KEY','test-only-openai-key')
    monkeypatch.setenv('OPENAI_MODEL','gpt-4.1-mini')
    monkeypatch.delenv('OPENAI_INPUT_USD_PER_MILLION',raising=False)
    monkeypatch.delenv('OPENAI_OUTPUT_USD_PER_MILLION',raising=False)
    def post(url,**kwargs):
        assert url=='https://api.openai.com/v1/chat/completions'
        assert kwargs['headers']['Authorization']=='Bearer test-only-openai-key'
        assert kwargs['json']['model']=='gpt-4.1-mini'
        assert kwargs['json']['response_format']['type']=='json_schema'
        assert 'include_reasoning' not in kwargs['json']
        return Response()
    monkeypatch.setattr(g.requests,'post',post)
    answer=g.generate('Read CSV?',HITS)
    assert answer.kind=='openai' and answer.cost_usd is None
    assert answer.usage['total_tokens']==110
    assert answer.supporting_quotes[1]==[HITS[0]['text']]


def test_auto_prefers_openai_and_missing_explicit_key_fails(monkeypatch):
    monkeypatch.setenv('GENERATOR','auto')
    monkeypatch.setenv('OPENAI_API_KEY','test-only-key')
    monkeypatch.setenv('GROQ_API_KEY','test-only-groq-key')
    assert g.configuration()[0]=='openai'
    monkeypatch.setenv('GENERATOR','openai')
    monkeypatch.delenv('OPENAI_API_KEY')
    with pytest.raises(g.GenerationError,match='OPENAI_API_KEY'):
        g.generate('Read CSV?',HITS)


def test_openai_refusal_and_incomplete_output(monkeypatch):
    monkeypatch.setenv('GENERATOR','openai')
    monkeypatch.setenv('OPENAI_API_KEY','test-only-key')
    class Refusal(Response):
        def json(self):return {'choices':[{'message':{'refusal':'Refused','content':None}}]}
    monkeypatch.setattr(g.requests,'post',lambda *a,**kw:Refusal())
    with pytest.raises(g.GenerationError,match='declined'):g.generate('Read CSV?',HITS)
    class Incomplete(Response):
        def json(self):return {'choices':[{'message':{'content':'{'},'finish_reason':'length'}]}
    monkeypatch.setattr(g.requests,'post',lambda *a,**kw:Incomplete())
    with pytest.raises(g.GenerationError,match='output limit'):g.generate('Read CSV?',HITS)


def test_openai_auth_error_does_not_expose_key(monkeypatch):
    monkeypatch.setenv('GENERATOR','openai');monkeypatch.setenv('OPENAI_API_KEY','secret-openai-value')
    response=Response();response.status_code=401;response.ok=False
    monkeypatch.setattr(g.requests,'post',lambda *a,**kw:response)
    with pytest.raises(g.GenerationError,match='OpenAI rejected') as error:g.generate('Read CSV?',HITS)
    assert 'secret-openai-value' not in str(error.value)


@pytest.mark.parametrize('provider',['openai','groq'])
def test_invalid_quote_repaired_once_and_usage_includes_both_calls(monkeypatch,provider):
    monkeypatch.setenv('GENERATOR',provider)
    monkeypatch.setenv(provider.upper()+'_API_KEY','test-only-key')
    monkeypatch.setenv('GROQ_FREE_TIER','false')
    monkeypatch.setenv(provider.upper()+'_INPUT_USD_PER_MILLION','1')
    monkeypatch.setenv(provider.upper()+'_OUTPUT_USD_PER_MILLION','2')
    calls=[]
    class BadQuote(Response):
        def json(self):
            body=super().json()
            body['choices'][0]['message']['content']=body['choices'][0]['message']['content'].replace(
                '"quote_id":1','"quote_id":999')
            return body
    def post(url,**kwargs):
        calls.append(kwargs['json'])
        return BadQuote() if len(calls)==1 else Response()
    monkeypatch.setattr(g.requests,'post',post)
    answer=g.generate('Read CSV?',HITS)
    assert len(calls)==2
    assert 'previous attempt failed quotation validation' in calls[1]['messages'][0]['content']
    assert calls[0]['messages'][1]==calls[1]['messages'][1]
    assert answer.supporting_quotes=={1:[HITS[0]['text']]}
    assert answer.usage==dict(prompt_tokens=200,completion_tokens=20,total_tokens=220)
    assert answer.cost_usd==pytest.approx(0.00024)


def test_repeated_invalid_quote_returns_labelled_source_passages(monkeypatch):
    monkeypatch.setenv('GENERATOR','openai')
    monkeypatch.setenv('OPENAI_API_KEY','test-only-key')
    calls=[]
    class BadQuote(Response):
        def json(self):
            body=super().json()
            body['choices'][0]['message']['content']=body['choices'][0]['message']['content'].replace(
                '"quote_id":1','"quote_id":999')
            return body
    def post(*args,**kwargs):calls.append(1);return BadQuote()
    monkeypatch.setattr(g.requests,'post',post)
    answer=g.generate('Read CSV?',HITS)
    assert len(calls)==2
    assert answer.kind=='extractive_fallback'
    assert 'could not be verified' in answer.text
    assert HITS[0]['text'] in answer.text
    assert 'Invented source quotation.' not in answer.text
    assert answer.citation_numbers==[1]
    assert answer.usage['total_tokens']==220


def test_quote_options_are_contiguous_source_excerpts():
    text='First sentence with enough text.\nA second sentence. '+('x'*650)
    normalized=' '.join(text.split())
    options=g.quotation_options(text)
    assert 'First sentence with enough text.' in options
    assert 'A second sentence.' in options
    assert all(12<=len(quote)<=300 and quote in normalized for quote in options)


def test_selected_quote_is_resolved_from_its_own_source():
    context=[{'quote_options':[{'text':'First original passage.'}]},
             {'quote_options':[{'text':'Second original passage.'},{'text':'Another exact excerpt.'}]}]
    payload={'abstained':False,'claims':[{'text':'Explanation','sources':[2],
                'quotes':[{'source':2,'quote_id':2,'quote':'Fabricated text'}]}]}
    g.resolve_quotations(payload,context)
    assert payload['claims'][0]['quotes'][0]['quote']=='Another exact excerpt.'
    assert g.validate_payload(payload,[{'text':'First original passage.'},
               {'text':'Second original passage. Another exact excerpt.'}])[1]==[2]


@pytest.mark.parametrize('source,quote_id',[(1,2),(0,1),(True,1),(1,True),(1,0),(2,1)])
def test_quote_selection_rejects_unknown_or_invalid_ids(source,quote_id):
    payload={'abstained':False,'claims':[{'quotes':[{'source':source,'quote_id':quote_id}]}]}
    with pytest.raises(g.QuotationError):
        g.resolve_quotations(payload,[{'quote_options':[{'text':'Original excerpt.'}]}])
