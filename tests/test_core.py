import json
from fastapi.testclient import TestClient
from librarian.engine import Engine, tokens
from librarian.corpus import DATA, make_chunk

def test_ids_change_with_content():
    s=dict(url='https://docs.python.org/3.12/test.html',title='test',section='test')
    assert make_chunk(s,'a',0)['id'] != make_chunk(s,'b',0)['id']

def test_tokenize_identifiers():
    assert 'read_text' in tokens('Path.read_text()')

def test_eval_sources_exist():
    chunks={json.loads(l)['id']:json.loads(l) for l in (DATA/'chunks.jsonl').read_text(encoding='utf-8').splitlines()}
    rows=[json.loads(l) for l in (DATA/'eval_questions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(r['answerable'] for r in rows)>=100
    for row in rows:
        if row['answerable']:
            assert row['relevant_ids']
            assert any(chunks[i]['text']==row['source_passage'] for i in row['relevant_ids'])

def test_api_validation_and_citations(monkeypatch):
    monkeypatch.setenv('GENERATOR','extractive')
    from librarian.app import app
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert client.post('/api/ask',json={'question':'x'}).status_code==422
        assert client.post('/api/ask',json={'question':'read csv','profile':'bad'}).status_code==422
        body={'question':'How do I read CSV rows as dictionaries?','page':'library/csv'}
        r=client.post('/api/ask',json=body).json()
        assert not r['abstained']
        assert all(c['text'] in r['answer'] and c['url'].startswith('https://docs.python.org/3.12/') for c in r['citations'])
        assert client.post('/api/ask',json=body).json()['cache_hit']
        assert client.post('/api/ask',json={'question':'read csv','page':'not-a-page'}).json()['abstained']
