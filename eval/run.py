"""Retrieval ablation, generation proxies and paired optimization benchmark."""
import argparse
import json
import time
import os
from pathlib import Path
from librarian.corpus import DATA, ROOT
from librarian.engine import Engine, fingerprint
from librarian.generation import configuration
import numpy as np

def run(split='test',limit=0,generator='extractive',pause=0):
    os.environ['GENERATOR']=generator
    if generator in ('groq','openai') and not os.getenv(generator.upper()+'_API_KEY'):
        raise SystemExit(f'Set {generator.upper()}_API_KEY in .env before running evaluation.')
    rows=[json.loads(l) for l in (DATA/'eval_questions.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['split']==split]
    if limit: rows=rows[:limit]
    engine=Engine()
    engine.search('How do I read CSV files?')  # warm local models; no unmeasured API calls
    folder=ROOT/'results' / generator;folder.mkdir(parents=True,exist_ok=True)
    ablation={}; details=[]
    answerable=[r for r in rows if r['answerable']]
    for mode in ['keyword','vector','hybrid','hybrid_rerank']:
        recall=[]; reciprocal=[]; timing=[]
        for row in answerable:
            hits,ms=engine.search(row['question'],mode=mode)
            ranks=[i+1 for i,h in enumerate(hits) if h['id'] in row['relevant_ids']]
            recall.append(bool(ranks)); reciprocal.append(1/min(ranks) if ranks else 0); timing.append(ms)
            details.append(dict(id=row['id'],mode=mode,hits=[h['id'] for h in hits],latency_ms=ms))
        ablation[mode]=dict(recall_at_5=float(np.mean(recall)),mrr_at_5=float(np.mean(reciprocal)),p50_ms=float(np.percentile(timing,50)),p95_ms=float(np.percentile(timing,95)),n=len(recall))
    profiles={}; per_profile={}
    # Interleave profiles per question to reduce time/order bias. Cold response cache throughout.
    for row in rows:
        for profile in (['baseline','optimized'] if int(row['id'][1:])%2 else ['optimized','baseline']):
            result=engine.ask(row['question'],profile,use_cache=False)
            citations=result['citations']
            correct=(not result['abstained'] and any(c['id'] in row['relevant_ids'] for c in citations) and row['answer'].lower() in result['answer'].lower()) if row['answerable'] else result['abstained']
            # Support is exact-copy evidence, not an LLM faithfulness assessment.
            supported=all(c['text'] in result['answer'] for c in citations) if result['answer_type']=='extractive' else None
            per_profile.setdefault(profile,[]).append(dict(id=row['id'],correct_proxy=bool(correct),exact_copy_support=supported,abstained=result['abstained'],latency_ms=result['latency_ms'],answer=result['answer'],citations=citations,usage=result['usage'],cost_usd=result['cost_usd'],human_correctness=None,human_faithfulness=None,human_citation_accuracy=None))
            if generator!='extractive' and pause: time.sleep(pause)
            (folder/f'answers_{split}_partial.json').write_text(json.dumps(per_profile,indent=2),encoding='utf-8')
            print(f'{generator}: completed {profile}/{row["id"]}',flush=True)
    for profile,items in per_profile.items():
        copy_scores=[r['exact_copy_support'] for r in items if r['citations'] and r['exact_copy_support'] is not None]
        profiles[profile]=dict(accuracy_proxy=float(np.mean([r['correct_proxy'] for r in items])),exact_copy_support=float(np.mean(copy_scores)) if copy_scores else None,abstention_accuracy=float(np.mean([r['abstained'] for r in items if not next(x for x in rows if x['id']==r['id'])['answerable']])) if any(not x['answerable'] for x in rows) else None,p50_ms=float(np.percentile([r['latency_ms'] for r in items],50)),p95_ms=float(np.percentile([r['latency_ms'] for r in items],95)),n=len(items),api_cost_usd=sum(r['cost_usd'] for r in items) if all(r['cost_usd'] is not None for r in items) else None,total_tokens=sum(r['usage'].get('total_tokens',0) for r in items))
    baseline=profiles['baseline']; optimized=profiles['optimized']
    reduction=1-optimized['p95_ms']/baseline['p95_ms']; loss=baseline['accuracy_proxy']-optimized['accuracy_proxy']
    result=dict(corpus_sha256=fingerprint(),split=split,label_warning='Automatically resolved weak labels; human review pending. Dev/test questions share topics. Correctness and support are proxies.',ablation=ablation,profiles=profiles,optimization=dict(p95_reduction=reduction,accuracy_loss_points=loss*100,target_met=bool(reduction>=.5 and loss<=.02)),hardware=__import__('platform').platform(),timestamp_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    result.update(generator=generator,model=configuration()[1] if generator!='extractive' else None,
                  runtime={'embedding_device':str(engine.embedder.device),'reranker_device':str(engine.reranker.model.device)})
    folder=ROOT/'results' / generator;folder.mkdir(parents=True,exist_ok=True)
    (folder/f'metrics_{split}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (folder/f'answers_{split}.json').write_text(json.dumps(per_profile,indent=2),encoding='utf-8')
    (folder/f'retrieval_{split}.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--generator',choices=['extractive','groq','openai'],default='extractive');p.add_argument('--pause',type=float,default=0)
    a=p.parse_args();run(a.split,a.limit,a.generator,a.pause)
