"""Recommend a logit gate from development evidence only; never change it automatically."""
import argparse
import json
import math
import os
from librarian.corpus import DATA,ROOT

DEV_NEGATIVES=[
    'Which Django middleware enables CSRF protection?',
    'How do I connect a Motor client to MongoDB Atlas?',
    'Which TensorFlow API loads saved neural network weights?',
    'How do I configure Spring Boot application properties?',
    'How do I create an Android activity in Kotlin?',
    'Which React hook manages component state?',
]

def recommend_threshold(rows):
    if not rows:raise ValueError('No development scores.')
    scores=[r['score'] for r in rows if r['score'] is not None and math.isfinite(r['score'])]
    candidates=sorted(set([0.0]+scores+([math.nextafter(max(scores),math.inf)] if scores else [])))
    def metrics(threshold):
        positives=[r for r in rows if r['relevant']]
        negatives=[r for r in rows if not r['relevant']]
        accepts=lambda r:r['score'] is not None and r['score']>=threshold
        rates=[]
        if positives:rates.append(sum(accepts(r) for r in positives)/len(positives))
        if negatives:rates.append(sum(not accepts(r) for r in negatives)/len(negatives))
        return sum(rates)/len(rates)
    best=max(candidates,key=lambda value:(metrics(value),-abs(value)))
    return {'threshold':best,'balanced_accuracy_proxy':metrics(best),'n':len(rows)}

def main(candidates=24):
    # Small matrix/vector queries do not benefit from large BLAS thread pools.
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','2')
    from librarian.engine import Engine,fingerprint
    engine=Engine();output=[]
    rows=[json.loads(line) for line in (DATA/'eval_questions.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['split']=='dev']
    rows += [dict(id=f'dev-negative-{i}',question=q,relevant_ids=[],split='dev') for i,q in enumerate(DEV_NEGATIVES,1)]
    for row in rows:
        hits,_=engine.search(row['question'],k=1,candidates=candidates)
        output.append(dict(id=row['id'],question=row['question'],score=hits[0]['rerank_score'] if hits else None,
                           relevant=bool(hits and hits[0]['id'] in row['relevant_ids'])))
        print(f'Scored {row["id"]}',flush=True)
    result={'corpus_sha256':fingerprint(),'candidate_budget':candidates,**recommend_threshold(output),'rows':output,
            'warning':'Weak dev relevance labels and six curated dev negatives; review manually. Scores are not probabilities. Test questions were not used. Recommendation does not change runtime configuration.'}
    folder=ROOT/'results';folder.mkdir(exist_ok=True)
    (folder/'dev_calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidates',type=int,default=24)
    args=parser.parse_args();main(args.candidates)
