"""Inspect dev confidence scores; never tune thresholds on the final test set."""
import json
from librarian.engine import Engine
from librarian.corpus import DATA
if __name__=='__main__':
    engine=Engine(); output=[]
    for line in (DATA/'eval_questions.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['split']!='dev':continue
        hits,_=engine.search(row['question'],k=1)
        output.append(dict(id=row['id'],question=row['question'],score=hits[0]['rerank_score'],relevant=hits[0]['id'] in row['relevant_ids']))
    (DATA/'dev_confidence.json').write_text(json.dumps(output,indent=2))
