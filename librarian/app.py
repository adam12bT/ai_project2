from contextlib import asynccontextmanager
import json
import threading
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .corpus import ROOT, DATA
from .engine import Engine
from .generation import GenerationError, configuration

lock=threading.Lock()
@asynccontextmanager
async def lifespan(app):
    app.state.engine=Engine()
    yield

app=FastAPI(title='Ask Python Docs',version='1.0.0',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

class Question(BaseModel):
    question: str=Field(min_length=3,max_length=1200)
    profile: str='optimized'
    page: str | None=None

class Feedback(BaseModel):
    question: str=Field(min_length=3,max_length=1200)
    helpful: bool
    comment: str=Field(default='',max_length=1000)

@app.get('/')
def home(): return FileResponse(ROOT/'static/index.html')

@app.get('/health')
def health(): return dict(status='ok',chunks=len(app.state.engine.chunks),version='3.12',generator=configuration()[0],model=configuration()[1])

@app.get('/api/pages')
def pages(): return sorted(set(c['page'] for c in app.state.engine.chunks))

@app.post('/api/ask')
def ask(body: Question):
    if body.profile not in ('baseline','optimized'): raise HTTPException(422,'Unknown profile')
    if not body.question.strip(): raise HTTPException(422,'Enter a question')
    # Serialize local inference to avoid device contention and cache races.
    try:
        with lock: return app.state.engine.ask(body.question.strip(),body.profile,body.page)
    except GenerationError as error:
        raise HTTPException(503,str(error)) from None

@app.post('/api/feedback')
def feedback(body: Feedback):
    with lock:
        with (DATA/'feedback.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({**body.model_dump(),'timestamp':datetime.now(timezone.utc).isoformat()})+'\n')
    return {'saved':True}
