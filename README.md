---
title: Ask Python Docs
emoji: 📚
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
---

# Ask Python Docs — Project 2: The Librarian

A RAG service over official Python 3.12 documentation. Local BM25 + neural vectors retrieve passages, a cross-encoder reranks them, and Groq generates a cited answer. Without a key, the app explicitly uses an extractive offline baseline. A FastAPI service powers the browser interface.

## Configure Groq

Open the prepared `.env` file and enter your own `GROQ_API_KEY`. Keep the key out of chat, screenshots and version control. `.env` is ignored by git and excluded from Docker builds. Obtain a key at https://console.groq.com/keys.

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
GENERATOR=auto
GROQ_FREE_TIER=true
```

`auto` uses Groq when a key exists; `groq` requires a key; `extractive` explicitly disables API calls. Restart the server after changing `.env`. Account-specific quotas still apply. Groq requires no local LLM download; the two small local models are for retrieval and reranking. Questions and retrieved public documentation passages are sent to Groq.

Citation identifiers and paragraph coverage are validated, but semantic faithfulness still requires evaluation. API and quota failures are shown explicitly. Provider references: https://console.groq.com/docs/api-reference and https://console.groq.com/docs/models.

## Setup (Windows / PowerShell, Python 3.11)

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m librarian.corpus
./.venv/Scripts/python.exe -m librarian.engine
./.venv/Scripts/python.exe -m eval.make_dataset
Copy-Item .env.example .env
```

Only use these setup steps for a fresh installation. This folder is already prepared; use the run command below. Initial setup requires internet for the documentation and retrieval models. The reranker downloads on first launch. Keep your existing .env file when using this prepared project.

## Run

```powershell
./.venv/Scripts/python.exe -m uvicorn librarian.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Interactive API reference: http://127.0.0.1:8000/docs.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/ask -Method Post -ContentType application/json -Body '{"question":"How do I read CSV rows as dictionaries?"}'
```

## Evaluate

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m eval.run --split dev
./.venv/Scripts/python.exe -m eval.run --split test
# Real Groq benchmark (uses your account quota):
./.venv/Scripts/python.exe -m eval.run --split test --generator groq --pause 12
# Semantic judge (additional API quota; same-model bias must be reviewed):
./.venv/Scripts/python.exe -m eval.judge --split test --pause 8
# Smaller, explicitly labeled semantic sample:
./.venv/Scripts/python.exe -m eval.judge --split test --limit 6 --pause 20
```

`results/extractive/metrics_test.json` and `results/groq/metrics_test.json` keep offline and Groq measurements separate. Both include retrieval recall@5, MRR@5, latency, correctness proxies, abstention accuracy and the optimization decision. Groq records token usage. Exact-copy support only applies to extractive answers; semantic faithfulness requires judge/human evaluation. `answers_test.json` in each directory is the review artifact. Timings exclude startup and downloads, with cold response caches. This serial benchmark does not establish concurrent-load SLOs.

The corpus was repaired after these measurements to preserve code lines and decode UTF-8 correctly. Saved benchmark scores and chunk IDs refer to the earlier snapshot; see `results/README.md`. The current index and evaluation source labels have been rebuilt, but old scores must not be claimed as measurements of this repaired version.

The completed Groq run covers 65 test cases per profile: optimized correctness proxy 0.877, baseline 0.846; both reject all 10 negatives. Optimized p95 is 5,451.8 ms vs 6,930.0 ms baseline (21.3% reduction). The 50% optimization target was **not met**. Proxy scores are not human correctness; quotas/backoff and fixed per-query profile ordering limit timing conclusions. Hardware details are in `results/hardware.json`.

The dataset has 110 answerable cases and 10 out-of-scope cases. Questions are curated; passage labels are resolved automatically from documented symbol names and marked `needs_human_review`. Test questions paraphrase dev topics; this is not topic-independent generalization. Review source relevance and answer correctness before submission. Exact copying proves textual support, not that an answer fully addresses the question.

## Repository

| Path | Purpose |
|---|---|
| `librarian/corpus.py` | Download, section extraction, code-aware chunking, provenance |
| `librarian/engine.py` | Embeddings, BM25, hybrid fusion, reranking, abstention, cache |
| `librarian/app.py` | API, web UI, explicit feedback capture |
| `data/eval_questions.jsonl` | Questions, answers, source passages, split and review status |
| `eval/` | Dataset builder, ablation and paired benchmark, confidence inspection |
| `results/` | Actual machine-readable measurements and answers |
| `design.md` | Design choices and failure cases |
| `optimisation.md` | Baseline vs optimized protocol and measured results |
| `report.md` | Concise project report |
| `postmortem.md` | Problems, limitations and remaining work |

## Deployment

GitHub source: https://github.com/adam12bT/ai_project2

Deployment target: https://huggingface.co/spaces/adamnouassida5/the_dot_camp5

The GitHub Actions workflow in `.github/workflows/ci-cd.yml` builds the CPU Docker image, runs the tests, and checks the frontend, health endpoint and a cited API answer. Pull requests only run CI. Passing pushes to `main` also sync the runtime files to the existing Hugging Face Space, which rebuilds the app. A successful sync means the source was uploaded; check the Space build/runtime status before treating deployment as healthy.

Configure these secrets once:

- **GitHub repository > Settings > Secrets and variables > Actions:** `HF_TOKEN`, a Hugging Face write token permitted to update `adamnouassida5/the_dot_camp5`.
- **Hugging Face Space > Settings > Secrets:** `GROQ_API_KEY`, your Groq API key.
- **Hugging Face Space > Settings > Variables:** `GENERATOR=groq`, `GROQ_MODEL=openai/gpt-oss-20b`, `GROQ_FREE_TIER=true`.

Deployment replaces the Space's old source files with the runtime allowlist. The README configures Docker and port 8000. `.env`, models, caches, feedback and local evaluation outputs are excluded from sync. The Groq key is not needed by GitHub CI. HF secrets are managed separately from source files.

The image indexes the committed corpus and downloads public retrieval/reranker weights during build. Runtime model loading is offline. The container runs as UID 1000 with writable data and model-cache directories, as required by Spaces. This avoids re-downloading Python documentation during deployment.

```powershell
docker build -t ask-python-docs .
docker run --rm --env-file .env -p 8000:8000 ask-python-docs
```

The image downloads the documents and models during build; model startup may require a few GB of RAM. Configure a container host to expose port 8000 and use `/health` for readiness. A public URL requires a hosting account and an actual deployment. This repository does not claim a public deployment exists. Mount `/app/data` persistently if collecting feedback, retaining the existing corpus/index/cache contents. Feedback contains only voluntarily submitted questions and votes; do not submit personal data. Add authentication, rate limiting, HTTPS and storage retention before opening an unrestricted public endpoint.

## Cost model

With `GROQ_FREE_TIER=true`, API cost is recorded as zero under the user's free allowance; account quotas still apply. Hardware, electricity and hosting are excluded. For paid accounts, set that flag false and supply current token prices; otherwise cost is unknown. Python documentation license: https://docs.python.org/3.12/license.html. Keep attribution when redistributing the corpus. No equivalent printed-page count is claimed.

## Contributions

Implementation, dataset preparation, evaluation scripts and documentation were prepared with Codex assistance. Team members must add their actual names, individual contributions, review notes and explanations before submission. Do not claim unperformed user studies or instructor approval.

## Submission checklist requiring real-world work

- Obtain instructor approval of the corpus. Groq provides LLM generation; extractive mode is an offline baseline.
- Review the 120 evaluation labels; independently assess citation relevance and answer completeness.
- Recruit classmates who use Python; collect real feedback through the app and summarize it.
- Deploy to your chosen host and record its actual URL.
- Run the measured optimization and chunk-size comparisons; report whether the target was met.
- Record team contributions and connect the evaluation to your P1 harness if required.
