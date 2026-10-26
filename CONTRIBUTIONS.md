# Team RAG review plan

These implementations were prepared with AI assistance. The project owner confirmed that all three teammates reviewed their assigned changes and approved author attribution. Their RAG commits use account-linked GitHub noreply identities. Committer identity, ingestion, integration and non-RAG work remain under adam12bT.

| Assigned reviewer | RAG change | Review commit | Review branch |
|---|---|---|---|
| adam12bT | Definition-aware document chunking and exact API citation anchors | `995e8f1` | `feature/adam-function-chunks` |
| skanderbouricha1 | Weighted BM25/vector fusion, candidate selection and metadata filtering | `a9e77d7` | `feature/skanderbouricha1-hybrid-retrieval` |
| skander2904 | Cross-encoder ranking, per-passage evidence rejection and dev calibration | `30ef514` | `feature/skander2904-reranking` |
| MehdiBouzouaya | Exact source quotations for generated claims | `1d244ba` | `feature/MehdiBouzouaya-grounded-generation` |

The review branches are stacked: later branches include earlier changes. Review each listed commit's diff to inspect one person's assigned change. `team/rag-improvements` integrates everything. The project owner approved attribution and integration after confirming teammate review; the reviewed commits are preserved when merged to main.

## Review acceptance checks

- Ingestion: `python -m pytest tests/test_corpus.py`; verify that neighboring API definitions remain separate and each citation links to its symbol anchor.
- Retrieval: `python -m pytest tests/test_retrieval.py`; inspect weight validation, deterministic fusion, and filtering before both rankings. Compare ablations with `python -m eval.run --split dev`.
- Reranking: `python -m pytest tests/test_reranking.py tests/test_calibration.py`; inspect unrelated-tail rejection. Run `python -m eval.calibrate --candidates 24` using development questions only. Any recommended threshold needs manual review; it is not a probability or automatic runtime change.
- Generation: `python -m pytest tests/test_generation.py`; verify that invented quotations and uncited source claims are rejected. Matching text improves traceability but does not prove semantic entailment.

Each teammate should be ready to explain the reviewed implementation and make any further corrections. Review/attribution approval was confirmed by the project owner in this session. AI assistance remains disclosed. Non-RAG integration and documentation commits remain under adam12bT.

## Integration validation

- Full local suite: 35 tests passed.
- Two real Groq requests (CSV and pathlib) returned answers with source-matched quotations.
- Development calibration ran on 55 dev questions plus six independent dev negatives. Its recommended logit threshold was about 3.98, with a balanced weak-label proxy score of 0.752. This is not test accuracy and has not been applied automatically; see results/dev_calibration.json for the exact corpus hash, scores and caveats.
- Historical benchmark files remain labeled as older-snapshot measurements in results/README.md.
