# Ask Python Docs — The Librarian

Generation update: the service uses Groq when a key is configured; extractive mode is the offline baseline. Benchmark generator: groq. Groq end-to-end results require a configured key. Citation syntax validation does not prove semantic faithfulness.

## 1. Corpus and users

Official Python 3.12 tutorial and selected library documentation: 73 source HTML pages, 784 sections and 2632 chunks. These HTML pages contain variable amounts of content; no equivalent printed-page count is claimed. Intended users are classmates learning Python. Real-user recruitment, feedback and instructor corpus approval remain pending. Corpus hash: `696a87e561405c090c012875231d68142291cba6908159cc1a6b86af8246e1ea`. License and source: https://docs.python.org/3.12/license.html and https://docs.python.org/3.12/.

## 2. Pipeline design

The loader groups paragraphs, definition terms, lists and code by section, retaining version, title and URL. The default chunk size is 220 words with up to 40 words of whole-line overlap. MiniLM produces normalized embeddings stored in a NumPy matrix. BM25 and dense search are combined using reciprocal rank fusion and a MiniLM cross-encoder reranker. Groq generates structured claims from three passages; the app validates source numbers and attaches citations. The model is openai/gpt-oss-20b, selected from the account's live model list. Extractive mode is a separate offline baseline. Retrieval abstains below a logit threshold of zero; this is a heuristic, not calibrated confidence. Generation can also abstain if evidence is insufficient. No local LLM weights are required for Groq.

## 3. Retrieval evaluation

The 120-case dataset has 110 answerable questions and 10 out-of-scope questions. Half of answerable questions are development cases; test includes 55 answerable paraphrases plus 10 negatives. Relevant passage IDs are automatically resolved from symbol mentions and require human review. This creates weak relevance labels; incidental mentions can over-credit retrieval. Dev/test share topics and cannot establish unseen-topic generalization.

| Method | Recall@5 | MRR@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| keyword | 0.945 | 0.852 | 6.1 | 9.0 |
| vector | 0.909 | 0.772 | 16.0 | 19.8 |
| hybrid | 0.964 | 0.842 | 17.4 | 25.9 |
| hybrid_rerank | 0.945 | 0.874 | 92.8 | 113.8 |


Development-only chunk sensitivity (weak source labels are remapped for each index):

| Chunk configuration | Dev recall@5 (reranked) |
|---|---:|
| 140_20 words / overlap | 0.855 |
| 220_40 words / overlap | 0.891 |
| 320_60 words / overlap | 0.891 |


The 220-word option is retained because it matches the 320-word recall with shorter contexts.

Recall is the fraction of answerable questions with at least one labeled relevant passage among five hits, rather than document-level recall over all possible relevant sources. MRR uses the first such passage. All ablations use the same candidate budget. Latency excludes model startup and downloads and reflects serial local inference.

## 4. Answer evaluation

The correctness proxy checks for a relevant cited passage plus the expected symbol, or abstention for a negative question. This is not human semantic correctness. Exact-copy support only applies to extractive mode; it is null for Groq rather than presented as LLM faithfulness. Answers and citations are saved separately in results/groq and results/extractive. A separate semantic judge checks correctness, faithfulness and citation accuracy; same-model judging is biased and must be supplemented by human review. Human judgments remain pending. Abstention is measured separately on out-of-scope cases.

Same-model LLM judge for answerable cases; deterministic expected-abstention check for negatives. Not human evaluation. Evenly spaced sample of at most 6 cases per profile; not a full-test score.

| Profile | n | Correct | Faithful | Citations |
|---|---:|---:|---:|---:|
| optimized | 6 | 1.000 | 1.000 | 0.833 |
| baseline | 6 | 0.833 | 0.833 | 0.833 |


## 5. Optimization

Baseline reranks 24 candidates, optimized reranks 8. Models, corpus, output size and questions stay fixed. Profiles are interleaved per query after a warm-up, with answer caching disabled. Normal operation also has a bounded exact-query cache; its cache hits are not counted as the primary optimization evidence.

| Profile | Correctness proxy | p50 ms | p95 ms |
|---|---:|---:|---:|
| optimized | 0.877 | 1411.2 | 5451.8 |
| baseline | 0.846 | 1626.6 | 6930.0 |

Measured p95 reduction: 21.3%. Accuracy proxy loss: -3.08 points. Numeric target met on proxy: False. Human correctness validation remains pending.


With GROQ_FREE_TIER=true, API cost is recorded as zero under the configured account allowance; quotas still apply. Local baseline inference has no API bill. Actual token usage is recorded, with paid rates configurable rather than hard-coded. The reduced candidate set can lose passages. Chunk-size sensitivity has a reproducible script; results are only claimed if corresponding output files exist.

## 6. Failure cases, costs and delivery

Tables, long code and nested definitions can lose structure. Embedding token limits can truncate chunks. Relevance scores do not guarantee answerability; a copied passage can be irrelevant. Third-party APIs are out of scope. Public deployment and real-user feedback have not been established by a local implementation.

Local model inference incurs no API bill but consumes hardware and electricity. The measured machine has CUDA available; model device selection is automatic, so Docker/CPU timings can differ. For monthly host cost H and N requests, fixed allocation is H/N; extrapolating to 100N requests is valid only while capacity holds. The serialized service needs load testing before estimating production p95. Keep corpus and hardware details with measurements. The repository includes setup, API/UI, Dockerfile, eval data, ablations, optimization and postmortem. Team names and real contributions must be added before submission. No API keys or personal records are included.
