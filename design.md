# Design

## Corpus and users

Python 3.12 official documentation: tutorial plus 62 selected library pages, 73 source pages total. Intended users are students learning Python and classmates implementing coursework. These users have not yet been recruited. Public documentation contains no student records. URLs are version-pinned, but upstream patch documentation can change: raw HTML is cached and the normalized corpus SHA-256 is recorded in `data/corpus_manifest.json`. Archived chunks preserve what was actually indexed. Source dates/authors are not fabricated; upstream URLs, section titles, documentation version and content hashes are retained instead.

## Chunking

BeautifulSoup extracts content from the documentation body, removes navigation markers, groups by the nearest section, and preserves code-block newlines and indentation. Whole lines are packed toward 220 whitespace-delimited words with up to 40 words of whole-line overlap. Long individual prose lines are split; long code lines remain a known limitation. Word counts are not tokenizer counts; embedding inputs can still be truncated. Tables are incompletely represented because extraction targets paragraphs, definition terms, lists and preformatted blocks. Sections and URLs are always retained; content-based IDs prevent silently reusing an index after a corpus edit.

The development sensitivity experiment tested 140/20, 220/40 and 320/60 in isolated data directories, using the same questions and shared model/raw-document caches. Reranked recall@5 was 0.855, 0.891 and 0.891 respectively. The 220-word default is retained as the smaller configuration matching the larger one's recall. Weak source labels are remapped for each index, so this is a sensitivity estimate rather than independent gold relevance. The main corpus remains unchanged during these experiments; results are saved in results/chunks_*.json.

## Index and retrieval

`all-MiniLM-L6-v2` encodes title, section and content into normalized vectors. A NumPy matrix allows exact cosine search at this scale without operating a separate vector service. BM25 is built in memory over identifiers and natural-language words. Startup checks the index against corpus and model identifiers. Matrix search is O(n); migrate to an approximate index only if corpus growth justifies it.

Keyword and dense candidate rankings are merged by reciprocal rank fusion: sum of 1/(60 + rank). RRF scores are rank aggregates, never answer confidence. The reranker is `ms-marco-MiniLM-L-6-v2`; it scores query/passage pairs. Baseline reranks 24 candidates; optimized reranks 8. Both return three passages. The ablation includes keyword only, vector only, hybrid, and hybrid + reranker with the same candidate limit. Filtering uses an exact allowlisted corpus page match; a missing page yields no answer. Queries preserve identifiers rather than using a rewrite model.

## Answers, citations, abstention

With a configured key, Groq `openai/gpt-oss-20b` generates structured claims from only the question and retrieved passages. Each claim must reference valid evidence IDs; the application attaches numbered citations and rejects malformed output. Syntax validation does not prove semantic entailment. Without a key, `auto` explicitly selects extractive mode; `GENERATOR=groq` fails clearly if no key exists. Provider errors are surfaced. The initial Llama model returned 404 for this account, so the default was chosen from the live model list. Token usage and configured free-tier assumptions are recorded. The default cross-encoder logit threshold is 0, configurable with `RERANK_THRESHOLD`; this is a heuristic, not calibrated probability. Add independent dev negatives before tuning; do not tune against the test negatives. Question and document content are treated as data, and no retrieved commands are executed.

## Operations

FastAPI serves static assets and JSON endpoints. A bounded 128-entry LRU answer cache keys exact query, profile and page; it is per-process and does not cache across index reloads. API inference is serialized to avoid device contention and cache races, limiting concurrent throughput. Feedback stores explicit votes and questions locally without IP addresses. Browser rendering uses `textContent`, avoiding HTML interpretation of user queries and passages. The core UI works with system fonts if external fonts are unavailable.

## Failure cases

- Dense retrieval confuses neighboring APIs; lexical matching misses paraphrases.
- Whole-section anchors can be less precise than function anchors.
- Tables, nested definition structures and long code samples can lose context or be truncated.
- Symbol-based labels over-credit incidental mentions; source passage review is required.
- High relevance does not establish answerability. The rejection threshold can abstain incorrectly.
- Extractive support is not semantic correctness or completeness.
- Async Python documentation is only partially covered; third-party libraries are out of scope.
- Cached performance applies to repeated exact queries, not unseen questions.
- Local p95 timing does not establish behavior under a production workload.

## References

- https://docs.python.org/3.12/
- https://docs.python.org/3.12/license.html
- https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- https://fastapi.tiangolo.com/deployment/docker/
