# Postmortem

## Problems encountered

1. The execution sandbox blocked outbound network access. Documentation and dependency downloads required explicit tool escalation. Downloads are reproducible through setup and cached afterward.
2. The task requires a genuine, source-grounded evaluation rather than a tutorial dataset. Curated symbol questions allow repeatable initial measurements, but automatic passage selection is weak supervision. The labels explicitly require human review, and dev/test share topics.
3. The initial no-key service used extraction. The user selected Groq for LLM generation, so the service now generates structured claims with source IDs. The initial Llama model returned 404 for the user's account; a live model listing identified openai/gpt-oss-20b. JSON object mode produced missing fields, so strict structured output was added. The account's 8,000-token/minute limit required pacing, bounded backoff and progress files. Real API calls, rather than mocks alone, exposed these issues.
4. The speed/accuracy target cannot be assumed. Reducing reranker candidates is a measured experiment and may fail to cut p95 in half or may miss correct passages. The reports consume actual benchmark output.
5. Code blocks and definition lists complicate chunking. Indentation is retained, but some table structure and context can be lost, and neural model token limits differ from whitespace word limits.
6. Drive C: ran out of space during the first model setup. After space became available, indexing completed. A Windows Arrow/torch DLL import conflict crashed standalone evaluation; loading optional Arrow before torch fixed the conflict in this shared Python environment. An isolated environment is preferable for a clean clone.
7. The first offline benchmark achieved roughly 46% lower p95, with no correctness-proxy loss. That falls short of the 50% target; generation/network delays can further reduce the end-to-end gain. These results are not represented as a successful optimization target.

## Lessons

Separate citation support, relevance and correctness. A copied passage is supported by construction yet can still be unhelpful. Keep corpus hashes so stale embeddings fail loudly. Report startup, warm inference and cache hits separately. Never substitute mock feedback for user testing or a Dockerfile for a deployed URL.

## Remaining external work

Instructor approval; independent review of gold labels and answers; real classmate feedback; a deployment account and public URL; actual team contributions; P1 harness integration if the class requires it. The implementation and reports do not claim those steps happened.
