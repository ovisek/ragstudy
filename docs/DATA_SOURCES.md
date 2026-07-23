# Dataset acquisition and normalisation worksheet

Fill in this worksheet as each source is acquired. Do not alter a dataset after
the hash has been entered in `results/data_register.json`.

| Dataset | Official source | Corpus construction | Questions | Required evidence |
|---|---|---|---|---|
| UltraDomain Agriculture / Legal | Hugging Face `TommyChien/UltraDomain` | De-duplicate `context` by its supplied context ID or text hash; retain document title/authors in metadata. | `input`, `answers`, and label from the selected domain. | Original context ID and answer list. |
| GraphRAG-Bench Medical / Novel | Hugging Face `GraphRAG-Bench/GraphRAG-Bench` plus its official corpus files | Use the released source documents, not answer/evidence strings reconstructed as documents. | Published `id`, `question`, `answer`, `question_type`, and `evidence`. | Released evidence/evidence relations. |
| HotpotQA fullwiki validation | Hugging Face `hotpotqa/hotpot_qa`, `fullwiki` validation | Build a de-duplicated document corpus from title + sentence arrays; do not include question or reference-answer fields. | All selected validation IDs, stratified by `type` and `level`. | `supporting_facts` title/sentence positions. |
| Kevin Scott Podcasts | Microsoft GraphRAG benchmarking datasets | Unzip supplied transcript input files without edits. | All 125 published thematic queries. | Query CSV ID and query text; no fabricated fact-answer gold labels. |

Before `prepare`, save a one-line JSONL document/question record in the
canonical format described in the README. Save raw downloads outside version
control under `data/raw/<dataset>/`; canonical inputs are copied to
`data/prepared/<dataset>/` by the tool.

GraphRAG-Bench's current published dataset exposes Medical and Novel subsets.
Report that fact in the methodology rather than describing it as a CS-textbook
dataset. Kevin Scott supports global-sensemaking evaluation and should not be
forced into fact-retrieval scoring.

