# Graph RAG thesis experiment harness

This repository is the reproducible execution record for the LightRAG vs.
GraphRAG thesis experiment. It deliberately separates expensive, external
work (downloads and model calls) from preparation, logging, scoring, and
analysis. Nothing is downloaded and no API call is made unless the relevant
command is explicitly run.

## What this provides

- A frozen protocol and environment manifest with file hashes.
- A canonical JSONL contract for the four datasets, deterministic sampling,
  document de-duplication, and update splits.
- An instrumented NaiveRAG implementation using a configured OpenAI-compatible
  endpoint, plus command adapters for the official GraphRAG and LightRAG
  implementations.
- Append-only index/query traces, token/cost accounting, RAGAS/RAGChecker
  imports, paired statistical analysis, and thesis-ready CSV/Markdown outputs.

## Install in this Codex workspace

The complete experiment environment requires Python 3.13.x. Run these commands
from the workspace root:

```bash
cd /Users/ovisek/Documents/Codex/2026-07-22/bas

# Create or update .venv with the Python version in .python-version and install
# the research harness plus every optional dependency group.
uv sync --all-extras

# Confirm that the local command is available and the included tests pass.
uv run ragstudy --help
uv run python -m unittest discover -s tests -v
```

No system-wide package installation is required. If you prefer an activated
shell for the rest of the session, run:

```bash
source .venv/bin/activate
```

Otherwise, keep using `uv run ragstudy` and `uv run python` explicitly.

## Before the first paid run

1. Create your study-specific configuration without overwriting the template:

   ```bash
   cp -n config/experiment.example.toml config/experiment.toml
   ```

2. Edit `config/experiment.toml` and set the
   exact model IDs, dated prices, endpoint, and API-key environment variable.
   Do not put a secret in the TOML file.

3. Set the API key only in your current shell, then run the local command:

   ```bash
   export OPENAI_API_KEY="your-key-here"
   .venv/bin/ragstudy --config config/experiment.toml init
   ```

   Archive the generated manifest with the thesis. Download one source at a
   time with `.venv/bin/ragstudy acquire --dataset <name>
   --confirm-download`. Populate the canonical source files described below,
   then use `prepare`.
   A source-specific acquisition worksheet is in `docs/DATA_SOURCES.md`.

## Canonical input contract

For each dataset, provide two JSONL files. These files are deliberately
provider-independent, so all systems see the same corpus and questions.

`documents.jsonl`

```json
{"id":"stable-document-id","text":"source text","title":"optional title","metadata":{"source":"official URL"}}
```

`questions.jsonl`

```json
{"id":"stable-question-id","question":"...","reference_answer":"...","evidence":["supporting source text or ids"],"strata":{"question_type":"bridge"}}
```

The preparation command deduplicates documents by SHA-256 of the original
text, preserves the first stable ID, emits an immutable `evaluation_set.jsonl`,
and records the complete provenance in the data register.

### UltraDomain conversion

UltraDomain domain splits contain a question, answers, and a repeated source
context on each row. Convert a split into the two canonical inputs before
running `prepare`:

```bash
python scripts/convert_ultradomain_jsonl.py \
  --input data/raw/ultradomain_agriculture/agriculture.jsonl \
  --domain agriculture
```

This writes `agriculture_documents.jsonl` (one record per `context_id`) and
`agriculture_questions.jsonl` (one record per source row) beside the raw split.
It maps `_id` to the question ID, `input` to `question`, `answers` to
`reference_answer`, and `context_id` to both the document ID and question
evidence. It refuses to overwrite existing output; pass `--force` only when
regenerating those inputs intentionally.

To create a separate, internally consistent small subset without changing the
full inputs, use a different output prefix and the deterministic limits:

```bash
python scripts/convert_ultradomain_jsonl.py \
  --input data/raw/ultradomain_agriculture/agriculture.jsonl \
  --domain agriculture_subset \
  --max-documents 6 \
  --max-questions 3
```

This writes `agriculture_subset_documents.jsonl` and
`agriculture_subset_questions.jsonl`. Documents are selected by stable ID;
questions are selected only when their evidence document is in that subset.

### GraphRAG-Bench Medical conversion

The released Medical corpus is one large source context. The following command
creates a small, source-preserving sample: six contiguous whitespace-aligned
chunks of that corpus plus the first three published questions. It never builds
documents from question answers or evidence strings.

```bash
python scripts/convert_graphrag_bench_json.py \
  --corpus data/raw/graphrag_bench_medical/Datasets/Corpus/medical.json \
  --questions data/raw/graphrag_bench_medical/Datasets/Questions/medical_questions.json \
  --output-prefix data/raw/graphrag_bench_medical/medical_subset
```

This produces `medical_subset_documents.jsonl` and
`medical_subset_questions.jsonl`; each question retains its published ID,
answer, question type, evidence, and evidence relations.

## Command sequence

```bash
# Freeze versions, operating environment, and the protocol.
ragstudy --config config/experiment.toml init

# Convert canonical source JSONL into the fixed, audited experiment input.
ragstudy --config config/experiment.toml prepare \
  --dataset ultradomain_agriculture \
  --documents /path/to/agriculture_documents.jsonl \
  --questions /path/to/agriculture_questions.jsonl

# Repeat prepare for every configured dataset, then verify all gates.
ragstudy --config config/experiment.toml validate

# Run each pilot independently. Record the run_id printed by each command.
ragstudy --config config/experiment.toml run --phase pilot --systems naive
ragstudy --config config/experiment.toml run --phase pilot --systems graphrag
ragstudy --config config/experiment.toml run --phase pilot --systems lightrag

# Freeze the exact three runs into one auditable pilot report. The active
# medical subset has 3 prepared questions versus pilot_size=20, so accepting
# it requires an explicit, recorded exception.
ragstudy --config config/experiment.toml collect-pilot \
  --dataset graphrag_bench_medical \
  --run naive=<NAIVE_RUN_ID> \
  --run graphrag=<GRAPHRAG_RUN_ID> \
  --run lightrag=<LIGHTRAG_RUN_ID> \
  --allow-short-pilot

# Only after pilot sign-off, run the fixed evaluation and the 90/10 update run.
ragstudy --config config/experiment.toml run --phase full --systems naive,graphrag,lightrag
ragstudy --config config/experiment.toml run-update --systems naive,graphrag,lightrag

# Import evaluator scores or use the optional evaluator integrations, analyse,
# and generate thesis tables/figures.
ragstudy --config config/experiment.toml score --input results/query_traces.jsonl
# Run this separately so evaluator version and cost remain isolated from system cost.
python scripts/run_ragas.py --config config/experiment.toml \
  --input results/evaluation/ragas_input.jsonl --output results/evaluation/ragas_metrics.jsonl
ragstudy --config config/experiment.toml score --import-metrics results/evaluation/ragas_metrics.jsonl --backend ragas
# RAGChecker needs a fixed extractor/checker model choice. Export its documented
# input shape, run its pinned CLI, then import its per-query metrics/coverage.
python scripts/export_ragchecker_input.py --input results/evaluation/ragchecker_input.jsonl \
  --output results/evaluation/ragchecker_input.json
ragstudy --config config/experiment.toml analyse
ragstudy --config config/experiment.toml thesis
```

### Incremental progress and resume

Index and query progress is printed immediately. Command-adapter stdout and
stderr are also appended to per-attempt files under
`results/runs/<run_id>/<dataset>/<system>/logs/`; the configured API-key value
is redacted from both destinations. Index attempts are recorded append-only in
`index_progress.jsonl`, and `index_summary.json` is refreshed after every
checkpoint unit.

A normal `run` always starts a fresh, independent run. Resume the newest
compatible incomplete run explicitly:

```bash
.venv/bin/ragstudy --config config/experiment.toml run \
  --phase pilot --systems lightrag --resume
```

Resume refuses to combine changed configuration, prepared documents, or
questions with an earlier run. It skips successful index units and successful
queries from that run, while retrying failed, interrupted, and pending work.
Each index unit is retried twice by default; change this with
`--index-retries N`.

Checkpoint granularity is backend-safe: NaiveRAG and LightRAG checkpoint each
document, while GraphRAG checkpoints its corpus-level workflow and reuses its
native cache. `Ctrl+C` records the active attempt as interrupted and exits 130.

### Collecting independent pilot runs

`results/run_summary.json` is only a convenience pointer to the most recently
finished run. Durable run summaries live under `results/runs/<run_id>/`, and
the global query trace is append-only. Do not use either global file directly
to compare independently launched systems.

After all three single-system pilots are complete, use `collect-pilot` with
their exact run IDs. Collection blocks incomplete indexes, missing questions,
changed prepared inputs, incompatible frozen settings, and missing index/query
usage. It ignores unrelated historical runs and trace rows. The resulting
immutable directory is:

```text
results/collections/<dataset>/pilot-<selection-hash>/
```

Review `report.md` for pilot sign-off and retain `collection_manifest.json`,
`summary.json`, `summary.csv`, `traces.jsonl`, `usage_events.jsonl`, and
`quality_metrics.jsonl` as the source artifacts. Running the same collection
command again returns the existing collection without rewriting it. Pilot
quality is deterministic and descriptive; paid evaluator runs and inferential
comparisons remain separate from collection.

The provider request budget and shared worker concurrency are configured independently:

```toml
[provider]
requests_per_minute = 30
embedding_dimension = 2048

[common]
concurrent_workers = 1
```

For a 40 RPM provider limit, `30` RPM preserves 25% headroom. GraphRAG uses its
native sliding-window limiter, while LightRAG spaces completion and embedding
requests through a shared pacer. `concurrent_workers` applies to whichever
systems are selected and defaults to `1` when omitted. These operational rate
and concurrency fields may be changed before `--resume`;
they are recorded in the run manifest but do not invalidate resume compatibility.
They can also be overridden for a single launch without editing TOML:

```bash
.venv/bin/ragstudy --config config/experiment.toml run \
  --phase pilot --systems graphrag --resume \
  --concurrent-workers 8 --requests-per-minute 30
```

GraphRAG fixes its worker-pool size when the indexing process starts. To change
it during a long run, interrupt the process and resume with a new override; the
native GraphRAG cache avoids repeating completed LLM calls.

If index failures remain, queries for that index are skipped and the command
exits nonzero. For diagnostic use only, queries may be forced against the
partial index:

```bash
.venv/bin/ragstudy --config config/experiment.toml run \
  --phase pilot --systems lightrag --resume --allow-partial-index
```

Such traces include `index_complete: false` and the failed unit IDs. They are
rerun after the index is completed and are never treated as complete traces by
resume.

## Adapter contract

NaiveRAG runs directly when `openai` is installed and the configured API key is
available. GraphRAG and LightRAG run through the included, version-pinned
bridges: `scripts/graphrag_bridge.py` (GraphRAG 3.1.0) and
`scripts/lightrag_bridge.py` (LightRAG 1.4.14). Their commands are already set
in `config/experiment.toml`; install the `systems` dependency extra before
running them.
Templates receive `{workspace}`, `{input}`, `{question}`, `{method}`,
`{output}`, and `{incremental_flag}`. The query command must write one JSON object to `{output}` with
`answer`, optional `contexts`, optional `usage`, and optional `metadata`.

This contract makes every external system produce the same query trace without
pretending that GraphRAG's or LightRAG's internal token accounting is identical.

RAGAS is invoked through `scripts/run_ragas.py` with the frozen judge and
embedding model fields. It only scores rows that have both retrieved contexts
and a reference answer; the remaining metric coverage is explicitly unavailable
rather than represented as a zero.
