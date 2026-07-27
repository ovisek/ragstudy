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

Run these commands from the workspace root:

```bash
cd /Users/ovisek/Documents/Codex/2026-07-22/bas

# Create once; it is safe to reuse on later sessions.
python3 -m venv .venv

# Install the research harness plus analysis, OpenAI-provider, dataset, and
# evaluation dependencies into this workspace only.
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[analysis,providers,data,evaluation,systems]'

# Confirm that the local command is available and the included tests pass.
.venv/bin/ragstudy --help
.venv/bin/python -m unittest discover -s tests -v
```

No system-wide installation is required. If you prefer an activated shell for
the rest of the session, run:

```bash
source .venv/bin/activate
```

Otherwise, keep using `.venv/bin/ragstudy` and `.venv/bin/python` explicitly.

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

### Local Ollama models

The native adapter, GraphRAG bridge, LightRAG bridge, and standalone RAGAS
script also support downloaded Ollama models. Replace the `[provider]` values
with an explicit local profile; use the exact tags reported by `ollama list`:

```toml
[provider]
type = "ollama"
base_url = "http://localhost:11434/v1"
generator_model = "llama3.2"
embedding_model = "nomic-embed-text"
judge_model = "llama3.2"
embedding_dimensions = 768
generator_input_per_million = 0.0
generator_output_per_million = 0.0
embedding_input_per_million = 0.0
judge_input_per_million = 0.0
judge_output_per_million = 0.0
```

Download the configured tags before running an experiment. The local profile
does not use `api_key_env` or any API-key environment variable:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama list
```

Set `embedding_dimensions` to the dimension emitted by the chosen embedding
model; LightRAG needs this value when it creates vector storage. Run the same
experiment commands after configuring Ollama, including all three systems and
the optional evaluator:

```bash
ragstudy --config config/experiment.toml run --phase pilot --systems naive,graphrag,lightrag
python scripts/run_ragas.py --config config/experiment.toml \
  --input results/evaluation/ragas_input.jsonl --output results/evaluation/ragas_metrics.jsonl
```

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

# First use a 20-question pilot. The command never changes the frozen sample.
ragstudy --config config/experiment.toml run --phase pilot --systems naive,graphrag,lightrag

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

`run` is resumable: an existing `(dataset, system, question_id, repetition)`
trace is never overwritten. Delete neither failures nor timeouts; correct the
configuration and start a new run identifier instead.

### Resume interrupted GraphRAG indexing in the same workspace

If GraphRAG indexing is interrupted (for example `Ctrl+C`), rerunning
`ragstudy ... run` creates a new `run_id` and a new workspace. To continue the
same GraphRAG workspace (and reuse its cache), rerun GraphRAG directly against
that existing `--root` directory:

```bash
# Use the same shell environment as the original run.
export OPENAI_API_KEY="your-key-here"

# Optional if your proxy uses a custom CA.
export SSL_CERT_FILE="/absolute/path/to/company-root-ca.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

.venv/bin/graphrag index \
  --root results/runs/<run_id>/<dataset>/graphrag \
  --method standard
```

Progress is written to:

```bash
tail -f results/runs/<run_id>/<dataset>/graphrag/logs/indexing-engine.log
```

Notes:

- `index_run.json` and `index_command_report.json` are written by the harness
  after `adapter.index(...)` returns. If the process is interrupted before
  completion, those final files may not exist yet for that run.
- To finish the full harness flow after manual GraphRAG completion, start a new
  `ragstudy run` for normal trace/manifests or add a resume mode in the harness.

## Adapter contract

NaiveRAG runs directly when `openai` is installed and the configured API key is
available. GraphRAG and LightRAG run through the included, version-pinned
bridges: `scripts/graphrag_bridge.py` (GraphRAG 3.1.0) and
`scripts/lightrag_bridge.py` (LightRAG 1.4.14). Their commands are already set
in `config/experiment.toml`; install the `systems` dependency extra before
running them.
Templates receive `{workspace}`, `{input}`, `{question}`, `{method}`, and
`{output}`. The query command must write one JSON object to `{output}` with
`answer`, optional `contexts`, optional `usage`, and optional `metadata`.

This contract makes every external system produce the same query trace without
pretending that GraphRAG's or LightRAG's internal token accounting is identical.

RAGAS is invoked through `scripts/run_ragas.py` with the frozen judge and
embedding model fields. It only scores rows that have both retrieved contexts
and a reference answer; the remaining metric coverage is explicitly unavailable
rather than represented as a zero.
