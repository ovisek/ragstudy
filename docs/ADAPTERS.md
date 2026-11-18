# Official-system adapter setup

The experiment runner does not vendor GraphRAG or LightRAG. Each must be
installed in a dedicated, version-pinned environment, and the exact package
version, command templates, and generated settings must be captured by
`ragstudy init`. This avoids an untraceable experiment where one system is
silently upgraded mid-study.

## Common adapter output

For every query, the configured `query_command` must create `{output}`:

```json
{
  "answer": "Generated answer",
  "contexts": [{"id": "stable-chunk-id", "text": "retrieved text", "rank": 1}],
  "usage": {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "retries": 0},
  "metadata": {"implementation_version": "pinned version", "search_mode": "local"}
}
```

Both included bridges append one record per provider attempt to
`{workspace}/usage_events.jsonl`. Records distinguish index/query and
generator/embedding calls, retain the model ID and success/failure status, and
label token counts as `provider`, `estimated`, or `partial_estimate`. Failed
calls receive an input-token estimate and an unknown (`null`) output count.
Bridge outputs and the pilot `run_summary.json` aggregate this journal by
operation and model; the journal remains the auditable source of truth across
retries and resumed runs. A query without context trace may still be scored for faithfulness
when valid, but it must be marked unavailable for retrieval-context metrics.

The native NaiveRAG adapter writes the same journal contract. It disables
opaque SDK retries and records each explicit provider attempt, allowing the
pilot collector to compare index/query and generator/embedding usage across all
three systems without reconstructing token splits from aggregate traces.

## GraphRAG

Pin the official Microsoft GraphRAG release in its own environment. The adapter
receives `{input}` as canonical JSONL and should materialise one deterministic
UTF-8 text file per document under the GraphRAG input directory, retaining the
stable document ID in the filename and body. Use the GraphRAG **standard**
indexer, not FastGraphRAG, because the thesis compares knowledge-graph RAG
architectures. Save GraphRAG's generated configuration and Parquet outputs in
`{workspace}`.

The index command must invoke `graphrag init` only when creating a new
workspace, set the frozen provider/model/chunk settings, then run the standard
index pipeline. The query command maps requested modes as follows:

| Harness mode | GraphRAG mode |
|---|---|
| `local` | `local` |
| `global` | `global` |
| `hybrid` | `local` plus the method recorded as a multi-hop proxy |

The wrapper must parse GraphRAG output and, where available, export source text
units/reports as trace contexts. In the update experiment the harness calls the
GraphRAG index command on the complete 100% corpus a second time; label that
report `full_reindex`.

## LightRAG

Pin the official HKUDS LightRAG release in a separate environment. The adapter
receives document JSONL with stable IDs. Initialise storage exactly once per
workspace, use the frozen provider/model and embedding settings, and insert
documents using those stable IDs/file paths. The update run is an explicit
second insertion of the 10% added document set into the existing workspace;
it must not delete the original index.

Map harness modes as follows:

| Harness mode | LightRAG mode |
|---|---|
| `local` | `local` |
| `global` | `global` |
| `hybrid` | `hybrid` |

Pass LightRAG's token tracker into each LLM role, including the embedding role,
and export its token tracker result into `usage`. Disable opaque SDK retries so
the bridge can journal and rate-limit each attempt explicitly. Export retrieved
chunk IDs/text where the version permits; otherwise write `metadata` explaining
the unavailable trace.

## Included command-template pattern

The project includes the two version-controlled bridge scripts. The active
configuration already has these command entries:

```toml
[systems.graphrag]
kind = "command"
index_checkpoint = "corpus"
index_command = ".venv/bin/python scripts/graphrag_bridge.py index --config config/experiment.toml --workspace {workspace} --input {input} --output {output}"
query_command = ".venv/bin/python scripts/graphrag_bridge.py query --config config/experiment.toml --workspace {workspace} --question {question} --method {method} --output {output}"
```

LightRAG uses `index_checkpoint = "document"` and accepts
`{incremental_flag}` in its index command. Every checkpoint attempt is journaled
before and after execution. GraphRAG remains corpus-granular because its graph,
community, and report workflows operate across the full corpus; its own cache
provides safe reuse when that corpus unit is retried. Store the bridge scripts
and their lock files in the study archive; the complete command template is
also embedded in the experiment manifest.

Set `systems.lightrag.chunk_llm_concurrency` to a positive integer to control
LightRAG's `llm_model_max_async` limit during chunk extraction. The default is
`1` when omitted, which is serial. This setting does not change embedding or
document concurrency; the active example sets it explicitly to `4`.
