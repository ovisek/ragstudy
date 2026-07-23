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

Usage fields are zero only when the upstream package cannot expose them. State
that limitation in the methodology and retain the package logs in the adapter
workspace. A query without context trace may still be scored for faithfulness
when valid, but it must be marked unavailable for retrieval-context metrics.

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

Pass LightRAG's token tracker into each LLM role where the pinned version
supports it, and export its token tracker result into `usage`. Export retrieved
chunk IDs/text where the version permits; otherwise write `metadata` explaining
the unavailable trace.

## Recommended command-template pattern

Create two version-controlled wrapper scripts in each pinned environment and
set TOML entries along these lines:

```toml
[systems.graphrag]
kind = "command"
index_command = "/absolute/pinned-env/bin/python adapters/graphrag_bridge.py index --workspace {workspace} --input {input} --output {output}"
query_command = "/absolute/pinned-env/bin/python adapters/graphrag_bridge.py query --workspace {workspace} --question {question} --method {method} --output {output}"
```

The same shape applies to LightRAG. Store the bridge scripts and their lock
files in the study archive; the complete command template is also embedded in
the experiment manifest.

