
# Experiment 1 : Run experiments on a single dataset end to end for all systems

## Naive RAG Pilot on `ultradomain_agriculture` data set

`config/experiment.toml` has only the following data set enabled:

```toml
...
[datasets.ultradomain_agriculture]
family = "ultradomain"
domain = "agriculture"
sample_size = 300
pilot_size = 20
source_url = "https://huggingface.co/datasets/TommyChien/UltraDomain"
licence = "Apache-2.0"
...
```

### Run experiment for the naive system on the enabled data set in pilot phase:
```sh
❯ ragstudy --config config/experiment.toml run --phase pilot --systems naive
{
  "indexed": 1,
  "queries_ok": 20,
  "queries_failed": 0,
  "queries_skipped": 0
}
```

### Generated Output Files:

```sh
❯ ls -alh results/runs/20260723T034904Z-57783f43/ultradomain_agriculture/naive                  
total 210568
drwxr-xr-x@ 4 mabhisek  staff   128B Jul 22 22:50 .
drwxr-xr-x@ 3 mabhisek  staff    96B Jul 22 22:49 ..
-rw-r--r--@ 1 mabhisek  staff   545B Jul 22 22:50 index_run.json
-rw-r--r--@ 1 mabhisek  staff    96M Jul 22 22:50 naive_index.jsonl
```

**Content of `index_run.json`**

```json
{
  "dataset": "ultradomain_agriculture",
  "elapsed_seconds": 106.00492954201763,
  "error": null,
  "phase": "pilot",
  "report": {
    "chunks": 2897,
    "incremental": false,
    "index_path": "/Users/mabhisek/Workspace/play/github.com/ragstudy/results/runs/20260723T034904Z-57783f43/ultradomain_agriculture/naive/naive_index.jsonl",
    "usage": {
      "api_calls": 46,
      "input_tokens": 2314057,
      "output_tokens": 0,
      "retries": 0
    }
  },
  "run_id": "20260723T034904Z-57783f43",
  "status": "ok",
  "system": "naive"
}
```

### Export evaluation inputs from the naive traces

Command:

```sh
❯ ragstudy --config config/experiment.toml score --input results/query_traces.jsonl
{
  "ragas_rows": 20,
  "ragchecker_rows": 20,
  "metric_rows": 20
}
```

Creates: 
`results/evaluation/ragas_input.jsonl`, `results/evaluation/ragchecker_input.jsonl`

Also computes deterministic metrics (`evidence_recall_proxy`, `answer_exact_match_proxy`) → `results/metric_results.jsonl`

### Run RAGAS evaluation on exported inputs

Command: 

```sh
python scripts/run_ragas.py --config config/experiment.toml --input results/evaluation/ragas_input.jsonl --output experiments/results/first_pass/ragas_metrics.jsonl
```

Computes: `answer_relevancy`, `context_precision`, `faithfulness`, `context_recall`

### Import RAGAS metrics back

Command: 

```
> ragstudy --config config/experiment.toml score \
  --import-metrics experiments/results/first_pass/ragas_metrics.jsonl \
  --backend ragas
{
  "ragas_rows": 20,
  "ragchecker_rows": 20,
  "metric_rows": 20,
  "imported_metric_rows": 68
}
```

Appends RAGAS metrics to `results/metric_results.jsonl`

### Optionally analyze naive results

Command: `ragstudy --config experiment.toml analyse`

```
ragstudy --config config/experiment.toml analyse 
{
  "summary_rows": 8,
  "comparison_rows": 0
}
```

### Bootstrapped Summaries and Comparisons

| Metric | Coverage | Mean | Median | Reading |
|--------|----------|------|--------|---------|
| answer_relevancy | 20/20 | 0.925 | 0.948 | Generated answers generally address the question. |
| context_precision | 20/20 | 0.827 | 1.000 | Retrieved top-5 chunks are often relevant, but there are notable bad cases. |
| context_recall | 20/20 | 0.800 | 1.000 | Most questions retrieve the gold-supported material, but 4 questions are weak. |
| faithfulness | 8/20 | 0.912 | 0.952 | Promising where available, but insufficient coverage to report as an overall baseline result. |

<!-- Generates bootstrapped summaries and comparisons (currently all systems together)

## Phase 2: Run Graph Methods & Re-evaluate (later)
Run GraphRAG and LightRAG



Command: ragstudy --config experiment.toml run --phase pilot --systems graphrag,lightrag
Appends traces to same query_traces.jsonl
Re-run score for all systems

Same command as step 1; exports new inputs for graph method traces
Can run RAGAS separately on just the new traces, then import, or import all together -->

## Graph RAG Pilot on `ultradomain_agriculture` data set

```sh
ragstudy --config config/experiment.toml run --phase pilot --systems graphrag
```


## Light RAG Pilot on `ultradomain_agriculture` data set

```sh
ragstudy --config config/experiment.toml run --phase pilot --systems lightrag
```


## Evaluation of Graph and Light RAG Pilots


