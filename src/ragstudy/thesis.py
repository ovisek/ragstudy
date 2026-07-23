from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import Config
from .io import read_jsonl


def render_thesis_outputs(config: Config) -> Path:
    analysis = config.output_root / "analysis" / "summary.csv"
    if not analysis.exists():
        raise FileNotFoundError("Run `ragstudy analyse` before generating thesis output")
    lines = ["# Thesis Results Draft", "", "## Reproducibility and Experimental Controls", "", "The experiment used the frozen configuration and environment manifest stored with this output. All methods received the same prepared corpus, fixed evaluation set, common chunk size/overlap, answer prompt, model identifiers, and temperature. Raw traces include failures, timeouts, API usage, and cost estimates.", "", "## Results", "", "Use `analysis/summary.csv` as the source of truth for the values below. Do not manually transcribe figures or tables.", "", "| Dataset | System | Measure | Mean | Median | 95% CI |", "|---|---|---:|---:|---:|---:|"]
    import csv
    with analysis.open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            lines.append(f"| {row['dataset']} | {row['system']} | {row['measure']} | {float(row['mean']):.4f} | {float(row['median']):.4f} | [{row['ci95_low']}, {row['ci95_high']}] |")
    lines += ["", "## Interpretation template", "", "Report paired comparisons only when the corresponding `p_value_bh` in `analysis/paired_comparisons.csv` is at or below 0.05. Describe practical magnitude together with statistical significance. Keep evaluator cost separate from retrieval/generation cost, and state any metrics marked unavailable rather than treating them as zero.", "", "## Limitations", "", "The reported evidence applies to the frozen software versions, dataset snapshots, models, price date, and sampled questions in the manifest. Architecture-specific internal settings are disclosed in the same manifest and adapter command reports."]
    output = config.output_root / "thesis" / "results_draft.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output

