from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .io import sha256_file, write_json


HF_REPOS = {
    "ultradomain_agriculture": "TommyChien/UltraDomain",
    "ultradomain_legal": "TommyChien/UltraDomain",
    "graphrag_bench_medical": "GraphRAG-Bench/GraphRAG-Bench",
    "graphrag_bench_novel": "GraphRAG-Bench/GraphRAG-Bench",
    "hotpotqa_fullwiki": "hotpotqa/hotpot_qa",
}


def acquire(config: Config, dataset: str) -> dict[str, object]:
    """Download an immutable official source snapshot only when explicitly invoked."""
    destination = config.data_root / "raw" / dataset
    destination.mkdir(parents=True, exist_ok=True)
    if dataset in HF_REPOS:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install data dependencies: pip install -e '.[data]'") from exc
        revision = snapshot_download(repo_id=HF_REPOS[dataset], repo_type="dataset", local_dir=str(destination))
        source = {"kind": "huggingface_snapshot", "repo": HF_REPOS[dataset], "revision": revision}
    elif dataset == "kevin_scott_podcasts":
        checkout = destination / "repository"
        if checkout.exists():
            raise RuntimeError(f"Refusing to overwrite existing checkout: {checkout}")
        result = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/microsoft/graphrag-benchmarking-datasets.git", str(checkout)], capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        revision = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        source = {"kind": "git", "repo": "https://github.com/microsoft/graphrag-benchmarking-datasets.git", "revision": revision}
    else:
        raise ValueError(f"No official acquisition route registered for {dataset}")
    files = []
    for item in sorted(path for path in destination.rglob("*") if path.is_file() and ".git" not in path.parts):
        files.append({"path": str(item.relative_to(destination)), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    report = {"dataset": dataset, "acquired_at": datetime.now(timezone.utc).isoformat(), "source": source, "files": files}
    write_json(destination / "acquisition_manifest.json", report)
    return report
