from __future__ import annotations

import os
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
            from huggingface_hub import configure_http_backend, snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install data dependencies: pip install -e '.[data]'") from exc
        insecure_used = False
        ssl_recovery = None
        try:
            revision = snapshot_download(repo_id=HF_REPOS[dataset], repo_type="dataset", local_dir=str(destination))
        except Exception as exc:
            message = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in message or "SSLCertVerificationError" in message:
                # First recovery: clear potentially broken certificate bundle overrides.
                had_override = any(os.environ.get(name) for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"))
                if had_override:
                    saved = {name: os.environ.get(name) for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")}
                    for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
                        os.environ.pop(name, None)
                    try:
                        revision = snapshot_download(repo_id=HF_REPOS[dataset], repo_type="dataset", local_dir=str(destination))
                    except Exception:
                        for name, value in saved.items():
                            if value:
                                os.environ[name] = value
                    else:
                        ssl_recovery = "cleared_cert_env_overrides"

                # Last-resort recovery: user explicitly opts into insecure SSL bypass.
                if ssl_recovery is None and os.getenv("RAGSTUDY_ALLOW_INSECURE_SSL", "").strip().lower() in {"1", "true", "yes"}:
                    def _insecure_backend():
                        import requests

                        session = requests.Session()
                        session.verify = False
                        return session

                    configure_http_backend(_insecure_backend)
                    revision = snapshot_download(repo_id=HF_REPOS[dataset], repo_type="dataset", local_dir=str(destination))
                    insecure_used = True
                if ssl_recovery is None and not insecure_used:
                    raise RuntimeError(
                        "TLS certificate verification failed while downloading from Hugging Face. "
                        "Try unsetting REQUESTS_CA_BUNDLE and SSL_CERT_FILE, or set a valid CA bundle. "
                        "If you must bypass verification temporarily, set RAGSTUDY_ALLOW_INSECURE_SSL=1."
                    ) from exc
            else:
                raise
        source = {"kind": "huggingface_snapshot", "repo": HF_REPOS[dataset], "revision": revision}
        if ssl_recovery:
            source["ssl_recovery"] = ssl_recovery
        if insecure_used:
            source["ssl_recovery"] = "insecure_verify_false"
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
