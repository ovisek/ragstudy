from __future__ import annotations

import unittest
from pathlib import Path

from ragstudy.cli import apply_run_overrides, parser
from ragstudy.config import Config


class RunOverrideTest(unittest.TestCase):
    def test_runtime_overrides_mutate_only_operational_config(self) -> None:
        config = Config(
            path=Path("config/experiment.toml"),
            root=Path("."),
            data={
                "common": {"concurrent_workers": 1},
                "provider": {"requests_per_minute": 30},
                "systems": {"graphrag": {}, "lightrag": {}},
            },
        )
        args = parser().parse_args([
            "run", "--phase", "pilot", "--systems", "graphrag", "--resume",
            "--concurrent-workers", "8", "--requests-per-minute", "25",
        ])

        apply_run_overrides(config, args)

        self.assertEqual(config.data["common"]["concurrent_workers"], 8)
        self.assertEqual(config.data["provider"]["requests_per_minute"], 25)
        self.assertEqual(config.data["systems"], {"graphrag": {}, "lightrag": {}})

    def test_collect_pilot_parser_accepts_exact_runs_and_short_override(self) -> None:
        args = parser().parse_args([
            "collect-pilot", "--dataset", "medical",
            "--run", "naive=r1", "--run", "graphrag=r2", "--run", "lightrag=r3",
            "--allow-short-pilot",
        ])
        self.assertEqual(args.command, "collect-pilot")
        self.assertEqual(args.run, ["naive=r1", "graphrag=r2", "lightrag=r3"])
        self.assertTrue(args.allow_short_pilot)


if __name__ == "__main__":
    unittest.main()
