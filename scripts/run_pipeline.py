from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from codeswitch_pipeline.pipeline import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate English-Spanish code-switch datasets.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["sample", "finetune", "datasets", "evaluate", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to the pipeline YAML config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override the number of sampled prompts and final generated examples per dataset.",
    )
    args = parser.parse_args()
    run_stage(
        stage=args.stage,
        config_path=args.config,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
