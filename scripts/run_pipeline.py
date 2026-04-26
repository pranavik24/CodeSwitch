from __future__ import annotations

import argparse

from codeswitch_pipeline.pipeline import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate English-Spanish code-switch datasets.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["sample", "datasets", "evaluate", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to the pipeline YAML config.",
    )
    args = parser.parse_args()
    run_stage(stage=args.stage, config_path=args.config)


if __name__ == "__main__":
    main()
