import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodrive_vla.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an autonomous-driving VLA policy baseline.")
    parser.add_argument("--config", default="configs/example.yaml", help="Path to experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(
        "evaluation placeholder: "
        f"dataset={config.data.dataset_name}, "
        f"episodes={config.eval.episodes}"
    )


if __name__ == "__main__":
    main()
