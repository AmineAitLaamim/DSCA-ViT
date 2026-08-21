from __future__ import annotations

import argparse

from src.config import load_config
from src.inference import load_inference_model, predict_path
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/inference.json")
    args = parser.parse_args()

    config = load_config(args.config)
    model, device = load_inference_model(config, args.checkpoint)
    results = predict_path(model, args.input, config, device)
    write_json(results, args.output)
    print(results[:5])


if __name__ == "__main__":
    main()

