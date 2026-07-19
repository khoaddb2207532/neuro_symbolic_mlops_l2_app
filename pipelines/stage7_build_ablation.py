"""Stage 7: materialize ablation tables from existing search logs only."""
import argparse
import json
import os

import pandas as pd

from src.utils.config import load_params


def main(params_path: str) -> None:
    params = load_params(params_path)
    output_dir = os.path.join("reports", "ablation")
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    if os.path.isdir(params["logs_dir"]):
        for name in sorted(os.listdir(params["logs_dir"])):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(params["logs_dir"], name)
            rows = []
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        rows.append(json.loads(line))
            target = os.path.join(output_dir, name.replace(".jsonl", ".csv"))
            pd.DataFrame(rows).to_csv(target, index=False)
            manifest.append({"source": path, "table": target, "trials": len(rows)})
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)
