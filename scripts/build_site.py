"""Inject current results into site/index.html (the <script id="results-data"> block).

    python -I scripts/build_site.py [--results results/week1.csv]
"""
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean(o):
    """JSON has no NaN; JSON.parse in the page rejects it, so write null instead."""
    if isinstance(o, float):
        return None if math.isnan(o) else round(o, 6)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


def inject(html, block_id, data):
    blob = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html, n = re.subn(rf'(<script id="{block_id}" type="application/json">)(.*?)(</script>)',
                      lambda m: m.group(1) + blob + m.group(3), html, flags=re.S)
    assert n == 1, f"{block_id} block not found"
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/week1.csv")
    res = ROOT / "results"
    df = pd.read_csv(ROOT / ap.parse_args().results)
    g = df.groupby(["method", "decoder", "split", "metric"])
    s = pd.DataFrame({"mean": g["value"].mean(), "std": g["value"].std(ddof=1), "n_seeds": g["value"].size(),
                      "n_cond": g["n_conditions"].first()}).reset_index()
    s["std"] = s["std"].fillna(0.0)
    runs = df[df.decoder != "baseline"][["method", "decoder", "seed"]].drop_duplicates()
    tuning = pd.read_csv(res / "tuning.csv").to_dict("records") if (res / "tuning.csv").exists() else []
    cov = read_json(ROOT / "embeddings" / "coverage.json")
    cov = {k: {"dim": v["dim"], "n_covered": v["n_covered"], "missing": v["missing"]} for k, v in cov.items()}
    data = {"summary": s.to_dict("records"), "tuning": tuning, "best_hparams": read_json(res / "best_hparams.json"),
            "coverage": cov, "runs_done": len(runs), "runs_planned": 24}
    p = ROOT / "site" / "index.html"
    html = inject(p.read_text(), "results-data", data)
    html = inject(html, "map-data", read_json(res / "perturbation_map.json"))
    p.write_text(html)
    print(f"injected {len(s)} summary rows ({len(runs)} decoder runs) and the perturbation map into {p}")


if __name__ == "__main__":
    main()
