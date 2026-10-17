#!/usr/bin/env python
"""Train the operators from the command line and save a checkpoint.

    python scripts/train.py --preset quick
    python scripts/train.py --preset notebook --out checkpoints
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat_operator.config import PRESETS, get_preset          # noqa: E402
from heat_operator.persistence import save_bundle              # noqa: E402
from heat_operator.pipeline import run_pipeline                # noqa: E402
from heat_operator.progress import Progress                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    ap.add_argument("--out", default="checkpoints", help="checkpoint root directory")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = get_preset(args.preset)
    t0 = time.time()
    bundle = run_pipeline(cfg, Progress())
    path = save_bundle(bundle, Path(args.out) / args.preset)

    m = bundle.meta
    print(f"\nDone in {time.time() - t0:.0f}s -> {path}")
    print(f"  active-learning test MAE : {m['final_test_mean_err']:.3f} degC "
          f"(max {m['final_test_max_err']:.3f})")
    print(f"  nonlinear-operator MAE   : {m['nonlinear_test_mae']:.3f} degC")
    if m["baseline_history"]:
        print(f"  random-baseline test MAE : {m['baseline_history'][-1]['err']:.3f} degC")


if __name__ == "__main__":
    main()
