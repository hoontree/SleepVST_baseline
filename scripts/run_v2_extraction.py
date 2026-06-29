"""v2 respiratory proxy extraction — standalone runner (no Hydra required).

Usage:
    # all videos, 60 workers
    python scripts/run_v2_extraction.py

    # custom worker count
    python scripts/run_v2_extraction.py --workers 30

    # specific records only
    python scripts/run_v2_extraction.py --records A2019-EM-01-0022,A2021-EM-01-0020

    # re-process already-done epochs
    python scripts/run_v2_extraction.py --no-skip

    # custom output dir (default: data/resp_proxy_video_epochs_v2)
    python scripts/run_v2_extraction.py --output-dir /path/to/output
"""

import argparse
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from src.data.preprocess.respiratory_pipeline_mp import process_all_videos
from src.utils.logger import setup_logging


def build_cfg(args):
    base = OmegaConf.load(ROOT / "config/preprocess/respiratory_v2.yaml")

    overrides = {}
    if args.workers is not None:
        overrides["multiprocessing"] = {"enabled": True, "num_workers": args.workers}
    if args.output_dir is not None:
        overrides["output"] = dict(base.output)
        overrides["output"]["dir"] = args.output_dir
    if args.no_skip:
        overrides["skip_existing"] = False
    if args.records:
        ids = [r.strip() for r in args.records.split(",") if r.strip()]
        overrides["filter"] = {
            "enabled": True,
            "record_ids": ids,
            "epoch_ranges": {},
        }

    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))

    return base


def main():
    ap = argparse.ArgumentParser(description="Run v2 respiratory proxy extraction")
    ap.add_argument("--workers", type=int, default=None,
                    help="Multiprocessing workers (default: from config, 60)")
    ap.add_argument("--records", type=str, default=None,
                    help="Comma-separated record IDs to process (default: all)")
    ap.add_argument("--no-skip", action="store_true",
                    help="Re-process epochs that already have output files")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory")
    ap.add_argument("--config", type=str,
                    default=str(ROOT / "config/preprocess/respiratory_v2.yaml"),
                    help="Path to YAML config (default: config/preprocess/respiratory_v2.yaml)")
    args = ap.parse_args()

    cfg = build_cfg(args)

    setup_logging(cfg.log.dir, cfg.log.name)
    log = logging.getLogger(__name__)

    log.info("=" * 56)
    log.info("  v2 Respiratory Extraction")
    log.info(f"  output    : {cfg.output.dir}")
    log.info(f"  workers   : {cfg.multiprocessing.num_workers}")
    log.info(f"  skip_done : {cfg.skip_existing}")
    if cfg.filter.enabled:
        log.info(f"  records   : {list(cfg.filter.record_ids)}")
    log.info("=" * 56)

    results = process_all_videos(cfg)

    ok  = sum(1 for r in results if r.startswith("Successfully"))
    skp = sum(1 for r in results if r.startswith("Skipped"))
    err = len(results) - ok - skp

    log.info("=" * 56)
    log.info(f"  Done  : {ok}")
    log.info(f"  Skip  : {skp}")
    log.info(f"  Error : {err}")
    log.info("=" * 56)

    if err:
        for r in results:
            if r.startswith("Error") or r.startswith("Failed"):
                log.error(r)
        sys.exit(1)


if __name__ == "__main__":
    main()
