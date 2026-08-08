#!/usr/bin/env python
"""
End-to-end pipeline: load -> temporal stats -> refine -> features -> train ->
predict -> figures.

Usage
-----
    # Runs on real GEBCO grids in data/raw/ if present, else generates a
    # synthetic demo dataset so the whole pipeline still executes.
    python scripts/run_pipeline.py

    # Force the synthetic demo (no downloads needed):
    python scripts/run_pipeline.py --synthetic

    # Use the official EEZ polygon mask (needs geopandas + polygon file):
    python scripts/run_pipeline.py --polygon
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``src`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, load_config  # noqa: E402
from src.data.download_gebco import validate_raw  # noqa: E402
from src.data.load_bathymetry import load_all, make_synthetic_gebco  # noqa: E402
from src.features.terrain_features import compute_terrain  # noqa: E402
from src.models.dataset import build_table  # noqa: E402
from src.models.predict import predict_grid  # noqa: E402
from src.models.train import load_model, train_and_evaluate  # noqa: E402
from src.temporal.temporal_refinement import refine, temporal_statistics  # noqa: E402
from src.visualization import plots  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate synthetic GEBCO grids instead of using data/raw/.")
    ap.add_argument("--polygon", action="store_true",
                    help="Mask to the official EEZ polygon (requires geopandas).")
    ap.add_argument("--strategy", default="robust", choices=["robust", "latest"],
                    help="Temporal refinement strategy.")
    ap.add_argument("--config", default=None, help="Path to config.yaml.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()

    # ---- 1. Data --------------------------------------------------------- #
    status = validate_raw(cfg.gebco)
    if args.synthetic or not any(status.values()):
        log.info("Generating synthetic GEBCO grids (demo mode).")
        make_synthetic_gebco(cfg.gebco, cfg.study_area)

    stack = load_all(cfg.gebco, cfg.study_area, use_polygon=args.polygon)
    log.info("Loaded depth stack: %s", dict(stack.sizes))

    # ---- 2. Temporal ----------------------------------------------------- #
    stats = temporal_statistics(stack)
    refined = refine(stack, stats, strategy=args.strategy)
    log.info("Refined surface built with strategy=%s", args.strategy)

    # ---- 3. Terrain features on the refined surface ---------------------- #
    terrain = compute_terrain(refined)

    # ---- 4. Modelling table --------------------------------------------- #
    # Carry temporal_std through as a feature.
    stats_for_table = stats[["temporal_std", "temporal_range"]]
    table = build_table(refined, terrain, stats_for_table, cfg.model)
    log.info("Training table: %d rows x %d features", len(table),
             len([f for f in cfg.model.features if f in table.columns]))

    # ---- 5. Train + evaluate -------------------------------------------- #
    metrics = train_and_evaluate(table, cfg.model)
    log.info("Metrics: %s", metrics["point_accuracy"])
    log.info("Interval: %s", metrics["interval"])

    # ---- 6. Predict on grid + figures ----------------------------------- #
    model = load_model()
    pred = predict_grid(model, table, refined)
    pred.to_netcdf(cfg_output_path())
    log.info("Wrote gridded predictions to %s", cfg_output_path())

    plots.plot_depth(refined, "Refined GEBCO depth - Nigerian EEZ",
                     "refined_depth.png")
    plots.plot_uncertainty(pred["uncertainty_total"])
    plots.plot_temporal_change(stats)
    log.info("Figures written to outputs/figures/")

    log.info("Pipeline complete.")


def cfg_output_path() -> Path:
    from src.config import PROCESSED_DIR
    return PROCESSED_DIR / "eez_refined_bathymetry_uncertainty.nc"


if __name__ == "__main__":
    main()
