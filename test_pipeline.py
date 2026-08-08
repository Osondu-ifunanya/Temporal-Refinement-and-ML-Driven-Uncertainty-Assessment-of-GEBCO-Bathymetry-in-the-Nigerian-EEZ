"""
Fast unit tests exercising each stage on small synthetic grids.

Run with:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GebcoConfig, ModelConfig, StudyArea  # noqa: E402
from src.data.load_bathymetry import make_synthetic_gebco  # noqa: E402
from src.features.terrain_features import compute_terrain  # noqa: E402
from src.models.dataset import build_table  # noqa: E402
from src.models.train import train_and_evaluate  # noqa: E402
from src.temporal.temporal_refinement import refine, temporal_statistics  # noqa: E402


@pytest.fixture(scope="module")
def stack():
    area = StudyArea(lon_min=2.0, lon_max=3.0, lat_min=1.0, lat_max=2.0)
    gebco = GebcoConfig(versions=[2021, 2022, 2023])
    return make_synthetic_gebco(gebco, area, resolution_deg=0.05, write=False).rename(
        {gebco.depth_var: "elevation"}
    )


def test_synthetic_shapes(stack):
    assert "year" in stack.dims
    assert stack.sizes["year"] == 3
    assert stack.sizes["lat"] > 5 and stack.sizes["lon"] > 5


def test_temporal_statistics(stack):
    depth = (-stack["elevation"]).to_dataset(name="depth")
    stats = temporal_statistics(depth)
    for var in ("temporal_mean", "temporal_std", "temporal_trend", "n_valid"):
        assert var in stats
    assert float(stats["temporal_std"].max()) >= 0.0


def test_refine_strategies(stack):
    depth = (-stack["elevation"]).to_dataset(name="depth")
    for strategy in ("robust", "latest"):
        refined = refine(depth, strategy=strategy)
        assert refined.dims == ("lat", "lon")
        assert np.isfinite(refined.values[np.isfinite(refined.values)]).all()


def test_terrain_features(stack):
    depth = (-stack["elevation"]).isel(year=-1)
    terrain = compute_terrain(depth)
    for var in ("slope", "aspect_sin", "aspect_cos", "roughness", "tpi", "curvature"):
        assert var in terrain
    assert float(terrain["slope"].max()) >= 0.0


def test_train_end_to_end(stack):
    depth_ds = (-stack["elevation"]).to_dataset(name="depth")
    stats = temporal_statistics(depth_ds)
    refined = refine(depth_ds, strategy="robust")
    terrain = compute_terrain(refined)
    cfg = ModelConfig(n_estimators=60)
    table = build_table(refined, terrain, stats[["temporal_std", "temporal_range"]], cfg)
    assert len(table) > 20
    metrics = train_and_evaluate(table, cfg)
    assert metrics["point_accuracy"]["rmse_m"] >= 0
    # Conformal interval should be near or above nominal coverage.
    assert metrics["interval"]["empirical_coverage"] >= 0.7
