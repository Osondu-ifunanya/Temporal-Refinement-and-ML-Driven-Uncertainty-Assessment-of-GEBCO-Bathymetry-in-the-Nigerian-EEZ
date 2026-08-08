# Temporal Refinement and ML-Driven Uncertainty Assessment of GEBCO Bathymetry in the Nigerian EEZ

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

A reproducible pipeline that harmonises multiple annual **GEBCO** gridded
bathymetry releases over the **Nigerian Exclusive Economic Zone (EEZ)**, derives
temporal and seafloor-terrain descriptors, and trains a machine-learning model
that produces a **refined depth estimate** together with **calibrated,
conformalised per-cell uncertainty**.

Rather than asking only *"how deep is the seabed here?"*, this project also
answers *"how much should we trust that number, and where?"* — the question that
actually matters for navigation safety, offshore engineering, habitat mapping,
and blue-economy planning in the Gulf of Guinea.

---

## Why this matters

The GEBCO grid is revised every year as new soundings and predicted-depth models
are ingested, and its quality is highly uneven — dense multibeam survey lines sit
next to cells interpolated from sparse or model-predicted data. This pipeline
turns that release history and spatial heterogeneity into two products:

1. **A temporally-refined best-estimate depth surface** that exploits the full
   set of releases instead of a single year.
2. **A spatial uncertainty field** that is *calibrated* — its 90 % prediction
   intervals actually contain the truth ~90 % of the time on held-out data.

---

## Highlights

- **Temporal refinement** across GEBCO 2020–2024 with recency-weighted,
  outlier-trimmed compositing.
- **Empirical uncertainty prior** from inter-release disagreement.
- **Terrain descriptors** (slope, aspect, roughness, TPI, curvature) with
  latitude-corrected cell spacing.
- **Three-way uncertainty model**: heteroscedastic quantile intervals
  (gradient boosting) + epistemic ensemble spread (random forest) + the
  temporal prior, all wrapped in **split-conformal calibration** for a
  finite-sample coverage guarantee.
- **Runs with zero downloads**: a deterministic synthetic GEBCO generator lets
  the whole pipeline execute for CI, teaching, or demos.
- Clean, tested, `config`-driven, and packaged.

---

## Repository structure

```
gebco-eez-refinement/
├── config/
│   └── config.yaml               # study area, GEBCO versions, model settings
├── src/
│   ├── config.py                 # typed config + loader
│   ├── data/
│   │   ├── download_gebco.py      # portal instructions + raw-file validation
│   │   ├── load_bathymetry.py     # load real grids OR synthesise demo grids
│   │   └── nigerian_eez.py        # bbox + official-polygon masking
│   ├── features/
│   │   └── terrain_features.py    # slope/aspect/roughness/TPI/curvature
│   ├── temporal/
│   │   └── temporal_refinement.py # temporal stats + refined surface
│   ├── models/
│   │   ├── dataset.py             # grids -> tidy training table
│   │   ├── uncertainty.py         # quantile + ensemble + conformal
│   │   ├── train.py               # fit, evaluate, persist
│   │   └── predict.py             # rasterise predictions back to grid
│   └── visualization/
│       └── plots.py               # maps + calibration figures
├── scripts/
│   └── run_pipeline.py           # end-to-end CLI
├── notebooks/
│   └── 01_explore_and_run.py     # jupytext walkthrough
├── tests/
│   └── test_pipeline.py          # fast unit tests for every stage
├── docs/
│   └── methodology.md            # full scientific write-up
├── data/                         # raw / interim / processed (git-ignored)
├── outputs/                      # figures / tables / models (git-ignored)
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Dogiye12/Machine-Learning-for-Seafloor-Geomorphological-Classification-in-the-Nigerian-EEZ-.git
cd Machine-Learning-for-Seafloor-Geomorphological-Classification-in-the-Nigerian-EEZ-

# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate gebco-eez
```

---

## Quick start

Run the entire pipeline on synthetic demo data — no downloads required:

```bash
python scripts/run_pipeline.py --synthetic
```

This will:

1. generate synthetic GEBCO grids under `data/raw/`,
2. compute temporal statistics and a refined surface,
3. derive terrain descriptors,
4. train the uncertainty model and print metrics,
5. write `data/processed/eez_refined_bathymetry_uncertainty.nc` and figures.

Example console output:

```
RMSE=46.57 m  MAE=32.83 m  R2=0.999  coverage=0.902 (nominal 0.90)  width=200.9 m
```

The empirical coverage (0.902) landing on the 0.90 nominal target is the whole
point — the intervals are calibrated, not just plausible-looking.

---

## Using real GEBCO data

1. Get the bounding box and filenames to request:

   ```bash
   python -m src.data.download_gebco
   ```

2. Download one **sub-set** grid per year from the
   [GEBCO download portal](https://download.gebco.net/) for the printed bounds,
   saving them as `data/raw/GEBCO_<year>.nc` (NetCDF, variable `elevation`).

3. Run the pipeline (it auto-detects real files):

   ```bash
   python scripts/run_pipeline.py --strategy robust
   ```

4. For a publication-grade mask, drop the Marine Regions Nigeria EEZ polygon at
   `data/raw/eez/nigeria_eez.geojson`, install the geo extras, and add
   `--polygon`:

   ```bash
   pip install "geopandas>=0.14" shapely
   python scripts/run_pipeline.py --polygon
   ```

---

## Configuration

Everything is driven by `config/config.yaml` (overriding the typed defaults in
`src/config.py`). Common edits:

```yaml
study_area: { lon_min: 2.0, lon_max: 8.5, lat_min: 1.0, lat_max: 6.5 }
gebco:      { versions: [2020, 2021, 2022, 2023, 2024], reference_version: 2024 }
model:      { lower_quantile: 0.05, upper_quantile: 0.95, n_estimators: 400 }
```

---

## Outputs

| Path | Contents |
|------|----------|
| `data/processed/eez_refined_bathymetry_uncertainty.nc` | refined depth, calibrated interval half-width, epistemic component, interval bounds |
| `outputs/figures/refined_depth.png` | refined bathymetry map |
| `outputs/figures/uncertainty_map.png` | calibrated uncertainty map |
| `outputs/figures/temporal_change.png` | inter-release disagreement + depth trend |
| `outputs/tables/metrics.json` | accuracy + calibration metrics |
| `outputs/models/uncertainty_model.joblib` | fitted model |

---

## Method in one paragraph

Annual GEBCO releases are aligned to a common grid; per-cell temporal mean, std,
range and trend are computed, and the inter-release std is retained as a
data-driven uncertainty prior. A recency-weighted, MAD-trimmed composite gives
the refined surface, on which terrain descriptors are derived. A tidy table
(coordinates + terrain + temporal prior → depth) trains two quantile gradient
boosters (5th/95th percentile) for heteroscedastic intervals and a random forest
whose per-tree spread supplies epistemic uncertainty. Split-conformal
calibration then inflates the interval so its coverage matches the nominal level
on held-out data. Full details in [`docs/methodology.md`](docs/methodology.md).

---

## Tests

```bash
pytest -q
```

The suite exercises the synthetic generator, temporal statistics, both
refinement strategies, terrain features, and the full train/evaluate loop on
small grids in seconds.

---

## Data & attribution

Bathymetry: **GEBCO Compilation Group** gridded bathymetry data —
<https://www.gebco.net/>. Please cite the specific GEBCO release(s) you use.
EEZ boundary (optional polygon mask): **Marine Regions** (VLIZ),
<https://www.marineregions.org/>.

The synthetic grids shipped for demos are *not* real bathymetry and must not be
used for any operational or navigational purpose.

---

## Citation

If you use this repository, please cite it via [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
