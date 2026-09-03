<p align="center">
	<img src="SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png" alt="StatsBomb" width="420">
</p>

# Spatial xG Model

An experimental Expected Goals (xG) model built with **StatsBomb Open Data** and
StatsBomb 360 freeze-frame data. The project extends conventional distance and
angle features with spatial context such as defender obstruction, goalkeeper
positioning, pressure, nearby players and assist characteristics.

This is an independent research project. It is not produced, sponsored or
endorsed by StatsBomb.

## What it does

- Extracts shot, event and freeze-frame information with `statsbombpy`.
- Builds spatial features with `shapely`, including the unobstructed shooting
	angle and players inside the shooting cone.
- Trains an `XGBClassifier` on `dataset_xg_male.csv`.
- Applies the trained model to the men's Euro, men's World Cup and women's CSV
	datasets, then compares results with the StatsBomb xG benchmark.

## Repository contents

| File | Purpose |
| --- | --- |
| `dataset_extractor.py` | Extracts and engineers shot-level datasets. |
| `xG_computation.py` | Trains and saves `xg_spatial_model_male.pkl`. |
| `data_analysis.py` | Scores the datasets and prints a comparison table. |
| `grid_search.py` | Searches model hyperparameters. |
| `model_config.py` | Stores the selected hyperparameters. |
| `dataset_xg_*.csv` | Prepared datasets used by the pipeline. |
| `competitions.csv` | Competition and season selections. |

## Requirements

Python 3.10 or newer is recommended. Install the packages used by the scripts:

```bash
python -m pip install pandas shapely statsbombpy xgboost scikit-learn joblib
```

## Usage

Run the commands from the repository root:

```bash
python dataset_extractor.py
python xG_computation.py
python data_analysis.py
```

The extraction step may require network access to StatsBomb Open Data. The
training step writes a local model file, which is intentionally not part of the
data attribution.

## Data source and attribution

Data used in this project comes from **StatsBomb Open Data**, including selected
StatsBomb 360 matches. If you reuse or publish analysis based on these data,
credit StatsBomb and retain this attribution:

> Data provided by StatsBomb. Used under the StatsBomb Open Data Terms & Conditions.

The official terms permit the data to be used for research and genuine interest
in football analytics. Review the current terms before redistributing data or
using the project commercially:

- [StatsBomb Open Data repository and terms](https://github.com/statsbomb/open-data)
- [StatsBomb Media Pack](https://statsbomb.com/media-pack/)
- [StatsBomb Open Data licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf)

The StatsBomb logo above is included for source attribution and follows the
project's use of the official Media Pack asset. StatsBomb names, logos and
trademarks remain the property of their respective owners.

## Limitations

This model is an independent experiment, not an official StatsBomb model. The
datasets, competitions and available 360 freeze frames are limited to the
matches exposed by StatsBomb Open Data. Reported results should therefore not be
treated as production-grade probabilities or as representative of all football.
