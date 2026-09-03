# Spatial-xg-Model

An advanced **Expected Goals (xG)** machine learning model built using **StatsBomb 360 open data**, incorporating spatial freeze-frame features (such as shooting cones, defender obstruction volumes, and goalkeeper positioning) powered by geometric data processing and XGBoost.

---

## 🚀 Key Features

- **Spatial Feature Engineering:** Moves beyond simple distance and angle by calculating physical obstruction volumes using `shapely` (e.g., defenders inside the shooting cone, goalkeeper positioning, and visible angle via vector dot products).
- **Machine Learning Pipeline:** Trains an `XGBoost` classifier on match data to estimate true goal probabilities based on tactical pressure.
- **Interactive Dashboard:** Built with `Streamlit` and `mplsoccer` to visualize player danger zones and spatial shooting profiles.

---

## 🛠️ Tech Stack

- **Python** (Core logic and data processing)
- **StatsBombPy** (Data extraction from StatsBomb Open Data API)
- **Shapely** (Spatial geometry and polygon intersections)
- **XGBoost & Scikit-Learn** (Model training and evaluation metrics)
- **Streamlit & Mplsoccer** (Interactive UI and pitch visualizations)

---
