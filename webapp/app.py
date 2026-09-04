"""Flask web app to explore the Spatial xG Model interactively.

Serves a page with a football pitch: the user places the shot location
(and, optionally, the goalkeeper and outfield defenders) plus a few shot
details, and the app runs the trained XGBoost model
(``xg_spatial_model_male.pkl``) to compute a live xG value.

The spatial feature engineering below intentionally mirrors the logic in
``dataset_extractor.py`` so that predictions made here are consistent with
how the training data was built.
"""

import math
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from shapely.geometry import Point, Polygon

# The ML pipeline (data_analysis.py, the trained model, the CSVs) lives one
# level up, at the project root — this app is just the web layer on top of it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis import comparison_table, run_analysis  # noqa: E402

warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------
# Pitch geometry (StatsBomb coordinates: 120 x 80, goal centred on x=120)
# ---------------------------------------------------------------------------
GOALPOST_1 = (120, 36)
GOALPOST_2 = (120, 44)
GOAL_CENTER = (120, 40)
PLAYER_RADIUS = 0.6

MODEL_PATH = PROJECT_ROOT / "xg_spatial_model_male.pkl"

app = Flask(__name__)

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
TRAIN_FEATURES = _bundle["features"]

# Dropdown choices shown in the UI (kept in sync with the training data).
CHOICES = {
    "body_part": ["Right Foot", "Left Foot", "Head", "Other"],
    "shot_type": ["Open Play", "Free Kick", "Penalty", "Corner"],
    "shot_technique": [
        "Normal",
        "Volley",
        "Half Volley",
        "Lob",
        "Diving Header",
        "Overhead Kick",
        "Backheel",
    ],
    "play_pattern": [
        "Regular Play",
        "From Counter",
        "From Free Kick",
        "From Corner",
        "From Goal Kick",
        "From Keeper",
        "From Kick Off",
        "From Throw In",
        "Other",
    ],
    "position_tiratore": [
        "Center Forward",
        "Left Center Forward",
        "Right Center Forward",
        "Left Wing",
        "Right Wing",
        "Left Attacking Midfield",
        "Right Attacking Midfield",
        "Center Attacking Midfield",
        "Left Center Midfield",
        "Right Center Midfield",
        "Left Midfield",
        "Right Midfield",
        "Left Defensive Midfield",
        "Right Defensive Midfield",
        "Center Defensive Midfield",
        "Left Wing Back",
        "Right Wing Back",
        "Left Back",
        "Right Back",
        "Left Center Back",
        "Right Center Back",
        "Center Back",
    ],
    "assist_altezza": ["Ground Pass", "Low Pass", "High Pass"],
    "assist_tecnica": ["Straight", "Inswinging", "Outswinging", "Through Ball"],
}


# ---------------------------------------------------------------------------
# Feature engineering (same formulas as dataset_extractor.py)
# ---------------------------------------------------------------------------
def calculate_distance_angle(shot_x, shot_y):
    goal_x, goal_y = GOAL_CENTER
    distance = math.hypot(goal_x - shot_x, goal_y - shot_y)

    if shot_x >= 120:
        return distance, 0.0

    v1_x, v1_y = 120.0 - shot_x, 36.0 - shot_y
    v2_x, v2_y = 120.0 - shot_x, 44.0 - shot_y
    dot = v1_x * v2_x + v1_y * v2_y
    mag1, mag2 = math.hypot(v1_x, v1_y), math.hypot(v2_x, v2_y)

    if mag1 * mag2 == 0:
        return distance, 0.0

    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_deg = math.degrees(math.acos(cos_theta))
    return distance, angle_deg


def unobstructed_angle(shot_x, shot_y, opponent_locations):
    ang_p1 = math.atan2(GOALPOST_1[1] - shot_y, GOALPOST_1[0] - shot_x)
    ang_p2 = math.atan2(GOALPOST_2[1] - shot_y, GOALPOST_2[0] - shot_x)
    lo, hi = sorted([ang_p1, ang_p2])
    total = hi - lo
    if total <= 0:
        return 0.0

    intervals = []
    for (px, py) in opponent_locations:
        d = math.hypot(px - shot_x, py - shot_y)
        if d <= 1e-6:
            continue
        if d <= PLAYER_RADIUS:
            intervals.append((lo, hi))
            continue
        half_w = math.asin(min(1.0, PLAYER_RADIUS / d))
        center = math.atan2(py - shot_y, px - shot_x)
        a, b = max(center - half_w, lo), min(center + half_w, hi)
        if a < b:
            intervals.append((a, b))

    if not intervals:
        return math.degrees(total)

    intervals.sort()
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))

    blocked = sum(b - a for a, b in merged)
    return math.degrees(max(0.0, total - blocked))


def calculate_frame_features(shot_x, shot_y, defenders, keeper, teammates_nearby_5m, n_players_visible):
    """Build the 360-frame-style features from user-placed markers."""
    shooting_cone = Polygon([(shot_x, shot_y), GOALPOST_1, GOALPOST_2])

    defenders_in_cone = 0
    gk_in_cone = 0
    min_defender_distance = 99.0
    opponents_nearby_5m = 0
    opponent_locations = [(d["x"], d["y"]) for d in defenders]

    for px, py in opponent_locations:
        dist = math.hypot(px - shot_x, py - shot_y)
        if dist <= 5.0:
            opponents_nearby_5m += 1
        if dist < min_defender_distance:
            min_defender_distance = dist
        if shooting_cone.intersects(Point(px, py).buffer(PLAYER_RADIUS)):
            defenders_in_cone += 1

    gk_distanza_da_porta = None
    gk_distanza_da_tiratore = None
    gk_offset_laterale = None

    if keeper is not None:
        gx, gy = keeper["x"], keeper["y"]
        opponent_locations.append((gx, gy))
        dist = math.hypot(gx - shot_x, gy - shot_y)
        if dist <= 5.0:
            opponents_nearby_5m += 1
        if dist < min_defender_distance:
            min_defender_distance = dist
        if shooting_cone.intersects(Point(gx, gy).buffer(PLAYER_RADIUS)):
            gk_in_cone = 1
        gk_distanza_da_porta = math.hypot(120.0 - gx, 40.0 - gy)
        gk_distanza_da_tiratore = dist
        gk_offset_laterale = abs(gy - 40.0)

    free_angle = unobstructed_angle(shot_x, shot_y, opponent_locations)

    return {
        "difensori_cono": defenders_in_cone,
        "portiere_cono": gk_in_cone,
        "distanza_min_difensore": min_defender_distance,
        "angolo_libero": free_angle,
        "n_giocatori_visibili": n_players_visible,
        "compagni_vicini_5m": teammates_nearby_5m,
        "avversari_vicini_5m": opponents_nearby_5m,
        "gk_distanza_da_porta": gk_distanza_da_porta,
        "gk_distanza_da_tiratore": gk_distanza_da_tiratore,
        "gk_offset_laterale": gk_offset_laterale,
    }


def build_feature_row(payload):
    shot = payload["shot"]
    shot_x, shot_y = float(shot["x"]), float(shot["y"])
    defenders = payload.get("defenders", [])
    keeper = payload.get("keeper")
    ctx = payload.get("context", {})
    details = payload.get("details", {})
    assist = payload.get("assist")

    distance, angle = calculate_distance_angle(shot_x, shot_y)
    frame_feats = calculate_frame_features(
        shot_x,
        shot_y,
        defenders,
        keeper,
        int(ctx.get("compagni_vicini_5m", 0)),
        int(ctx.get("n_giocatori_visibili", 15)),
    )

    if assist:
        assist_feats = {
            "assist_altezza": assist.get("altezza"),
            "assist_tecnica": assist.get("tecnica"),
            "assist_lunghezza": float(assist.get("lunghezza", 20)),
            "assist_cross": int(bool(assist.get("cross", False))),
            "assist_cutback": int(bool(assist.get("cutback", False))),
            "assist_through_ball": int(bool(assist.get("through_ball", False))),
        }
    else:
        assist_feats = {
            "assist_altezza": None,
            "assist_tecnica": None,
            "assist_lunghezza": None,
            "assist_cross": 0,
            "assist_cutback": 0,
            "assist_through_ball": 0,
        }

    row = {
        "minuto": int(ctx.get("minuto", 51)),
        "periodo": int(ctx.get("periodo", 1)),
        "differenza_reti": int(ctx.get("differenza_reti", 0)),
        "possesso_passaggi_precedenti": int(ctx.get("possesso_passaggi_precedenti", 5)),
        "distanza": distance,
        "angolo": angle,
        **frame_feats,
        "body_part": details.get("body_part", "Right Foot"),
        "shot_type": details.get("shot_type", "Open Play"),
        "shot_technique": details.get("shot_technique", "Normal"),
        "play_pattern": details.get("play_pattern", "Regular Play"),
        "position_tiratore": details.get("position_tiratore", "Center Forward"),
        "primo_tocco": int(bool(details.get("primo_tocco", False))),
        "porta_sguarnita": int(bool(details.get("porta_sguarnita", False))),
        "tiro_deviato": int(bool(details.get("tiro_deviato", False))),
        "sotto_pressione": int(bool(details.get("sotto_pressione", False))),
        **assist_feats,
        "distanza_x_angolo": distance * angle,
        "angolo_libero_su_angolo_tot": (
            frame_feats["angolo_libero"] / angle if angle > 0 else 0.0
        ),
        "pressione_relativa": (
            frame_feats["distanza_min_difensore"] / distance if distance > 0 else 99.0
        ),
    }
    return row, {"distanza": distance, "angolo": angle, **frame_feats}


def predict_xg(row):
    cat_cols = [
        "body_part",
        "shot_type",
        "shot_technique",
        "play_pattern",
        "position_tiratore",
        "assist_altezza",
        "assist_tecnica",
    ]
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=cat_cols)
    df = df.reindex(columns=TRAIN_FEATURES, fill_value=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    proba = MODEL.predict_proba(df)[:, 1][0]
    return float(proba)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", choices=CHOICES)


_dashboard_cache = None


def get_dashboard_data():
    """Score the comparison datasets once per process and cache the result.

    Mirrors what data_analysis.py prints on the CLI, reshaped for the
    dashboard template/charts.
    """
    global _dashboard_cache
    if _dashboard_cache is not None:
        return _dashboard_cache

    results = run_analysis()
    datasets = [
        {
            "label": label,
            "metrics": r["metrics"],
        }
        for label, r in results.items()
    ]
    table = comparison_table(results)
    _dashboard_cache = {
        "datasets": datasets,
        "table_rows": table.to_dict(orient="records"),
        "table_columns": table.columns.tolist(),
    }
    return _dashboard_cache


@app.route("/dashboard")
def dashboard():
    try:
        data = get_dashboard_data()
        error = None
    except FileNotFoundError as exc:
        data = None
        error = str(exc)
    return render_template("dashboard.html", data=data, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    if not payload or "shot" not in payload:
        return jsonify({"error": "Missing shot location."}), 400

    try:
        row, breakdown = build_feature_row(payload)
        xg = predict_xg(row)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 400

    return jsonify({"xg": xg, "features": breakdown})


if __name__ == "__main__":
    app.run(debug=True)
