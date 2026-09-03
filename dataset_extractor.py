import math
import warnings

import pandas as pd
from shapely.geometry import Point, Polygon
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

warnings.simplefilter("ignore", NoAuthWarning)

GOALPOST_1 = (120, 36)
GOALPOST_2 = (120, 44)
GOAL_CENTER = (120, 40)
PLAYER_RADIUS = 0.6


# ---------------------------------------------------------------------------
# 1. FEATURE SPAZIALI DI BASE (distanza, angolo grezzo) — dal tuo script
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


# ---------------------------------------------------------------------------
# 2. ANGOLO DI PORTA LIBERO — proietta l'ombra di ogni difensore sulla porta
#    e sottrae la copertura totale dall'angolo grezzo. Feature molto più
#    informativa del semplice "conteggio difensori nel cono".
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. FEATURE DAL FREEZE FRAME 360: difensori nel cono, portiere, pressione,
#    compagni di supporto, densità locale.
# ---------------------------------------------------------------------------
def calculate_frame_features(shot_x, shot_y, shot_frame):
    ball_point = Point(shot_x, shot_y)
    shooting_cone = Polygon([(shot_x, shot_y), GOALPOST_1, GOALPOST_2])

    defenders_in_cone = 0
    gk_in_cone = False
    min_defender_distance = 99.0
    opponent_locations = []
    gk_loc = None

    teammates_nearby_5m = 0
    opponents_nearby_5m = 0
    n_players_visible = len(shot_frame)

    for _, player in shot_frame.iterrows():
        loc = player["location"]
        p_point = Point(loc[0], loc[1])
        dist_to_ball = ball_point.distance(p_point)

        if player["teammate"]:
            if not player.get("actor", False) and dist_to_ball <= 5.0:
                teammates_nearby_5m += 1
            continue

        # da qui in poi: solo avversari
        opponent_locations.append((loc[0], loc[1]))
        if dist_to_ball <= 5.0:
            opponents_nearby_5m += 1
        if dist_to_ball < min_defender_distance:
            min_defender_distance = dist_to_ball

        if player.get("keeper", False):
            gk_loc = (loc[0], loc[1])

        player_area = p_point.buffer(PLAYER_RADIUS)
        if shooting_cone.intersects(player_area):
            if player.get("keeper", False):
                gk_in_cone = True
            else:
                defenders_in_cone += 1

    free_angle = unobstructed_angle(shot_x, shot_y, opponent_locations)

    # feature portiere
    if gk_loc is not None:
        gk_x, gk_y = gk_loc
        gk_distance_to_goal = math.hypot(120.0 - gk_x, 40.0 - gk_y)
        gk_distance_to_shot = math.hypot(gk_x - shot_x, gk_y - shot_y)
        gk_lateral_offset = abs(gk_y - 40.0)
    else:
        gk_distance_to_goal = None
        gk_distance_to_shot = None
        gk_lateral_offset = None

    return {
        "difensori_cono": defenders_in_cone,
        "portiere_cono": int(gk_in_cone),
        "distanza_min_difensore": min_defender_distance,
        "angolo_libero": free_angle,
        "n_giocatori_visibili": n_players_visible,
        "compagni_vicini_5m": teammates_nearby_5m,
        "avversari_vicini_5m": opponents_nearby_5m,
        "gk_distanza_da_porta": gk_distance_to_goal,
        "gk_distanza_da_tiratore": gk_distance_to_shot,
        "gk_offset_laterale": gk_lateral_offset,
    }


# ---------------------------------------------------------------------------
# 4. FEATURE SULL'ASSIST — cerca il passaggio chiave collegato al tiro
# ---------------------------------------------------------------------------
def calculate_assist_features(shot_row, events_df):
    key_pass_id = shot_row.get("shot_key_pass_id")
    if not isinstance(key_pass_id, str):
        return {
            "assist_altezza": None,
            "assist_tecnica": None,
            "assist_lunghezza": None,
            "assist_cross": 0,
            "assist_cutback": 0,
            "assist_through_ball": 0,
        }

    kp = events_df[events_df["id"] == key_pass_id]
    if kp.empty:
        return {
            "assist_altezza": None,
            "assist_tecnica": None,
            "assist_lunghezza": None,
            "assist_cross": 0,
            "assist_cutback": 0,
            "assist_through_ball": 0,
        }

    kp = kp.iloc[0]
    return {
        "assist_altezza": kp.get("pass_height"),
        "assist_tecnica": kp.get("pass_technique"),
        "assist_lunghezza": kp.get("pass_length"),
        "assist_cross": int(bool(kp.get("pass_cross", False))),
        "assist_cutback": int(bool(kp.get("pass_cut_back", False))),
        "assist_through_ball": int(bool(kp.get("pass_through_ball", False))),
    }


# ---------------------------------------------------------------------------
# 5. CONTESTO DI PARTITA — differenza reti al momento del tiro, minuto
# ---------------------------------------------------------------------------
def build_score_diff_lookup(events_df):
    """Per ogni team nel match, restituisce funzione (index) -> diff reti."""
    goals = events_df[
        (events_df["type"] == "Shot") & (events_df["shot_outcome"] == "Goal")
    ][["index", "team"]].sort_values("index")

    teams = events_df["team"].dropna().unique().tolist()

    def score_diff_before(shot_index, shot_team):
        opponent = next((t for t in teams if t != shot_team), None)
        team_goals = (
            (goals["team"] == shot_team) & (goals["index"] < shot_index)
        ).sum()
        opp_goals = (
            (goals["team"] == opponent) & (goals["index"] < shot_index)
        ).sum()
        return int(team_goals - opp_goals)

    return score_diff_before


def possession_passes_before_shot(events_df, shot_row):
    """Numero di passaggi completati nella stessa azione (possession) prima del tiro."""
    poss_id = shot_row.get("possession")
    if poss_id is None:
        return 0
    mask = (
        (events_df["possession"] == poss_id)
        & (events_df["index"] < shot_row["index"])
        & (events_df["type"] == "Pass")
    )
    return int(mask.sum())


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    # male
    #COMP_ID = 55
    #SEASON_IDS = [282, 43]
    # DS_NAME = "dataset_xg_male.csv"
    
    DS_NAME = "dataset_xg_male_wc.csv"
    COMP_ID = 43
    SEASON_IDS = [106]
    
    # female
    # COMP_ID = 53
    # SEASON_IDS = [315, 106]
    # DS_NAME = "dataset_xg_female.csv"

    all_matches_list = []
    for s_id in SEASON_IDS:
        try:
            matches_df = sb.matches(competition_id=COMP_ID, season_id=s_id)
            all_matches_list.append(matches_df)
        except Exception as e:
            print(f"Errore stagione {s_id}: {e}")

    if not all_matches_list:
        return

    matches_df = pd.concat(all_matches_list, ignore_index=True)
    match_ids = matches_df["match_id"].tolist()
    print(f"Trovate {len(match_ids)} partite. Inizio estrazione feature avanzate...")

    dataset = []

    for i, m_id in enumerate(match_ids):
        try:
            frames_360_df = sb.frames(match_id=m_id, fmt="dataframe")
            events_df = sb.events(match_id=m_id)
            shots = events_df[events_df["type"] == "Shot"].copy()
            score_diff_fn = build_score_diff_lookup(events_df)

            for _, shot in shots.iterrows():
                shot_id = shot["id"]
                shot_location = shot["location"]
                shot_frame = frames_360_df[frames_360_df["id"] == shot_id]

                if shot_frame.empty:
                    continue

                shot_x, shot_y = shot_location
                dist, angle = calculate_distance_angle(shot_x, shot_y)
                frame_feats = calculate_frame_features(shot_x, shot_y, shot_frame)
                assist_feats = calculate_assist_features(shot, events_df)

                is_goal = 1 if shot.get("shot_outcome") == "Goal" else 0

                row = {
                    "match_id": m_id,
                    "tiratore": shot["player"],
                    "squadra": shot.get("team"),
                    "minuto": shot.get("minute"),
                    "periodo": shot.get("period"),
                    "differenza_reti": score_diff_fn(shot["index"], shot.get("team")),
                    "possesso_passaggi_precedenti": possession_passes_before_shot(
                        events_df, shot
                    ),
                    "distanza": dist,
                    "angolo": angle,
                    **frame_feats,
                    "body_part": shot.get("shot_body_part", "Other"),
                    "shot_type": shot.get("shot_type", "Open Play"),
                    "shot_technique": shot.get("shot_technique", "Normal"),
                    "play_pattern": shot.get("play_pattern"),
                    "position_tiratore": shot.get("position"),
                    "primo_tocco": int(bool(shot.get("shot_first_time", False))),
                    "porta_sguarnita": int(bool(shot.get("shot_open_goal", False))),
                    "tiro_deviato": int(bool(shot.get("shot_deflected", False))),
                    "sotto_pressione": int(bool(shot.get("under_pressure", False))),
                    **assist_feats,
                    # feature derivate
                    "distanza_x_angolo": dist * angle,
                    "angolo_libero_su_angolo_tot": (
                        frame_feats["angolo_libero"] / angle if angle > 0 else 0.0
                    ),
                    "pressione_relativa": (
                        frame_feats["distanza_min_difensore"] / dist if dist > 0 else 99.0
                    ),
                    # benchmark, NON usare come input del modello
                    "xg_statsbomb_benchmark": shot.get("shot_statsbomb_xg"),
                    "goal": is_goal,
                }
                dataset.append(row)

            print(f"[{i + 1}/{len(match_ids)}] Match {m_id}: Elaborato.")
        except Exception as e:
            print(f"[{i + 1}/{len(match_ids)}] Match {m_id}: Dati 360 non disponibili ({e}).")

    if dataset:
        final_df = pd.DataFrame(dataset)
        final_df.to_csv(DS_NAME, index=False)
        print(f"\nDataset avanzato salvato! Raccolti {len(final_df)} tiri.")
        print(f"Colonne totali: {len(final_df.columns)}")


if __name__ == "__main__":
    main()