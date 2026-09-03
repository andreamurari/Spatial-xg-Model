from statsbombpy import sb
from shapely.geometry import Point, Polygon
import pandas as pd
import math
import warnings
from statsbombpy.api_client import NoAuthWarning

warnings.simplefilter('ignore', NoAuthWarning)

GOALPOST_1 = (120, 36)
GOALPOST_2 = (120, 44)
PLAYER_RADIUS = 0.6 

def calculate_spatial_and_pressure_features(shot_x, shot_y, shot_frame):
    goal_x, goal_y = 120.0, 40.0
    
    # 1. Distanza e Angolo
    distance = math.sqrt((goal_x - shot_x)**2 + (goal_y - shot_y)**2)
    
    if shot_x >= 120:
        return distance, 0.0, 0, False, 99.0 # 99 yard = nessun difensore vicino
        
    v1_x, v1_y = 120.0 - shot_x, 36.0 - shot_y
    v2_x, v2_y = 120.0 - shot_x, 44.0 - shot_y
    dot_product = (v1_x * v2_x) + (v1_y * v2_y)
    mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
    mag_v2 = math.sqrt(v2_x**2 + v2_y**2)
    
    if mag_v1 * mag_v2 == 0:
        angle_deg = 0.0
    else:
        cos_theta = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
        angle_deg = math.degrees(math.acos(cos_theta))

    ball_point = Point(shot_x, shot_y)
    shooting_cone = Polygon([(shot_x, shot_y), GOALPOST_1, GOALPOST_2])

    defenders_in_cone = 0
    gk_in_cone = False
    min_defender_distance = 99.0 # Valore iniziale alto

    for _, player in shot_frame.iterrows():
        if not player['teammate']:
            player_loc = player['location']
            player_point = Point(player_loc[0], player_loc[1])
            
            # Calcoliamo la distanza euclidea pura dal pallone al difensore (Pressione)
            dist_to_ball = ball_point.distance(player_point)
            if dist_to_ball < min_defender_distance:
                min_defender_distance = dist_to_ball
            
            # Controllo cono e volume (come prima)
            player_area = player_point.buffer(PLAYER_RADIUS)
            if shooting_cone.intersects(player_area):
                if player.get('keeper', False):
                    gk_in_cone = True
                else:
                    defenders_in_cone += 1
                    
    return distance, angle_deg, defenders_in_cone, gk_in_cone, min_defender_distance

def main():
    COMP_ID = 55 # Sostituisci con l'ID competizione desiderato
    SEASON_IDS = [282, 43] 
    
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
    match_ids = matches_df['match_id'].tolist()
    print(f"Trovate {len(match_ids)} partite. Inizio estrazione feature avanzate...")
    
    dataset = []
    
    for i, m_id in enumerate(match_ids):
        try:
            frames_360_df = sb.frames(match_id=m_id, fmt='dataframe')
            events_df = sb.events(match_id=m_id)
            shots = events_df[events_df['type'] == 'Shot'].copy()
            
            for _, shot in shots.iterrows():
                shot_id = shot['id']
                shot_location = shot['location']
                shot_frame = frames_360_df[frames_360_df['id'] == shot_id]
                
                if shot_frame.empty:
                    continue
                    
                dist, angle, defs, gk, min_def_dist = calculate_spatial_and_pressure_features(
                    shot_location[0], shot_location[1], shot_frame
                )
                
                is_goal = 1 if shot.get('shot_outcome') == 'Goal' else 0
                body_part = shot.get('shot_body_part', 'Other')
                shot_type = shot.get('shot_type', 'Open Play')
                
                dataset.append({
                    'match_id': m_id,
                    'tiratore': shot['player'],
                    'distanza': dist,
                    'angolo': angle,
                    'difensori_cono': defs,
                    'portiere_cono': int(gk),
                    'distanza_min_difensore': min_def_dist, # Nuova feature di pressione
                    'body_part': body_part,
                    'shot_type': shot_type,
                    'goal': is_goal
                })
            print(f"[{i+1}/{len(match_ids)}] Match {m_id}: Elaborato.")
        except Exception:
            print(f"[{i+1}/{len(match_ids)}] Match {m_id}: Dati 360 non disponibili.")

    if dataset:
        final_df = pd.DataFrame(dataset)
        final_df.to_csv("dataset_xg_spaziale_male.csv", index=False)
        print(f"\nDataset avanzato salvato! Raccolti {len(final_df)} tiri.")

if __name__ == "__main__":
    main()