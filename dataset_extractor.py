from statsbombpy import sb
from shapely.geometry import Point, Polygon
import pandas as pd
import math
import warnings
from statsbombpy.api_client import NoAuthWarning

warnings.simplefilter('ignore', NoAuthWarning)

"""competitions = sb.competitions()
competitions.to_csv("competitions.csv", index=False)"""

GOALPOST_1 = (120, 36)
GOALPOST_2 = (120, 44)
PLAYER_RADIUS = 0.6 

def calculate_spatial_features(shot_x, shot_y, shot_frame):
    goal_x, goal_y = 120.0, 40.0
    distance = math.sqrt((goal_x - shot_x)**2 + (goal_y - shot_y)**2)
    
    if shot_x >= 120:
        return distance, 0.0, 0, False
        
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

    for _, player in shot_frame.iterrows():
        if not player['teammate']:
            player_loc = player['location']
            player_point = Point(player_loc[0], player_loc[1])
            player_area = player_point.buffer(PLAYER_RADIUS)
            
            if shooting_cone.intersects(player_area):
                if player.get('keeper', False):
                    gk_in_cone = True
                else:
                    defenders_in_cone += 1
                    
    return distance, angle_deg, defenders_in_cone, gk_in_cone

def main():
    # Esempio: Mondiali 2022 (competition_id=43, season_id=106)
    # Sostituisci con i codici della competizione che preferisci
    COMP_ID = 53
    SEASON_IDS = [315, 106] # Nota: usa una lista di interi
    
    all_matches_list = []
    
    for s_id in SEASON_IDS:
        print(f"Recupero le partite per la competizione {COMP_ID}, stagione {s_id}...")
        try:
            matches_df = sb.matches(competition_id=COMP_ID, season_id=s_id)
            all_matches_list.append(matches_df)
        except Exception as e:
            print(f"Stagione {s_id} non disponibile o errore: {e}")
            
    # Uniamo tutte le stagioni in un unico DataFrame di partite
    if all_matches_list:
        matches_df = pd.concat(all_matches_list, ignore_index=True)
        match_ids = matches_df['match_id'].tolist()
        print(f"\nTrovate {len(match_ids)} partite totali tra le stagioni selezionate.")
    else:
        print("Nessuna partita trovata per le stagioni indicate.")
        match_ids = []

    print(izio := f"Trovate {len(match_ids)} partite. Inizio estrazione dati a 360°...")
    
    dataset = []
    
    for i, m_id in enumerate(match_ids):
        print(f"Elaborazione match {i+1}/{len(match_ids)} (ID: {m_id})...")
        try:
            events_df = sb.events(match_id=m_id)
            frames_360_df = sb.frames(match_id=m_id, fmt='dataframe')
            shots = events_df[events_df['type'] == 'Shot'].copy()
            
            for _, shot in shots.iterrows():
                shot_id = shot['id']
                shot_location = shot['location']
                shot_frame = frames_360_df[frames_360_df['id'] == shot_id]
                
                if shot_frame.empty:
                    continue
                    
                dist, angle, defs, gk = calculate_spatial_features(
                    shot_location[0], shot_location[1], shot_frame
                )
                
                is_goal = 1 if shot.get('shot_outcome') == 'Goal' else 0
                
                dataset.append({
                    'match_id': m_id,
                    'tiratore': shot['player'],
                    'distanza': dist,
                    'angolo': angle,
                    'difensori_cono': defs,
                    'portiere_cono': int(gk),
                    'goal': is_goal
                })
        except Exception as e:
            print(f"Saltata partita {m_id} per errore: {e}")

    # Creazione del Dataset Finale
    final_df = pd.DataFrame(dataset)
    print(f"\nDataset completato! Raccolti {len(final_df)} tiri totali.")
    
    # Salvataggio in CSV per usi futuri senza riscaricare tutto
    final_df.to_csv("dataset_xg_spaziale_female.csv", index=False)
    print("Dataset salvato in 'dataset_xg_spaziale_female.csv'.")

if __name__ == "__main__":
    main()