import pandas as pd
import joblib
import warnings

warnings.simplefilter('ignore')

def main():
    print("Caricamento del modello salvato (.pkl)...")
    bundle = joblib.load("xg_spatial_model_male.pkl")
    model = bundle['model']
    train_features = bundle['features']

    print("Caricamento dei dataset (Maschile e Femminile)...")
    try:
        df_male = pd.read_csv("dataset_xg_male.csv")
        df_female = pd.read_csv("dataset_xg_female.csv") # Assicurati che il nome corrisponda al tuo file femminile
    except FileNotFoundError as e:
        print(f"Errore nel trovare i file CSV: {e}")
        return

    # Funzione di preprocessing e allineamento colonne
    def preprocess_df(df, expected_features):
        # One-Hot Encoding
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        meta_text_cols = ['tiratore', 'squadra'] 
        cols_to_encode = [col for col in cat_cols if col not in meta_text_cols]
        
        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
            
        # Allineiamo le colonne esattamente a quelle del modello di training (aggiunge 0 se manca, scarta se in eccesso)
        df_aligned = df.reindex(columns=expected_features, fill_value=0)
        return df, df_aligned

    print("Elaborazione e predizione xG per il dataset Maschile...")
    df_male_raw, X_male = preprocess_df(df_male, train_features)
    df_male_raw['predicted_xg'] = model.predict_proba(X_male)[:, 1]

    print("Elaborazione e predizione xG per il dataset Femminile...")
    df_female_raw, X_female = preprocess_df(df_female, train_features)
    df_female_raw['predicted_xg'] = model.predict_proba(X_female)[:, 1]

    # --- TABELLA COMPARATIVA ---
    print("\n" + "="*50)
    print("      CONFRONTO STATISTICO: MASCHILE vs FEMMINILE")
    print("="*50)

    total_shots_m = len(df_male_raw)
    total_goals_m = df_male_raw['goal'].sum()
    total_xg_m = df_male_raw['predicted_xg'].sum()
    avg_dist_m = df_male_raw['distanza'].mean()

    total_shots_f = len(df_female_raw)
    total_goals_f = df_female_raw['goal'].sum()
    total_xg_f = df_female_raw['predicted_xg'].sum()
    avg_dist_f = df_female_raw['distanza'].mean()

    comparison_df = pd.DataFrame({
        'Metrica': [
            'Tiri Totali', 
            'Gol Effettivi', 
            'xG Totale (Modello)', 
            'Distanza Media Tiro (yard)', 
            'Conversione Reale (%)', 
            'Conversione Predetta xG (%)'
        ],
        'Calcio Maschile': [
            total_shots_m, 
            total_goals_m, 
            round(total_xg_m, 2), 
            round(avg_dist_m, 2),
            f"{(total_goals_m/total_shots_m)*100:.2f}%",
            f"{(total_xg_m/total_shots_m)*100:.2f}%"
        ],
        'Calcio Femminile': [
            total_shots_f, 
            total_goals_f, 
            round(total_xg_f, 2), 
            round(avg_dist_f, 2),
            f"{(total_goals_f/total_shots_f)*100:.2f}%",
            f"{(total_xg_f/total_shots_f)*100:.2f}%"
        ]
    })

    print(comparison_df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()