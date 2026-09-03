import pandas as pd
import joblib
import warnings

warnings.simplefilter('ignore')

def main():
    print("Caricamento del modello salvato (.pkl)...")
    bundle = joblib.load("xg_spatial_model_male.pkl")
    model = bundle['model']
    train_features = bundle['features']

    print("Caricamento dei dataset (Maschile Euro, Maschile Mondiali e Femminile)...")
    try:
        df_male_euro = pd.read_csv("dataset_xg_male.csv")
        df_male_wc = pd.read_csv("dataset_xg_male_wc.csv")
        df_female = pd.read_csv("dataset_xg_female.csv")
    except FileNotFoundError as e:
        print(f"Errore nel trovare i file CSV: {e}")
        return

    # Funzione di preprocessing e allineamento colonne
    def preprocess_df(df, expected_features):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        meta_text_cols = ['tiratore', 'squadra'] 
        cols_to_encode = [col for col in cat_cols if col not in meta_text_cols]
        
        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
            
        df_aligned = df.reindex(columns=expected_features, fill_value=0)
        return df, df_aligned

    print("Elaborazione e predizione xG per Maschile (Euro)...")
    df_m_euro_raw, X_m_euro = preprocess_df(df_male_euro, train_features)
    df_m_euro_raw['predicted_xg'] = model.predict_proba(X_m_euro)[:, 1]

    print("Elaborazione e predizione xG per Maschile (Mondiali)...")
    df_m_wc_raw, X_m_wc = preprocess_df(df_male_wc, train_features)
    df_m_wc_raw['predicted_xg'] = model.predict_proba(X_m_wc)[:, 1]

    print("Elaborazione e predizione xG per il dataset Femminile...")
    df_female_raw, X_female = preprocess_df(df_female, train_features)
    df_female_raw['predicted_xg'] = model.predict_proba(X_female)[:, 1]

    # --- ESTRAZIONE METRICHE (Inclusi i dati StatsBomb) ---
    def get_metrics(df):
        shots = len(df)
        goals = df['goal'].sum()
        total_xg_model = df['predicted_xg'].sum()
        
        # Gestione sicura nel caso la colonna di benchmark StatsBomb abbia nomi leggermente diversi o NaN
        sb_col = 'xg_statsbomb_benchmark' if 'xg_statsbomb_benchmark' in df.columns else None
        total_xg_sb = df[sb_col].sum() if sb_col and not df[sb_col].isna().all() else 0.0
        
        avg_dist = df['distanza'].mean()
        
        real_conv = (goals / shots) * 100 if shots > 0 else 0
        pred_conv_model = (total_xg_model / shots) * 100 if shots > 0 else 0
        pred_conv_sb = (total_xg_sb / shots) * 100 if shots > 0 and total_xg_sb > 0 else 0
        
        return [
            shots, 
            goals, 
            round(total_xg_model, 2), 
            round(total_xg_sb, 2),
            round(avg_dist, 2), 
            f"{real_conv:.2f}%", 
            f"{pred_conv_model:.2f}%",
            f"{pred_conv_sb:.2f}%" if total_xg_sb > 0 else "N/A"
        ]

    metrics_m_euro = get_metrics(df_m_euro_raw)
    metrics_m_wc = get_metrics(df_m_wc_raw)
    metrics_female = get_metrics(df_female_raw)

    # --- TABELLA COMPARATIVA AMPLIATA ---
    print("\n" + "="*85)
    print("       CONFRONTO COMPLETO: MODELLO PROPRIO vs STATSBOMB BENCHMARK")
    print("="*85)

    comparison_df = pd.DataFrame({
        'Metrica': [
            'Tiri Totali', 
            'Gol Effettivi', 
            'xG Totale (Nostro Modello)', 
            'xG Totale (StatsBomb)', 
            'Distanza Media Tiro (yard)', 
            'Conversione Reale (%)', 
            'Conversione Nostro xG (%)',
            'Conversione StatsBomb xG (%)'
        ],
        'Maschile (Euro)': metrics_m_euro,
        'Maschile (Mondiali)': metrics_m_wc,
        'Calcio Femminile': metrics_female
    })

    print(comparison_df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()