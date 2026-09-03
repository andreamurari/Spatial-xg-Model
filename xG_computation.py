import pandas as pd
import joblib
from xgboost import XGBClassifier
import warnings
from model_config import BEST_PARAMS  # Importiamo i parametri calcolati in automatico

warnings.simplefilter('ignore')

def main():
    print("Caricamento del dataset maschile...")
    try:
        df_male = pd.read_csv("dataset_xg_male.csv")
    except FileNotFoundError:
        print("Errore: Impossibile trovare 'dataset_xg_male.csv'.")
        return

    # One-Hot Encoding identico a prima per allineare le colonne
    cat_cols = df_male.select_dtypes(include=['object']).columns.tolist()
    meta_text_cols = ['tiratore', 'squadra'] 
    cols_to_encode = [col for col in cat_cols if col not in meta_text_cols]
    
    if cols_to_encode:
        df_male = pd.get_dummies(df_male, columns=cols_to_encode, drop_first=True)

    # Definizione feature e target
    exclude_cols = ['match_id', 'tiratore', 'squadra', 'goal', 'xg_statsbomb_benchmark']
    features = [col for col in df_male.columns if col not in exclude_cols]
    target = 'goal'

    X = df_male[features]
    y = df_male[target]

    print(f"Dataset pronto: {len(X)} tiri, {len(features)} feature.")

    # Inizializziamo il modello usando direttamente i parametri dinamici dal file di configurazione
    print("\nCreazione del modello con i parametri ottimali da 'model_config.py'...")
    best_xgboost_model = XGBClassifier(**BEST_PARAMS)

    # Addestramento sul 100% dei dati per creare il benchmark definitivo
    print("Addestramento del modello in corso...")
    best_xgboost_model.fit(X, y)

    # Salvataggio del modello e della lista delle feature
    model_bundle = {
        'model': best_xgboost_model,
        'features': features
    }
    
    model_filename = "xg_spatial_model_male.pkl"
    joblib.dump(model_bundle, model_filename)
    
    print(f"\nModello salvato con successo in '{model_filename}'!")
    print("Ora siamo pronti per applicarlo al dataset femminile o a qualsiasi nuova partita.")

if __name__ == "__main__":
    main()