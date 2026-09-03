import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier
import warnings

warnings.simplefilter('ignore')

def main():
    print("Caricamento del dataset maschile...")
    try:
        df_male = pd.read_csv("dataset_xg_spaziale_male.csv")
    except FileNotFoundError:
        print("Errore: Impossibile trovare 'dataset_xg_spaziale_male.csv'.")
        return

    # Feature e Target
    features = ['distanza', 'angolo', 'difensori_cono', 'portiere_cono']
    target = 'goal'

    X = df_male[features]
    y = df_male[target]

    # Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Definizione della Griglia dei Parametri da testare
    # Qui diciamo a Python quali combinazioni provare
    param_grid = {
        'max_depth': [3, 4, 5, 6],                  # Profondità degli alberi
        'learning_rate': [0.01, 0.03, 0.05, 0.1],     # Velocità di apprendimento
        'n_estimators': [50, 100, 200],               # Numero di alberi
        'subsample': [0.8, 1.0]                       # Frazione di dati da usare per albero (evita overfitting)
    }

    print("\nInizializzazione di GridSearchCV...")
    # Usiamo XGBClassifier di base
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)

    # Configuriamo la ricerca: 
    # cv=5 significa 5-fold cross-validation
    # scoring='neg_log_loss' perché vogliamo minimizzare la log loss
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=5,
        verbose=1,
        n_jobs=-1 # Sfrutta tutti i core della CPU
    )

    print("Ricerca dei parametri ottimali in corso (potrebbe richiedere qualche secondo)...")
    grid_search.fit(X_train, y_train)

    # 2. Estrazione del Miglior Modello
    best_model = grid_search.best_estimator_

    print("\n" + "="*40)
    print("       MIGLIORI PARAMETRI TROVATI")
    print("="*40)
    for param, value in grid_search.best_params_.items():
        print(f" - {param}: {value}")
    print("="*40)

    # 3. Valutazione del Modello Ottimizzato sul Test Set
    preds_proba = best_model.predict_proba(X_test)[:, 1]
    
    loss = log_loss(y_test, preds_proba)
    auc = roc_auc_score(y_test, preds_proba)

    print("\n--- Performance Modello Ottimizzato (Test Set) ---")
    print(f"Log Loss: {loss:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

if __name__ == "__main__":
    main()