import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, RocCurveDisplay
from xgboost import XGBClassifier
import warnings

warnings.simplefilter('ignore')

def main():
    print("Caricamento del dataset...")
    try:
        df_male = pd.read_csv("dataset_xg_spaziale_male.csv")
        df_male = pd.get_dummies(df_male, columns=['body_part', 'shot_type', 'shot_technique'], drop_first=True)
    except FileNotFoundError:
        print("Errore: Impossibile trovare 'dataset_xg_spaziale_male.csv'.")
        return

    # Feature e Target
    exclude_cols = ['match_id', 'tiratore', 'goal', 'xG']
    features = [col for col in df_male.columns if col not in exclude_cols]
    target = 'goal'

    X = df_male[features]
    y = df_male[target]

    # Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Definizione della Griglia dei Parametri da testare
    param_grid = {
        'max_depth': [3, 4, 5, 6],                  
        'learning_rate': [0.01, 0.03, 0.05, 0.1],     
        'n_estimators': [50, 100, 200],               
        'subsample': [0.8, 1.0]                       
    }

    print("\nInizializzazione di GridSearchCV...")
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=5,
        verbose=1,
        n_jobs=-1 
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

    # --- 4. PLOT DELLA CURVA ROC-AUC ---
    print("\nGenerazione del grafico della curva ROC...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    RocCurveDisplay.from_predictions(
        y_test, 
        preds_proba, 
        name="XGBoost xG Spaziale",
        ax=ax
    )
    
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Caso Casuale (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasso di Falsi Positivi (1 - Specificità)", fontsize=11)
    plt.ylabel("Tasso di Veri Positivi (Sensibilità)", fontsize=11)
    plt.title(f"Curva ROC - Modello xG Spaziale (AUC = {auc:.4f})", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()