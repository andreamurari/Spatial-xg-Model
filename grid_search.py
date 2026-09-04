import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import log_loss, roc_auc_score, RocCurveDisplay
from xgboost import XGBClassifier
import warnings

warnings.simplefilter('ignore')

def main():
    print("Caricamento del dataset...")
    try:
        df_male = pd.read_csv("dataset_xg_male.csv") # Aggiornato al nome corretto del file usato finora
    except FileNotFoundError:
        print("Errore: Impossibile trovare il file del dataset.")
        return

    # Individuiamo automaticamente tutte le colonne di tipo testuale (object)
    cat_cols = df_male.select_dtypes(include=['object']).columns.tolist()

    # Rimuoviamo eventuali colonne testuali che non sono feature
    meta_text_cols = ['tiratore', 'squadra']
    cols_to_encode = [col for col in cat_cols if col not in meta_text_cols]

    if cols_to_encode:
        print(f"Applicazione One-Hot Encoding sulle colonne categoriche: {cols_to_encode}")
        df_male = pd.get_dummies(df_male, columns=cols_to_encode, drop_first=True)

    # Escludiamo dal set delle feature gli ID, il target, il tiratore e l'eventuale xG di benchmark StatsBomb
    exclude_cols = ['match_id', 'tiratore', 'squadra', 'goal', 'xg_statsbomb_benchmark']
    features = [col for col in df_male.columns if col not in exclude_cols]
    target = 'goal'

    X = df_male[features]
    y = df_male[target]

    print(f"Feature totali utilizzate per il modello: {len(features)}")

    # Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Definizione dello spazio di ricerca dei parametri.
    # Con appena ~2500 tiri e un tasso di gol al ~10%, il rischio principale
    # e' l'overfitting: oltre ai parametri "classici" (profondita', numero di
    # alberi, learning rate, subsample), qui includiamo esplicitamente le leve
    # di regolarizzazione di XGBoost:
    #   - min_child_weight: peso minimo richiesto in un nodo figlio prima di
    #     poter splittare ulteriormente -> valori alti scoraggiano split su
    #     pochi esempi rumorosi (rilevante con solo ~250 gol totali).
    #   - gamma: riduzione di loss minima richiesta per accettare uno split.
    #   - reg_lambda / reg_alpha: penalizzazione L2 / L1 sui pesi delle foglie.
    #   - colsample_bytree: sottocampiona le 73 feature ad ogni albero.
    # Il precedente grid (4 iperparametri, valori fissi) aveva scelto il
    # combo piu' conservativo testato (learning_rate=0.01, max_depth=3), cioe'
    # proprio il bordo della griglia: segno che serviva esplorare ancora piu'
    # regolarizzazione, non altri valori di quei 4 parametri soltanto.
    #
    # Con 9 iperparametri lo spazio cartesiano esploderebbe (>1 milione di
    # combinazioni): una GridSearchCV esaustiva non e' piu' praticabile, quindi
    # passiamo a RandomizedSearchCV, che campiona un numero fisso di
    # combinazioni casuali dallo spazio e resta comunque scored su
    # 'neg_log_loss' con 5-fold CV, come prima.
    param_distributions = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        'n_estimators': [100, 150, 200, 300, 400],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 10, 15],
        'gamma': [0, 0.05, 0.1, 0.2, 0.5],
        'reg_lambda': [1, 2, 5, 10],
        'reg_alpha': [0, 0.1, 0.5, 1],
    }
    N_ITER = 150

    print(f"\nInizializzazione di RandomizedSearchCV ({N_ITER} combinazioni campionate)...")
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring='neg_log_loss',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    print("Ricerca dei parametri ottimali in corso...")
    search.fit(X_train, y_train)

    # 2. Estrazione del Miglior Modello e dei Parametri
    best_model = search.best_estimator_
    best_params = search.best_params_

    print("\n" + "="*40)
    print("       MIGLIORI PARAMETRI TROVATI")
    print("="*40)
    for param, value in best_params.items():
        print(f" - {param}: {value}")
    print("="*40)

    # --- SCRITTURA AUTOMATICA DEL FILE DI CONFIGURAZIONE ---
    config_filename = "model_config.py"
    print(f"\nSalvataggio automatico dei parametri nel file '{config_filename}'...")
    with open(config_filename, "w") as f:
        f.write("# File di configurazione generato automaticamente dal Randomized Search\n")
        f.write("BEST_PARAMS = {\n")
        for param, value in best_params.items():
            f.write(f"    '{param}': {value},\n")
        f.write("    'random_state': 42,\n")
        f.write("    'eval_metric': 'logloss'\n")
        f.write("}\n")
    print("File di configurazione creato con successo!")

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
        name="XGBoost xG Spaziale Avanzato",
        ax=ax
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Caso Casuale (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasso di Falsi Positivi (1 - Specificità)", fontsize=11)
    plt.ylabel("Tasso di Veri Positivi (Sensibilità)", fontsize=11)
    plt.title(f"Curva ROC - Modello xG Avanzato (AUC = {auc:.4f})", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.show()

if __name__ == "__main__":
    main()
