from pathlib import Path

import pandas as pd
import joblib
import warnings

warnings.simplefilter('ignore')

# Resolved against this file's own location so the pipeline scripts and CSVs
# are found correctly regardless of the caller's working directory (e.g. the
# webapp/ Flask app importing this module from a sibling folder).
BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    'Male Euro': 'dataset_xg_male.csv',
    'Male World Cup': 'dataset_xg_male_wc.csv',
    'Female Euro': 'dataset_xg_female.csv',
}

METRIC_LABELS = [
    'Total Shots',
    'Actual Goals',
    'Total xG (Our Model)',
    'Total xG (StatsBomb)',
    'Avg. Shot Distance (yd)',
    'Real Conversion (%)',
    'Our xG Conversion (%)',
    'StatsBomb xG Conversion (%)',
]


def preprocess_df(df, expected_features):
    """One-hot encode + align columns to the features the model was trained on."""
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    meta_text_cols = ['tiratore', 'squadra']
    cols_to_encode = [col for col in cat_cols if col not in meta_text_cols]

    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    df_aligned = df.reindex(columns=expected_features, fill_value=0)
    return df, df_aligned


def get_metrics(df):
    """Numeric metrics for one scored dataset (used by both the CLI table and the dashboard)."""
    shots = len(df)
    goals = int(df['goal'].sum())
    total_xg_model = float(df['predicted_xg'].sum())

    sb_col = 'xg_statsbomb_benchmark' if 'xg_statsbomb_benchmark' in df.columns else None
    total_xg_sb = float(df[sb_col].sum()) if sb_col and not df[sb_col].isna().all() else 0.0

    avg_dist = float(df['distanza'].mean())

    real_conv = (goals / shots) * 100 if shots > 0 else 0.0
    pred_conv_model = (total_xg_model / shots) * 100 if shots > 0 else 0.0
    pred_conv_sb = (total_xg_sb / shots) * 100 if shots > 0 and total_xg_sb > 0 else 0.0

    return {
        'shots': shots,
        'goals': goals,
        'total_xg_model': total_xg_model,
        'total_xg_sb': total_xg_sb,
        'avg_distance': avg_dist,
        'real_conversion': real_conv,
        'model_conversion': pred_conv_model,
        'sb_conversion': pred_conv_sb if total_xg_sb > 0 else None,
    }


def calibration_curve(df, n_bins=10):
    """Bin shots by predicted xG (quantiles) and compare mean predicted vs actual goal rate."""
    scored = df[['predicted_xg', 'goal']].dropna().copy()
    if len(scored) < n_bins:
        return []
    try:
        scored['bin'] = pd.qcut(scored['predicted_xg'], q=n_bins, duplicates='drop')
    except ValueError:
        return []
    grouped = scored.groupby('bin', observed=True).agg(
        predicted=('predicted_xg', 'mean'),
        actual=('goal', 'mean'),
        n=('goal', 'size'),
    )
    return [
        {'predicted': round(float(r.predicted), 4), 'actual': round(float(r.actual), 4), 'n': int(r.n)}
        for r in grouped.itertuples()
    ]


def run_analysis(model_path=None, n_calibration_bins=10):
    """Score every comparison dataset with the trained model.

    Returns a dict keyed by dataset label with the scored dataframe, its
    metrics and a calibration curve, plus the comparison table used by the
    CLI printout. Raises FileNotFoundError if the model or a dataset CSV is
    missing.
    """
    if model_path is None:
        model_path = BASE_DIR / 'xg_spatial_model_male.pkl'
    bundle = joblib.load(model_path)
    model = bundle['model']
    train_features = bundle['features']

    results = {}
    for label, csv_path in DATASETS.items():
        df_raw, X = preprocess_df(pd.read_csv(BASE_DIR / csv_path), train_features)
        df_raw['predicted_xg'] = model.predict_proba(X)[:, 1]
        metrics = get_metrics(df_raw)
        results[label] = {
            'df': df_raw,
            'metrics': metrics,
            'calibration': calibration_curve(df_raw, n_calibration_bins),
        }

    return results


def comparison_table(results):
    def fmt(metrics):
        return [
            metrics['shots'],
            metrics['goals'],
            round(metrics['total_xg_model'], 2),
            round(metrics['total_xg_sb'], 2),
            round(metrics['avg_distance'], 2),
            f"{metrics['real_conversion']:.2f}%",
            f"{metrics['model_conversion']:.2f}%",
            f"{metrics['sb_conversion']:.2f}%" if metrics['sb_conversion'] is not None else "N/A",
        ]

    return pd.DataFrame({
        'Metric': METRIC_LABELS,
        **{label: fmt(r['metrics']) for label, r in results.items()},
    })


def main():
    print("Caricamento del modello salvato (.pkl) e dei dataset di confronto...")
    try:
        results = run_analysis()
    except FileNotFoundError as e:
        print(f"Errore nel trovare i file necessari: {e}")
        return

    print("\n" + "=" * 85)
    print("       CONFRONTO COMPLETO: MODELLO PROPRIO vs STATSBOMB BENCHMARK")
    print("=" * 85)
    print(comparison_table(results).to_string(index=False))
    print("=" * 85)


if __name__ == "__main__":
    main()
