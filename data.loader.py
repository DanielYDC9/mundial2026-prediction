import pandas as pd

def load_data():
    try:
        future_df = pd.read_csv('future_match_probabilities_baseline.csv')
    except FileNotFoundError:
        raise FileNotFoundError("No se encontró 'future_match_probabilities_baseline.csv'. Asegúrate de crearlo con el contenido que proporcionaste.")

    try:
        hist_df = pd.read_csv('historical_matches.csv')
    except FileNotFoundError:
        raise FileNotFoundError("No se encontró 'historical_matches.csv'. Descárgalo de Kaggle: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 y guárdalo en la carpeta.")

    # Preprocesa histórico: Calcula resultado (0=win_home, 1=draw, 2=win_away)
    # Asume columnas: home_score, away_score (ajusta si tu CSV histórico tiene nombres diferentes, e.g., 'home_goals', 'away_goals')
    hist_df['resultado'] = 0
    hist_df.loc[hist_df['home_score'] == hist_df['away_score'], 'resultado'] = 1
    hist_df.loc[hist_df['home_score'] < hist_df['away_score'], 'resultado'] = 2

    # Simula ELO si no está en histórico (para este ejemplo; en real, usa una librería como elopy si necesitas calcularlo)
    if 'home_elo' not in hist_df.columns:
        hist_df['home_elo'] = 1700  # Valor promedio placeholder
        hist_df['away_elo'] = 1700
    hist_df['elo_diff'] = hist_df['home_elo'] - hist_df['away_elo']
    hist_df['home_injury_flag'] = 0  # Placeholder
    hist_df['away_injury_flag'] = 0

    # Maneja NaNs en future_df
    future_df['home_elo'].fillna(1700, inplace=True)
    future_df['away_elo'].fillna(1700, inplace=True)
    future_df['elo_diff'].fillna(0, inplace=True)
    future_df['home_injury_flag'].fillna(0, inplace=True)
    future_df['away_injury_flag'].fillna(0, inplace=True)

    return hist_df, future_df