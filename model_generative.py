import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import joblib

# Cargar datos y modelo base
historical_df = pd.read_csv('historical_matches.csv')
historical_df['date'] = pd.to_datetime(historical_df['date'])
historical_df = historical_df[historical_df['date'] >= '2000-01-01'].copy()

historical_df['result'] = 0
historical_df.loc[historical_df['home_score'] == historical_df['away_score'], 'result'] = 1
historical_df.loc[historical_df['home_score'] < historical_df['away_score'], 'result'] = 2

historical_df['neutral'] = historical_df['neutral'].map({'FALSE': 0, 'TRUE': 1}).fillna(0).astype(int)
historical_df['elo_diff'] = 0  # Placeholder

le = joblib.load('label_encoder.pkl')
historical_df['home_team_enc'] = le.transform(historical_df['home_team'])
historical_df['away_team_enc'] = le.transform(historical_df['away_team'])

features = ['home_team_enc', 'away_team_enc', 'neutral', 'elo_diff']
X = historical_df[features]
y = historical_df['result']

# Generar datos sintéticos con CTGAN
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(historical_df[features + ['result']])

synthesizer = CTGANSynthesizer(metadata, epochs=50)  # Ajusta epochs para más calidad
synthesizer.fit(historical_df[features + ['result']])

synthetic_data = synthesizer.sample(5000)  # Genera 5000 rows sintéticas

# Augmentar datos
X_aug = pd.concat([X, synthetic_data[features]], ignore_index=True)
y_aug = pd.concat([y, synthetic_data['result']], ignore_index=True)

X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug)

# Re-entrenar modelo mejorado
model_improved = XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
model_improved.fit(X_train_aug, y_train_aug)

# Evaluar mejora
probs_aug = model_improved.predict_proba(X_test_aug)
preds_aug = model_improved.predict(X_test_aug)
print(f"Accuracy mejorada: {accuracy_score(y_test_aug, preds_aug):.4f}")
print(f"Log Loss mejorada: {log_loss(y_test_aug, probs_aug):.4f}")

# Guardar modelo mejorado
joblib.dump(model_improved, 'model_improved.pkl')

# Predicciones en future (carga future_df como en base)
future_df = pd.read_csv('future_match_probabilities_baseline.csv')
future_df['neutral'] = 1
future_df['home_team_enc'] = le.transform(future_df['home_team'])
future_df['away_team_enc'] = le.transform(future_df['away_team'])
future_df['elo_diff'] = future_df['elo_diff'].fillna(0)

future_X = future_df[features]
future_probs_improved = model_improved.predict_proba(future_X)
future_df[['p_home_win_gen', 'p_draw_gen', 'p_away_win_gen']] = future_probs_improved
future_df.to_csv('predictions_improved.csv', index=False)
print("Predicciones mejoradas guardadas en predictions_improved.csv")