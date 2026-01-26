import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Cargando datos... (puede tardar 20-40 segundos la primera vez)")

# 1. Cargar datos con UTF-8 para caracteres especiales
future_df = pd.read_csv('future_match_probabilities_baseline.csv', encoding='utf-8')
historical_df = pd.read_csv('historical_matches.csv', encoding='utf-8')  # O 'historical_matches.csv'

print(f"Histórico columnas: {historical_df.columns.tolist()}")
print(f"Histórico shape: {historical_df.shape}")
print(f"Futuro columnas: {future_df.columns.tolist()}")
print(f"Futuro shape: {future_df.shape}")

# Normaliza nombres de equipos (e.g., 'Curasao' -> 'Curaçao')
historical_df['home_team'] = historical_df['home_team'].str.replace('Curasao', 'Curaçao', case=False)
historical_df['away_team'] = historical_df['away_team'].str.replace('Curasao', 'Curaçao', case=False)
future_df['home_team'] = future_df['home_team'].str.replace('Curasao', 'Curaçao', case=False)
future_df['away_team'] = future_df['away_team'].str.replace('Curasao', 'Curaçao', case=False)

# 2. Preprocesar histórico
historical_df['date'] = pd.to_datetime(historical_df['date'])
historical_df = historical_df[historical_df['date'] >= '2000-01-01'].copy()

historical_df['result'] = 0
historical_df.loc[historical_df['home_score'] == historical_df['away_score'], 'result'] = 1
historical_df.loc[historical_df['home_score'] < historical_df['away_score'], 'result'] = 2

historical_df['neutral'] = historical_df['neutral'].map({'FALSE': 0, 'TRUE': 1}).fillna(0).astype(int)

# 3. Añadir neutral a future
future_df['neutral'] = 1

# 4. Codificar equipos (todos únicos, incluyendo variaciones)
all_teams = pd.concat([historical_df['home_team'], historical_df['away_team'], 
                       future_df['home_team'], future_df['away_team']]).unique()
le = LabelEncoder()
le.fit(all_teams)

historical_df['home_team_enc'] = le.transform(historical_df['home_team'])
historical_df['away_team_enc'] = le.transform(historical_df['away_team'])

future_df['home_team_enc'] = le.transform(future_df['home_team'])
future_df['away_team_enc'] = le.transform(future_df['away_team'])

# 5. Features (añadí injury flags como 0, para checkboxes en app)
historical_df['home_injury_flag'] = 0
historical_df['away_injury_flag'] = 0
future_df['home_injury_flag'] = 0
future_df['away_injury_flag'] = 0

features = ['home_team_enc', 'away_team_enc', 'neutral', 'home_injury_flag', 'away_injury_flag']

# Añadir elo_diff
historical_df['elo_diff'] = 0
future_df['elo_diff'] = future_df['elo_diff'].fillna(0)
features += ['elo_diff']

# 6. Preparar datos
X = historical_df[features]
y = historical_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Entrenar
model = XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 8. Evaluar
probs_test = model.predict_proba(X_test)
preds_test = model.predict(X_test)
print("\n📊 Resultados del modelo:")
print(f"Accuracy: {accuracy_score(y_test, preds_test):.4f}")
print(f"Log Loss:  {log_loss(y_test, probs_test):.4f}")

# 9. Predecir
future_X = future_df[features]
future_probs = model.predict_proba(future_X)
future_df[['p_home_win_ml', 'p_draw_ml', 'p_away_win_ml']] = future_probs

# 10. Guardar
future_df.to_csv('predictions_worldcup2026.csv', index=False, encoding='utf-8')
joblib.dump(model, 'model_base.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\n✅ ¡Listo! Re-ejecuta la app.")