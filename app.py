import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

model = joblib.load('model_base.pkl')  # O 'model_improved.pkl' si usas generativa
le = joblib.load('label_encoder.pkl')

st.title('Predicción Mundial 2026 ⚽')

# Carga teams únicos de ambos CSV para dropdown completo
future_df_load = pd.read_csv('future_match_probabilities_baseline.csv', encoding='utf-8')
all_teams = sorted(set(future_df_load['home_team'].unique()).union(future_df_load['away_team'].unique()))

home_team = st.selectbox('Equipo Local', all_teams)
away_team = st.selectbox('Equipo Visitante', all_teams)
elo_diff = st.number_input('Diferencia ELO (local - visitante)', value=0)
neutral = st.checkbox('Partido Neutral', value=True)
injury_home = st.checkbox('Lesión Local')
injury_away = st.checkbox('Lesión Visitante')

if st.button('Predecir'):
    try:
        home_enc = le.transform([home_team])[0]
        away_enc = le.transform([away_team])[0]
    except ValueError as e:
        st.error(f"Error: Equipo no reconocido. Verifica acentos (e.g., 'Curaçao' con ç). Detalle: {e}")
    else:
        input_data = pd.DataFrame({
            'home_team_enc': [home_enc],
            'away_team_enc': [away_enc],
            'neutral': [1 if neutral else 0],
            'home_injury_flag': [1 if injury_home else 0],
            'away_injury_flag': [1 if injury_away else 0],
            'elo_diff': [elo_diff]
        })
        probs = model.predict_proba(input_data)[0]
        st.write(f"Victoria Local ({home_team}): {probs[0]:.2%}")
        st.write(f"Empate: {probs[1]:.2%}")
        st.write(f"Victoria Visitante ({away_team}): {probs[2]:.2%}")

# Viz integrada
future_df = pd.read_csv('predictions_worldcup2026.csv', encoding='utf-8')
st.plotly_chart(px.bar(future_df, x='group', y='p_home_win_ml', color='home_team', title='Probs Victoria Local por Grupo'))