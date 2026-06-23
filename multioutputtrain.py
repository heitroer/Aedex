# train_multioutput.py
import datetime
import requests
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

print("=== Passagem 1: Coletando dados da API InfoDengue ===")
ano_atual = datetime.datetime.now().year
url = f"https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=53&ey_end={ano_atual}"

try:
    resp = requests.get(url, timeout=30)
    df = pd.DataFrame(resp.json())
    df['ano'] = df['SE'].astype(str).str[:4].astype(int)
    df['semana'] = df['SE'].astype(str).str[4:].astype(int)
    df = df.sort_values(['ano', 'semana']).reset_index(drop=True)
except Exception as e:
    print(f"Erro ao coletar dados: {e}")
    exit()

# Conversão numérica e tratamento de NaNs
for col in df.columns:
    if col not in ['data_iniSE', 'Localidade_id', 'versao_modelo', 'municipio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'casos_est' in df.columns:
    df['casos'] = df['casos_est']

df = df.dropna(subset=['casos']).reset_index(drop=True)

# Interpolação de variáveis meteorológicas
for col in ['tempmed', 'umidmed', 'tempmin', 'tempmax', 'umidmin', 'umidmax']:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

print("=== Passagem 2: Engenharia de Recursos (Features) ===")
df['log_casos'] = np.log1p(df['casos'])
df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)

# Criação de Lags e Médias Móveis
for lag in range(1, 9): 
    df[f'log_lag{lag}'] = df['log_casos'].shift(lag)
    
df['media_mov_4sem'] = df['log_casos'].shift(1).rolling(4).mean()
df['media_mov_8sem'] = df['log_casos'].shift(1).rolling(8).mean()

if 'tempmed' in df.columns:
    df['temp_lag2'] = df['tempmed'].shift(2)
    df['temp_lag4'] = df['tempmed'].shift(4)
    df['temp_4sem'] = df['tempmed'].rolling(4).mean().shift(2)
if 'umidmed' in df.columns:
    df['umid_lag2'] = df['umidmed'].shift(2)
    df['umid_lag4'] = df['umidmed'].shift(4)
for col in ['p_rt1', 'Rt', 'nivel']:
    if col in df.columns: 
        df[f'{col}_lag1'] = df[col].shift(1)

feature_cols = [
    'sem_seno', 'sem_cos', 'log_lag1', 'log_lag2', 'log_lag3', 'log_lag4',
    'log_lag5', 'log_lag6', 'log_lag7', 'log_lag8', 'media_mov_4sem', 'media_mov_8sem',
    'temp_lag2', 'temp_lag4', 'temp_4sem', 'umid_lag2', 'umid_lag4', 'p_rt1_lag1', 'Rt_lag1', 'nivel_lag1'
]

print("=== Passagem 3: Engenharia de Targets (Multi-Output) ===")
# Alvos deslocados negativamente apontando para o futuro
df['target_sem1'] = df['log_casos'].shift(-1)
df['target_sem2'] = df['log_casos'].shift(-2)
df['target_sem3'] = df['log_casos'].shift(-3)
df['target_sem4'] = df['log_casos'].shift(-4)

# Divisão Temporal: Treino (Até 2024) e Teste/Validação (2025 em diante)
df_treino = df[df['ano'] <= 2024].dropna(subset=feature_cols + ['target_sem1', 'target_sem2', 'target_sem3', 'target_sem4']).reset_index(drop=True)
df_teste = df[df['ano'] >= 2025].dropna(subset=feature_cols + ['target_sem1', 'target_sem2', 'target_sem3', 'target_sem4']).reset_index(drop=True)

X_train = df_treino[feature_cols]
X_test = df_teste[feature_cols]

print(f"Instâncias de Treino: {len(df_treino)} | Instâncias de Validação/Teste: {len(df_teste)}")
print("\n=== Passagem 4: Treinamento e Cálculo de R² por Horizonte ===")

r2_scores = {}

for horizon in range(1, 5):
    y_train = df_treino[f'target_sem{horizon}']
    y_test = df_teste[f'target_sem{horizon}']
    
    # Instanciação do XGBoost parametrizado para evitar overfitting
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    # Predições de teste (em escala logarítmica)
    preds = model.predict(X_test)
    
    # Cálculo do R² na escala log original de modelagem
    score_r2 = r2_score(y_test, preds)
    r2_scores[horizon] = score_r2
    
    # Salvamento do modelo específico para o horizonte temporal correspondente
    filename = f"modelo_aedex_sem{horizon}.pkl"
    joblib.dump(model, filename)
    print(f"-> Modelo Semana {horizon} salvo como '{filename}' | R² de Teste: {score_r2 * 100:.2f}%")

print("\nTreinamento concluído com sucesso. Os 4 modelos estão prontos para o ambiente de produção.")
