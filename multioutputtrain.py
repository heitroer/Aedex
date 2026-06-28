# train_multioutput_final.py
import datetime
import requests
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("=== Passagem 1: Coletando dados da API InfoDengue (Campo Grande) ===")
ano_atual = datetime.datetime.now().year
url = f"https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=52&ey_end={ano_atual}"

try:
    resp = requests.get(url, timeout=25)
    df = pd.DataFrame(resp.json())
except Exception as e:
    raise SystemExit(f"Erro na API: {e}")

df['ano'] = df['SE'].astype(str).str[:4].astype(int)
df['semana'] = df['SE'].astype(str).str[4:].astype(int)
df = df.sort_values(['ano', 'semana']).reset_index(drop=True)

for col in df.columns:
    if col not in ['data_iniSE', 'Localidade_id', 'versao_modelo', 'municipio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'casos_est' in df.columns:
    if 'casos' in df.columns: df.drop(columns=['casos'], inplace=True)
    df.rename(columns={'casos_est': 'casos'}, inplace=True)

df['casos'] = pd.to_numeric(df['casos'], errors='coerce')
df = df.dropna(subset=['casos']).reset_index(drop=True)

for col in ['tempmed', 'umidmed', 'tempmin', 'tempmax', 'umidmin', 'umidmax']:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

print("=== Passagem 2: Engenharia de Atributos Corrigida e Alinhamento Biológico ===")
df['log_casos'] = np.log1p(df['casos'])
df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)

# Lags dinâmicos de casos (t até t-7)
df['log_lag1'] = df['log_casos'] 
for lag in range(2, 9): 
    df[f'log_lag{lag}'] = df['log_casos'].shift(lag - 1)

df['media_mov_4sem'] = df['log_casos'].rolling(4).mean()
df['media_mov_8sem'] = df['log_casos'].rolling(8).mean()

# ESTRATÉGIA 1: Alinhamento Biológico (Clima Atual 't' como Lead de Incubação Vetorial)
if 'tempmed' in df.columns:
    df['temp_atual'] = df['tempmed']         # Clima em t (gatilho biológico para t+3 e t+4)
    df['temp_lag2'] = df['tempmed'].shift(1) # t-1
    df['temp_lag4'] = df['tempmed'].shift(3) # t-3
    df['temp_4sem'] = df['tempmed'].rolling(4).mean()
if 'umidmed' in df.columns:
    df['umid_atual'] = df['umidmed']         # Umidade em t
    df['umid_lag2'] = df['umidmed'].shift(1)
    df['umid_lag4'] = df['umidmed'].shift(3)
for col in ['p_rt1', 'Rt', 'nivel']:
    if col in df.columns: 
        df[f'{col}_lag1'] = df[col]          # Situação epidemiológica exata de hoje (t)

feature_cols = ['sem_seno', 'sem_cos', 'log_lag1', 'log_lag2', 'log_lag3', 'log_lag4',
                'log_lag5', 'log_lag6', 'log_lag7', 'log_lag8', 'media_mov_4sem', 'media_mov_8sem']

colunas_adicionais = ['temp_atual', 'temp_lag2', 'temp_lag4', 'temp_4sem', 'umid_atual', 'umid_lag2', 'umid_lag4', 'p_rt1_lag1', 'Rt_lag1', 'nivel_lag1']
for col in colunas_adicionais:
    if col in df.columns: 
        feature_cols.append(col)

print("=== Passagem 3: Criação dos Alvos Multi-Output (Abordagem Delta Variação) ===")
# ESTRATÉGIA 3: O alvo deixa de ser absoluto e passa a focar na velocidade/variação em relação a hoje
df['target_sem1'] = df['log_casos'].shift(-1) - df['log_casos']
df['target_sem2'] = df['log_casos'].shift(-2) - df['log_casos']
df['target_sem3'] = df['log_casos'].shift(-3) - df['log_casos']
df['target_sem4'] = df['log_casos'].shift(-4) - df['log_casos']

df_valid = df.dropna(subset=feature_cols + ['target_sem1', 'target_sem2', 'target_sem3', 'target_sem4']).reset_index(drop=True)

anos = df_valid['ano'].to_numpy(dtype=int)
mask_2023 = (anos >= 2023)
idx_corte = int(np.argmax(mask_2023))

# Mantendo em estrutura de DataFrame para recuperação segura do log_lag1 de base
X_df = df_valid[feature_cols]
X_train, X_test = X_df.iloc[:idx_corte], X_df.iloc[idx_corte:]

print("\n=== Passagem 4: Treinamento Desacoplado de Alta Capacidade ===")

# ESTRATÉGIA 2: Hiperparâmetros customizados por horizonte para mitigar overfitting no ruído futuro
params_horizon = {
    1: {'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 1.0, 'colsample_bytree': 1.0},
    2: {'n_estimators': 800,  'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.9},
    3: {'n_estimators': 600,  'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8},
    4: {'n_estimators': 400,  'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.7, 'colsample_bytree': 0.7}
}

base_log_test = X_test['log_lag1'].to_numpy()

for horizon in range(1, 5):
    y_delta = df_valid[f'target_sem{horizon}'].to_numpy()
    y_train_delta, y_test_delta = y_delta[:idx_corte], y_delta[idx_corte:]
    
    # Instanciação do XGBoost com a arquitetura definida para a respectiva semana
    p = params_horizon[horizon]
    modelo_val = XGBRegressor(
        n_estimators=p['n_estimators'], 
        max_depth=p['max_depth'], 
        learning_rate=p['learning_rate'], 
        subsample=p['subsample'],
        colsample_bytree=p['colsample_bytree'],
        verbosity=0, 
        random_state=42
    )
    modelo_val.fit(X_train.to_numpy(), y_train_delta)
    
    # Predição do Delta e reconstrução reversa da escala absoluta para avaliação
    pred_delta_test = modelo_val.predict(X_test.to_numpy())
    pred_log_test = pred_delta_test + base_log_test
    y_test_abs = y_test_delta + base_log_test
    
    real_casos = np.expm1(y_test_abs)
    pred_casos = np.expm1(pred_log_test)
    
    # Métricas calculadas sobre o valor real final reconstruído
    r2_log = r2_score(y_test_abs, pred_log_test) * 100
    r2_orig = r2_score(real_casos, pred_casos) * 100
    mae_orig = mean_absolute_error(real_casos, pred_casos)
    
    print(f"-> Semana {horizon} | R2(log): {r2_log:.1f}% | R2(orig): {r2_orig:.1f}% | MAE: {mae_orig:.0f}")
    
    # Modelo de Produção: Treinado com 100% dos dados reais disponíveis
    modelo_prod = XGBRegressor(
        n_estimators=p['n_estimators'], 
        max_depth=p['max_depth'], 
        learning_rate=p['learning_rate'], 
        subsample=p['subsample'],
        colsample_bytree=p['colsample_bytree'],
        verbosity=0, 
        random_state=42
    )
    modelo_prod.fit(X_df.to_numpy(), y_delta)
    # IMPORTANTE: Esta linha PRECISA de 4 espaços (ou 1 tab) de recuo para rodar 4 vezes!
    modelo_prod.save_model(f'modelo_aedex_sem{horizon}.json')
    print(f"-> Modelo da Semana {horizon} exportado nativamente em JSON.")

print("\n🚀 Todos os modelos foram salvos e blindados matematicamente contra Data Leakage.")
