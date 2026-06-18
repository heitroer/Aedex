import os
import requests
import pandas as pd
import numpy as np
import datetime
from datetime import date
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuracao de ambiente
DIRETORIO_SAIDA = '/mnt/user-data/outputs'
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
hoje = date.today().strftime('%Y%m%d')

print("Aedex - Sistema de Previsao")

# Coleta de dados via API
ano_atual = datetime.datetime.now().year
url = f"https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=52&ey_end={ano_atual}"

try:
    resp = requests.get(url, timeout=25)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
except Exception as e:
    raise SystemExit(f"Erro na API: {e}")

# Pre-processamento
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

# Imputacao de dados climaticos
for col in ['tempmed', 'umidmed', 'tempmin', 'tempmax', 'umidmin', 'umidmax']:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

# Engenharia de atributos
df['log_casos'] = np.log1p(df['casos'])
df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)

for lag in range(1, 9): df[f'log_lag{lag}'] = df['log_casos'].shift(lag)
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
    if col in df.columns: df[f'{col}_lag1'] = df[col].shift(1)

feature_cols = ['sem_seno', 'sem_cos', 'log_lag1', 'log_lag2', 'log_lag3', 'log_lag4',
                'log_lag5', 'log_lag6', 'log_lag7', 'log_lag8', 'media_mov_4sem', 'media_mov_8sem']
for col in ['temp_lag2', 'temp_lag4', 'temp_4sem', 'umid_lag2', 'umid_lag4', 'p_rt1_lag1', 'Rt_lag1', 'nivel_lag1']:
    if col in df.columns: feature_cols.append(col)

df = df.dropna(subset=feature_cols + ['log_casos']).reset_index(drop=True)

# Treino e Validacao
X, y = df[feature_cols].to_numpy(), df['log_casos'].to_numpy()
anos = df['ano'].to_numpy(dtype=int)
mask_2023 = (anos >= 2023)
idx_corte = int(np.argmax(mask_2023)) if mask_2023.sum() >= 10 else int(len(X) * 0.80)

X_train, X_test = X[:idx_corte], X[idx_corte:]
y_train, y_test = y[:idx_corte], y[idx_corte:]

modelo_val = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.03, verbosity=0, random_state=42)
modelo_val.fit(X_train, y_train)

# Métricas
pred_log_test = modelo_val.predict(X_test)
real_casos, pred_casos = np.expm1(y_test), np.expm1(pred_log_test)
residuos = real_casos - pred_casos

r2_log = r2_score(y_test, pred_log_test) * 100
r2_orig = r2_score(real_casos, pred_casos) * 100
mae_orig = mean_absolute_error(real_casos, pred_casos)

print(f"R2(log): {r2_log:.1f}% | R2(orig): {r2_orig:.1f}% | MAE: {mae_orig:.0f}")

# Modelo de Producao e Projecao
modelo_prod = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.03, verbosity=0, random_state=42)
modelo_prod.fit(X, y)

historico_log = df['log_casos'].tail(8).tolist()
ultima_linha = df.iloc[-1].copy()
features_atuais = ultima_linha[feature_cols].copy()
sem_sim = int(ultima_linha['semana'])
log_futuro = []

for _ in range(4):
    sem_sim = sem_sim + 1 if sem_sim < 52 else 1
    features_atuais['sem_seno'], features_atuais['sem_cos'] = np.sin(2 * np.pi * sem_sim / 52.0), np.cos(2 * np.pi * sem_sim / 52.0)
    for lag in range(1, 9): features_atuais[f'log_lag{lag}'] = historico_log[-lag]
    features_atuais['media_mov_4sem'], features_atuais['media_mov_8sem'] = np.mean(historico_log[-4:]), np.mean(historico_log[-8:])

    p = float(modelo_prod.predict(features_atuais.to_frame().T.to_numpy())[0])
    log_futuro.append(p)
    historico_log.append(p)

# Grafico SHAP
try:
    import shap
    explainer = shap.TreeExplainer(modelo_val)
    shap_values = explainer.shap_values(X_test)
    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values().tail(10)
    ax_s.barh(imp.index, imp.values, color=plt.cm.RdBu(np.linspace(0.2, 0.8, len(imp))))
    ax_s.set_title('Aedex - SHAP Importance')
    plt.savefig(os.path.join(DIRETORIO_SAIDA, f'aedex_shap_{hoje}.png'))
    plt.close()
except ImportError: pass

# Dashboard visual
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('#0f1923')
gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)
C_R, C_P, C_F, C_T, C_G = '#58a6ff', '#f78166', '#161b22', '#e6edf3', '#30363d'

# Plot Linha do Tempo
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(C_F)
ax1.plot(range(len(y_train)), y_train, color='#8b949e', alpha=0.7)
ax1.plot(range(idx_corte, len(y)), y_test, color=C_R)
ax1.plot(range(idx_corte, len(y)), pred_log_test, color=C_P, ls='--')
ax1.plot(range(len(y)-1, len(y)+4), [y[-1]] + log_futuro, color='#39ff14', marker='o')
ax1.set_title('Aedex - Previsao Log(Casos)', color=C_T)

# Escala Original
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(C_F)
ax2.plot(real_casos, color=C_R)
ax2.plot(pred_casos, color=C_P, ls='--')
ax2.set_title('Escala Original', color=C_T)

# Importancia de Features
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(C_F)
imp_xgb = pd.Series(modelo_val.feature_importances_, index=feature_cols).sort_values().tail(10)
ax3.barh(imp_xgb.index, imp_xgb.values, color='steelblue')
ax3.set_title('XGBoost Importance', color=C_T)

# Residuos
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor(C_F)
ax4.bar(range(len(residuos)), residuos, color=[C_P if r > 0 else C_R for r in residuos])
ax4.set_title('Residuos', color=C_T)

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(colors=C_T)
    ax.grid(True, alpha=0.1, color=C_G)
    for s in ax.spines.values(): s.set_edgecolor(C_G)

plt.savefig(os.path.join(DIRETORIO_SAIDA, f'aedex_dashboard_{hoje}.png'), facecolor=fig.get_facecolor())
plt.show()

# Export final
joblib.dump(modelo_prod, 'modelo_aedex.pkl')
print("Processo concluido.")
