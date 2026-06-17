import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import PIL.Image as Image

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Aedex - Monitoramento Inteligente", layout="wide", initial_sidebar_state="expanded")

# 2. CSS PARA TEMA ESCURO
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] { background-color: #1e293b; border: 1px solid #334155; padding: 1.2rem; border-radius: 12px; }
    .protocol-container { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; }
    h1, h2, h3, h4 { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

# 3. BARRA LATERAL (SIDEBAR)
with st.sidebar:
    st.markdown("<h2 style='color:#3b82f6; text-align: center;'>Aedex</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color:#94a3b8; margin-bottom: 20px;'>Painel Operacional Avançado</div>", unsafe_allow_html=True)
    st.divider()
    st.info("Monitoramento preditivo de arboviroses em Campo Grande/MS.")
    st.success("Conexão com API: Estável")

# 4. CABEÇALHO
col_logo, col_titulo = st.columns([1, 6])
with col_titulo:
    st.markdown("<h1 style='padding-bottom: 0;'>Monitoramento Epidemiológico Inteligente</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Projeção de casos de Dengue e gestão de risco em tempo real.</p>", unsafe_allow_html=True)

# 5. LÓGICA DE BACKEND
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_aedex.pkl")

@st.cache_data(ttl=86400)
def puxar_dados_api():
    url = "https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=52&ey_end=2026"
    try:
        resp = requests.get(url, timeout=25)
        df = pd.DataFrame(resp.json())
        df['ano'] = df['SE'].astype(str).str[:4].astype(int)
        df['semana'] = df['SE'].astype(str).str[4:].astype(int)
        df = df.sort_values(['ano', 'semana']).reset_index(drop=True)
        # Limpeza básica
        df['casos'] = pd.to_numeric(df['casos_est'], errors='coerce')
        return df.dropna(subset=['casos']).reset_index(drop=True)
    except: return None

df = puxar_dados_api()
modelo_prod = carregar_modelo()

# Processamento
df['log_casos'] = np.log1p(df['casos'])
df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)
for lag in range(1, 17): df[f'log_lag{lag}'] = df['log_casos'].shift(lag)
df['media_mov_4sem'] = df['log_casos'].shift(1).rolling(4).mean()
df['media_mov_8sem'] = df['log_casos'].shift(1).rolling(8).mean()
feature_cols = ['sem_seno', 'sem_cos'] + [f'log_lag{i}' for i in range(1, 17)] + ['media_mov_4sem', 'media_mov_8sem']
df_limpo = df.dropna(subset=feature_cols + ['log_casos']).reset_index(drop=True)
df_limpo['predicao_casos'] = np.expm1(modelo_prod.predict(df_limpo[feature_cols].astype(float))).astype(int)

# Projeção Futura
historico_log = df_limpo['log_casos'].tail(16).tolist()
ultima_linha = df_limpo.iloc[-1].copy()
features_atuais = ultima_linha[feature_cols].copy()
sem_sim, ano_sim = int(ultima_linha['semana']), int(ultima_linha['ano'])
log_futuro, labels_futuro = [], []

for _ in range(4):
    sem_sim += 1
    if sem_sim > 52: sem_sim = 1; ano_sim += 1
    labels_futuro.append(f"{sem_sim}/{ano_sim}")
    features_atuais['sem_seno'] = np.sin(2 * np.pi * sem_sim / 52.0)
    features_atuais['sem_cos'] = np.cos(2 * np.pi * sem_sim / 52.0)
    for lag in range(1, 17): features_atuais[f'log_lag{lag}'] = historico_log[-lag]
    features_atuais['media_mov_4sem'] = np.mean(historico_log[-4:])
    features_atuais['media_mov_8sem'] = np.mean(historico_log[-8:])
    p = float(modelo_prod.predict(pd.DataFrame([features_atuais])[feature_cols].astype(float))[0])
    log_futuro.append(p); historico_log.append(p)

casos_projetados_fim = int(np.expm1(log_futuro[-1]))

# Indicadores de Risco
if casos_projetados_fim < 50: status, cor = "AZUL — ESTÁVEL", "#3b82f6"
elif casos_projetados_fim < 150: status, cor = "AMARELO — ATENÇÃO", "#eab308"
elif casos_projetados_fim < 400: status, cor = "LARANJA — ALERTA", "#f97316"
else: status, cor = "VERMELHO — CRISE", "#ef4444"

# 6. DASHBOARD UI
st.markdown("### Visão Geral do Risco")
col1, col2, col3 = st.columns(3)
col1.metric("Casos Estimados (4 sem.)", f"{casos_projetados_fim}")
col2.metric("Última Atualização", f"Semana {int(ultima_linha['semana'])}/{int(ultima_linha['ano'])}")
col3.markdown(f"**Nível de Risco**<br><div style='background-color: {cor}; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. GRÁFICO COM SELETOR DE TEMPO
st.markdown("### Validação Histórica e Horizonte Preditivo")

# SELETOR DE JANELA TEMPORAL
col_sel, _ = st.columns([1, 3])
with col_sel:
    janela = st.selectbox("Selecione o período de visualização (semanas):", [4, 12, 26, 52], index=2)

df_ultimas = df_limpo.tail(janela).copy()
df_ultimas['label'] = df_ultimas['semana'].astype(str) + "/" + df_ultimas['ano'].astype(str)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ultimas['label'], y=df_ultimas['casos'], mode='lines+markers', name='Casos Reais', line=dict(color='#3b82f6', width=3)))
fig.add_trace(go.Scatter(x=df_ultimas['label'], y=df_ultimas['predicao_casos'], mode='lines', name='Predição (Passado)', line=dict(color='#10b981', width=2, dash='dot')))

# Projeção futura no gráfico
labels_futuro_com_ultimo = [df_ultimas['label'].iloc[-1]] + labels_futuro
casos_futuros_reais = [df_ultimas['casos'].iloc[-1]] + [np.expm1(x) for x in log_futuro]
fig.add_trace(go.Scatter(x=labels_futuro_com_ultimo, y=casos_futuros_reais, mode='lines+markers', name='Projeção Aedex', line=dict(color='#f97316', width=4, dash='dash')))

fig.update_layout(
    plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font=dict(color='#f8fafc'),
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#334155'), yaxis=dict(showgrid=True, gridcolor='#334155')
)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
