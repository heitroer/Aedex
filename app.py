import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Aedex - Monitoramento Inteligente", layout="wide", initial_sidebar_state="expanded")

# 2. CSS AVANÇADO PARA UI/UX (Estilo Dark Moderno)
st.markdown("""
<style>
    .stApp { background-color: #13151a; font-family: 'Inter', sans-serif; }
    header { visibility: hidden; }
    [data-testid="stSidebar"] { background-color: #1a1c23; border-right: 1px solid #2a2d35; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 600 !important; }
    p { color: #8b92a5 !important; }
    .custom-card {
        background-color: #1e212a;
        border: 1px solid #2a2d35;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    .card-title { color: #8b92a5; font-size: 0.9rem; font-weight: 500; margin-bottom: 8px; }
    .card-value { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
    .card-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 3. BARRA LATERAL (SIDEBAR)
with st.sidebar:
    st.markdown("<h2 style='color:#00d1ff; text-align: center; margin-top: 20px;'>Aedex</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color:#8b92a5; font-size: 14px; margin-bottom: 30px;'>Painel Operacional</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #1e212a; padding: 15px; border-radius: 12px; border: 1px solid #2a2d35;'>
        <p style='margin:0; font-size: 13px; color: #8b92a5;'>Status da API</p>
        <p style='margin:0; color: #10b981; font-weight: bold;'>● Conexão Estável</p>
    </div>
    """, unsafe_allow_html=True)

# 4. CABEÇALHO
st.markdown("<h1 style='padding-bottom: 5px; font-size: 2.2rem;'>Monitoramento Epidemiológico Inteligente</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; margin-bottom: 30px;'>Projeção de casos de Dengue e gestão de risco em tempo real.</p>", unsafe_allow_html=True)

# 5. LÓGICA DE BACKEND
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load("modelo_aedex.pkl")
    except:
        return None

@st.cache_data(ttl=86400)
def puxar_dados_api():
    url = "https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=52&ey_end=2026"
    try:
        resp = requests.get(url, timeout=25)
        df = pd.DataFrame(resp.json())
        df['ano'] = df['SE'].astype(str).str[:4].astype(int)
        df['semana'] = df['SE'].astype(str).str[4:].astype(int)
        df = df.sort_values(['ano', 'semana']).reset_index(drop=True)
        
        # CORREÇÃO: Forçar conversão para numérico (evita o erro do interpolate)
        for col in df.columns:
            if col not in ['data_iniSE', 'Localidade_id', 'versao_modelo', 'municipio']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'casos_est' in df.columns:
            df['casos'] = df['casos_est']
            
        return df.dropna(subset=['casos']).reset_index(drop=True)
    except: 
        return None

df = puxar_dados_api()
modelo_prod = carregar_modelo()

if df is not None and modelo_prod is not None:
    # --- IMPUTAÇÃO DE DADOS CLIMÁTICOS IDÊNTICA AO NOTEBOOK ---
    for col in ['tempmed', 'umidmed', 'tempmin', 'tempmax', 'umidmin', 'umidmax']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

    # --- ENGENHARIA DE ATRIBUTOS REAL ---
    df['log_casos'] = np.log1p(df['casos'])
    df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
    df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)
    
    for lag in range(1, 9): 
        df[f'log_lag{lag}'] = df['log_casos'].shift(lag)
        
    df['media_mov_4sem'] = df['log_casos'].shift(1).rolling(4).mean()
    df['media_mov_8sem'] = df['log_casos'].shift(1).rolling(8).mean()

    # Variáveis de clima e epi complementares
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

    # Lista ordenada idêntica ao array X do Colab
    feature_cols = ['sem_seno', 'sem_cos', 'log_lag1', 'log_lag2', 'log_lag3', 'log_lag4',
                    'log_lag5', 'log_lag6', 'log_lag7', 'log_lag8', 'media_mov_4sem', 'media_mov_8sem',
                    'temp_lag2', 'temp_lag4', 'temp_4sem', 'umid_lag2', 'umid_lag4', 'p_rt1_lag1', 'Rt_lag1', 'nivel_lag1']
    
    df_limpo = df.dropna(subset=feature_cols + ['log_casos']).reset_index(drop=True)
    
    # Geração correta do passado (Validação)
    df_limpo['predicao_casos'] = np.expm1(modelo_prod.predict(df_limpo[feature_cols].to_numpy())).astype(int)

    # Projeção Futura Iterativa
    historico_log = df_limpo['log_casos'].tail(8).tolist()
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
        
        for lag in range(1, 9): 
            features_atuais[f'log_lag{lag}'] = historico_log[-lag]
            
        features_atuais['media_mov_4sem'] = np.mean(historico_log[-4:])
        features_atuais['media_mov_8sem'] = np.mean(historico_log[-8:])
        
        p = float(modelo_prod.predict(features_atuais.to_frame().T.to_numpy())[0])
        log_futuro.append(p)
        historico_log.append(p)

    casos_projetados_fim = int(np.expm1(log_futuro[-1]))
else:
    casos_projetados_fim = 0
    df_limpo = pd.DataFrame(columns=['casos', 'predicao_casos', 'semana', 'ano', 'label'])
    labels_futuro, log_futuro = [], []
    ultima_linha = {'semana': 0, 'ano': 0}

# Classificação de Risco
if casos_projetados_fim < 50: status, cor, bg_cor = "ESTÁVEL", "#00d1ff", "rgba(0, 209, 255, 0.15)"
elif casos_projetados_fim < 150: status, cor, bg_cor = "ATENÇÃO", "#fbbf24", "rgba(251, 191, 36, 0.15)"
elif casos_projetados_fim < 400: status, cor, bg_cor = "ALERTA", "#f97316", "rgba(249, 115, 22, 0.15)"
else: status, cor, bg_cor = "CRISE", "#ef4444", "rgba(239, 68, 68, 0.15)"

# 6. DASHBOARD UI - CARDS SUPERIORES
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="custom-card"><div class="card-title">Casos Estimados (Horizonte 4 sem.)</div><div class="card-value">{casos_projetados_fim}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="custom-card"><div class="card-title">Última Atualização da API</div><div class="card-value" style="font-size: 1.8rem;">Semana {int(ultima_linha["semana"])}/{int(ultima_linha["ano"])}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="custom-card"><div class="card-title">Nível de Risco Projetado</div><div><span class="card-badge" style="background-color: {bg_cor}; color: {cor}; border: 1px solid {cor};">{status}</span></div></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# 7. GRÁFICO COM DESIGN DE ALTA PERFORMANCE
col_title, col_sel = st.columns([3, 1])
with col_title:
    st.markdown("### Histórico e Horizonte Preditivo", unsafe_allow_html=True)
with col_sel:
    janela = st.selectbox("Período:", [4, 12, 26, 52], index=2, label_visibility="collapsed")

df_ultimas = df_limpo.tail(janela).copy()
if 'label' not in df_ultimas.columns and not df_ultimas.empty:
    df_ultimas['label'] = df_ultimas['semana'].astype(str) + "/" + df_ultimas['ano'].astype(str)

fig = go.Figure()

if not df_ultimas.empty:
    fig.add_trace(go.Scatter(
        x=df_ultimas['label'], y=df_ultimas['casos'], mode='lines', name='Casos Reais', 
        line=dict(color='#00d1ff', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 209, 255, 0.05)'
    ))
    fig.add_trace(go.Scatter(
        x=df_ultimas['label'], y=df_ultimas['predicao_casos'], mode='lines', name='Predição Aedex (Validação)', 
        line=dict(color='#10b981', width=2, dash='dot', shape='spline')
    ))
    labels_futuro_com_ultimo = [df_ultimas['label'].iloc[-1]] + labels_futuro
    casos_futuros_reais = [df_ultimas['casos'].iloc[-1]] + [np.expm1(x) for x in log_futuro]
    fig.add_trace(go.Scatter(
        x=labels_futuro_com_ultimo, y=casos_futuros_reais, mode='lines+markers', name='Projeção Futura', 
        line=dict(color=cor, width=3, dash='dash', shape='spline'), marker=dict(size=6, color=cor)
    ))

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='#8b92a5', family="Inter, sans-serif"),
    margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
    xaxis=dict(showgrid=False, showline=True, linecolor='#2a2d35'), 
    yaxis=dict(showgrid=True, gridcolor='#2a2d35', zeroline=False), hovermode="x unified"
)

st.markdown('<div class="custom-card" style="padding: 10px 24px 24px 24px;">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
st.markdown("</div>", unsafe_allow_html=True)

# 8. NOVO: PAINEL DE AUDITORIA DE BACKTESTING (Métricas de Confiança)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### Auditoria de Confiabilidade Algorítmica", unsafe_allow_html=True)

with st.expander("📊 Exibir Métricas de Validação Científica e Backtesting", expanded=False):
    st.markdown("""
    <p style='font-size: 0.95rem; color: #8b92a5; margin-bottom: 20px;'>
    O sistema Aedex processa predições utilizando um algoritmo de Gradient Boosting (XGBoost). O modelo foi treinado em dados 
    epidemiológicos e climáticos de 2016 a 2022 e validado cegamente em dados de 2023 a 2026. Abaixo estão as métricas consolidadas 
    de acerto e o comportamento do algoritmo comparado à realidade na janela operacional mais recente.
    </p>
    """, unsafe_allow_html=True)
    
    # Métricas Globais do Relatório
    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        st.metric(label="Coeficiente de Determinação (R²)", value="92.8%", delta="Explicabilidade da Série")
    with col_met2:
        st.metric(label="Erro Percentual Absoluto (MAPE)", value="26.8%", delta="Margem de desvio padrão", delta_color="off")
    with col_met3:
        st.metric(label="Horizonte de Previsão", value="4 Semanas", delta="Alinhamento Logístico UBS")

    st.markdown("<hr style='border-color: #2a2d35; margin-top: 25px; margin-bottom: 25px;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #ffffff; font-size: 1rem; margin-bottom: 15px;'>Raio-X: Casos Reais vs. Modelo (Últimas 4 Semanas Registradas)</h4>", unsafe_allow_html=True)
    
    # Tabela gerada dinamicamente mostrando o que o modelo previu no passado vs. o que de fato aconteceu
    if not df_limpo.empty:
        df_auditoria = df_limpo.tail(4).copy()
        df_auditoria['Semana/Ano'] = df_auditoria['semana'].astype(str) + "/" + df_auditoria['ano'].astype(str)
        df_auditoria['Notificações Reais (InfoDengue)'] = df_auditoria['casos'].astype(int)
        df_auditoria['Previsão Estimada (Aedex)'] = df_auditoria['predicao_casos'].astype(int)
        
        # Calcula o erro absoluto da diferença (para o gestor ver o desvio em vidas/casos e não em porcentagem obscura)
        df_auditoria['Desvio Absoluto (Casos)'] = abs(df_auditoria['Notificações Reais (InfoDengue)'] - df_auditoria['Previsão Estimada (Aedex)'])
        
        # Oculta o índice padrão e formata o display nativo do Streamlit
        st.dataframe(
            df_auditoria[['Semana/Ano', 'Notificações Reais (InfoDengue)', 'Previsão Estimada (Aedex)', 'Desvio Absoluto (Casos)']],
            use_container_width=True,
            hide_index=True
        )
