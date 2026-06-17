import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import PIL.Image as Image

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Aedex - Monitoramento Inteligente", layout="wide", initial_sidebar_state="expanded")

# 2. CSS MODERNO PARA UI/UX
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #1A5FFF;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. BARRA LATERAL (SIDEBAR)
with st.sidebar:
    st.markdown("<h2 style='color:#1A5FFF; text-align: center;'>Aedex</h2>", unsafe_allow_html=True)
    st.caption("<div style='text-align: center; margin-bottom: 20px;'>Painel Operacional Avançado</div>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("**Sobre o Sistema**")
    st.info("Solução desenvolvida para análise epidemiológica preditiva e suporte direto à decisão clínica nas UBS de Campo Grande/MS.")
    
    st.markdown("**Metodologia Aplicada**")
    st.markdown("""
    - **Fonte:** API InfoDengue
    - **Geocódigo:** 5002704 (CG/MS)
    - **Motor Preditivo:** XGBoost Regressor
    - **Horizonte:** 4 Semanas
    """)
    st.divider()
    st.success("Conexão com API Estável")

# 4. CABEÇALHO PRINCIPAL
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    try:
        logo_img = Image.open('logo.png')
        st.image(logo_img, width='stretch')
    except Exception:
        st.markdown("<h1 style='color:#1A5FFF;'>AEDEX</h1>", unsafe_allow_html=True)

with col_titulo:
    st.markdown("<h1 style='color:#0f172a; padding-bottom: 0;'>Monitoramento Epidemiológico Inteligente</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:1.1rem; margin-top: -10px;'>Projeção de casos de Dengue e gestão de risco em tempo real.</p>", unsafe_allow_html=True)

st.markdown("---")

# 5. LÓGICA DE BACKEND
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_aedex.pkl")

try:
    modelo_prod = carregar_modelo()
except Exception as e:
    st.error("Erro: O arquivo 'modelo_aedex.pkl' não foi encontrado.")
    st.stop()

@st.cache_data(ttl=86400)
def puxar_dados_api():
    url = "https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=52&ey_end=2026"
    try:
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
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
        return df
    except Exception as e:
        return None

df = puxar_dados_api()

if df is not None:
    # Engenharia de Atributos Avançada (Gera exatamente 20 variáveis: 2 sazonais + 16 lags + 2 médias)
    df['log_casos'] = np.log1p(df['casos'])
    df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
    df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)

    # Criando 16 lags para alinhar com o shape esperado pelo XGBoost
    for lag in range(1, 17): 
        df[f'log_lag{lag}'] = df['log_casos'].shift(lag)
        
    df['media_mov_4sem'] = df['log_casos'].shift(1).rolling(4).mean()
    df['media_mov_8sem'] = df['log_casos'].shift(1).rolling(8).mean()

    # Construindo a lista de features estruturada idêntica ao modelo
    feature_cols = ['sem_seno', 'sem_cos'] + [f'log_lag{i}' for i in range(1, 17)] + ['media_mov_4sem', 'media_mov_8sem']
    
    df_limpo = df.dropna(subset=feature_cols + ['log_casos']).reset_index(drop=True)

    # Loop de Projeção Iterativa (Histórico expandido para 16 posições)
    historico_log = df_limpo['log_casos'].tail(16).tolist()
    ultima_linha = df_limpo.iloc[-1].copy()
    features_atuais = ultima_linha[feature_cols].copy()
    sem_sim = int(ultima_linha['semana'])
    ano_sim = int(ultima_linha['ano'])
    log_futuro = []
    labels_futuro = []

    for _ in range(4):
        sem_sim += 1
        if sem_sim > 52:
            sem_sim = 1
            ano_sim += 1
        labels_futuro.append(f"{sem_sim}/{ano_sim}")
        
        features_atuais['sem_seno'] = np.sin(2 * np.pi * sem_sim / 52.0)
        features_atuais['sem_cos'] = np.cos(2 * np.pi * sem_sim / 52.0)
        
        for lag in range(1, 17): 
            features_atuais[f'log_lag{lag}'] = historico_log[-lag]
            
        features_atuais['media_mov_4sem'] = np.mean(historico_log[-4:])
        features_atuais['media_mov_8sem'] = np.mean(historico_log[-8:])
        
        # Garante a formatação correta dos dados estruturados antes da inferência
        input_df = pd.DataFrame([features_atuais])
        input_df = input_df[feature_cols].astype(float)
            
        try:
            p = float(modelo_prod.predict(input_df)[0])
            log_futuro.append(p)
            historico_log.append(p)
        except ValueError as e:
            st.error(f"**Erro de Alinhamento:** O modelo espera dados diferentes. Erro: `{e}`")
            st.stop()

    casos_projetados_fim = int(np.expm1(log_futuro[-1]))

    # Definição de Classificação de Risco Epidemiológico
    if casos_projetados_fim < 50:
        status, cor = "AZUL — ESTÁVEL", "#1A5FFF"
        protocolo = "Situação de normalidade. Manter vigilância passiva de notificações nas unidades."
    elif casos_projetados_fim < 150:
        status, cor = "AMARELO — ATENÇÃO", "#fca326"
        protocolo = "Início de sazonalidade. Revisar estoques de testes rápidos NS1 e insumos de hidratação nas UBS."
    elif casos_projetados_fim < 400:
        status, cor = "LARANJA — ALERTA", "#db6d28"
        protocolo = "Pré-surto. Gatilho de contingência nível 1: Ampliar equipes de triagem e abrir salas de hidratação rápida."
    else:
        status, cor = "VERMELHO — CRISE", "#cf222e"
        protocolo = "Emergência de Saúde Pública: Criação de polos de atendimento específico e mutirões intersetoriais."

    # 6. DASHBOARD UI - Indicadores Preditivos
    st.markdown("### Visão Geral do Risco")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Casos Estimados (Em 4 sem.)", value=f"{casos_projetados_fim}", delta="Projeção Dinâmica", delta_color="normal")
    with col2:
        st.metric(label="Última Atualização Real", value=f"Semana {int(ultima_linha['semana'])}", delta=f"Ano {int(ultima_linha['ano'])}", delta_color="off")
    with col3:
        st.markdown("<p style='color: #64748b; font-weight: 600; font-size: 1.05rem; margin-bottom: 5px;'>Nível de Risco Operacional</p>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: {cor}; color: white; padding: 12px; border-radius: 8px; font-weight: 700; text-align: center; font-size: 1.1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>{status}</div>", unsafe_allow_html=True)

    # 7. PROTOCOLO OPERACIONAL CLINICO
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"#### Diretriz de Enfrentamento Clínico (UBS)")
        st.markdown(f"<p style='font-size: 1.1rem; color: #334155;'>{protocolo}</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. GRÁFICO DE SÉRIES TEMPORAIS COM PLOTLY
    st.markdown("### Linha de Tendência e Horizonte Preditivo")
    
    df_ultimas = df_limpo.tail(15).copy()
    df_ultimas['label'] = df_ultimas['semana'].astype(str) + "/" + df_ultimas['ano'].astype(str)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_ultimas['label'], 
        y=df_ultimas['casos'], 
        mode='lines+markers', 
        name='Casos Reais (InfoDengue)', 
        line=dict(color='#1A5FFF', width=4),
        marker=dict(size=8, color='#1A5FFF')
    ))
    
    labels_futuro_com_ultimo = [df_ultimas['label'].iloc[-1]] + labels_futuro
    casos_futuros_reais = [df_ultimas['casos'].iloc[-1]] + [np.expm1(x) for x in log_futuro]
    
    fig.add_trace(go.Scatter(
        x=labels_futuro_com_ultimo, 
        y=casos_futuros_reais, 
        mode='lines+markers', 
        name='Projeção Aedex', 
        line=dict(color='#fca326', width=4, dash='dash'), 
        marker=dict(size=10, symbol='diamond', color='#fca326')
    ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Semana Epidemiológica"),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Número de Casos Real/Estimado")
    )
    
    st.plotly_chart(fig, width='stretch')

else:
    st.error("Não foi possível estabelecer conexão para processamento dos dados da API.")
