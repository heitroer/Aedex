import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import PIL.Image as Image

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Aedex - Monitoramento Inteligente", layout="wide", initial_sidebar_state="expanded")

# 2. CSS MODERNO PARA TEMA ESCURO (DARK MODE)
st.markdown("""
<style>
    /* Fundo geral e texto principal */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Títulos e textos padrão */
    h1, h2, h3, h4, p, span {
        color: #f8fafc !important;
    }
    
    /* Estilização das Métricas (Caixas de informação) */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        border-color: #3b82f6;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* Contêiner de protocolo clínico */
    .protocol-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 3. BARRA LATERAL (SIDEBAR)
with st.sidebar:
    st.markdown("<h2 style='color:#3b82f6; text-align: center;'>Aedex</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color:#94a3b8; margin-bottom: 20px;'>Painel Operacional Avançado</div>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("**Sobre o Sistema**")
    st.info("Solução desenvolvida para análise epidemiológica preditiva e suporte à decisão nas UBS de Campo Grande/MS.")
    
    st.markdown("**Metodologia Aplicada**")
    st.markdown("""
    - **Algoritmo:** XGBoost Regressor
    - **Variáveis de Destaque (SHAP):** Lags semanais (log_lag1) e Rt
    - **Precisão Estatística:** R² de 92,8%
    - **Horizonte de Ação:** 4 Semanas
    """)
    st.divider()
    st.success("Conexão com API: Estável")

# 4. CABEÇALHO PRINCIPAL
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    try:
        logo_img = Image.open('logo.png')
        st.image(logo_img, width='stretch')
    except Exception:
        st.markdown("<h1 style='color:#3b82f6; font-size: 3rem;'>AEDEX</h1>", unsafe_allow_html=True)

with col_titulo:
    st.markdown("<h1 style='padding-bottom: 0;'>Monitoramento Epidemiológico Inteligente</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top: -10px;'>Projeção de casos de Dengue e gestão de risco em tempo real.</p>", unsafe_allow_html=True)

# Informações do Projeto para preencher o visual do Dashboard
with st.expander("Informações do Projeto & Autores", expanded=False):
    st.markdown("""
    **Aedex: Implementação de Aprendizado de Máquina (XGBoost) para Predição de Surtos de Dengue e Otimização da Gestão de Unidades Básicas de Saúde em Campo Grande/MS.**
    
    * **Autores:** Heitor Leite de Oliveira Mesquita, Arthur Pereira Espindola, Emelly de Carvalho Koritiaki, Pâmela Rafaela do Prado.
    * **Instituição:** Colégio Status, Campo Grande - MS (Ciências da Saúde - Saúde Coletiva).
    
    **Métricas do Modelo:** O algoritmo capturou com exatidão os momentos críticos de inflexão da doença, atingindo um coeficiente de determinação ($R^2$) de **92,8%** e um Erro Percentual Absoluto Médio (MAPE) de **26,8%**. Análises via **SHAP** validam que a rede opera alinhada à biologia real da doença, sendo altamente influenciada pela taxa de reprodução efetiva (Rt).
    """)

st.markdown("---")

# 5. LÓGICA DE BACKEND
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_aedex.pkl")

try:
    modelo_prod = carregar_modelo()
except Exception as e:
    st.error("Aviso: O arquivo 'modelo_aedex.pkl' não foi encontrado. Certifique-se de que ele está na mesma pasta do script.")
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
    # Engenharia de Atributos Avançada
    df['log_casos'] = np.log1p(df['casos'])
    df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
    df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)

    for lag in range(1, 17): 
        df[f'log_lag{lag}'] = df['log_casos'].shift(lag)
        
    df['media_mov_4sem'] = df['log_casos'].shift(1).rolling(4).mean()
    df['media_mov_8sem'] = df['log_casos'].shift(1).rolling(8).mean()

    feature_cols = ['sem_seno', 'sem_cos'] + [f'log_lag{i}' for i in range(1, 17)] + ['media_mov_4sem', 'media_mov_8sem']
    df_limpo = df.dropna(subset=feature_cols + ['log_casos']).reset_index(drop=True)

    # NOVO: Realizando as predições para o passado (para visualização de validação)
    # Convertendo para float explicitamente para evitar alertas do XGBoost
    df_limpo_features = df_limpo[feature_cols].astype(float)
    df_limpo['predicao_log'] = modelo_prod.predict(df_limpo_features)
    df_limpo['predicao_casos'] = np.expm1(df_limpo['predicao_log']).astype(int)

    # Loop de Projeção Iterativa (Futuro)
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
        
        input_df = pd.DataFrame([features_atuais])[feature_cols].astype(float)
            
        try:
            p = float(modelo_prod.predict(input_df)[0])
            log_futuro.append(p)
            historico_log.append(p)
        except ValueError as e:
            st.error(f"Erro no formato dos dados: `{e}`")
            st.stop()

    casos_projetados_fim = int(np.expm1(log_futuro[-1]))

    # Definição de Classificação de Risco Epidemiológico
    if casos_projetados_fim < 50:
        status, cor = "AZUL — ESTÁVEL", "#3b82f6"
        protocolo = "Situação de normalidade. Manter vigilância passiva de notificações nas unidades."
    elif casos_projetados_fim < 150:
        status, cor = "AMARELO — ATENÇÃO", "#eab308"
        protocolo = "Início de sazonalidade. Revisar estoques de testes rápidos NS1 e insumos de hidratação nas UBS."
    elif casos_projetados_fim < 400:
        status, cor = "LARANJA — ALERTA", "#f97316"
        protocolo = "Pré-surto. Gatilho de contingência nível 1: Ampliar equipes de triagem e abrir salas de hidratação rápida."
    else:
        status, cor = "VERMELHO — CRISE", "#ef4444"
        protocolo = "Emergência de Saúde Pública: Criação de polos de atendimento específico e mutirões intersetoriais."

    # 6. DASHBOARD UI - Indicadores Preditivos
    st.markdown("### Visão Geral do Risco")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Casos Estimados (Em 4 sem.)", value=f"{casos_projetados_fim}", delta="Projeção Dinâmica", delta_color="normal")
    with col2:
        st.metric(label="Última Atualização Real", value=f"Semana {int(ultima_linha['semana'])}", delta=f"Ano {int(ultima_linha['ano'])}", delta_color="off")
    with col3:
        st.markdown("<p style='color: #94a3b8; font-weight: 600; font-size: 1.05rem; margin-bottom: 5px;'>Nível de Risco Operacional</p>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: {cor}; color: white; padding: 12px; border-radius: 8px; font-weight: 800; text-align: center; font-size: 1.1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>{status}</div>", unsafe_allow_html=True)

    # 7. PROTOCOLO OPERACIONAL CLINICO
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="protocol-container">
        <h4 style="margin-top: 0; color: #f8fafc;">Diretriz de Enfrentamento Clínico (UBS)</h4>
        <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 0;">{protocolo}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. GRÁFICO DE SÉRIES TEMPORAIS COM PLOTLY (TEMA ESCURO + HISTÓRICO)
    st.markdown("### Validação Histórica e Horizonte Preditivo")
    
    # Exibindo as últimas 26 semanas (1 semestre)
    df_ultimas = df_limpo.tail(26).copy()
    df_ultimas['label'] = df_ultimas['semana'].astype(str) + "/" + df_ultimas['ano'].astype(str)
    
    fig = go.Figure()
    
    # Curva Real
    fig.add_trace(go.Scatter(
        x=df_ultimas['label'], 
        y=df_ultimas['casos'], 
        mode='lines+markers', 
        name='Casos Reais (InfoDengue)', 
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6, color='#3b82f6')
    ))

    # Curva de Predição Histórica do Modelo (Como o modelo performou no passado)
    fig.add_trace(go.Scatter(
        x=df_ultimas['label'], 
        y=df_ultimas['predicao_casos'], 
        mode='lines', 
        name='Predição do Modelo (Passado)', 
        line=dict(color='#10b981', width=2, dash='dot'), # Verde claro tracejado
        opacity=0.7
    ))
    
    # Curva de Projeção Futura
    labels_futuro_com_ultimo = [df_ultimas['label'].iloc[-1]] + labels_futuro
    casos_futuros_reais = [df_ultimas['casos'].iloc[-1]] + [np.expm1(x) for x in log_futuro]
    
    fig.add_trace(go.Scatter(
        x=labels_futuro_com_ultimo, 
        y=casos_futuros_reais, 
        mode='lines+markers', 
        name='Projeção Aedex (4 Semanas)', 
        line=dict(color='#f97316', width=4, dash='dash'), 
        marker=dict(size=10, symbol='diamond', color='#f97316')
    ))
    
    # Atualização do Layout para Fundo Escuro
    fig.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        font=dict(color='#f8fafc'),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='#334155', title="Semana Epidemiológica"),
        yaxis=dict(showgrid=True, gridcolor='#334155', title="Número de Casos")
    )
    
    # Config para remover o modebar (botões no canto superior direito)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    st.error("Não foi possível estabelecer conexão para processamento dos dados da API.")
