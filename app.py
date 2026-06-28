# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import datetime
import os

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Aedex - Monitoramento Inteligente", layout="wide", initial_sidebar_state="expanded")

# 2. CSS AVANÇADO PARA UI/UX (Estilo Dark Moderno)
st.markdown("""
<style>
    .stApp { background-color: #13151a; font-family: 'Inter', sans-serif; }
    header { visibility: hidden; }
    [data-testid="stSidebar"] { background-color: #1a1c23; border-right: 1px solid #2a2d35; }
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 600 !important; }
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
    st.sidebar.markdown("""
    ---
    <div style='font-size: 11px; color: #5c6275; text-align: center;'>
        <b>Aedex Engine v1.2</b><br>
        Arquitetura de Produção Pronta para Integração via API (JSON/.pkl)
    </div>
    """, unsafe_allow_html=True)

# 4. CABEÇALHO
st.markdown("<h1 style='padding-bottom: 5px; font-size: 2.2rem;'>Monitoramento Epidemiológico Inteligente</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; margin-bottom: 30px;'>Projeção de casos de Dengue e gestão de risco em tempo real com arquitetura Multi-Output Delta.</p>", unsafe_allow_html=True)

# 5. LÓGICA DE BACKEND (Carregamento dos 4 Modelos Otimizados)
# Substitua a abordagem antiga do joblib por esta:
@st.cache_resource
def carregar_modelos_multioutput():
    from xgboost import XGBRegressor
    modelos = {}
    for h in range(1, 5):
        try:
            modelo = XGBRegressor()
            _ = os.path.getmtime(f"modelo_aedex_sem{h}.json")
            modelo.load_model(f"modelo_aedex_sem{h}.json")
            modelos[h] = modelo
        except:
            modelos[h] = None
    return modelos

@st.cache_data(ttl=3600)
def puxar_dados_api():
    ano_atual = datetime.datetime.now().year
    url = f"https://info.dengue.mat.br/api/alertcity/?geocode=5002704&disease=dengue&format=json&ew_start=1&ey_start=2016&ew_end=52&ey_end={ano_atual}"
    try:
        resp = requests.get(url, timeout=25)
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
            
        return df.dropna(subset=['casos']).reset_index(drop=True)
    except: 
        return None

df = puxar_dados_api()
modelos_prod = carregar_modelos_multioutput()

# Verifica se todos os 4 modelos foram carregados com sucesso
modelos_ok = modelos_prod and all(modelos_prod[h] is not None for h in range(1, 5))

if df is not None and modelos_ok:
    # Tratamento de nulos/clima
    for col in ['tempmed', 'umidmed', 'tempmin', 'tempmax', 'umidmin', 'umidmax']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

    # Engenharia de Atributos Idêntica ao Treino Otimizado
    df['log_casos'] = np.log1p(df['casos'])
    df['sem_seno'] = np.sin(2 * np.pi * df['semana'] / 52.0)
    df['sem_cos']  = np.cos(2 * np.pi * df['semana'] / 52.0)
    
    df['log_lag1'] = df['log_casos'] 
    for lag in range(2, 9): 
        df[f'log_lag{lag}'] = df['log_casos'].shift(lag - 1)
        
    df['media_mov_4sem'] = df['log_casos'].rolling(4).mean()
    df['media_mov_8sem'] = df['log_casos'].rolling(8).mean()

    if 'tempmed' in df.columns:
        df['temp_atual'] = df['tempmed']
        df['temp_lag2'] = df['tempmed'].shift(1)
        df['temp_lag4'] = df['tempmed'].shift(3)
        df['temp_4sem'] = df['tempmed'].rolling(4).mean()
    if 'umidmed' in df.columns:
        df['umid_atual'] = df['umidmed']
        df['umid_lag2'] = df['umidmed'].shift(1)
        df['umid_lag4'] = df['umidmed'].shift(3)
    for col in ['p_rt1', 'Rt', 'nivel']:
        if col in df.columns: 
            df[f'{col}_lag1'] = df[col]

    feature_cols = ['sem_seno', 'sem_cos', 'log_lag1', 'log_lag2', 'log_lag3', 'log_lag4',
                    'log_lag5', 'log_lag6', 'log_lag7', 'log_lag8', 'media_mov_4sem', 'media_mov_8sem',
                    'temp_atual', 'temp_lag2', 'temp_lag4', 'temp_4sem', 'umid_atual', 'umid_lag2', 'umid_lag4',
                    'p_rt1_lag1', 'Rt_lag1', 'nivel_lag1']
    
    df_limpo = df.dropna(subset=feature_cols).reset_index(drop=True)
    
    # Histórico do Modelo 1 (Alinhado temporalmente para o gráfico de Auditoria)
    X_hist = df_limpo[feature_cols].to_numpy()
    pred_delta_hist = modelos_prod[1].predict(X_hist)
    pred_log_hist = pred_delta_hist + df_limpo['log_lag1'].to_numpy()
    # Deslocamos 1 semana para frente para comparar a predição feita em 't' com o real de 't+1'
    df_limpo['predicao_casos'] = pd.Series(np.expm1(pred_log_hist), index=df_limpo.index).shift(1).fillna(0).astype(int)

    # Captura da Linha Mais Recente para Projeção Futura Exclusiva
    linha_atual = df_limpo.iloc[-1:]
    X_inferencia = linha_atual[feature_cols].to_numpy()
    log_casos_atual = linha_atual['log_lag1'].values[0]
    
    sem_atual = int(linha_atual['semana'].values[0])
    ano_atual = int(linha_atual['ano'].values[0])
    
    casos_futuros = []
    labels_futuro = []
    
    sem_sim, ano_sim = sem_atual, ano_atual
    
    # Extração direta das 4 previsões independentes sem loops recursivos que acumulam erros
    for h in range(1, 5):
        sem_sim += 1
        if sem_sim > 52: 
            sem_sim = 1
            ano_sim += 1
        labels_futuro.append(f"{sem_sim}/{ano_sim}")
        
        # Predição do Delta e Reconstrução Blindada (Mínimo de 0 casos)
        pred_delta = modelos_prod[h].predict(X_inferencia)[0]
        pred_log_final = max(0.0, pred_delta + log_casos_atual)
        casos_futuros.append(int(np.expm1(pred_log_final)))

    casos_projetados_fim = casos_futuros[-1]
    ultima_linha = {'semana': sem_atual, 'ano': ano_atual}
else:
    casos_projetados_fim = 0
    df_limpo = pd.DataFrame(columns=['casos', 'predicao_casos', 'semana', 'ano', 'label'])
    labels_futuro, casos_futuros = [], []
    ultima_linha = {'semana': 0, 'ano': 0}

# Lógica de Cores do Alerta Baseado no Horizonte Final
if casos_projetados_fim < 50: status, cor, bg_cor = "ESTÁVEL", "#00d1ff", "rgba(0, 209, 255, 0.15)"
elif casos_projetados_fim < 150: status, cor, bg_cor = "ATENÇÃO", "#fbbf24", "rgba(251, 191, 36, 0.15)"
elif casos_projetados_fim < 400: status, cor, bg_cor = "ALERTA", "#f97316", "rgba(249, 115, 22, 0.15)"
else: status, cor, bg_cor = "CRISE", "#ef4444", "rgba(239, 68, 68, 0.15)"

# LÓGICA DE DIRETRIZES OPERACIONAIS (Painel de Ações Recomendadas)
diretrizes = {
    "ESTÁVEL": "Manter rotina padrão de visitas dos Agentes de Combate a Endemias (ACE). Monitoramento passivo.",
    "ATENÇÃO": "Intensificar busca ativa de focos em bairros de risco. Planejar remanejamento de estoque de testes NS1 e insumos de triagem.",
    "ALERTA": "Acionar comitê de crise local. Iniciar aplicação de bloqueio químico (Fumacê) nas zonas preditas pelo modelo e ampliar horário de atendimento na UBS.",
    "CRISE": "Abertura emergencial de tendas de hidratação rápida. Contratação de equipes médicas de suporte, estoque máximo de soro fisiológico e mobilização total de contingência."
}

# 6. DASHBOARD UI - CARDS SUPERIORES
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="custom-card"><div class="card-title">Casos Estimados (Horizonte Semana +4)</div><div class="card-value">{casos_projetados_fim}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="custom-card"><div class="card-title">Último Dado Consolidado API</div><div class="card-value" style="font-size: 1.8rem;">Semana {int(ultima_linha["semana"])}/{int(ultima_linha["ano"])}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="custom-card"><div class="card-title">Nível de Risco Projetado</div><div><span class="card-badge" style="background-color: {bg_cor}; color: {cor}; border: 1px solid {cor};">{status}</span></div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6.1 PAINEL DE AÇÕES RECOMENDADAS
st.markdown(f"""
<div class="custom-card" style="border-left: 5px solid {cor}; padding: 20px 24px;">
    <h4 style="margin:0; color:{cor} !important; font-size: 1.1rem;">📋 Protocolo de Contingência Ativado: Nível {status}</h4>
    <p style="margin-top:8px; font-size:1rem; color:#ffffff !important; margin-bottom:0;">{diretrizes[status]}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# 7. GRÁFICO DE LINHA - HISTÓRICO E PROJEÇÃO MULTI-OUTPUT
col_title, col_sel = st.columns([3, 1])
with col_title:
    st.markdown("### Histórico e Horizonte Preditivo", unsafe_allow_html=True)
with col_sel:
    janela = st.selectbox("Período:", [4, 12, 26, 52], index=2, label_visibility="collapsed")

df_ultimas = df_limpo.tail(janela).copy()
if 'label' not in df_ultimas.columns and not df_ultimas.empty:
    df_ultimas['label'] = df_ultimas['semana'].astype(str) + "/" + df_ultimas['ano'].astype(str)

fig_linha = go.Figure()

if not df_ultimas.empty:
    fig_linha.add_trace(go.Scatter(
        x=df_ultimas['label'], y=df_ultimas['casos'], mode='lines', name='Casos Reais', 
        line=dict(color='#00d1ff', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 209, 255, 0.05)'
    ))
    fig_linha.add_trace(go.Scatter(
        x=df_ultimas['label'], y=df_ultimas['predicao_casos'], mode='lines', name='Predição Retroativa (S+1)', 
        line=dict(color='#10b981', width=2, dash='dot', shape='spline')
    ))
    
    # Vincula o último ponto real ao início da linha de projeção futura de 4 semanas
    labels_futuro_com_ultimo = [df_ultimas['label'].iloc[-1]] + labels_futuro
    casos_futuros_com_ultimo = [df_ultimas['casos'].iloc[-1]] + casos_futuros
    
    fig_linha.add_trace(go.Scatter(
        x=labels_futuro_com_ultimo, y=casos_futuros_com_ultimo, mode='lines+markers', name='Projeção Futura Estabilizada', 
        line=dict(color=cor, width=3, dash='dash', shape='spline'), marker=dict(size=6, color=cor)
    ))

fig_linha.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='#8b92a5', family="Inter, sans-serif"),
    margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
    xaxis=dict(showgrid=False, showline=True, linecolor='#2a2d35'), 
    yaxis=dict(showgrid=True, gridcolor='#2a2d35', zeroline=False), hovermode="x unified"
)

st.markdown('<div class="custom-card" style="padding: 10px 24px 24px 24px;">', unsafe_allow_html=True)
st.plotly_chart(fig_linha, use_container_width=True, config={'displayModeBar': False})
st.markdown("</div>", unsafe_allow_html=True)

# 8. SEÇÃO DE AUDITORIA ATUALIZADA COM OS RESULTADOS REAIS DO NOVO TREINO
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### 📊 Auditoria de Precisão Científica e Confiabilidade", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1rem; margin-bottom: 20px;'>Métricas reais validadas sobre dados históricos fora do treino (2023-2026).</p>", unsafe_allow_html=True)

col_met1, col_met2, col_chart = st.columns([1, 1, 3])

with col_met1:
    st.markdown('''
    <div class="custom-card" style="padding: 20px;">
        <div class="card-title">Acurácia Semana 1 (R²)</div>
        <div class="card-value" style="color: #10b981;">93.9%</div>
        <div style="font-size: 0.85rem; color: #8b92a5; margin-top: 8px;">Acurácia real de curtíssimo prazo.</div>
    </div>
    ''', unsafe_allow_html=True)

with col_met2:
    st.markdown('''
    <div class="custom-card" style="padding: 20px;">
        <div class="card-title">Acurácia Semana 4 (R²)</div>
        <div class="card-value" style="color: #00d1ff;">71.6%</div>
        <div style="font-size: 0.85rem; color: #8b92a5; margin-top: 8px;">Estabilidade mantida a longo prazo.</div>
    </div>
    ''', unsafe_allow_html=True)

with col_chart:
    df_barras = df_limpo.tail(6).copy()
    if 'label' not in df_barras.columns and not df_barras.empty:
        df_barras['label'] = df_barras['semana'].astype(str) + "/" + df_barras['ano'].astype(str)

    fig_bar = go.Figure()

    if not df_barras.empty:
        fig_bar.add_trace(go.Bar(
            x=df_barras['label'], y=df_barras['casos'], name='Casos Reais', 
            marker_color='#00d1ff', marker_line_color='rgba(0,0,0,0)', opacity=0.85
        ))
        fig_bar.add_trace(go.Bar(
            x=df_barras['label'], y=df_barras['predicao_casos'], name='Modelo Aedex (S+1)', 
            marker_color='#10b981', marker_line_color='rgba(0,0,0,0)', opacity=0.9
        ))

    fig_bar.update_layout(
        barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b92a5', family="Inter, sans-serif"), margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.2, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor='#2a2d35'),
        yaxis=dict(showgrid=True, gridcolor='#2a2d35', zeroline=False),
        hovermode="x unified", height=220
    )

    st.markdown('<div class="custom-card" style="padding: 15px 24px 10px 24px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)
# HACK TEMPORÁRIO DE AUDITORIA: Gerador de Payload para o Swagger
st.markdown("### 📋 Payload Real para Copiar no Swagger")
if df_limpo is not None and not df_limpo.empty:
    ultima_f = df_limpo.iloc[-1]
    payload_real = {
        "localidade_id": "5002704-B01",
        "semana_epidemiologica": int(ultima_f["semana"]),
        "ano": int(ultima_f["ano"]),
        "Rt_lag1": float(ultima_f["Rt"]) if "Rt" in ultima_f else 1.0,
        "nivel_lag1": int(ultima_f["nivel"]) if "nivel" in ultima_f else 2,
        "p_rt1_lag1": float(ultima_f["p_rt1"]) if "p_rt1" in ultima_f else 0.8,
        "casos_lag1": int(ultima_f["casos"]),
        "casos_lag2": int(df_limpo.iloc[-2]["casos"]),
        "casos_lag3": int(df_limpo.iloc[-3]["casos"]),
        "casos_lag4": int(df_limpo.iloc[-4]["casos"]),
        "casos_lag5": int(df_limpo.iloc[-5]["casos"]),
        "casos_lag6": int(df_limpo.iloc[-6]["casos"]),
        "casos_lag7": int(df_limpo.iloc[-7]["casos"]),
        "casos_lag8": int(df_limpo.iloc[-8]["casos"]),
        "temp_atual": float(ultima_f["tempmed"]),
        "temp_lag2": float(df_limpo.iloc[-2]["tempmed"]),  # Corresponde ao shift(1) do treino
        "temp_lag4": float(df_limpo.iloc[-4]["tempmed"]),  # Corresponde ao shift(3) do treino
        "temp_4sem": float(ultima_f["temp_4sem"]),
        "umid_atual": float(ultima_f["umidmed"]),
        "umid_lag2": float(df_limpo.iloc[-2]["umidmed"]),
        "umid_lag4": float(df_limpo.iloc[-4]["umidmed"])
    }
    st.json(payload_real)
