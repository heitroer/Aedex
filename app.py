# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import datetime

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
@st.cache_resource
def carregar_modelos_multioutput():
    modelos = {}
    for h in range(1, 5):
        try:
            modelos[h] = joblib.load(f"modelo_aedex_sem{h}.pkl")
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
