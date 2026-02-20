import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import numpy as np
from scipy.stats import pearsonr

# ==========================================
# 1. METODOLOGIA E COLETA DE DADOS (APIs IBGE)
# ==========================================
# Decisão Metodológica: 
# 1. IPCA (Mensal, RMs) será agregado para Trimestral (Média do trimestre) para parear com a PNAD[cite: 17].
# 2. As RMs (N7) serão mapeadas para as Grandes Regiões (N2) usando o primeiro dígito do código do IBGE.
#    (1: Norte, 2: Nordeste, 3: Sudeste, 4: Sul, 5: Centro-Oeste).

def fetch_ibge_data():
    # API IPCA (Agregado 7060, Var 63 - Variação Mensal %, N7 - Regiões Metropolitanas)
    url_ipca = "https://servicodados.ibge.gov.br/api/v3/agregados/7060/periodos/202001-202412/variaveis/63?localidades=N7[all]"
    
    # API PNAD (Agregado 4099, Var 4099 - Taxa de desocupação %, N2 - Grandes Regiões)
    url_pnad = "https://servicodados.ibge.gov.br/api/v3/agregados/4099/periodos/202001-202404/variaveis/4099?localidades=N2[all]"
    
    try:
        resp_ipca = requests.get(url_ipca, timeout=15).json()
        resp_pnad = requests.get(url_pnad, timeout=15).json()
    except Exception as e:
        return pd.DataFrame() # Retorna vazio em caso de falha de conexão

    # Parse IPCA
    ipca_records = []
    for serie in resp_ipca[0]['resultados']:
        rm_name = serie['localidade']['nome']
        rm_id = serie['localidade']['id']
        macro_code = str(rm_id)[0] # 1 a 5
        for periodo, valor in serie['series'][0].items():
            if valor != '...':
                ipca_records.append({'Mes': periodo, 'RM': rm_name, 'Macro_ID': macro_code, 'IPCA': float(valor)})
    
    df_ipca = pd.DataFrame(ipca_records)
    
    # Transformar Mensal em Trimestral
    df_ipca['Ano'] = df_ipca['Mes'].str[:4]
    df_ipca['Trimestre'] = pd.to_datetime(df_ipca['Mes'], format='%Y%m').dt.quarter
    df_ipca['Trimestre_Cod'] = df_ipca['Ano'] + "0" + df_ipca['Trimestre'].astype(str)
    
    # Agregar IPCA por Macro Região e Trimestre (Média da inflação mensal do trimestre das RMs)
    df_ipca_trim = df_ipca.groupby(['Macro_ID', 'Trimestre_Cod'])['IPCA'].mean().reset_index()
    # Aproximação da inflação acumulada do trimestre (Soma simples para variação pequena)
    df_ipca_trim['IPCA_Acum_Trim'] = df_ipca_trim['IPCA'] * 3 

    # Parse PNAD
    pnad_records = []
    macro_map = {'1': 'Norte', '2': 'Nordeste', '3': 'Sudeste', '4': 'Sul', '5': 'Centro-Oeste'}
    
    for serie in resp_pnad[0]['resultados']:
        macro_id = serie['localidade']['id']
        for periodo, valor in serie['series'][0].items():
            if valor != '...':
                pnad_records.append({'Trimestre_Cod': periodo, 'Macro_ID': macro_id, 'Desemprego': float(valor)})
    
    df_pnad = pd.DataFrame(pnad_records)
    
    # Merge Final
    df_final = pd.merge(df_pnad, df_ipca_trim, on=['Macro_ID', 'Trimestre_Cod'], how='inner')
    df_final['Regiao'] = df_final['Macro_ID'].map(macro_map)
    
    # Formatação de data para gráficos
    df_final['Data'] = pd.PeriodIndex(df_final['Trimestre_Cod'].str[:4] + "Q" + df_final['Trimestre_Cod'].str[-1], freq='Q').to_timestamp()
    
    return df_final

df = fetch_ibge_data()

# ==========================================
# 2. CONSTRUÇÃO DO DASHBOARD (PLOTLY/DASH)
# ==========================================
app = dash.Dash(__name__)
app.title = "Dashboard Econométrico: TCC"

if df.empty:
    app.layout = html.Div("Erro ao carregar dados da API do IBGE. Verifique sua conexão.")
else:
    # 1. Gráfico de Barras: Comparação Direta
    df_bar = df.groupby('Regiao')[['Desemprego', 'IPCA_Acum_Trim']].mean().reset_index()
    fig_bar = go.Figure(data=[
        go.Bar(name='Tx. Desemprego Média (%)', x=df_bar['Regiao'], y=df_bar['Desemprego'], marker_color='indianred'),
        go.Bar(name='IPCA Médio Trimestral (%)', x=df_bar['Regiao'], y=df_bar['IPCA_Acum_Trim'], marker_color='lightsalmon')
    ])
    fig_bar.update_layout(barmode='group', title="Médias do Período (2020-2024) por Região")

    # 2. Gráfico de Linhas com Tendência
    fig_line = px.line(df, x='Data', y='Desemprego', color='Regiao', 
                       title="Evolução Temporal do Desemprego por Grande Região", markers=True)
    
    # 3. Gráfico de Pizza: Participação (Desemprego Médio)
    fig_pie = px.pie(df_bar, values='Desemprego', names='Regiao', 
                     title="Proporção Relativa das Taxas Médias de Desemprego")
    
    # 4. Mapa Regional Analítico
    # Usando coordenadas aproximadas das macro-regiões para o mapa de dispersão geográfica
    coords = {'Norte': [-3.11, -60.02], 'Nordeste': [-12.97, -38.51], 'Sudeste': [-23.55, -46.63], 
              'Sul': [-30.03, -51.23], 'Centro-Oeste': [-15.79, -47.88]}
    df_map = df_bar.copy()
    df_map['lat'] = df_map['Regiao'].map(lambda x: coords[x][0])
    df_map['lon'] = df_map['Regiao'].map(lambda x: coords[x][1])
    
    fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", hover_name="Regiao", 
                                hover_data=["Desemprego", "IPCA_Acum_Trim"],
                                color="Desemprego", size="Desemprego",
                                color_continuous_scale=px.colors.sequential.Reds, size_max=40, zoom=3,
                                title="Mapa Analítico: Intensidade do Desemprego e Inflação")
    fig_map.update_layout(mapbox_style="carto-positron")

    # 5. Gráfico de Dispersão Econométrico + Regressão Linear + Pearson
    fig_scatter = px.scatter(df, x="IPCA_Acum_Trim", y="Desemprego", color="Regiao", trendline="ols",
                             title="Relação Econométrica: Inflação vs Desemprego (Curva de Phillips Regional)",
                             labels={"IPCA_Acum_Trim": "Inflação (IPCA Trimestral %)", "Desemprego": "Taxa de Desocupação (%)"})
    
    # Cálculo Global do Pearson (r)
    r, p_value = pearsonr(df['IPCA_Acum_Trim'], df['Desemprego'])

    app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px', 'maxWidth': '1200px', 'margin': 'auto'}, children=[
        html.H1("Impactos da Inflação sobre o Desemprego nas Grandes Regiões (2020-2024)", style={'textAlign': 'center'}),
        html.Hr(),
        
        # Mapa
        dcc.Graph(figure=fig_map),
        html.P("Este gráfico responde ao objetivo de delimitar espacialmente o estudo ao demonstrar que existem assimetrias regionais significativas na distribuição das taxas de desocupação ao longo do território nacional.", style={'fontStyle': 'italic', 'marginBottom': '40px'}),

        # Barras
        dcc.Graph(figure=fig_bar),
        html.P("Este gráfico responde ao objetivo de comparar as regiões ao demonstrar que o nível estrutural de desemprego varia drasticamente (ex: Nordeste vs Sul), enquanto a variação inflacionária média apresenta maior homogeneidade no médio prazo.", style={'fontStyle': 'italic', 'marginBottom': '40px'}),

        # Linhas
        dcc.Graph(figure=fig_line),
        html.P("Este gráfico responde ao objetivo de analisar o comportamento temporal ao demonstrar que os choques econômicos da pandemia geraram picos sistêmicos de desemprego, seguidos de trajetórias de recuperação que diferem em velocidade entre as Grandes Regiões.", style={'fontStyle': 'italic', 'marginBottom': '40px'}),

        # Pizza
        dcc.Graph(figure=fig_pie),
        html.P("Este gráfico responde ao objetivo de ilustrar a assimetria ao demonstrar a participação relativa do 'peso' da taxa de desemprego de cada região na composição do cenário nacional.", style={'fontStyle': 'italic', 'marginBottom': '40px'}),

        # Dispersão
        dcc.Graph(figure=fig_scatter),
        html.P(f"Este gráfico responde ao objetivo principal (III) do projeto ao calcular a correlação estatística entre as séries. O coeficiente de Pearson global calculado é r = {r:.3f} (p-valor: {p_value:.3f}), demonstrando a direção e a força da relação entre pressão inflacionária e mercado de trabalho regional.", style={'fontStyle': 'italic'})
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
