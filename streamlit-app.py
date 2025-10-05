
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard de Risque de Portefeuille",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")

@st.cache_data
def download_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        if data.empty:
            return None, "Aucune donnée retournée. Vérifiez les tickers et la plage de dates."
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        return data.dropna(how='all'), None
    except Exception as e:
        return None, f"Une erreur est survenue lors du téléchargement : {e}"

@st.cache_data
def calculate_portfolio_returns(data, weights):
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def calculate_risk_metrics(returns, confidence_level=0.95):
    if returns.empty:
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan, None)

    alpha = 1 - confidence_level

    # Méthode Analytique (Paramétrique)
    mean = np.mean(returns)
    std = np.std(returns)
    var_analytical = norm.ppf(alpha, loc=mean, scale=std)
    es_analytical = mean - std * norm.pdf(norm.ppf(alpha)) / alpha

    # Méthode Historique
    var_historical = np.percentile(returns, alpha * 100)
    es_historical = np.mean(returns[returns <= var_historical])

    # Méthode Monte Carlo
    num_simulations = 10000
    simulated_returns = np.random.normal(mean, std, num_simulations)
    var_mc = np.percentile(simulated_returns, alpha * 100)
    es_mc = np.mean(simulated_returns[simulated_returns <= var_mc])

    return (var_analytical, es_analytical), (var_historical, es_historical), (var_mc, es_mc, simulated_returns)


st.sidebar.header("🔧 Paramètres du Portefeuille")

tickers_input = st.sidebar.text_input(
    "Symboles des actions (séparés par des virgules)",
    "AAPL, MSFT, GOOGL, JPM"
).upper()
selected_tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]

weights_input = st.sidebar.text_input(
    "Poids correspondants (séparés par des virgules)",
    "0.25, 0.25, 0.25, 0.25"
)
try:
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
except ValueError:
    st.sidebar.error("Veuillez entrer des poids numériques valides.")
    weights = []

if not selected_tickers:
    st.warning("Veuillez entrer au moins un symbole d'action pour commencer.")
    st.stop()

if len(selected_tickers) != len(weights):
    st.sidebar.warning("Le nombre de tickers doit correspondre au nombre de poids.")
    st.stop()

if not np.isclose(sum(weights), 1.0):
    st.sidebar.warning(f"La somme des poids doit être égale à 1 (actuellement : {sum(weights):.2f}).")
    st.stop()

st.sidebar.header("🗓️ Période et Risque")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Date de début", pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("Date de fin", pd.to_datetime("today"))

confidence_level = st.sidebar.slider(
    "Niveau de confiance pour la VaR/ES", 0.80, 0.99, 0.95, step=0.01, format="%.2f"
)

st.title("🛡️ Dashboard d'Analyse de Risque de Portefeuille")
st.markdown(f"Analyse du portefeuille **{', '.join(selected_tickers)}** sur la période du `{start_date}` au `{end_date}`.")

data, error_message = download_data(selected_tickers, start_date, end_date)

if error_message:
    st.error(error_message)
else:
    portfolio_returns = calculate_portfolio_returns(data, weights)
    
    (var_a, es_a), (var_h, es_h), (var_mc, es_mc, sim_returns) = calculate_risk_metrics(portfolio_returns, confidence_level)
        
    st.header("📈 Performance et Risque")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rendement Annuel Moyen", f"{portfolio_returns.mean() * 252:.2%}")
    with col2:
        st.metric("Volatilité Annuelle", f"{portfolio_returns.std() * np.sqrt(252):.2%}")
    with col3:
        sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
        st.metric("Ratio de Sharpe", f"{sharpe:.2f}")

    st.subheader("Performance Cumulative")
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Portefeuille', line=dict(color='royalblue', width=2)))
    fig_perf.update_layout(title="Croissance d'1€ investi", xaxis_title="Date", yaxis_title="Rendement Cumulé", yaxis_tickformat=".2%")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.header(f"💰 Value at Risk (VaR) et Expected Shortfall (ES) à {confidence_level:.0%}")
    st.markdown("La **VaR** est la perte maximale attendue sur un horizon de temps donné pour un niveau de confiance. L'**ES** est la perte moyenne attendue *sachant que la perte est supérieure à la VaR*.")

    results_data = {
        "Méthode": ["Analytique", "Historique", "Monte Carlo"],
        "VaR (perte max.)": [f"{var_a:.2%}", f"{var_h:.2%}", f"{var_mc:.2%}"],
        "ES (perte moyenne si > VaR)": [f"{es_a:.2%}", f"{es_h:.2%}", f"{es_mc:.2%}"]
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.header("📊 Distribution des Rendements")
    
    fig_dist = go.Figure()
    returns_to_plot = sim_returns if len(sim_returns) > 0 else portfolio_returns
    
    fig_dist.add_trace(go.Histogram(x=returns_to_plot, nbinsx=100, name='Distribution', marker_color='#636EFA'))
    
    fig_dist.add_vline(x=var_h, line_dash="dash", line_color="red", annotation_text=f"VaR Hist: {var_h:.2%}", annotation_position="top left")
    fig_dist.add_vline(x=var_mc, line_dash="dash", line_color="orange", annotation_text=f"VaR MC: {var_mc:.2%}", annotation_position="top right")

    fig_dist.update_layout(
        title_text='Distribution des Rendements Quotidiens et Seuils de VaR',
        xaxis_title='Rendements Quotidiens',
        yaxis_title='Fréquence',
        xaxis_tickformat=".2%"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.info("Cette application est un outil de démonstration. Les résultats ne constituent pas un conseil en investissement.")