# 🛡️ Portfolio Risk Dashboard App

**A decision-support web application built with Streamlit for investors to assess portfolio risk through dynamic VaR and Expected Shortfall simulations.**

This interactive dashboard provides a comprehensive framework for market risk analysis. Users can define a custom multi-asset portfolio, select a historical period, and instantly compute and visualize key risk metrics using three industry-standard methodologies.

---

## 🚀 Live Demo

This application is deployed and publicly accessible via Streamlit Community Cloud. Click the link below to launch and interact with the dashboard directly in your browser — no installation required.

**[➡️ Launch the Portfolio Risk Dashboard App](https://portfolio-risk-dashboard-app-ugzlqfi5fb23ue66xg7wvb.streamlit.app/)**

---

## ✨ Features

-   **Custom Portfolio Definition:** Enter any combination of stock tickers (from Yahoo Finance) and their corresponding weights.
-   **Flexible Time Period:** Select any historical date range for the analysis.
-   **Interactive Risk Parameters:** Adjust the confidence level for VaR and ES calculations using a simple slider.
-   **Multi-Method Analysis:** Instantly compute and compare risk metrics from three different models:
    1.  **Analytical (Parametric)**: Assumes returns follow a normal distribution.
    2.  **Historical Simulation**: Non-parametric method based directly on past returns.
    3.  **Monte Carlo Simulation**: Generates thousands of random future scenarios.
-   **Dynamic Performance Metrics:** Automatically calculates and displays key performance indicators like Annualized Return, Annualized Volatility, and the Sharpe Ratio.
-   **Interactive Visualizations:** All charts are built with Plotly for a rich, interactive user experience (zoom, pan, hover-to-inspect).