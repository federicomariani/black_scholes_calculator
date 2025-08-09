import streamlit as st
import pandas as pd
from black_scholes import black_scholes, implied_volatility
import plotly.graph_objects as go

# Valori iniziali
default_params = {
    "S": 100.0,
    "K": 100.0,
    "r": 1.0,
    "sigma": 20.0,
    "T": 1.0,
    "q": 0.0,
    "option_type": "call",
    "market_price": 0.0
}

# Inizializza session state se non esiste
for key, value in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Pulsante reset
#def reset_params():
    #for key, value in default_params.items():
        #st.session_state[key] = value

#st.sidebar.button("🔄 Reset parameters", on_click=reset_params)

st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

# ===== Sidebar inputs =====
st.sidebar.header("Option parameters")

S = st.sidebar.number_input("Underlying price (S)", value=st.session_state["S"], step=0.5)
K = st.sidebar.number_input("Strike (K)", value=st.session_state["K"], step=0.5)
r = st.sidebar.number_input("Risk-free rate r (%)", value=st.session_state["r"], step=0.1) / 100
sigma = st.sidebar.number_input("Volatility σ (%)", value=st.session_state["sigma"], step=0.1) / 100
T = st.sidebar.number_input("Time to maturity (years)", value=st.session_state["T"], step=0.1)
q = st.sidebar.number_input("Dividend yield q (%)", value=st.session_state["q"], step=0.1) / 100
option_type = st.sidebar.radio("Option type", ("call", "put"), index=0 if st.session_state["option_type"]=="call" else 1)
market_price = st.sidebar.number_input("Market price (for IV calculation)", value=st.session_state["market_price"], step=0.1)

# ===== Calcoli Black-Scholes =====
results = black_scholes(S, K, r, sigma, T, option_type, q)

# ===== Mostra risultati =====
st.title("📈 Black-Scholes Option Pricing")

st.subheader("Price e Greeks")
df = pd.DataFrame(results, index=[0]).T
df.columns = ["Value"]
st.dataframe(df.style.format("{:.6f}"))

# ===== Calcolo IV se prezzo di mercato inserito =====
if market_price > 0:
    iv = implied_volatility(market_price, S, K, r, T, option_type, q)
    if pd.notna(iv):
        st.success(f"Estimated implied volatility: {iv*100:.4f}%")
    else:
        st.error("Implied volatility could not be calculated.")

import numpy as np
import matplotlib.pyplot as plt

# ===== Grafico 1: Payoff a scadenza =====
st.subheader("Payoff diagram at expiration")

S_range = np.linspace(0.5*S, 1.5*S, 100)
if option_type == "call":
    payoff = np.maximum(S_range - K, 0)
else:
    payoff = np.maximum(K - S_range, 0)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=S_range, y=payoff,
    mode='lines', name="Payoff at maturity",
    line=dict(color="blue")
))
fig1.add_vline(x=S, line=dict(color="gray", dash="dash"), annotation_text="Current price S", annotation_font_color="black")
fig1.update_layout(
    xaxis_title="Underlying price",
    yaxis_title="Payoff",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis = dict(color = "black", tickfont = dict(color = "black"), title=dict(text="Underlying price", font=dict(color="black"))),
    yaxis = dict(color = "black", tickfont = dict(color = "black"),  title=dict(text="Payoff", font=dict(color="black")))
)
st.plotly_chart(fig1, use_container_width=True)

# ===== Grafico 2: Prezzo opzione vs S =====
st.subheader("Theoretical price vs. Underlying price")

S_vals = np.linspace(0.5*S, 1.5*S, 50)
prices = [black_scholes(s, K, r, sigma, T, option_type, q)["price"] for s in S_vals]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=S_vals, y=prices,
    mode='lines', name=f"Price {option_type}",
    line=dict(color="green")
))
fig2.add_vline(x=S, line=dict(color="gray", dash="dash"), annotation_text="Current price S", annotation_font_color="black")
fig2.update_layout(
    xaxis_title="Underlying price S",
    yaxis_title="Option price",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis = dict(color = "black", tickfont = dict(color = "black"), title=dict(text="Underlying price", font=dict(color="black"))),
    yaxis = dict(color = "black", tickfont = dict(color = "black"),  title=dict(text="Option price", font=dict(color="black")))
)
st.plotly_chart(fig2, use_container_width=True)

import plotly.graph_objects as go

# ===== Grafico 3: Prezzo vs σ =====
st.subheader("Theoretical price vs. Volatility σ")

sigma_vals = np.linspace(0.01, 1.0, 50)
prices_sigma = [black_scholes(S, K, r, sig, T, option_type, q)["price"] for sig in sigma_vals]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=sigma_vals*100, y=prices_sigma,
    mode='lines', name="Theoretical price",
    line=dict(color="purple")
))
fig3.add_vline(x=sigma*100, line=dict(color="gray", dash="dash"), annotation_text="σ current", annotation_font_color = "black")
fig3.update_layout(
    xaxis_title="Volatilità (%)",
    yaxis_title="Prezzo opzione",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis = dict(color = "black", tickfont = dict(color = "black"), title=dict(text="Volatility (%)", font=dict(color="black"))),
    yaxis = dict(color = "black", tickfont = dict(color = "black"),  title=dict(text="Option price", font=dict(color="black")))
)
st.plotly_chart(fig3, use_container_width=True)

# ===== Grafico 4: Superficie 3D Prezzo(S, σ) =====
st.subheader("Theoretical price surface: S × σ")

S_grid = np.linspace(0.5*S, 1.5*S, 30)
sigma_grid = np.linspace(0.05, 0.8, 30)  # 5% a 80%
S_mesh, sigma_mesh = np.meshgrid(S_grid, sigma_grid)

price_mesh = np.zeros_like(S_mesh)
for i in range(S_mesh.shape[0]):
    for j in range(S_mesh.shape[1]):
        price_mesh[i, j] = black_scholes(S_mesh[i, j], K, r, sigma_mesh[i, j], T, option_type, q)["price"]

fig4 = go.Figure(data=[go.Surface(
    x=S_mesh,
    y=sigma_mesh*100,
    z=price_mesh,
    colorscale="Viridis"
)])
fig4.update_layout(
    scene=dict(
        xaxis=dict(
            title=dict(text="Price S", font=dict(color="black")),
            tickfont=dict(color="black"),
            color="black"
        ),
        yaxis=dict(
            title=dict(text="Volatility (%)", font=dict(color="black")),
            tickfont=dict(color="black"),
            color="black"
        ),
        zaxis=dict(
            title=dict(text="Option price", font=dict(color="black")),
            tickfont=dict(color="black"),
            color="black"
        )
    ),
    paper_bgcolor="white",
    plot_bgcolor="white"
)
st.plotly_chart(fig4, use_container_width=True)

# streamlit run app.py