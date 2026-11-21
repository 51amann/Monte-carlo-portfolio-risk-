import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG + VISUAL STYLE
# ============================================================
st.set_page_config(
    page_title="Monte Carlo Risk Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 15% 0%, #172554 0%, #0b1220 38%, #080d18 100%);
        color: #e5edf7;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101b31 0%, #0b1220 100%);
        border-right: 1px solid #263653;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #14233b, #101a2c);
        border: 1px solid #293d5c;
        padding: 17px 18px;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    [data-testid="stMetricLabel"] { color: #9fb2cc !important; }
    [data-testid="stMetricValue"] { color: #f1f6ff !important; }
    h1, h2, h3 { color: #f1f6ff !important; }
    .eyebrow {
        color: #42d6c5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .subtle { color: #9fb2cc; }
    .panel {
        background: rgba(18, 31, 52, .88);
        border: 1px solid #293d5c;
        border-radius: 16px;
        padding: 18px 20px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #16b8a6, #4778e8);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
    }
    div.stButton > button:hover {
        border: 1px solid #74e5d8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Quant Finance · Portfolio Risk</div>', unsafe_allow_html=True)
st.title("Monte Carlo Risk Lab")
st.markdown(
    '<p class="subtle">An interactive portfolio simulation and decision-support dashboard. '
    'Change the assumptions, run the model, and inspect the distribution of possible outcomes.</p>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.markdown("## ⚙️ Model inputs")
st.sidebar.caption("Set your portfolio and simulation assumptions.")

investment = st.sidebar.number_input(
    "Initial investment ($)", min_value=100.0, max_value=100_000_000.0,
    value=100_000.0, step=5_000.0, format="%.2f"
)
tickers_text = st.sidebar.text_input(
    "Stock tickers (comma-separated)", value="AAPL, MSFT, NVDA, AMZN, GOOGL"
)
weights_text = st.sidebar.text_input(
    "Portfolio weights (%)", value="30, 25, 20, 15, 10",
    help="Enter one weight per ticker, in the same order. Weights must total 100."
)
lookback = st.sidebar.selectbox("Historical lookback", ["1y", "2y", "5y", "10y"], index=2)
simulations = st.sidebar.select_slider(
    "Number of simulations", options=[1000, 2500, 5000, 10000, 20000, 50000], value=10000
)
horizon = st.sidebar.slider("Forecast horizon (trading days)", 5, 252, 30, 5)
confidence = st.sidebar.selectbox("Risk confidence level", [90, 95, 99], index=1)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

run_model = st.sidebar.button("▶ Run Monte Carlo simulation", use_container_width=True)

# ============================================================
# INPUT VALIDATION + MODEL
# ============================================================
def parse_portfolio(ticker_input, weight_input):
    tickers = [x.strip().upper() for x in ticker_input.split(",") if x.strip()]
    try:
        weights = np.array([float(x.strip()) for x in weight_input.split(",") if x.strip()], dtype=float)
    except ValueError:
        raise ValueError("Weights must be numeric, separated by commas.")

    if not tickers:
        raise ValueError("Enter at least one ticker.")
    if len(tickers) != len(weights):
        raise ValueError("The number of weights must match the number of tickers.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Remove duplicate ticker symbols.")
    if np.any(weights < 0):
        raise ValueError("This starter model does not support negative weights or short positions.")
    if not np.isclose(weights.sum(), 100.0, atol=0.05):
        raise ValueError(f"Weights must total 100%. Current total: {weights.sum():.2f}%.")
    return tickers, weights / 100.0


def run_simulation(tickers, weights, period, n_sims, n_days, initial_value, conf, random_seed):
    raw = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No market data was returned. Check ticker symbols or try another lookback.")

    # yfinance normally returns a MultiIndex for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Could not find adjusted closing prices in the downloaded data.")
        prices = raw["Close"]
    else:
        # Defensive handling for a single-ticker response.
        if "Close" not in raw.columns:
            raise ValueError("Could not find closing prices in the downloaded data.")
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    prices = prices.reindex(columns=tickers).dropna(how="any")
    if prices.shape[1] != len(tickers) or len(prices) < 30:
        raise ValueError("Insufficient overlapping price history for all selected tickers.")

    # Log returns are used so compounded simulated prices stay positive.
    log_returns = np.log(prices / prices.shift(1)).dropna()
    portfolio_log_returns = log_returns.dot(weights)

    daily_mean = float(portfolio_log_returns.mean())
    daily_vol = float(portfolio_log_returns.std(ddof=1))
    if not np.isfinite(daily_mean) or not np.isfinite(daily_vol) or daily_vol <= 0:
        raise ValueError("Could not estimate valid historical portfolio returns and volatility.")

    rng = np.random.default_rng(int(random_seed))
    simulated_log_returns = rng.normal(
        loc=daily_mean,
        scale=daily_vol,
        size=(int(n_sims), int(n_days)),
    )

    cumulative_log_returns = np.cumsum(simulated_log_returns, axis=1)
    values = np.empty((int(n_sims), int(n_days) + 1), dtype=float)
    values[:, 0] = initial_value
    values[:, 1:] = initial_value * np.exp(cumulative_log_returns)

    final_values = values[:, -1]
    losses = initial_value - final_values
    var_value = float(np.percentile(losses, conf))
    tail_losses = losses[losses >= var_value]
    es_value = float(tail_losses.mean()) if len(tail_losses) else var_value

    return {
        "prices": prices,
        "values": values,
        "final_values": final_values,
        "losses": losses,
        "mean": float(final_values.mean()),
        "median": float(np.median(final_values)),
        "prob_loss": float(np.mean(final_values < initial_value) * 100),
        "var": var_value,
        "es": es_value,
        "p5": float(np.percentile(final_values, 5)),
        "p50": float(np.percentile(final_values, 50)),
        "p95": float(np.percentile(final_values, 95)),
        "daily_mean": daily_mean,
        "daily_vol": daily_vol,
        "confidence": conf,
    }


# ============================================================
# RUN / STORE RESULTS
# ============================================================
if run_model:
    try:
        tickers, weights = parse_portfolio(tickers_text, weights_text)
        with st.spinner("Downloading market data and simulating possible portfolio outcomes..."):
            result = run_simulation(
                tickers, weights, lookback, simulations, horizon,
                investment, confidence, seed
            )
        st.session_state["mc_result"] = result
        st.session_state["mc_inputs"] = {
            "tickers": tickers,
            "weights": weights,
            "investment": investment,
            "simulations": simulations,
            "horizon": horizon,
            "lookback": lookback,
            "confidence": confidence,
        }
    except Exception as exc:
        st.error(f"Model could not run: {exc}")

# ============================================================
# DASHBOARD OUTPUT
# ============================================================
if "mc_result" not in st.session_state:
    st.info("Set your assumptions in the sidebar, then click **Run Monte Carlo simulation**.")
    st.markdown("### What this dashboard will show")
    a, b, c = st.columns(3)
    a.metric("Simulation paths", f"{simulations:,}")
    b.metric("Forecast horizon", f"{horizon} trading days")
    c.metric("Risk confidence", f"{confidence}%")
    st.markdown("""
    <div class="panel">
      <b>Model methodology</b><br><br>
      Historical adjusted prices → daily log returns → weighted portfolio returns →
      normal-distribution Monte Carlo simulation → compounded portfolio paths →
      VaR, Expected Shortfall and outcome analysis.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

result = st.session_state["mc_result"]
inputs = st.session_state["mc_inputs"]
final_values = result["final_values"]
values = result["values"]
losses = result["losses"]
conf = result["confidence"]
currency = "$"

st.markdown("### Portfolio overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Expected final value", f"{currency}{result['mean']:,.0f}",
          delta=f"{(result['mean']/inputs['investment']-1)*100:+.2f}%")
m2.metric("Median final value", f"{currency}{result['median']:,.0f}",
          delta=f"{(result['median']/inputs['investment']-1)*100:+.2f}%")
m3.metric("Probability of loss", f"{result['prob_loss']:.1f}%",
          delta="Chance of finishing below start", delta_color="off")
m4.metric(f"{conf}% Value at Risk", f"{currency}{result['var']:,.0f}",
          delta="Loss threshold", delta_color="off")

m5, m6, m7, m8 = st.columns(4)
m5.metric(f"{conf}% Expected Shortfall", f"{currency}{result['es']:,.0f}")
m6.metric("5th percentile value", f"{currency}{result['p5']:,.0f}")
m7.metric("95th percentile value", f"{currency}{result['p95']:,.0f}")
m8.metric("Historical daily volatility", f"{result['daily_vol']*100:.3f}%")

st.markdown("---")
left, right = st.columns([1.55, 1])

with left:
    st.markdown("### Final portfolio value distribution")
    hist_df = pd.DataFrame({"Final portfolio value": final_values})
    fig_hist = px.histogram(
        hist_df, x="Final portfolio value", nbins=65,
        color_discrete_sequence=["#32c9bd"], opacity=0.88,
    )
    fig_hist.add_vline(x=inputs["investment"], line_dash="dash",
                       line_color="#f5b942", annotation_text="Initial investment")
    fig_hist.add_vline(x=result["p50"], line_dash="dot",
                       line_color="#a78bfa", annotation_text="Median")
    fig_hist.update_layout(
        template="plotly_dark", height=390, bargap=0.06,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,22,38,.6)",
        xaxis_title="Final portfolio value ($)", yaxis_title="Number of simulations",
        margin=dict(l=10, r=10, t=35, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    st.markdown("### Outcome percentiles")
    pct_df = pd.DataFrame({
        "Outcome": ["5th percentile", "Median", "95th percentile"],
        "Portfolio value": [result["p5"], result["p50"], result["p95"]],
    })
    fig_pct = px.bar(
        pct_df, x="Outcome", y="Portfolio value", text_auto=",.0f",
        color="Outcome",
        color_discrete_map={
            "5th percentile": "#f97373",
            "Median": "#a78bfa",
            "95th percentile": "#32c9bd",
        },
    )
    fig_pct.update_layout(
        template="plotly_dark", height=390, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,22,38,.6)",
        yaxis_title="Portfolio value ($)", xaxis_title="",
        margin=dict(l=10, r=10, t=35, b=10),
    )
    st.plotly_chart(fig_pct, use_container_width=True)

st.markdown("### Simulated portfolio paths")
path_fig = go.Figure()
days = np.arange(values.shape[1])
for i in range(min(100, values.shape[0])):
    path_fig.add_trace(go.Scatter(
        x=days, y=values[i], mode="lines",
        line=dict(color="rgba(50,201,189,0.16)", width=1),
        showlegend=False, hovertemplate="Day %{x}<br>Value $%{y:,.0f}<extra></extra>",
    ))
path_fig.add_hline(y=inputs["investment"], line_dash="dash",
                   line_color="#f5b942", annotation_text="Initial investment")
path_fig.update_layout(
    template="plotly_dark", height=440,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,22,38,.6)",
    xaxis_title="Trading day", yaxis_title="Portfolio value ($)",
    margin=dict(l=10, r=10, t=25, b=10),
)
st.plotly_chart(path_fig, use_container_width=True)

st.markdown("### Loss distribution and tail risk")
loss_fig = px.histogram(
    pd.DataFrame({"Simulated loss": losses}),
    x="Simulated loss", nbins=65,
    color_discrete_sequence=["#8b7cf6"],
)
loss_fig.add_vline(x=result["var"], line_dash="dash", line_color="#f5b942",
                   annotation_text=f"{conf}% VaR")
loss_fig.add_vline(x=result["es"], line_dash="dot", line_color="#fb7185",
                   annotation_text=f"{conf}% ES")
loss_fig.update_layout(
    template="plotly_dark", height=360,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,22,38,.6)",
    xaxis_title="Loss ($); negative values represent gains",
    yaxis_title="Number of simulations",
    margin=dict(l=10, r=10, t=35, b=10),
)
st.plotly_chart(loss_fig, use_container_width=True)

# ============================================================
# PORTFOLIO + INTERPRETATION
# ============================================================
st.markdown("### Portfolio allocation")
allocation = pd.DataFrame({
    "Ticker": inputs["tickers"],
    "Weight": inputs["weights"],
})
alloc_fig = px.pie(
    allocation, names="Ticker", values="Weight", hole=0.62,
    color_discrete_sequence=["#32c9bd", "#668cf2", "#a78bfa", "#f5b942", "#fb7185"],
)
alloc_fig.update_traces(textposition="inside", textinfo="label+percent")
alloc_fig.update_layout(
    template="plotly_dark", height=330,
    paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=15, b=10),
)
st.plotly_chart(alloc_fig, use_container_width=True)

st.markdown("### Model interpretation")
var_pct = result["var"] / inputs["investment"] * 100
es_pct = result["es"] / inputs["investment"] * 100
st.markdown(f"""
<div class="panel">
  <b>What the simulation says</b><br><br>
  Across <b>{inputs['simulations']:,}</b> simulated paths over
  <b>{inputs['horizon']}</b> trading days, the average final portfolio value is
  <b>${result['mean']:,.2f}</b>. The simulated probability of finishing below the
  initial investment is <b>{result['prob_loss']:.2f}%</b>.<br><br>
  At the <b>{conf}% confidence level</b>, the estimated loss threshold is
  <b>${result['var']:,.2f}</b> ({var_pct:.2f}% of the initial investment).
  The average loss among outcomes at or beyond that threshold is
  <b>${result['es']:,.2f}</b> ({es_pct:.2f}% of the initial investment).
</div>
""", unsafe_allow_html=True)

st.caption(
    "Model limitations: this is a simplified parametric Monte Carlo model. "
    "It assumes normally distributed, independent daily portfolio log returns with "
    "constant historical mean and volatility. It does not model changing correlations, "
    "jumps, volatility clustering, fees, taxes, liquidity or market impact. "
    "VaR is a quantile, not a maximum-loss guarantee. Outputs are estimates, not forecasts."
)

# Downloadable summary
summary = pd.DataFrame({
    "Metric": [
        "Initial investment", "Expected final value", "Median final value",
        "Probability of loss (%)", f"{conf}% VaR", f"{conf}% Expected Shortfall",
        "5th percentile final value", "95th percentile final value",
    ],
    "Value": [
        inputs["investment"], result["mean"], result["median"],
        result["prob_loss"], result["var"], result["es"], result["p5"], result["p95"],
    ],
})
st.download_button(
    "⬇ Download risk report (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="monte_carlo_risk_report.csv",
    mime="text/csv",
)
