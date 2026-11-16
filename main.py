import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

print ("my first quant project")

stocks = ["AAPL" , "NVDA" , "AMZN" , "GOOGL"]

prices = yf.download(
    stocks,
    period="5y",
    auto_adjust=True
) ["Close"]
prices = prices.dropna()

## Calculating historical returns
returns = prices.pct_change().dropna()

## Calculate portfolio returns 
weights = np.array([0.30, 0.25, 0.15, 0.10])
portfolio_returns = returns.dot(weights)

# Initial portfolio investment
initial_investment = 100000

# Number of simulations
num_simulations = 10000

# Number of trading days
num_days = 30

# Expected daily portfolio return
daily_mean = portfolio_returns.mean()

# Daily portfolio volatility
daily_volatility = portfolio_returns.std()

## Creating Random number generator
rng = np.random.default_rng(42)

# Generate random returns 
simulated_returns = rng.normal(
    loc=daily_mean,
    scale=daily_volatility,
    size=(num_simulations, num_days))

## INSPECTING SIULATION ##
print(simulated_returns.shape)
print(simulated_returns[:5])

plt.figure(figsize=(12, 6))

plt.hist(
    simulated_returns.flatten(),
    bins=100,
    density=True,
    alpha=0.7,
    edgecolor="black"
)

plt.axvline(
    daily_mean,
    linestyle="--",
    label="Expected Daily Return"
)

plt.title("Monte Carlo Simulated Returns Distribution")

plt.xlabel("Daily Portfolio Returns")

plt.ylabel("Probability Density")

plt.legend()

plt.grid(alpha=0.3)

plt.show()