import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from arch import arch_model


# DATA RETRIEVAL & CLEANING

# Download raw data from Yahoo Finance
raw_data = yf.download('CL=F', start='2018-01-01', end='2026-03-28', auto_adjust=True)

# Select only the closing prices and filter out non-positive values (e.g., 2020 crash anomalies)
prices = raw_data['Close'].copy()
prices.columns = ['WTI']
prices = prices[prices > 0]

# Calculate log returns and drop missing values
log_returns = np.log(prices / prices.shift(1)).dropna()

# Plot 1: Daily Log Returns
plt.figure(figsize=(14, 5))
plt.plot(log_returns.index, log_returns, color='blue', linewidth=0.8)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('WTI Daily Log Returns (2018-2026)')
plt.ylabel('Log Returns')
plt.show()


# VOLATILITY MODELING (ROLLING & GARCH)

# 30-day rolling annualized volatility
rolling_vol = log_returns.rolling(30).std() * np.sqrt(252)
mean_vol = rolling_vol.mean().item()

# Plot 2: Rolling Volatility vs Historical Mean
plt.figure(figsize=(14, 5))
plt.plot(rolling_vol.index, rolling_vol, color="darkorange", linewidth=1.2)
plt.axhline(mean_vol, color="red", linewidth=1, linestyle="--")
plt.title("WTI 30-Day Rolling Volatility (Annualized)")
plt.ylabel("Volatility")
plt.show()

# Create a clean master DataFrame
data_clean = pd.DataFrame({
    'price': prices.squeeze(),
    'return': log_returns.squeeze(),
    'vol_30d': rolling_vol.squeeze()
}).dropna()

# GARCH Model Initialization 
returns_pct = log_returns.squeeze() * 100
model = arch_model(returns_pct, mean='constant', vol='GARCH', p=1, q=1, dist='t')
garch_fit = model.fit(disp='off')
print(garch_fit.summary())

# Annualized GARCH conditional volatility
garch_vol = garch_fit.conditional_volatility * np.sqrt(252) / 100

# Plot 3: GARCH Volatility vs Rolling Volatility
plt.figure(figsize=(14, 5))
plt.plot(garch_vol.index, garch_vol, color='blue', linewidth=1, label='GARCH')
plt.plot(rolling_vol.index, rolling_vol.squeeze(), color='orange', linewidth=1, alpha=0.7, label='Rolling 30d')
plt.axhline(mean_vol, color='red', linewidth=1, linestyle='--', label='Historical Mean')
plt.title('GARCH Volatility vs 30-Day Rolling Volatility - WTI')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# VOLATILITY REGIME CLASSIFICATION

# Define threshold and smooth the GARCH volatility to avoid noise
vol_threshold = garch_vol.mean()
data_clean['garch_vol'] = garch_vol.loc[data_clean.index].values
data_clean['garch_vol_smooth'] = data_clean['garch_vol'].rolling(5).mean()
data_clean = data_clean.dropna()

# Classification function: 1 for High Volatility, 0 for Low Volatility
def classify_regime(volatility):
    return 1 if volatility > vol_threshold else 0

data_clean['regime'] = data_clean['garch_vol_smooth'].apply(classify_regime)

# Plot 4: WTI Price with Volatility Regimes Highlighted
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(data_clean.index, data_clean['price'], color='steelblue', linewidth=1)

in_regime = False
start_date = None

for date, row in data_clean.iterrows():
    if row['regime'] == 1 and not in_regime:
        start_date = date
        in_regime = True
    elif row['regime'] == 0 and in_regime:
        ax.axvspan(start_date, date, alpha=0.2, color='red')
        in_regime = False

if in_regime:
    ax.axvspan(start_date, data_clean.index[-1], alpha=0.2, color='red')

ax.set_title('WTI Price with Smoothed Volatility Regimes (Red = High Vol)')
ax.set_ylabel("Price USD")
plt.show()


# BLACK-SCHOLES PRICING (STRADDLE)

def black_scholes(price, strike, ttm, rate, vol, opt_type):
    """Calculates theoretical option prices using the Black-Scholes model."""
    d1 = (np.log(price / strike) + (rate + 0.5 * vol**2) * ttm) / (vol * np.sqrt(ttm))
    d2 = d1 - vol * np.sqrt(ttm)
    if opt_type == "call":
        return price * norm.cdf(d1) - strike * np.exp(-rate * ttm) * norm.cdf(d2)
    else:
        return strike * np.exp(-rate * ttm) * norm.cdf(-d2) - price * norm.cdf(-d1)

# Calculate daily theoretical straddle premiums using GARCH volatility
TIME_TO_MATURITY = 30 / 252
RISK_FREE_RATE = 0.05
straddle_premiums = []

for i in range(len(data_clean)):
    s_i = data_clean["price"].iloc[i]
    k_i = round(s_i / 5) * 5  # Round to nearest 5 for dynamic strike
    sigma_i = data_clean["garch_vol_smooth"].iloc[i]
    
    call_i = black_scholes(s_i, k_i, TIME_TO_MATURITY, RISK_FREE_RATE, sigma_i, "call")
    put_i = black_scholes(s_i, k_i, TIME_TO_MATURITY, RISK_FREE_RATE, sigma_i, "put")
    straddle_premiums.append(call_i + put_i)

data_clean["straddle_premium"] = straddle_premiums


# ADAPTIVE SPOT TRADING STRATEGY (INVERTED / OPERATION MIRROR)

# Technical Indicators Setup
rolling_20 = data_clean['price'].rolling(20)
data_clean['BB_middle'] = rolling_20.mean()
data_clean['BB_upper'] = data_clean['BB_middle'] + (2.5 * rolling_20.std())
data_clean['BB_lower'] = data_clean['BB_middle'] - (2.5 * rolling_20.std())
data_clean['SMA_50'] = data_clean['price'].rolling(50).mean()

# ADX Calculation
diff = data_clean['price'].diff()
true_range = data_clean['price'].rolling(2).max() - data_clean['price'].rolling(2).min()
plus_dm = diff.clip(lower=0).rolling(14).mean()
data_clean['ADX'] = (100 * (plus_dm / true_range.rolling(14).mean())).rolling(14).mean()

# Strategy Conditions (Inverted logic based on preliminary underperformance)
conditions = [
    (data_clean['regime'] == 0) & (data_clean['price'] < data_clean['BB_lower']), # Was Buy, now Short
    (data_clean['regime'] == 0) & (data_clean['price'] > data_clean['BB_upper']), # Was Sell, now Buy
    (data_clean['regime'] == 1) & (data_clean['ADX'] > 15) & (data_clean['price'] > data_clean['SMA_50']), # Was Buy, now Short
    (data_clean['regime'] == 1) & (data_clean['ADX'] > 15) & (data_clean['price'] < data_clean['SMA_50'])  # Was Sell, now Buy
]
choices = [-1, 1, -1, 1]

data_clean['spot_signal'] = np.select(conditions, choices, default=0)

# Strategy Returns and Cumulative Performance
data_clean['spot_strategy_return'] = data_clean['return'] * data_clean['spot_signal'].shift(1)
data_clean['cum_spot_market'] = (1 + data_clean['return'].fillna(0)).cumprod()
data_clean['cum_spot_strategy'] = (1 + data_clean['spot_strategy_return'].fillna(0)).cumprod()

# Plot 5: Final Strategy Performance
plt.figure(figsize=(14, 5))
plt.plot(data_clean.index, data_clean['cum_spot_strategy'], color='blue', label='Inverted Adaptive Strategy')
plt.plot(data_clean.index, data_clean['cum_spot_market'], color='gray', alpha=0.5, label='Buy & Hold WTI')
plt.title('Performance of Inverted Adaptive Spot Strategy on WTI')
plt.legend()
plt.show()

# Final Metrics Output
sharpe_spot = (data_clean['spot_strategy_return'].mean() / data_clean['spot_strategy_return'].std()) * np.sqrt(252)
print("\n--- Final Strategy Performance ---")
print(f"Sharpe Ratio: {round(sharpe_spot, 2)}")
print("Signals Count:")
print(data_clean['spot_signal'].value_counts())