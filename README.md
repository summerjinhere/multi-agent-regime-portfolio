# Multi-Agent Portfolio System v3.0

## 🎯 Overview

A comprehensive multi-agent portfolio optimization system with:
- **5 Sectors × 8 Stocks = 40 Equities**
- **5 Agents**: MarketData, ML (HMM), Quant (MS-VECM), Risk, Portfolio
- **Streamlit Dashboard**: Backtest results + Real-time monitoring
- **Statistical Validation**: Bootstrap, DSR, Newey-West

---

## 📁 Project Structure

```
enhanced_agent_v3/
├── main.py                 # Entry point
├── orchestrator.py         # Agent orchestration & backtest engine
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py         # System configuration
│
├── agents/
│   ├── base_agent.py       # Abstract base class
│   ├── market_data_agent.py    # Data fetching (Yahoo Finance)
│   ├── ml_agent.py         # HMM regime detection
│   ├── quant_agent.py      # MS-VECM expected returns
│   ├── risk_manager_agent.py   # VaR, position sizing
│   └── portfolio_agent.py  # Black-Litterman optimization
│
├── events/
│   └── message_bus.py      # Pub/sub event system
│
├── dashboard/
│   └── app.py              # Streamlit dashboard
│
└── results/                # Backtest outputs
```

---

## 🚀 Quick Start

### Installation

```bash
cd enhanced_agent_v3
pip install -r requirements.txt
```

### Run Backtest

```bash
python main.py --backtest
```

### Launch Dashboard

```bash
python main.py --dashboard
# or
streamlit run dashboard/app.py
```

### Run Both

```bash
python main.py --backtest --dashboard
```

---

## 📊 Asset Universe (55 Assets)

### Equities (40 stocks)

| Sector | Tickers |
|--------|---------|
| **Technology** | AAPL, MSFT, NVDA, AVGO, ORCL, ADBE, CRM, AMD |
| **Healthcare** | LLY, UNH, JNJ, MRK, ABBV, TMO, AMGN, PFE |
| **Financials** | JPM, V, MA, BAC, WFC, MS, GS, AXP |
| **Consumer** | AMZN, TSLA, HD, MCD, NKE, LOW, SBUX, TGT |
| **Energy** | XOM, CVX, COP, EOG, SLB, MPC, PSX, OXY |

### Fixed Income (5 ETFs)
TLT, IEF, SHY, LQD, HYG

### Commodities (4 ETFs)
GLD, USO, SLV, DBC

### Others (2 ETFs)
VNQ (REITs), SPY (Benchmark)

---

## 🤖 Agent Architecture

### 1. MarketDataAgent
- Fetches data from Yahoo Finance
- Caches data locally
- Calculates returns and quality metrics

### 2. MLAgent (HMM Regime Detection)
- 3-state Gaussian HMM
- Features: Returns, Volatility
- Outputs: Regime (Bull/Transition/Bear), Probabilities

### 3. QuantAgent (MS-VECM)
- Johansen cointegration test
- Vector Error Correction Model
- Regime-adjusted expected returns
- Ledoit-Wolf covariance shrinkage

### 4. RiskManagerAgent
- VaR/CVaR calculation
- Drawdown monitoring
- Position limits by regime
- Risk parity budgeting

### 5. PortfolioAgent
- Black-Litterman model
- Mean-variance optimization
- Turnover constraints
- Rebalancing decisions

---

## ⚙️ Configuration

Edit `config/settings.py` or create `config/agent_config.yaml`:

```yaml
market_data:
  start_date: '2015-01-01'
  end_date: '2025-11-30'
  sectors:
    Technology: [AAPL, MSFT, NVDA, ...]

ml:
  n_components: 3
  covariance_type: 'full'

quant:
  lookback_window: 252
  use_shrinkage: true

risk:
  max_drawdown_limit: 0.20
  var_confidence: 0.95
  regime_params:
    0:  # Bull
      risk_multiplier: 1.0
      max_position: 0.30
    2:  # Bear
      risk_multiplier: 0.70
      max_position: 0.20

portfolio:
  use_black_litterman: true
  rebalance_threshold: 0.03

backtest:
  initial_capital: 1000000
  start_date: '2020-01-01'
  rebalance_freq: 'M'
```

---

## 📈 Dashboard Features

### Backtest Results View
- Performance metrics (CAGR, Sharpe, Max DD)
- Cumulative returns vs benchmark
- Drawdown chart
- Regime distribution

### Real-time Monitor View
- Current regime indicator
- Portfolio value
- Risk level
- Live price chart

### Analytics View
- Monthly returns heatmap
- Rolling metrics (252-day)
- Return distribution
- Q-Q plot

---

## 📊 Output Files

After running backtest:

```
results/
├── returns.csv           # Daily returns
├── portfolio_values.csv  # Portfolio value series
├── weights_history.csv   # Historical weights
└── metrics.csv           # Performance metrics
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📚 Methodology

### Regime Detection (HMM)
- Gaussian Hidden Markov Model with 3 states
- States ordered by volatility: Bull (low) → Bear (high)
- Features: log returns, rolling volatility

### Expected Returns (MS-VECM)
1. Johansen cointegration test on asset classes
2. If cointegrated: Fit VECM, forecast returns
3. If not: Fall back to mean-variance
4. Apply regime-based multipliers

### Portfolio Optimization
1. Black-Litterman: Combine market equilibrium with views
2. Mean-variance with risk aversion γ (regime-dependent)
3. Position limits and sector constraints
4. Transaction cost awareness

---

## ⚠️ Disclaimer

This software is for educational and research purposes only.
Not financial advice. Past performance does not guarantee future results.

---

## 📝 Version History

### v3.0 (December 2025)
- Full universe (55 assets across 5 sectors)
- All 5 agents (MarketData, ML, Quant, Risk, Portfolio)
- MS-VECM for cointegrated assets
- Streamlit dashboard with 3 views
- Event-driven architecture
