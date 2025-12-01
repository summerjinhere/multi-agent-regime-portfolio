# Enhanced Multi-Agent Portfolio Optimization System

## Version 2.0.0 | November 2025

A comprehensive portfolio optimization framework integrating:
- **Regime-adaptive risk management** with variable risk aversion
- **TWAP/VWAP execution algorithms** for optimal trade execution
- **Institutional flow tracking** with piggyback trading (10% profit target)
- **NLP sentiment analysis** from social media (English only, nuance-aware)
- **Enhanced regime detection** using futures positioning and investor flows
- **Rigorous statistical validation** (OLS, Bootstrap, Deflated Sharpe Ratio)

---

## 📁 Project Structure

```
enhanced_agent/
├── __init__.py              # Package initialization
├── main.py                  # Main orchestrator coordinating all agents
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── config/
│   ├── __init__.py
│   └── settings.py         # System configuration with regime parameters
│
├── models/
│   ├── __init__.py
│   ├── enhanced_regime_detector.py  # HMM + Futures + Flow regime detection
│   └── regime_optimizer.py          # Mean-variance with regime-adaptive γ
│
├── execution/
│   ├── __init__.py
│   └── execution_algos.py  # TWAP, VWAP, Adaptive execution
│
├── agents/
│   ├── __init__.py
│   └── institutional_flow_agent.py  # Flow detection & piggyback trading
│
├── nlp/
│   ├── __init__.py
│   └── sentiment_analyzer.py  # NLP sentiment with nuance learning
│
├── analysis/
│   ├── __init__.py
│   └── statistical_analysis.py  # OLS, Bootstrap, DSR validation
│
├── data/
│   ├── __init__.py
│   └── data_loader.py      # Data loading and export utilities
│
├── backtesting/
│   └── __init__.py
│
├── events/
│   └── __init__.py
│
└── utils/
    └── __init__.py
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone or extract the package
cd enhanced_agent

# Install dependencies
pip install -r requirements.txt

# For HMM support (recommended)
pip install hmmlearn

# For sentiment analysis with transformers (optional)
pip install transformers torch
```

### Basic Usage

```python
from main import MultiAgentOrchestrator
from config.settings import SystemConfig
from data.data_loader import DataLoader

# Initialize
config = SystemConfig()
orchestrator = MultiAgentOrchestrator(config)

# Load data (or use synthetic)
loader = DataLoader()
prices, volumes = loader.generate_synthetic(
    tickers=config.universe,
    start_date='2018-01-01',
    end_date='2025-11-30'
)

# Run backtest
result = orchestrator.run_backtest(prices, volumes)

# Print results
orchestrator.print_results(result)
```

### Using Real Data (Yahoo Finance)

```python
loader = DataLoader()
prices = loader.load_yahoo(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    start_date='2020-01-01',
    end_date='2025-11-30'
)
```

---

## ⚙️ Key Features

### 1. Regime-Adaptive Risk Aversion

The optimizer adjusts risk aversion (γ) based on detected market regime:

| Regime | Risk Aversion (γ) | Max Position | Risk Multiplier |
|--------|-------------------|--------------|-----------------|
| Bull (Low Vol) | 2.0 | 30% | 1.00 |
| Normal | 3.0 | 25% | 0.90 |
| Bear (High Vol) | 5.0 | 20% | 0.70 |
| Crisis | 8.0 | 15% | 0.50 |
| Recovery | 2.5 | 28% | 0.85 |

```python
# Customize risk aversion
from models.regime_optimizer import RegimeAdaptiveOptimizer

optimizer = RegimeAdaptiveOptimizer(tickers)
optimizer.set_regime_risk_aversion(regime=3, risk_aversion=10.0)  # Crisis
```

### 2. TWAP/VWAP Execution

```python
from execution.execution_algos import ExecutionEngine, ExecutionAlgo, OrderSide

engine = ExecutionEngine({
    'twap_intervals': 10,
    'max_participation': 0.10  # Max 10% of volume
})

# Submit TWAP order
order = engine.submit_order(
    ticker='AAPL',
    side=OrderSide.BUY,
    quantity=1000,
    algo=ExecutionAlgo.TWAP,
    duration_minutes=30
)

# Or VWAP for volume-following
order = engine.submit_order(
    ticker='AAPL',
    side=OrderSide.BUY,
    quantity=1000,
    algo=ExecutionAlgo.VWAP
)
```

### 3. Institutional Flow & Piggyback Trading

```python
from agents.institutional_flow_agent import PiggybackTrader, InstitutionalFlowDetector

# Detect institutional accumulation
detector = InstitutionalFlowDetector({
    'accumulation_threshold': 0.15,
    'lookback_days': 20
})

signal = detector.analyze_flow(ticker, prices, volumes)

# Piggyback trade with 10% profit target
trader = PiggybackTrader({
    'take_profit_pct': 0.10,  # Exit at +10%
    'stop_loss_pct': 0.05,    # Exit at -5%
    'short_tracking_enabled': True
})

position = trader.evaluate_entry(
    ticker='AAPL',
    signal=signal,
    current_price=150.0,
    available_capital=100000
)
```

### 4. NLP Sentiment Analysis

```python
from nlp.sentiment_analyzer import NuanceSentimentAnalyzer, MarketSentimentAgent

# Nuance-aware analyzer (English only)
analyzer = NuanceSentimentAnalyzer(use_transformer=False)

text = "AAPL looking incredibly bullish! 🚀 Major breakout incoming"
sentiment, confidence, keywords = analyzer.analyze_text(text)
# sentiment: 0.75, confidence: 0.8, keywords: [('bullish', 0.9), ('breakout', 0.85)]

# For transformer-based (requires transformers package)
analyzer = NuanceSentimentAnalyzer(use_transformer=True)  # Uses FinBERT
```

### 5. Enhanced Regime Detection

Combines multiple signals:
- HMM on price/volatility (40% weight)
- Nasdaq futures positioning (20% weight)
- Institutional/Retail/Foreign investor flows (25% weight)
- Sentiment analysis (15% weight)

```python
from models.enhanced_regime_detector import EnhancedRegimeDetector

detector = EnhancedRegimeDetector(n_regimes=5, config={
    'hmm_weight': 0.40,
    'futures_weight': 0.20,
    'flow_weight': 0.25,
    'sentiment_weight': 0.15
})

# Fit on historical data
detector.fit_hmm(returns, volatility)

# Get regime assessment
assessment = detector.detect_regime(
    current_return=0.01,
    current_volatility=0.02,
    futures_data={'commercial_net': 10000, ...},
    flow_data={'institutional_buy': 1e6, ...},
    sentiment_score=0.3
)

print(f"Regime: {assessment.primary_regime}")
print(f"Risk Level: {assessment.risk_level:.2f}")
print(f"Recommended Exposure: {assessment.recommended_exposure:.1%}")
```

### 6. Statistical Validation

```python
from analysis.statistical_analysis import (
    StatisticalValidator, DeflatedSharpeRatio, BlockBootstrap
)

# Full validation suite
validator = StatisticalValidator(confidence_level=0.95)
results = validator.full_validation(
    returns=strategy_returns,
    benchmark_returns=spy_returns,
    n_trials=10  # Number of strategy variations tried
)

# Individual tests
dsr = DeflatedSharpeRatio.compute(returns, n_trials=10)
print(f"Sharpe: {dsr.sharpe_ratio:.2f}, t-stat: {dsr.t_statistic:.2f}")
print(f"PSR: {dsr.psr:.1%}, DSR: {dsr.dsr:.1%}")

# Bootstrap
bootstrap = BlockBootstrap(block_size=21, n_bootstrap=10000)
boot_result = bootstrap.bootstrap_sharpe(returns)
print(f"95% CI: [{boot_result.ci_lower:.2f}, {boot_result.ci_upper:.2f}]")
```

---

## 📊 Configuration

Edit `config/settings.py` or create a YAML config:

```python
from config.settings import SystemConfig

config = SystemConfig(
    # Universe
    universe=['AAPL', 'MSFT', 'GOOGL', ...],
    benchmark='SPY',
    
    # Capital
    initial_capital=1_000_000,
    risk_free_rate=0.05,
    
    # Rebalancing
    rebalance_frequency='weekly',  # daily, weekly, monthly
    max_turnover=0.30,
    
    # Risk limits
    max_sector_exposure=0.35,
    max_drawdown_limit=0.20,
    var_limit_95=0.03
)
```

---

## 📈 Output Metrics

The system calculates comprehensive metrics:

**Performance:**
- Total Return, CAGR, Volatility
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Calmar Ratio

**Risk:**
- VaR (95%), CVaR (95%)
- Beta, Alpha
- Win Rate

**Statistical:**
- t-statistic, p-value
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Bootstrap Confidence Interval

---

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📚 References

1. Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series.
2. Bailey, D.H. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
3. Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts.
4. Lo, A.W. (2002). The statistics of Sharpe ratios.
5. Nystrup, P. et al. (2017). Long memory of financial time series and HMM.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🔄 Changelog

### v2.0.0 (November 2025)
- Added regime-adaptive risk aversion
- Implemented TWAP/VWAP execution algorithms
- Added institutional flow tracking with piggyback trading
- Integrated NLP sentiment analysis with nuance learning
- Added futures positioning to regime detection
- Enhanced statistical validation (DSR, Bootstrap)
- Comprehensive documentation and examples
