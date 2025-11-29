# Multi-Agent Portfolio Optimization System Architecture

## Executive Summary

This document outlines the transformation of the existing HMM + MS-VECM + Black-Litterman portfolio optimization model into a **Multi-Agent System (MAS)**. The architecture preserves all existing quantitative functionality while introducing autonomous agents that communicate through a centralized message bus.

---

## 1. Agent Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR AGENT                                  │
│                    (Coordinates workflow, handles failures)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA AGENT                                      │
│  • Fetches price data (Yahoo Finance)                                           │
│  • Preprocesses & validates data quality                                        │
│  • Publishes: MarketDataEvent                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
┌────────────────────────────────┐    ┌────────────────────────────────────────────┐
│        QUANT AGENT             │    │              ML AGENT                       │
│  • MS-VECM cointegration       │    │  • HMM regime detection                    │
│  • Johansen test               │    │  • Feature engineering                     │
│  • Expected returns forecast   │    │  • State probabilities                     │
│  • Covariance estimation       │    │  • Regime transition matrix                │
│  Publishes: QuantSignalEvent   │    │  Publishes: RegimeEvent                    │
└────────────────────────────────┘    └────────────────────────────────────────────┘
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RISK MANAGER AGENT                                      │
│  • Position sizing (Kelly Criterion / Risk Parity)                              │
│  • Drawdown limits & circuit breakers                                           │
│  • Regime-dependent risk constraints                                            │
│  • VaR / CVaR estimation                                                        │
│  • Correlation regime monitoring                                                │
│  Publishes: RiskAdjustedSignalEvent                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PORTFOLIO AGENT                                         │
│  • Black-Litterman optimization                                                 │
│  • Transaction cost modeling (regime-dependent slippage)                        │
│  • Rebalancing decision engine                                                  │
│  • Generates: BUY / SELL / HOLD signals                                         │
│  Publishes: TradeSignalEvent                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION AGENT (Optional)                              │
│  • Order management                                                             │
│  • Slippage tracking                                                            │
│  • Fill simulation (backtest) / Real execution (live)                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Definitions

### 2.1 Base Agent Interface

All agents inherit from a common `BaseAgent` class providing:
- Lifecycle management (`start`, `stop`, `pause`)
- Event subscription/publication via message bus
- Logging and error handling
- State persistence (for recovery)

### 2.2 Market Data Agent

**Responsibility**: Data acquisition, validation, and distribution

| Component | Source Module | Description |
|-----------|---------------|-------------|
| `DataFetcher` | `data_loader.py` | Yahoo Finance API wrapper |
| `DataValidator` | New | Quality checks (missing data, outliers, corporate actions) |
| `DataCache` | New | Local caching with TTL |

**Events Published**:
- `MarketDataEvent`: Contains `prices_df`, `metadata`, `timestamp`

**Events Consumed**: None (initiates pipeline)

### 2.3 Quant Agent

**Responsibility**: Econometric modeling and expected return estimation

| Component | Source Module | Description |
|-----------|---------------|-------------|
| `MSVECMAllocator` | `ms_vecm.py` | Markov-Switching VECM |
| `CointTester` | `ms_vecm.py` | Johansen cointegration test |
| `CovEstimator` | New | Ledoit-Wolf shrinkage, DCC-GARCH |

**Events Published**:
- `QuantSignalEvent`: Contains `expected_returns`, `cov_matrix`, `cointegration_rank`, `confidence`

**Events Consumed**:
- `MarketDataEvent`
- `RegimeEvent` (for regime-conditional parameters)

### 2.4 ML Agent

**Responsibility**: Regime detection and prediction

| Component | Source Module | Description |
|-----------|---------------|-------------|
| `RegimeDetector` | `hmm_regime.py` | Gaussian HMM |
| `FeatureEngineer` | New | Returns, volatility, momentum features |
| `RegimePredictor` | New | Regime transition probabilities |

**Events Published**:
- `RegimeEvent`: Contains `current_regime`, `regime_probs`, `transition_matrix`, `regime_duration`

**Events Consumed**:
- `MarketDataEvent`

### 2.5 Risk Manager Agent

**Responsibility**: Risk assessment and constraint enforcement

| Component | Source Module | Description |
|-----------|---------------|-------------|
| `RiskMetrics` | New | VaR, CVaR, Max Drawdown |
| `PositionSizer` | New | Kelly Criterion, Risk Parity |
| `CircuitBreaker` | New | Drawdown limits, volatility triggers |
| `CorrelationMonitor` | `visualize_process.py` | Rolling correlation stability |

**Events Published**:
- `RiskAdjustedSignalEvent`: Contains `risk_adjusted_returns`, `position_limits`, `risk_budget`, `alerts`

**Events Consumed**:
- `QuantSignalEvent`
- `RegimeEvent`
- `MarketDataEvent`

**Risk Rules (Regime-Dependent)**:

| Regime | Max Position | Leverage | Stop-Loss | Slippage |
|--------|-------------|----------|-----------|----------|
| Bull (0) | 25% | 1.0x | -5% | 5 bps |
| Transition (1) | 15% | 0.8x | -3% | 10 bps |
| Bear (2) | 10% | 0.5x | -2% | 25 bps |

### 2.6 Portfolio Agent

**Responsibility**: Optimization and trade signal generation

| Component | Source Module | Description |
|-----------|---------------|-------------|
| `BlackLittermanOptimizer` | `portfolio_optimizer.py` | BL model |
| `RebalanceEngine` | `backtester.py` (logic) | Turnover optimization |
| `TradeDecisionEngine` | New | BUY/SELL/HOLD logic |

**Events Published**:
- `TradeSignalEvent`: Contains `target_weights`, `trades` (list of `{ticker, action, weight, urgency}`)

**Events Consumed**:
- `RiskAdjustedSignalEvent`
- `RegimeEvent`
- `MarketDataEvent`

**Decision Logic**:
```python
if abs(current_weight - target_weight) < threshold:
    action = "HOLD"
elif target_weight > current_weight:
    action = "BUY"
else:
    action = "SELL"
```

---

## 3. Communication Protocol

### 3.1 Message Bus Design

Using an **Event-Driven Architecture** with a publish-subscribe pattern:

```python
class MessageBus:
    """Central event dispatcher"""
    
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = defaultdict(list)
        self._event_log: List[Event] = []
    
    def subscribe(self, event_type: Type[Event], handler: Callable):
        self._subscribers[event_type].append(handler)
    
    def publish(self, event: Event):
        self._event_log.append(event)
        for handler in self._subscribers[type(event)]:
            handler(event)
```

### 3.2 Event Schema

```python
@dataclass
class Event:
    timestamp: datetime
    source_agent: str
    correlation_id: str  # For tracing

@dataclass
class MarketDataEvent(Event):
    prices: pd.DataFrame
    asset_universe: List[str]
    data_quality_score: float

@dataclass
class RegimeEvent(Event):
    current_regime: int
    regime_probabilities: Dict[int, float]
    transition_matrix: np.ndarray
    regime_duration: int  # Days in current regime

@dataclass
class QuantSignalEvent(Event):
    expected_returns: pd.Series
    covariance_matrix: pd.DataFrame
    cointegration_rank: int
    forecast_confidence: float

@dataclass
class RiskAdjustedSignalEvent(Event):
    risk_adjusted_returns: pd.Series
    position_limits: Dict[str, float]
    risk_budget: float
    var_95: float
    cvar_95: float
    alerts: List[str]

@dataclass
class TradeSignalEvent(Event):
    target_weights: Dict[str, float]
    trades: List[TradeOrder]
    rebalance_reason: str

@dataclass
class TradeOrder:
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    current_weight: float
    target_weight: float
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
```

---

## 4. Directory Structure

```
agent_portfolio_system/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Abstract base class
│   ├── market_data_agent.py    # Data fetching
│   ├── ml_agent.py             # HMM regime detection
│   ├── quant_agent.py          # MS-VECM
│   ├── risk_manager_agent.py   # Risk management
│   ├── portfolio_agent.py      # Optimization & signals
│   └── orchestrator.py         # Pipeline coordination
│
├── events/
│   ├── __init__.py
│   ├── base_event.py           # Event dataclasses
│   └── message_bus.py          # Pub/sub implementation
│
├── models/                     # Preserved from original
│   ├── __init__.py
│   ├── hmm_regime.py           # RegimeDetector
│   ├── ms_vecm.py              # MSVECMAllocator
│   └── portfolio_optimizer.py  # PortfolioOptimizer
│
├── risk/
│   ├── __init__.py
│   ├── metrics.py              # VaR, CVaR, Drawdown
│   ├── position_sizing.py      # Kelly, Risk Parity
│   └── circuit_breaker.py      # Stop-loss, limits
│
├── utils/
│   ├── __init__.py
│   ├── visualizer.py           # Preserved
│   ├── logger.py               # Structured logging
│   └── config.py               # Configuration management
│
├── backtesting/
│   ├── __init__.py
│   ├── backtest_engine.py      # Agent-based backtester
│   └── metrics.py              # Performance analytics
│
├── data/
│   └── market_data.csv         # Cached data
│
├── reports/
│   └── report.html             # Generated reports
│
├── config/
│   ├── agent_config.yaml       # Agent parameters
│   └── risk_config.yaml        # Risk limits
│
├── main.py                     # Entry point
├── run_backtest.py             # Backtest runner
└── requirements.txt
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure
1. Event system (`events/`)
2. Base agent class
3. Message bus
4. Configuration management

### Phase 2: Agent Migration
1. Market Data Agent (from `data_loader.py`)
2. ML Agent (from `hmm_regime.py`)
3. Quant Agent (from `ms_vecm.py`)
4. Risk Manager Agent (new)
5. Portfolio Agent (from `portfolio_optimizer.py`)

### Phase 3: Orchestration
1. Pipeline coordinator
2. Error handling & recovery
3. Agent-based backtester

### Phase 4: Enhancement
1. Advanced risk metrics
2. Real-time capabilities
3. Monitoring dashboard

---

## 6. Key Design Decisions

### 6.1 Why Event-Driven?
- **Loose Coupling**: Agents don't know about each other directly
- **Testability**: Each agent can be tested in isolation
- **Extensibility**: New agents can subscribe to existing events
- **Audit Trail**: All events are logged for debugging

### 6.2 Preserving Original Logic
- All core algorithms (HMM, VECM, BL) remain unchanged
- Agents wrap existing classes, not replace them
- Backward compatibility with existing backtest results

### 6.3 Risk Manager as Gatekeeper
- All signals pass through risk manager before execution
- Regime-dependent constraints prevent excessive risk in bear markets
- Circuit breakers can halt trading during extreme volatility

---

## 7. Configuration Example

```yaml
# config/agent_config.yaml

market_data_agent:
  data_source: "yahoo"
  cache_ttl_hours: 24
  lookback_years: 10

ml_agent:
  hmm_states: 3
  covariance_type: "full"
  feature_window: 10

quant_agent:
  vecm_lookback: 252
  forecast_horizon: 5
  use_shrinkage: true

risk_manager:
  max_drawdown_limit: 0.15
  var_confidence: 0.95
  regime_risk_multipliers:
    0: 1.0   # Bull
    1: 0.8   # Transition
    2: 0.5   # Bear

portfolio_agent:
  use_black_litterman: true
  rebalance_threshold: 0.05
  max_turnover: 0.30
  transaction_cost_bps: 10
```

---

## 8. Next Steps

Ready to implement? The suggested order:
1. `events/` module (foundation)
2. `agents/base_agent.py`
3. `agents/market_data_agent.py`
4. `agents/ml_agent.py`
5. `agents/quant_agent.py`
6. `agents/risk_manager_agent.py`
7. `agents/portfolio_agent.py`
8. `agents/orchestrator.py`
9. Integration testing
10. Backtest validation

Shall I proceed with the implementation?
