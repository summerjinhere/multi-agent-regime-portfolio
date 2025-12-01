# 문제 진단 및 수정 사항

## 발견된 주요 문제점

### 1. R-Squared 0.018 (매우 낮음)
**원인:** 
- Cash 할당이 과도하게 높음 (20%+)
- Risk budget이 너무 보수적 (최소 20%까지 떨어질 수 있었음)
- Bear regime에서 risk_multiplier가 0.5로 너무 낮음

**수정:**
```python
# risk_manager_agent.py - _calculate_risk_budget()
# 기존: budget = max(budget, 0.2)
# 수정: budget = max(budget, 0.90)  # 최소 90% 투자
```

### 2. Total Return 0.59% (5년간 거의 0%)
**원인:**
- `backtest_engine.py`에서 `_update_portfolio_value()`가 sector weights만 사용
- `detailed_weights` (실제 주식 레벨)가 반영되지 않음
- 포트폴리오 수익률이 실제 주식 수익률과 연결되지 않음

**수정:**
```python
# backtest_engine.py
self.current_detailed_weights: Dict[str, float] = {}  # 추가

def _execute_trades(self, date, trade_signal):
    # detailed_weights 저장
    if trade_signal.detailed_weights:
        self.current_detailed_weights = trade_signal.detailed_weights.copy()

def _update_portfolio_value(self, from_date, to_date):
    # current_detailed_weights 사용하여 실제 수익률 계산
    for ticker, weight in self.current_detailed_weights.items():
        # 개별 주식 수익률 계산
```

### 3. Volatility 0.00%
**원인:**
- 위의 수익률 계산 문제로 인해 포트폴리오 가치가 거의 변하지 않음
- 모든 기간에서 동일한 값 유지

**수정:**
- `_update_portfolio_value()` 함수 전면 재작성
- Sector → 개별 주식 매핑 로직 추가

### 4. Dashboard 연결 안됨
**원인:**
- `TradeSignalEvent`에 `detailed_weights` 필드가 없거나 비어있음
- `final_portfolio_detailed.csv` 파일이 생성되지 않거나 비어있음

**수정:**
```python
# base_event.py - TradeSignalEvent
@dataclass
class TradeSignalEvent(Event):
    # ... 기존 필드들 ...
    detailed_weights: Dict[str, float] = field(default_factory=dict)  # 추가!
```

```python
# portfolio_agent.py - _generate_detailed_allocation()
# Sector weights를 개별 주식으로 분해하는 로직 강화
```

## 수정된 파일 목록

| 파일 | 주요 변경 내용 |
|------|--------------|
| `events/base_event.py` | `TradeSignalEvent`에 `detailed_weights` 필드 추가 |
| `agents/risk_manager_agent.py` | 최소 risk_budget 90%로 상향, regime multipliers 완화 |
| `agents/portfolio_agent.py` | `_generate_detailed_allocation()` 로직 강화 |
| `backtesting/backtest_engine.py` | `_update_portfolio_value()` 전면 재작성, detailed_weights 사용 |
| `main.py` | `save_portfolio_files()` 함수 추가, 에러 핸들링 강화 |
| `config/agent_config.yaml` | 파라미터 최적화 |

## 수정된 Risk Parameters

### 기존 (너무 보수적)
```yaml
regime_params:
  0:  # Bull
    risk_multiplier: 1.0
  1:  # Transition
    risk_multiplier: 0.8
  2:  # Bear
    risk_multiplier: 0.5  # 50%만 투자!
```

### 수정 후 (적절한 노출)
```yaml
regime_params:
  0:  # Bull
    risk_multiplier: 1.0
  1:  # Transition
    risk_multiplier: 0.95
  2:  # Bear
    risk_multiplier: 0.85  # 85% 유지
```

## 예상 성능 개선

| 지표 | 기존 | 예상 개선 |
|------|------|----------|
| Cash Allocation | 20%+ | <10% |
| R-Squared (vs SPY) | 0.018 | 0.5-0.7 |
| Total Return (5Y) | 0.59% | Market-like |
| Volatility | 0.00% | 10-20% |

## 사용 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 백테스트 실행
python main.py --backtest

# 3. 라이브 예측
python main.py

# 4. 생성되는 파일
# - final_portfolio.csv (자산 클래스 레벨)
# - final_portfolio_detailed.csv (개별 주식 레벨) ← Dashboard용
# - backtest_results.csv (백테스트 결과)
# - weights_history.csv (시간별 가중치)
# - regime_history.csv (시간별 레짐)
```

## Dashboard 연결

Dashboard는 `final_portfolio_detailed.csv` 파일을 읽어야 합니다:

```python
import pandas as pd

# Dashboard에서 읽기
weights = pd.read_csv('final_portfolio_detailed.csv', index_col=0, header=None)
weights.columns = ['Weight']

# 예시 출력:
# AAPL     0.06
# MSFT     0.06
# NVDA     0.06
# ...
# TLT      0.10
# GLD      0.05
# Cash     0.05
```
