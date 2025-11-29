class RiskManagerAgent:
    def __init__(self, transaction_cost=0.001):
        self.transaction_cost = transaction_cost
        # Slippage Map: Regime -> Slippage
        self.slippage_map = {0: 0.0005, 1: 0.0010, 2: 0.0025}

    def calculate_cost(self, turnover, capital, regime):
        """
        Calculates total cost including commission and regime-dependent slippage.
        """
        current_slippage = self.slippage_map.get(regime, 0.0010)
        
        traded_volume = turnover * capital
        
        commission_cost = traded_volume * self.transaction_cost
        slippage_cost = traded_volume * current_slippage
        
        total_cost = commission_cost + slippage_cost
        
        print(f"[RiskManager] Regime {regime} -> Slippage {current_slippage*10000:.0f}bps. Total Cost: ${total_cost:.2f}")
        return total_cost

    def check_limits(self, weights):
        """
        Enforce risk limits (e.g., max allocation to one asset).
        """
        # Example: Max 30% in any single stock (excluding ETFs/Indices if needed)
        # For now, just a pass-through or simple clamp
        # clamped_weights = {k: min(v, 0.30) for k, v in weights.items()}
        # Normalize
        # total = sum(clamped_weights.values())
        # return {k: v/total for k, v in clamped_weights.items()}
        return weights
