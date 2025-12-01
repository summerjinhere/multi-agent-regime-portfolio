# Enhanced Multi-Agent Portfolio System
# =====================================
# 
# A comprehensive portfolio optimization framework featuring:
# - Regime-adaptive risk aversion
# - TWAP/VWAP execution algorithms
# - Institutional flow tracking & piggyback trading
# - NLP sentiment analysis
# - Comprehensive statistical validation (DSR, Bootstrap, OLS)
#
# Usage:
# ------
# from main import MultiAgentOrchestrator
# from config.settings import SystemConfig
# 
# config = SystemConfig()
# orchestrator = MultiAgentOrchestrator(config)
# result = orchestrator.run_backtest(prices, volumes)
#
# Author: Anonymous
# Date: November 2025

__version__ = "2.0.0"
__author__ = "Anonymous"

from config.settings import SystemConfig, load_config
from main import MultiAgentOrchestrator
