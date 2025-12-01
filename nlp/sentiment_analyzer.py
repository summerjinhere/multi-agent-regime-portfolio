"""
NLP Sentiment Analysis Module
=============================

Features:
- Multi-source sentiment collection (Twitter, Reddit, News, StockTwits)
- English-only text processing
- Nuance-aware sentiment classification using transformers
- Context-aware word embedding for financial terminology
- Real-time sentiment scoring per ticker

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Container for sentiment analysis results"""
    ticker: str
    timestamp: datetime
    overall_score: float  # -1 (bearish) to +1 (bullish)
    positive_count: int
    negative_count: int
    neutral_count: int
    total_mentions: int
    confidence: float  # 0-1, based on sample size and agreement
    sources: Dict[str, float] = field(default_factory=dict)  # Per-source scores
    top_keywords: List[Tuple[str, float]] = field(default_factory=list)


class FinancialLexicon:
    """
    Domain-specific financial sentiment lexicon with nuance handling.
    Includes context-aware word scoring.
    """
    
    # Positive financial terms with intensity scores (0-1)
    POSITIVE_TERMS = {
        # Strong positive
        'bullish': 0.9, 'surge': 0.85, 'soar': 0.9, 'rally': 0.8,
        'breakout': 0.85, 'moon': 0.95, 'rocket': 0.9, 'skyrocket': 0.95,
        'outperform': 0.75, 'beat': 0.7, 'exceed': 0.7, 'crush': 0.8,
        'blowout': 0.85, 'stellar': 0.8, 'exceptional': 0.75,
        
        # Moderate positive
        'upgrade': 0.65, 'buy': 0.6, 'accumulate': 0.6, 'overweight': 0.55,
        'growth': 0.5, 'profit': 0.55, 'gain': 0.5, 'rise': 0.45,
        'increase': 0.4, 'improve': 0.45, 'positive': 0.5, 'strong': 0.5,
        'momentum': 0.55, 'upside': 0.6, 'opportunity': 0.5,
        
        # Mild positive
        'stable': 0.3, 'steady': 0.3, 'resilient': 0.4, 'recovery': 0.45,
        'rebound': 0.5, 'support': 0.35, 'hold': 0.25, 'confidence': 0.4
    }
    
    # Negative financial terms with intensity scores (0-1)
    NEGATIVE_TERMS = {
        # Strong negative
        'bearish': 0.9, 'crash': 0.95, 'plunge': 0.9, 'collapse': 0.95,
        'tank': 0.85, 'dump': 0.8, 'selloff': 0.8, 'freefall': 0.95,
        'disaster': 0.9, 'catastrophe': 0.95, 'bankruptcy': 0.95,
        
        # Moderate negative
        'downgrade': 0.65, 'sell': 0.6, 'underweight': 0.55, 'miss': 0.6,
        'decline': 0.5, 'drop': 0.5, 'fall': 0.45, 'loss': 0.55,
        'decrease': 0.4, 'negative': 0.5, 'weak': 0.5, 'concern': 0.45,
        'risk': 0.4, 'warning': 0.55, 'caution': 0.25,
        
        # Mild negative
        'volatile': 0.35, 'uncertainty': 0.4, 'pressure': 0.35,
        'headwind': 0.45, 'challenge': 0.35, 'slow': 0.3, 'flat': 0.25
    }
    
    # Negation words that flip sentiment
    NEGATIONS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                 'nowhere', 'hardly', 'barely', 'rarely', "don't", "doesn't",
                 "didn't", "won't", "wouldn't", "couldn't", "shouldn't"}
    
    # Intensifiers that amplify sentiment
    INTENSIFIERS = {
        'very': 1.3, 'extremely': 1.5, 'highly': 1.3, 'incredibly': 1.4,
        'absolutely': 1.4, 'totally': 1.3, 'completely': 1.3, 'massive': 1.4,
        'huge': 1.3, 'significant': 1.2, 'substantial': 1.2
    }
    
    # Diminishers that reduce sentiment
    DIMINISHERS = {
        'slightly': 0.7, 'somewhat': 0.8, 'fairly': 0.85, 'rather': 0.8,
        'marginally': 0.6, 'barely': 0.5, 'hardly': 0.5, 'little': 0.7
    }
    
    # Context-dependent terms (require surrounding context)
    CONTEXT_DEPENDENT = {
        'short': {'cover': 0.6, 'squeeze': 0.8, 'interest': -0.3, 'selling': -0.5},
        'call': {'option': 0.3, 'buy': 0.4, 'sweep': 0.5},
        'put': {'option': -0.3, 'buy': -0.4, 'volume': -0.3},
        'break': {'out': 0.7, 'down': -0.7, 'even': 0.1, 'support': -0.6},
        'dead': {'cat': -0.4, 'money': -0.3},
        'bag': {'holder': -0.6, 'holding': -0.5}
    }


class TextPreprocessor:
    """Clean and preprocess financial text for sentiment analysis"""
    
    # Ticker pattern
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
    
    # URL pattern
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    
    # Emoji sentiment mapping
    EMOJI_SENTIMENT = {
        '🚀': 0.8, '📈': 0.6, '💰': 0.5, '💵': 0.4, '🔥': 0.5,
        '💎': 0.6, '🙌': 0.5, '👍': 0.4, '✅': 0.3, '🎯': 0.4,
        '📉': -0.6, '💀': -0.7, '😱': -0.5, '😨': -0.4, '🤮': -0.6,
        '👎': -0.4, '❌': -0.3, '⚠️': -0.3, '🔻': -0.5, '😢': -0.3
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = TextPreprocessor.URL_PATTERN.sub('', text)
        
        # Convert to lowercase but preserve tickers
        tickers = TextPreprocessor.TICKER_PATTERN.findall(text)
        text = text.lower()
        
        # Remove special characters except $ for tickers
        text = re.sub(r'[^\w\s$@#]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text, tickers
    
    @staticmethod
    def extract_emoji_sentiment(text: str) -> float:
        """Extract sentiment from emojis"""
        emoji_scores = []
        for emoji, score in TextPreprocessor.EMOJI_SENTIMENT.items():
            count = text.count(emoji)
            if count > 0:
                emoji_scores.extend([score] * count)
        
        return np.mean(emoji_scores) if emoji_scores else 0.0
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        return text.split()


class NuanceSentimentAnalyzer:
    """
    Advanced sentiment analyzer with nuance understanding.
    Uses context windows and financial domain knowledge.
    """
    
    def __init__(self, use_transformer: bool = False):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        use_transformer : bool
            If True, use transformer model for better nuance detection.
            If False, use lexicon-based approach (faster, no dependencies).
        """
        self.lexicon = FinancialLexicon()
        self.preprocessor = TextPreprocessor()
        self.use_transformer = use_transformer
        self.transformer_model = None
        
        if use_transformer:
            self._load_transformer()
    
    def _load_transformer(self):
        """Load transformer model for sentiment analysis"""
        try:
            from transformers import pipeline
            # Use FinBERT for financial sentiment
            self.transformer_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            logger.info("Loaded FinBERT transformer model")
        except ImportError:
            logger.warning("Transformers not available, using lexicon-based analysis")
            self.use_transformer = False
        except Exception as e:
            logger.warning(f"Failed to load transformer: {e}")
            self.use_transformer = False
    
    def analyze_text(self, text: str) -> Tuple[float, float, List[Tuple[str, float]]]:
        """
        Analyze sentiment of a single text.
        
        Returns:
        --------
        Tuple of (sentiment_score, confidence, keywords)
        - sentiment_score: -1 to +1
        - confidence: 0 to 1
        - keywords: list of (word, contribution) tuples
        """
        original_text = text
        clean_text, tickers = self.preprocessor.clean_text(text)
        
        # Get emoji sentiment
        emoji_sentiment = self.preprocessor.extract_emoji_sentiment(original_text)
        
        if self.use_transformer and self.transformer_model:
            return self._transformer_analyze(clean_text, emoji_sentiment)
        else:
            return self._lexicon_analyze(clean_text, emoji_sentiment)
    
    def _lexicon_analyze(self, text: str, emoji_sentiment: float = 0.0
                        ) -> Tuple[float, float, List[Tuple[str, float]]]:
        """Lexicon-based sentiment analysis with nuance handling"""
        tokens = self.preprocessor.tokenize(text)
        
        if not tokens:
            return 0.0, 0.0, []
        
        scores = []
        keywords = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            base_score = 0.0
            multiplier = 1.0
            
            # Check for intensifiers/diminishers in previous token
            if i > 0:
                prev_token = tokens[i-1]
                if prev_token in self.lexicon.INTENSIFIERS:
                    multiplier = self.lexicon.INTENSIFIERS[prev_token]
                elif prev_token in self.lexicon.DIMINISHERS:
                    multiplier = self.lexicon.DIMINISHERS[prev_token]
            
            # Check for negation in window of 3 previous tokens
            negated = False
            for j in range(max(0, i-3), i):
                if tokens[j] in self.lexicon.NEGATIONS:
                    negated = True
                    break
            
            # Check positive terms
            if token in self.lexicon.POSITIVE_TERMS:
                base_score = self.lexicon.POSITIVE_TERMS[token]
                if negated:
                    base_score = -base_score * 0.7  # Negation reduces intensity
            
            # Check negative terms
            elif token in self.lexicon.NEGATIVE_TERMS:
                base_score = -self.lexicon.NEGATIVE_TERMS[token]
                if negated:
                    base_score = -base_score * 0.7
            
            # Check context-dependent terms
            elif token in self.lexicon.CONTEXT_DEPENDENT:
                context_dict = self.lexicon.CONTEXT_DEPENDENT[token]
                # Look at next token for context
                if i + 1 < len(tokens):
                    next_token = tokens[i+1]
                    if next_token in context_dict:
                        base_score = context_dict[next_token]
            
            # Apply multiplier
            final_score = base_score * multiplier
            
            if abs(final_score) > 0.1:
                scores.append(final_score)
                keywords.append((token, final_score))
            
            i += 1
        
        # Combine scores
        if scores:
            # Weighted average favoring stronger signals
            weights = np.abs(scores)
            sentiment = np.average(scores, weights=weights) if sum(weights) > 0 else 0
            
            # Include emoji sentiment
            if emoji_sentiment != 0:
                sentiment = 0.8 * sentiment + 0.2 * emoji_sentiment
            
            # Confidence based on number and agreement of signals
            confidence = min(1.0, len(scores) / 5) * (1 - np.std(scores) / 2)
            confidence = max(0.1, confidence)
        else:
            sentiment = emoji_sentiment * 0.5  # Lower confidence for emoji-only
            confidence = 0.2 if emoji_sentiment != 0 else 0.0
        
        # Clip to valid range
        sentiment = np.clip(sentiment, -1, 1)
        
        # Sort keywords by absolute contribution
        keywords.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return sentiment, confidence, keywords[:5]
    
    def _transformer_analyze(self, text: str, emoji_sentiment: float = 0.0
                            ) -> Tuple[float, float, List[Tuple[str, float]]]:
        """Transformer-based sentiment analysis"""
        try:
            result = self.transformer_model(text[:512])[0]  # Truncate to max length
            
            label = result['label'].lower()
            score = result['score']
            
            if label == 'positive':
                sentiment = score
            elif label == 'negative':
                sentiment = -score
            else:
                sentiment = 0.0
            
            # Blend with emoji sentiment
            if emoji_sentiment != 0:
                sentiment = 0.85 * sentiment + 0.15 * emoji_sentiment
            
            return sentiment, score, [('transformer', sentiment)]
            
        except Exception as e:
            logger.warning(f"Transformer analysis failed: {e}")
            return self._lexicon_analyze(text, emoji_sentiment)


class SentimentAggregator:
    """Aggregates sentiment from multiple sources and time periods"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.analyzer = NuanceSentimentAnalyzer(
            use_transformer=self.config.get('use_transformer', False)
        )
        self.history: Dict[str, List[SentimentScore]] = defaultdict(list)
    
    def analyze_mentions(self, ticker: str, 
                        mentions: List[Dict[str, str]]) -> SentimentScore:
        """
        Analyze a batch of mentions for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        mentions : List[Dict]
            List of {'text': str, 'source': str, 'timestamp': str}
        
        Returns:
        --------
        SentimentScore object
        """
        if not mentions:
            return SentimentScore(
                ticker=ticker,
                timestamp=datetime.now(),
                overall_score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                total_mentions=0,
                confidence=0.0
            )
        
        scores = []
        confidences = []
        all_keywords = []
        source_scores = defaultdict(list)
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for mention in mentions:
            text = mention.get('text', '')
            source = mention.get('source', 'unknown')
            
            # Skip non-English text (simple heuristic)
            if not self._is_english(text):
                continue
            
            sentiment, confidence, keywords = self.analyzer.analyze_text(text)
            
            scores.append(sentiment)
            confidences.append(confidence)
            all_keywords.extend(keywords)
            source_scores[source].append(sentiment)
            
            if sentiment > 0.2:
                positive_count += 1
            elif sentiment < -0.2:
                negative_count += 1
            else:
                neutral_count += 1
        
        if not scores:
            return SentimentScore(
                ticker=ticker,
                timestamp=datetime.now(),
                overall_score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                total_mentions=len(mentions),
                confidence=0.0
            )
        
        # Confidence-weighted average
        overall_score = np.average(scores, weights=confidences)
        overall_confidence = np.mean(confidences) * min(1.0, len(scores) / 10)
        
        # Aggregate per-source scores
        source_avg = {src: np.mean(s) for src, s in source_scores.items()}
        
        # Aggregate keywords
        keyword_scores = defaultdict(list)
        for kw, score in all_keywords:
            keyword_scores[kw].append(score)
        top_keywords = sorted(
            [(kw, np.mean(scores)) for kw, scores in keyword_scores.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        result = SentimentScore(
            ticker=ticker,
            timestamp=datetime.now(),
            overall_score=overall_score,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            total_mentions=len(mentions),
            confidence=overall_confidence,
            sources=source_avg,
            top_keywords=top_keywords
        )
        
        # Store in history
        self.history[ticker].append(result)
        
        return result
    
    def _is_english(self, text: str) -> bool:
        """Simple heuristic to detect English text"""
        # Check if most characters are ASCII
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return ascii_count / max(len(text), 1) > 0.8
    
    def get_sentiment_trend(self, ticker: str, 
                           hours: int = 24) -> Dict[str, float]:
        """Get sentiment trend over time period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [s for s in self.history[ticker] 
                 if s.timestamp > cutoff]
        
        if not recent:
            return {'current': 0.0, 'trend': 0.0, 'volatility': 0.0}
        
        scores = [s.overall_score for s in recent]
        
        # Current sentiment (most recent)
        current = scores[-1]
        
        # Trend (linear regression slope)
        if len(scores) >= 2:
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            trend = slope * len(scores)  # Total change over period
        else:
            trend = 0.0
        
        # Volatility
        volatility = np.std(scores) if len(scores) > 1 else 0.0
        
        return {
            'current': current,
            'trend': trend,
            'volatility': volatility,
            'n_observations': len(scores)
        }


class MarketSentimentAgent:
    """
    Full sentiment agent for market analysis.
    Integrates with the multi-agent system.
    """
    
    def __init__(self, tickers: List[str], config: Optional[Dict] = None):
        self.tickers = tickers
        self.config = config or {}
        self.aggregator = SentimentAggregator(config)
        self.last_update = {}
        self.sentiment_cache = {}
    
    def update_sentiment(self, ticker: str, 
                        data_source: 'DataCollector') -> SentimentScore:
        """
        Update sentiment for a ticker using data from collector.
        
        This is a placeholder - in production, integrate with actual
        social media APIs (Twitter, Reddit, etc.)
        """
        # Check if update is needed
        update_freq = self.config.get('update_frequency_minutes', 15)
        last = self.last_update.get(ticker)
        
        if last and (datetime.now() - last).seconds < update_freq * 60:
            return self.sentiment_cache.get(ticker)
        
        # Collect mentions (placeholder - implement actual data collection)
        mentions = self._collect_mentions(ticker, data_source)
        
        # Analyze
        result = self.aggregator.analyze_mentions(ticker, mentions)
        
        # Cache
        self.sentiment_cache[ticker] = result
        self.last_update[ticker] = datetime.now()
        
        return result
    
    def _collect_mentions(self, ticker: str, 
                         data_source) -> List[Dict[str, str]]:
        """
        Collect mentions from various sources.
        
        In production, implement actual API calls to:
        - Twitter API
        - Reddit API (r/wallstreetbets, r/stocks, r/investing)
        - StockTwits API
        - Financial news APIs
        """
        # Placeholder - return empty for now
        # Real implementation would make API calls
        return []
    
    def get_all_sentiments(self) -> Dict[str, SentimentScore]:
        """Get current sentiment for all tracked tickers"""
        return self.sentiment_cache.copy()
    
    def get_sentiment_signal(self, ticker: str) -> float:
        """
        Get trading signal from sentiment (-1 to +1).
        Combines current sentiment with trend.
        """
        if ticker not in self.sentiment_cache:
            return 0.0
        
        current = self.sentiment_cache[ticker]
        trend = self.aggregator.get_sentiment_trend(ticker)
        
        # Combine current level with trend
        signal = 0.7 * current.overall_score + 0.3 * trend['trend']
        
        # Adjust for confidence
        signal *= current.confidence
        
        return np.clip(signal, -1, 1)


# Simulated data collector for testing
class SimulatedDataCollector:
    """Generates simulated social media data for testing"""
    
    SAMPLE_POSITIVE = [
        "{ticker} looking bullish! 🚀 Major breakout incoming",
        "Just loaded up on ${ticker}, this thing is going to moon",
        "Strong earnings beat from {ticker}, upgrade incoming",
        "{ticker} momentum is incredible, adding more",
        "Technical breakout on ${ticker}, very bullish setup"
    ]
    
    SAMPLE_NEGATIVE = [
        "{ticker} looking weak, considering selling",
        "Bearish on ${ticker}, too much downside risk",
        "{ticker} missed estimates badly, dumping my shares",
        "Warning signs on {ticker}, be careful",
        "${ticker} in freefall, glad I sold earlier"
    ]
    
    SAMPLE_NEUTRAL = [
        "Anyone have thoughts on {ticker}?",
        "Watching ${ticker} today",
        "{ticker} trading sideways",
        "Not sure about {ticker} direction",
        "Holding my ${ticker} position for now"
    ]
    
    def get_mentions(self, ticker: str, n: int = 20) -> List[Dict]:
        """Generate simulated mentions"""
        import random
        
        mentions = []
        for _ in range(n):
            r = random.random()
            if r < 0.4:
                template = random.choice(self.SAMPLE_POSITIVE)
            elif r < 0.7:
                template = random.choice(self.SAMPLE_NEGATIVE)
            else:
                template = random.choice(self.SAMPLE_NEUTRAL)
            
            mentions.append({
                'text': template.format(ticker=ticker),
                'source': random.choice(['twitter', 'reddit', 'stocktwits']),
                'timestamp': datetime.now().isoformat()
            })
        
        return mentions


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = NuanceSentimentAnalyzer(use_transformer=False)
    
    test_texts = [
        "AAPL looking incredibly bullish! 🚀 Major breakout incoming",
        "Bearish on TSLA, too much downside risk ahead",
        "Not sure about MSFT direction, could go either way",
        "This stock is not good at all, selling everything",
        "Short squeeze incoming! Bears are getting destroyed 💎🙌"
    ]
    
    print("Sentiment Analysis Test Results")
    print("=" * 60)
    
    for text in test_texts:
        sentiment, confidence, keywords = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment:.3f}, Confidence: {confidence:.3f}")
        print(f"Keywords: {keywords}")
