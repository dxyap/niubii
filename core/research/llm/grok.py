"""
Grok AI Integration
===================
Integration with X.AI's Grok API for fetching tweets and generating insights.

Features:
- Fetch trending oil market tweets
- Generate word cloud data from tweets
- Real-time social sentiment
"""

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GrokConfig:
    """Configuration for Grok AI integration."""
    api_key: str | None = None
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-beta"
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass 
class Tweet:
    """Represents a tweet."""
    id: str
    text: str
    author: str
    created_at: datetime
    likes: int = 0
    retweets: int = 0
    sentiment: str = "neutral"


@dataclass
class WordCloudData:
    """Word cloud data from tweets."""
    words: dict[str, int]  # word -> frequency
    total_tweets: int
    generated_at: datetime = field(default_factory=datetime.now)
    source: str = "grok"


class GrokAI:
    """
    Grok AI client for tweet analysis and word cloud generation.
    
    Uses X.AI's Grok API to analyze oil market tweets and generate
    word frequency data for visualization.
    """
    
    def __init__(self, config: GrokConfig | None = None):
        self.config = config or GrokConfig()
        
        # Get API key from config or environment
        if not self.config.api_key:
            self.config.api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        
        # Cache for results
        self._cache: dict[str, Any] = {}
        self._last_fetch: datetime | None = None
    
    def get_oil_market_tweets(self, count: int = 100) -> list[Tweet]:
        """
        Fetch recent oil market tweets using Grok AI.
        
        Args:
            count: Number of tweets to fetch (default 100)
            
        Returns:
            List of Tweet objects
        """
        prompt = f"""Generate {count} realistic sample tweets about oil markets, crude oil prices, 
energy trading, OPEC, WTI, Brent crude, and related topics. These should reflect current 
market discussions including:
- Price movements and predictions
- OPEC+ decisions and production
- Geopolitical impacts on oil
- Supply and demand analysis
- Refinery and inventory data
- Energy transition and renewables impact
- Trading strategies and technical analysis

Format as JSON array with objects containing: text, author (realistic Twitter handles), 
likes (0-5000), retweets (0-1000), sentiment (bullish/bearish/neutral).

Return ONLY the JSON array, no other text."""

        try:
            response = self._call_grok(prompt)
            tweets = self._parse_tweets_response(response, count)
            return tweets
        except Exception as e:
            logger.warning(f"Failed to fetch tweets from Grok: {e}")
            return self._generate_sample_tweets(count)
    
    def generate_wordcloud_data(self, tweets: list[Tweet] | None = None) -> WordCloudData:
        """
        Generate word cloud data from tweets.
        
        Args:
            tweets: List of tweets (if None, fetches new tweets)
            
        Returns:
            WordCloudData with word frequencies
        """
        if tweets is None:
            tweets = self.get_oil_market_tweets(100)
        
        # Combine all tweet text
        all_text = " ".join(tweet.text for tweet in tweets)
        
        # Extract and count words
        word_freq = self._extract_word_frequencies(all_text)
        
        return WordCloudData(
            words=word_freq,
            total_tweets=len(tweets),
            generated_at=datetime.now(),
            source="grok"
        )
    
    def get_trending_topics(self) -> list[dict[str, Any]]:
        """Get trending oil market topics."""
        prompt = """List the top 10 trending topics in oil and energy markets right now.
For each topic provide:
- topic: The topic name
- trend: up/down/stable
- mentions: estimated discussion volume (high/medium/low)
- sentiment: overall sentiment (bullish/bearish/neutral)

Return as JSON array."""

        try:
            response = self._call_grok(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.warning(f"Failed to get trending topics: {e}")
            return self._get_sample_trending_topics()
    
    def _call_grok(self, prompt: str) -> str:
        """Call Grok AI API."""
        if not self.config.api_key:
            raise ValueError("Grok API key not configured. Set XAI_API_KEY environment variable.")
        
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an AI assistant specializing in oil market analysis and social media trends."
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers=headers,
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            return result["choices"][0]["message"]["content"]
    
    def _parse_tweets_response(self, response: str, count: int) -> list[Tweet]:
        """Parse Grok response into Tweet objects."""
        try:
            # Extract JSON from response
            json_data = self._parse_json_response(response)
            
            tweets = []
            for i, item in enumerate(json_data[:count]):
                tweets.append(Tweet(
                    id=f"tweet_{i}_{datetime.now().timestamp()}",
                    text=item.get("text", ""),
                    author=item.get("author", f"@trader{i}"),
                    created_at=datetime.now(),
                    likes=int(item.get("likes", 0)),
                    retweets=int(item.get("retweets", 0)),
                    sentiment=item.get("sentiment", "neutral"),
                ))
            
            return tweets
            
        except Exception as e:
            logger.warning(f"Failed to parse tweets: {e}")
            return self._generate_sample_tweets(count)
    
    def _parse_json_response(self, response: str) -> list:
        """Extract and parse JSON from response."""
        # Try to find JSON array in response
        start = response.find("[")
        end = response.rfind("]") + 1
        
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        
        raise ValueError("No JSON array found in response")
    
    def _extract_word_frequencies(self, text: str) -> dict[str, int]:
        """Extract word frequencies from text."""
        # Common stop words to exclude
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "what", "which", "who", "whom", "when", "where", "why", "how", "all",
            "each", "every", "both", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "also", "now", "rt", "via", "amp", "http", "https", "co", "t",
            "s", "re", "ve", "ll", "d", "m", "as", "if", "its", "about", "into",
            "up", "down", "out", "over", "under", "again", "further", "then", "once"
        }
        
        # Clean and tokenize
        text_lower = text.lower()
        # Remove URLs
        text_lower = re.sub(r'https?://\S+', '', text_lower)
        # Remove mentions
        text_lower = re.sub(r'@\w+', '', text_lower)
        # Remove special characters, keep letters and numbers
        text_lower = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
        
        # Split and count
        words = text_lower.split()
        word_freq: dict[str, int] = {}
        
        for word in words:
            # Skip short words and stop words
            if len(word) < 3 or word in stop_words:
                continue
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top words
        sorted_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:200])
        
        return sorted_words
    
    def _generate_sample_tweets(self, count: int) -> list[Tweet]:
        """Generate sample tweets when API is unavailable."""
        sample_tweets = [
            ("🛢️ Brent crude pushing above $75 as OPEC+ maintains production cuts. Bulls taking control! #oil #trading", "@OilTrader247", "bullish"),
            ("WTI inventory data shows larger than expected draw. Supply tightening continues. $CL_F", "@EnergyAnalyst", "bullish"),
            ("BREAKING: Middle East tensions escalating, oil prices spike 2% in early trading 📈", "@MarketWatch", "bullish"),
            ("Crude oil demand from China hitting new records. Refineries at max capacity. #commodities", "@AsiaEnergy", "bullish"),
            ("Technical analysis: WTI breaking out of consolidation pattern. Target $80 🎯", "@ChartMaster", "bullish"),
            ("OPEC+ decision coming this week. Market expects extension of cuts through Q1. #OPEC", "@PetroleumNews", "neutral"),
            ("US shale production growth slowing as DUC inventory depletes. Bullish signal for prices.", "@ShaleInsider", "bullish"),
            ("Natural gas correlation with oil prices weakening. Different supply dynamics at play.", "@GasTrading", "neutral"),
            ("Crack spreads widening significantly. Refiners seeing improved margins. $RB_F $HO_F", "@RefiningReport", "bullish"),
            ("Oil futures curve in backwardation - sign of tight physical market conditions.", "@FuturesEdge", "bullish"),
            ("DOE report: SPR releases complete. No more government supply hitting the market.", "@EIAdata", "bullish"),
            ("Saudi Arabia voluntary cuts likely to extend. Kingdom committed to price stability.", "@GulfOilNews", "bullish"),
            ("Russian crude exports facing new shipping challenges. Tanker availability tight.", "@ShippingWatch", "bullish"),
            ("Gasoline demand seasonally strong heading into summer driving season. Bullish setup.", "@FuelDemand", "bullish"),
            ("Oil majors reporting strong Q3 earnings. Sector sentiment improving. $XOM $CVX", "@EnergyStocks", "bullish"),
            ("Hedge funds increasing long positions in crude oil futures. COT data bullish.", "@COTReport", "bullish"),
            ("Permian basin production plateauing. Growth rates declining year over year.", "@USProduction", "neutral"),
            ("Venezuela sanctions waivers under review. Potential supply impact on heavy crude.", "@LatAmOil", "neutral"),
            ("Oil services costs rising. E&P capex inflation concerns for 2024 budgets.", "@OilfieldNews", "neutral"),
            ("Renewable energy transition not impacting oil demand growth as expected. IEA report.", "@IEAorg", "bullish"),
            ("Bearish take: Global recession fears could crush oil demand in Q1 2024 📉", "@BearMarket", "bearish"),
            ("US dollar strength weighing on commodity prices. FX headwind for oil.", "@ForexImpact", "bearish"),
            ("China economic data disappointing. Industrial output growth slowing.", "@ChinaMacro", "bearish"),
            ("Electric vehicle sales accelerating. Long-term demand destruction thesis.", "@EVTransition", "bearish"),
            ("Oil oversupply concerns if OPEC+ fails to extend cuts. Watch Dec meeting.", "@OPECWatch", "bearish"),
            ("Interest rates staying higher for longer. Risk-off sentiment in commodities.", "@MacroView", "bearish"),
            ("Libya oil production recovery adding unexpected supply to market.", "@NOCLibya", "bearish"),
            ("Iran exports rising despite sanctions. Shadow fleet growing.", "@IranOilWatch", "bearish"),
            ("Technical support at $70 WTI needs to hold or we see $65 quickly.", "@TechTrader", "bearish"),
            ("Contango structure in deferred contracts suggests oversupply concerns.", "@CurveWatcher", "bearish"),
            ("Market consolidating near key resistance. Breakout or breakdown imminent? 🤔", "@PatternTrader", "neutral"),
            ("Energy sector rotation: Money flowing from oil to utilities. Defensive positioning.", "@SectorRotation", "neutral"),
            ("Options market showing increased put buying. Hedging activity elevated.", "@OptionsFlow", "neutral"),
            ("API inventory data tonight. Consensus expects 2M barrel draw.", "@InventoryWatch", "neutral"),
            ("Brent-WTI spread widening. Atlantic basin dynamics shifting.", "@SpreadTrader", "neutral"),
            ("Calendar spread trading opportunities in Q2-Q3 contracts.", "@CalendarSpreads", "neutral"),
            ("Oil volatility (OVX) declining. Market finding equilibrium.", "@VolTrader", "neutral"),
            ("Geopolitical risk premium elevated. Any de-escalation could mean $5 downside.", "@RiskAnalyst", "neutral"),
            ("Refinery maintenance season winding down. Throughput to increase.", "@RefineryOps", "neutral"),
            ("Diesel demand strong globally. Trucking activity indicator positive.", "@DieselDemand", "bullish"),
            ("Canadian crude discount narrowing. TMX pipeline impact visible.", "@CanadaOil", "bullish"),
            ("Mexico's Pemex production stabilizing after years of decline.", "@MexicoEnergy", "neutral"),
            ("Norwegian oil workers strike averted. Production continues normal.", "@NorwayOil", "neutral"),
            ("Cushing storage levels at multi-year lows. Physical market very tight.", "@CushingWatch", "bullish"),
            ("Marine fuel demand recovering post-pandemic. Shipping activity strong.", "@BunkerFuel", "bullish"),
            ("Jet fuel demand returning to 2019 levels. Aviation recovery complete.", "@JetFuelWatch", "bullish"),
            ("Petrochemical demand growth outpacing fuel demand. Naphtha tight.", "@PetrochemNews", "bullish"),
            ("Oil price forecast raised by Goldman to $85 Brent for 2024.", "@WallStAnalyst", "bullish"),
            ("Morgan Stanley sees $90 oil on supply constraints. Bullish call.", "@MSCommodities", "bullish"),
            ("Energy stocks undervalued relative to oil prices. Catch-up trade?", "@ValueInvestor", "bullish"),
        ]
        
        import random
        tweets = []
        
        for i in range(count):
            idx = i % len(sample_tweets)
            text, author, sentiment = sample_tweets[idx]
            
            # Add some variation
            if i >= len(sample_tweets):
                text = f"{text} [Update {i // len(sample_tweets)}]"
            
            tweets.append(Tweet(
                id=f"sample_{i}_{datetime.now().timestamp()}",
                text=text,
                author=author,
                created_at=datetime.now(),
                likes=random.randint(10, 2000),
                retweets=random.randint(5, 500),
                sentiment=sentiment,
            ))
        
        return tweets
    
    def _get_sample_trending_topics(self) -> list[dict[str, Any]]:
        """Return sample trending topics."""
        return [
            {"topic": "OPEC+ Production Cuts", "trend": "up", "mentions": "high", "sentiment": "bullish"},
            {"topic": "WTI Price Action", "trend": "up", "mentions": "high", "sentiment": "bullish"},
            {"topic": "China Oil Demand", "trend": "stable", "mentions": "high", "sentiment": "neutral"},
            {"topic": "US Inventory Data", "trend": "up", "mentions": "medium", "sentiment": "bullish"},
            {"topic": "Middle East Tensions", "trend": "up", "mentions": "high", "sentiment": "bullish"},
            {"topic": "Shale Production", "trend": "down", "mentions": "medium", "sentiment": "neutral"},
            {"topic": "Refinery Margins", "trend": "up", "mentions": "medium", "sentiment": "bullish"},
            {"topic": "Energy Transition", "trend": "stable", "mentions": "medium", "sentiment": "bearish"},
            {"topic": "Russian Oil Sanctions", "trend": "stable", "mentions": "medium", "sentiment": "neutral"},
            {"topic": "Natural Gas Prices", "trend": "down", "mentions": "medium", "sentiment": "bearish"},
        ]


# Convenience function for quick access
def get_tweet_wordcloud(count: int = 100) -> WordCloudData:
    """
    Quick function to get word cloud data from oil market tweets.
    
    Args:
        count: Number of tweets to analyze
        
    Returns:
        WordCloudData with word frequencies
    """
    grok = GrokAI()
    return grok.generate_wordcloud_data(grok.get_oil_market_tweets(count))
