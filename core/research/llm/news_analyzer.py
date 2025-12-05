"""
News Analyzer
=============
LLM-powered news summarization and analysis for oil markets.

Features:
- News article summarization
- Key event extraction
- Impact assessment
- Multi-source aggregation
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)


class NewsSource(Enum):
    """News sources."""
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    WSJOURNAL = "wsj"
    OILPRICE = "oilprice"
    EIA = "eia"
    OPEC = "opec"
    CUSTOM = "custom"


class ImpactLevel(Enum):
    """Market impact level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEUTRAL = "NEUTRAL"


@dataclass
class NewsArticle:
    """A news article for analysis."""
    article_id: str
    title: str
    content: str
    source: NewsSource = NewsSource.CUSTOM
    url: Optional[str] = None
    published_at: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "source": self.source.value,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "tags": self.tags,
        }


@dataclass
class ArticleSummary:
    """Summary of a news article."""
    article_id: str
    title: str
    summary: str
    key_points: List[str]
    commodities_mentioned: List[str]
    impact_level: ImpactLevel
    impact_direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    extracted_entities: Dict[str, List[str]]  # organizations, people, locations
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "summary": self.summary,
            "key_points": self.key_points,
            "commodities_mentioned": self.commodities_mentioned,
            "impact_level": self.impact_level.value,
            "impact_direction": self.impact_direction,
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class AnalysisConfig:
    """Configuration for news analysis."""
    # LLM provider settings
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    api_key: Optional[str] = None
    
    # Analysis settings
    max_tokens: int = 500
    temperature: float = 0.3
    
    # Focus areas
    commodities: List[str] = field(default_factory=lambda: ["crude oil", "wti", "brent", "gasoline", "heating oil"])
    include_geopolitics: bool = True
    include_supply_demand: bool = True
    include_technical: bool = False
    
    # Output settings
    summary_length: str = "medium"  # short, medium, long
    extract_entities: bool = True


class NewsAnalyzer:
    """
    LLM-powered news analyzer for oil markets.
    
    Supports:
    - OpenAI GPT models
    - Anthropic Claude models
    - Local LLM endpoints
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        
        # Get API key from config or environment
        if not self.config.api_key:
            if self.config.provider == "openai":
                self.config.api_key = os.getenv("OPENAI_API_KEY")
            elif self.config.provider == "anthropic":
                self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Cache for recent analyses
        self._cache: Dict[str, ArticleSummary] = {}
        
        # Rate limiting
        self._request_count = 0
        self._last_reset = datetime.now()
    
    def analyze(self, article: NewsArticle) -> ArticleSummary:
        """
        Analyze a news article.
        
        Args:
            article: News article to analyze
            
        Returns:
            ArticleSummary with analysis results
        """
        # Check cache
        if article.article_id in self._cache:
            return self._cache[article.article_id]
        
        # Build prompt
        prompt = self._build_prompt(article)
        
        # Call LLM
        try:
            if self.config.provider == "openai":
                response = self._call_openai(prompt)
            elif self.config.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                response = self._fallback_analysis(article)
            
            summary = self._parse_response(article, response)
            
            # Cache result
            self._cache[article.article_id] = summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._fallback_analysis(article)
    
    def analyze_batch(self, articles: List[NewsArticle]) -> List[ArticleSummary]:
        """Analyze multiple articles."""
        return [self.analyze(article) for article in articles]
    
    def get_market_summary(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """
        Generate an overall market summary from multiple articles.
        
        Args:
            articles: List of news articles
            
        Returns:
            Market summary with sentiment and key themes
        """
        summaries = self.analyze_batch(articles)
        
        # Aggregate sentiment
        bullish = sum(1 for s in summaries if s.impact_direction == "bullish")
        bearish = sum(1 for s in summaries if s.impact_direction == "bearish")
        neutral = sum(1 for s in summaries if s.impact_direction == "neutral")
        
        total = len(summaries)
        
        # Determine overall sentiment
        if total == 0:
            overall_sentiment = "neutral"
            sentiment_score = 0
        else:
            sentiment_score = (bullish - bearish) / total
            if sentiment_score > 0.2:
                overall_sentiment = "bullish"
            elif sentiment_score < -0.2:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
        
        # Extract common themes
        all_key_points = []
        all_commodities = set()
        all_entities = {"organizations": set(), "people": set(), "locations": set()}
        
        for summary in summaries:
            all_key_points.extend(summary.key_points)
            all_commodities.update(summary.commodities_mentioned)
            for entity_type, entities in summary.extracted_entities.items():
                if entity_type in all_entities:
                    all_entities[entity_type].update(entities)
        
        # Find high-impact articles
        high_impact = [
            s.to_dict() for s in summaries 
            if s.impact_level == ImpactLevel.HIGH
        ]
        
        return {
            "article_count": total,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_breakdown": {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
            },
            "commodities_mentioned": list(all_commodities),
            "key_themes": all_key_points[:10],  # Top 10 key points
            "high_impact_articles": high_impact,
            "entities": {k: list(v) for k, v in all_entities.items()},
            "generated_at": datetime.now().isoformat(),
        }
    
    def _build_prompt(self, article: NewsArticle) -> str:
        """Build the analysis prompt."""
        commodities = ", ".join(self.config.commodities)
        
        prompt = f"""Analyze the following oil market news article and provide a structured summary.

ARTICLE:
Title: {article.title}
Source: {article.source.value}
Published: {article.published_at.strftime("%Y-%m-%d")}

{article.content}

INSTRUCTIONS:
1. Provide a concise summary (2-3 sentences)
2. Extract 3-5 key points
3. Identify commodities mentioned ({commodities})
4. Assess the potential market impact:
   - Impact Level: HIGH, MEDIUM, LOW, or NEUTRAL
   - Impact Direction: bullish, bearish, or neutral
   - Confidence (0-1)
5. Extract named entities (organizations, people, locations)

FORMAT YOUR RESPONSE AS JSON:
{{
    "summary": "...",
    "key_points": ["...", "..."],
    "commodities_mentioned": ["..."],
    "impact_level": "HIGH|MEDIUM|LOW|NEUTRAL",
    "impact_direction": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "entities": {{
        "organizations": ["..."],
        "people": ["..."],
        "locations": ["..."]
    }}
}}
"""
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key not configured")
        
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are an expert oil market analyst."},
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
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result["choices"][0]["message"]["content"]
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        if not self.config.api_key:
            raise ValueError("Anthropic API key not configured")
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        
        data = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers=headers,
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result["content"][0]["text"]
    
    def _parse_response(self, article: NewsArticle, response: str) -> ArticleSummary:
        """Parse LLM response into ArticleSummary."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            return ArticleSummary(
                article_id=article.article_id,
                title=article.title,
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                commodities_mentioned=data.get("commodities_mentioned", []),
                impact_level=ImpactLevel(data.get("impact_level", "NEUTRAL")),
                impact_direction=data.get("impact_direction", "neutral"),
                confidence=float(data.get("confidence", 0.5)),
                extracted_entities=data.get("entities", {}),
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse response: {e}")
            return self._fallback_analysis(article)
    
    def _fallback_analysis(self, article: NewsArticle) -> ArticleSummary:
        """Fallback analysis when LLM is unavailable."""
        # Simple keyword-based analysis
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        
        # Detect sentiment
        bullish_keywords = ["increase", "rise", "gain", "rally", "surge", "higher", "growth", "supply cut"]
        bearish_keywords = ["decrease", "fall", "drop", "decline", "lower", "weak", "surplus", "oversupply"]
        
        bullish_count = sum(1 for kw in bullish_keywords if kw in content_lower)
        bearish_count = sum(1 for kw in bearish_keywords if kw in content_lower)
        
        if bullish_count > bearish_count + 2:
            direction = "bullish"
        elif bearish_count > bullish_count + 2:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Detect commodities
        commodities = []
        for commodity in self.config.commodities:
            if commodity.lower() in content_lower or commodity.lower() in title_lower:
                commodities.append(commodity)
        
        # Detect impact level
        high_impact_keywords = ["opec", "sanctions", "war", "crisis", "emergency", "breakthrough"]
        medium_impact_keywords = ["inventory", "production", "demand", "export", "import"]
        
        if any(kw in content_lower for kw in high_impact_keywords):
            impact = ImpactLevel.HIGH
        elif any(kw in content_lower for kw in medium_impact_keywords):
            impact = ImpactLevel.MEDIUM
        else:
            impact = ImpactLevel.LOW
        
        # Generate simple summary
        sentences = article.content.split(".")
        summary = ". ".join(sentences[:2]) + "." if len(sentences) > 1 else article.content[:200]
        
        return ArticleSummary(
            article_id=article.article_id,
            title=article.title,
            summary=summary,
            key_points=[article.title],
            commodities_mentioned=commodities or ["crude oil"],
            impact_level=impact,
            impact_direction=direction,
            confidence=0.5,
            extracted_entities={"organizations": [], "people": [], "locations": []},
        )
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()
