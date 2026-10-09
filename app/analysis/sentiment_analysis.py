"""NLP Sentiment Analysis Engine for News Feeds and Social Media Signals."""

import time
from typing import Any


class SentimentAnalysisEngine:
    """Extracts entities, event classification, and computes weighted social/news sentiment scores."""

    def __init__(self) -> None:
        self.bullish_keywords = {
            "all-time high", "breakout", "inflow", "etf", "surge", "adoption",
            "partnership", "bullish", "upgrade", "recovery", "accumulation", "institutional"
        }
        self.bearish_keywords = {
            "hack", "crackdown", "lawsuit", "liquidation", "dump", "bearish",
            "fraud", "ban", "outflow", "recession", "insolvency", "sec"
        }

    def analyze_news(self, news_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculates time-decayed, credibility-weighted sentiment scores across news feeds."""
        if not news_items:
            # No items means no measurement. A 0.0 "NEUTRAL" here was a fabricated reading that the
            # trajectory model then used as a sentiment input.
            return {"overall_score": None, "classification": "NO_DATA", "item_count": 0}

        now = time.time()
        weighted_scores = []
        entities = set()
        events = []

        for item in news_items:
            title = item.get("title", "").lower()
            cred = item.get("credibility_weight", 0.75)
            ts = item.get("timestamp", now)
            hours_old = max(0.0, (now - ts) / 3600.0)
            time_decay = 1.0 / (1.0 + 0.1 * hours_old)  # Half-life weight decay

            score = 0.0
            pos_matches = sum(1 for kw in self.bullish_keywords if kw in title)
            neg_matches = sum(1 for kw in self.bearish_keywords if kw in title)

            if pos_matches > neg_matches:
                score = 0.6 + min(0.35, pos_matches * 0.15)
            elif neg_matches > pos_matches:
                score = -0.6 - min(0.35, neg_matches * 0.15)
            else:
                # No keyword evidence either way: neutral. The former 0.05 was a small invented
                # bullish bias on every headline with no bullish or bearish keyword.
                score = 0.0

            weighted_scores.append(score * cred * time_decay)

            # Entity & Event parsing
            if "btc" in title or "bitcoin" in title:
                entities.add("Bitcoin (BTC)")
            if "eth" in title or "ethereum" in title:
                entities.add("Ethereum (ETH)")
            if "etf" in title:
                events.append("ETF Inflow Acceleration")
            if "hashrate" in title:
                events.append("Network Hashrate Milestone")

        avg_score = round(sum(weighted_scores) / len(weighted_scores), 3) if weighted_scores else 0.0
        sentiment_label = "BULLISH" if avg_score >= 0.25 else ("BEARISH" if avg_score <= -0.25 else "NEUTRAL")

        return {
            "overall_score": avg_score,
            "classification": sentiment_label,
            "item_count": len(news_items),
            "extracted_entities": list(entities),
            "detected_events": list(set(events)),
            # Reddit and Twitter sentiment were the news score scaled by 1.1 and 0.95, and
            # news_confidence was a fixed 0.88. Neither social feed is read here, so both are
            # None (not measured) and the news confidence is withheld.
            "social_breakdown": {
                "news_confidence": None,
                "reddit_sentiment": None,
                "twitter_sentiment": None,
                "social_feed_status": "not_measured: no reddit or twitter feed is read by this engine",
            }
        }


sentiment_engine = SentimentAnalysisEngine()
