"""Tests for the scoring pipeline."""

import pytest
from src.scoring.weighted_scorer import WeightedScorer
from src.scoring.normalizer import normalize_score, assign_tier, build_risk_result


@pytest.fixture
def scorer():
    return WeightedScorer()


# ------------------------------------------------------------------
# WeightedScorer
# ------------------------------------------------------------------

def test_scorer_returns_dict(scorer):
    features = {
        "velocity": 0.05, "fan_out_ratio": 5.0, "betweenness": 0.3,
        "retention_time": 10.0, "pagerank": 0.02, "cross_channel_jumps": 4.0,
        "burst_score": 1.5,
    }
    result = scorer.score(features)
    assert "weighted_score" in result
    assert "contributions" in result
    assert "missing" in result


def test_scorer_score_in_range(scorer):
    features = {
        "velocity": 0.1, "fan_out_ratio": 10.0, "betweenness": 0.5,
        "retention_time": 5.0, "pagerank": 0.1, "cross_channel_jumps": 6.0,
        "burst_score": 2.0,
    }
    result = scorer.score(features)
    assert 0.0 <= result["weighted_score"] <= 1.0


def test_scorer_high_fraud_score_exceeds_normal(scorer):
    fraud_features = {
        "velocity": 0.5, "fan_out_ratio": 20.0, "betweenness": 0.9,
        "retention_time": 2.0, "pagerank": 0.5, "cross_channel_jumps": 10.0,
        "burst_score": 5.0,
    }
    normal_features = {
        "velocity": 0.0001, "fan_out_ratio": 0.1, "betweenness": 0.01,
        "retention_time": 86400.0, "pagerank": 0.001, "cross_channel_jumps": 0.0,
        "burst_score": 0.1,
    }
    fraud_score = scorer.score(fraud_features)["weighted_score"]
    normal_score = scorer.score(normal_features)["weighted_score"]
    assert fraud_score > normal_score


def test_scorer_missing_features_reported(scorer):
    result = scorer.score({"velocity": 0.01})
    assert len(result["missing"]) > 0


# ------------------------------------------------------------------
# Normalizer
# ------------------------------------------------------------------

def test_normalize_clamps():
    assert normalize_score(-0.5) == 0.0
    assert normalize_score(1.5) == 1.0
    assert normalize_score(0.5) == pytest.approx(0.5)


def test_assign_tier():
    assert assign_tier(0.10) == "LOW"
    assert assign_tier(0.45) == "MEDIUM"
    assert assign_tier(0.72) == "HIGH"
    assert assign_tier(0.91) == "CRITICAL"


def test_build_risk_result_structure():
    result = build_risk_result(
        account_id="ACC123",
        weighted_score=0.72,
        anomaly_boost=0.05,
        contributions={"velocity": 0.9, "fan_out": 0.7, "centrality": 0.4},
        missing=[],
    )
    assert result["account_id"] == "ACC123"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(result["top_drivers"], list)
    assert result["top_drivers"][0]["feature"] == "velocity"  # highest first
