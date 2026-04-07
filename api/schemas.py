"""
Pydantic request/response schemas for the FastAPI layer.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


# ------------------------------------------------------------------
# Ingest
# ------------------------------------------------------------------

class TransactionIn(BaseModel):
    txn_id:       str   = Field(..., description="Unique transaction identifier")
    from_account: str   = Field(..., description="Sending account ID")
    to_account:   str   = Field(..., description="Receiving account ID")
    amount:       float = Field(..., gt=0, description="Transaction amount (positive)")
    channel:      str   = Field(..., description="Payment channel (ACH, WIRE, CARD, P2P, ATM)")
    timestamp:    float = Field(..., description="Unix epoch timestamp")
    label:        Optional[str] = Field(None, description="Ground-truth label (normal/fraud)")

    @field_validator("channel")
    @classmethod
    def channel_must_be_valid(cls, v: str) -> str:
        valid = {"ACH", "WIRE", "CARD", "P2P", "ATM"}
        if v.upper() not in valid:
            raise ValueError(f"channel must be one of {valid}")
        return v.upper()


class BatchTransactionsIn(BaseModel):
    transactions: List[TransactionIn] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested:     int
    graph_nodes:  int
    graph_edges:  int
    message:      str


# ------------------------------------------------------------------
# Risk scoring
# ------------------------------------------------------------------

class FeatureScoreItem(BaseModel):
    feature: str
    score:   float


class RiskResponse(BaseModel):
    account_id:       str
    risk_score:       float
    tier:             str
    weighted_score:   float
    anomaly_boost:    float
    top_drivers:      List[FeatureScoreItem]
    cluster:          Optional[Dict[str, Any]] = None
    text_report:      Optional[str] = None
    missing_features: List[str] = []


# ------------------------------------------------------------------
# Clusters
# ------------------------------------------------------------------

class ClusterResponse(BaseModel):
    community_id:        int
    members:             List[str]
    size:                int
    avg_risk:            float
    max_risk:            float
    density:             float
    cluster_risk_score:  float
    is_mule_ring:        bool
    has_cycle:           bool


# ------------------------------------------------------------------
# Graph stats
# ------------------------------------------------------------------

class GraphStatsResponse(BaseModel):
    nodes:              int
    edges:              int
    total_transactions: int
    is_directed:        bool


# ------------------------------------------------------------------
# SAR
# ------------------------------------------------------------------

class SARStatusUpdate(BaseModel):
    reviewed_by: str = Field(default="compliance_officer")

class SARResponse(BaseModel):
    sar_id:              str
    status:              str
    created_at:          str
    reviewed_at:         Optional[str]
    reviewed_by:         Optional[str]
    cluster_id:          Optional[int]
    times_flagged:       int
    pattern_type:        str
    cluster_risk_score:  float
    is_mule_ring:        bool
    member_count:        int
    total_amount_moved:  float
    subject_accounts:    List[Dict[str, Any]]
    narrative:           str
    timeframe:           Dict[str, Any]
    all_members:         List[str]
    evidence:            Dict[str, Any]
