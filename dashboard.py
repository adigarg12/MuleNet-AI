"""
Fraud Detection Dashboard — Streamlit GUI

Run:
    .venv/Scripts/streamlit run dashboard.py
"""

import os, sys, tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="FraudGraph AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  *, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* ── Fix Streamlit expander arrow rendering as "arrows_down" text ── */
  [data-testid="stExpanderToggleIcon"] { display: none !important; }
  details > summary svg { display: inline !important; }

  /* ── App background ── */
  [data-testid="stAppViewContainer"] { background: #0a0e17; }
  [data-testid="stMain"] { background: #0a0e17; }
  .main .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

  /* ── Nav radio → pill buttons ── */
  div[data-testid="stRadio"] > label { display: none; }
  div[data-testid="stRadio"] > div { gap: 4px !important; flex-direction: column !important; }
  div[data-testid="stRadio"] > div > label {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 9px 14px !important;
    color: #8b949e !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  div[data-testid="stRadio"] > div > label:hover {
    background: #161b22 !important;
    color: #c9d1d9 !important;
  }
  div[data-testid="stRadio"] > div > label[data-selected="true"] {
    background: #1f2d3d !important;
    color: #58a6ff !important;
    border-left: 3px solid #1f6feb !important;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #30363d; }
  .metric-card .label {
    font-size: 0.72rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;
  }
  .metric-card .value {
    font-size: 1.9rem; font-weight: 700; color: #e6edf3;
    margin-top: 6px; line-height: 1;
  }
  .metric-card .value.red    { color: #f85149; }
  .metric-card .value.orange { color: #e67e22; }
  .metric-card .value.green  { color: #3fb950; }
  .metric-card .value.blue   { color: #58a6ff; }

  /* ── KPI banner cards ── */
  .kpi-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
  }
  .kpi-card.red::before   { background: #f85149; }
  .kpi-card.orange::before{ background: #e67e22; }
  .kpi-card.blue::before  { background: #1f6feb; }
  .kpi-card.green::before { background: #3fb950; }
  .kpi-card .kpi-label {
    font-size: 0.72rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;
  }
  .kpi-card .kpi-value {
    font-size: 2.4rem; font-weight: 700; color: #e6edf3;
    margin-top: 8px; line-height: 1;
  }
  .kpi-card .kpi-sub {
    font-size: 0.78rem; color: #8b949e; margin-top: 4px;
  }

  /* ── Section titles ── */
  .section-title {
    font-size: 0.88rem; font-weight: 600; color: #c9d1d9;
    border-left: 3px solid #1f6feb; padding-left: 10px;
    margin: 22px 0 12px; letter-spacing: 0.02em;
    text-transform: uppercase;
  }

  /* ── Page title ── */
  h1 { color: #e6edf3 !important; font-weight: 700 !important; font-size: 1.7rem !important; }
  h2, h3 { color: #c9d1d9 !important; }

  /* ── Dataframe ── */
  div[data-testid="stDataFrame"] {
    border: 1px solid #21262d; border-radius: 10px; overflow: hidden;
  }

  /* ── Divider ── */
  hr { border-color: #21262d !important; margin: 1.2rem 0 !important; }

  /* ── Expander ── */
  details[data-testid="stExpander"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    margin-bottom: 10px;
    overflow: hidden;
  }
  details[data-testid="stExpander"] summary {
    padding: 14px 18px;
    font-size: 0.88rem;
    font-weight: 500;
    color: #c9d1d9;
    background: #0d1117;
  }
  details[data-testid="stExpander"] summary:hover { background: #161b22; }
  details[data-testid="stExpander"][open] summary { border-bottom: 1px solid #21262d; }
  details[data-testid="stExpander"] > div { padding: 0; }

  /* ── SAR status badge ── */
  .sar-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .sar-badge.pending  { background: #3d2a0a; color: #e67e22; border: 1px solid #5a3d14; }
  .sar-badge.approved { background: #0d2a14; color: #3fb950; border: 1px solid #196127; }
  .sar-badge.dismissed{ background: #1c2128; color: #8b949e; border: 1px solid #30363d; }

  /* ── SAR card ── */
  .sar-card {
    background: #0d1117; border: 1px solid #21262d; border-radius: 14px;
    padding: 20px 22px; margin-bottom: 14px;
    transition: border-color 0.2s;
  }
  .sar-card.pending  { border-left: 4px solid #e67e22; }
  .sar-card.approved { border-left: 4px solid #3fb950; }
  .sar-card.dismissed{ border-left: 4px solid #30363d; }

  /* ── Account header card ── */
  .account-header {
    border-radius: 14px; padding: 20px 24px; margin-bottom: 18px;
  }

  /* ── Neighbor row ── */
  .nb-row {
    padding: 6px 0; border-bottom: 1px solid #161b22;
    display: flex; align-items: center; gap: 8px;
  }

  /* ── Info/warning message ── */
  .empty-state {
    text-align: center; padding: 48px 24px;
    color: #6e7681; font-size: 0.9rem;
  }
  .empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }

  /* ── Slider + checkbox ── */
  .stSlider > label { color: #8b949e !important; font-size: 0.82rem; }
  .stCheckbox > label { color: #c9d1d9 !important; }

  /* ── Text input ── */
  .stTextInput > div > div > input {
    background: #0d1117 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important;
    font-size: 0.9rem !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #1f6feb !important; box-shadow: 0 0 0 3px rgba(31,111,235,0.15) !important;
  }

  /* ── Plotly tooltip z-index fix ── */
  .js-plotly-plot .plotly .modebar { background: transparent !important; }

  /* ── Sidebar branding ── */
  .sidebar-brand {
    padding: 4px 0 18px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 18px;
  }
  .sidebar-brand .title {
    font-size: 1.15rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.01em;
  }
  .sidebar-brand .subtitle {
    font-size: 0.75rem; color: #6e7681; margin-top: 3px;
  }

  /* ── Sidebar stat pill ── */
  .stat-pill {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; background: #161b22; border-radius: 8px;
    margin-bottom: 6px; border: 1px solid #21262d;
  }
  .stat-pill .sp-label { font-size: 0.78rem; color: #8b949e; }
  .stat-pill .sp-value { font-size: 0.88rem; font-weight: 600; color: #e6edf3; }
  .stat-pill .sp-value.red { color: #f85149; }
  .stat-pill .sp-value.orange { color: #e67e22; }
  .stat-pill .sp-value.green { color: #3fb950; }

  /* ── Narrative box ── */
  .narrative-box {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 14px 18px; font-size: 0.88rem; line-height: 1.7;
    color: #c9d1d9; white-space: pre-wrap;
  }

  /* ── Timeline caption ── */
  .timeline-caption {
    font-size: 0.78rem; color: #6e7681;
    background: #161b22; border-radius: 8px; padding: 8px 14px;
    border: 1px solid #21262d; margin-top: 8px;
    display: inline-block;
  }

  /* ── Stmetric overrides ── */
  [data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #6e7681 !important; text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
  div[data-testid="metric-container"] {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 12px; padding: 16px 18px;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TIER_COLOR = {"CRITICAL": "#f85149", "HIGH": "#e67e22", "MEDIUM": "#d29922", "LOW": "#3fb950"}
TIER_BG    = {"CRITICAL": "#2a0d0d", "HIGH": "#2a1a05", "MEDIUM": "#2a2205", "LOW": "#071a0e"}
TIER_BORDER= {"CRITICAL": "#5a1a1a", "HIGH": "#5a3010", "MEDIUM": "#5a4a10", "LOW": "#0d3a1a"}

CHART_BASE = dict(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
)
_AXES = dict(gridcolor="#161b22", zerolinecolor="#21262d", linecolor="#21262d")

def _chart(fig, height=None, margin=None):
    m = margin or dict(t=12, b=36, l=48, r=12)
    fig.update_layout(**CHART_BASE, margin=m, xaxis=_AXES, yaxis=_AXES)
    if height:
        fig.update_layout(height=height)
    return fig

def _score_style(col):
    styles = []
    for v in col:
        try:
            r = float(v)
            red   = int(220 * r)
            green = int(160 * (1 - r))
            styles.append(f"background-color: rgba({red},{green},30,0.4); color: #e6edf3")
        except Exception:
            styles.append("")
    return styles

def style_scores(df, col):
    return df.style.apply(_score_style, subset=[col])

    return fig

# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training GNN on transaction graph…")
def load_pipeline():
    from src.ingestion.batch_loader import load_all
    from src.graph.graph_store import graph_store
    from src.graph.graph_builder import graph_summary
    from src.gnn.embedder import gnn_embedder
    from src.scoring.normalizer import assign_tier
    from src.network.community_detection import detect_communities
    from src.network.cluster_scorer import score_all_clusters

    data_dir = os.path.join(os.path.dirname(__file__), "data", "synthetic")
    txns = list(load_all(data_dir))
    graph_store.load_fresh(txns)
    G = graph_store.get_graph()

    fraud_count, total_count = {}, {}
    for t in txns:
        for acc in [t["from_account"], t["to_account"]]:
            total_count[acc] = total_count.get(acc, 0) + 1
            if t.get("label") in ("fraud", 1, "1"):
                fraud_count[acc] = fraud_count.get(acc, 0) + 1
    labels = {a: fraud_count.get(a, 0) / total_count[a] for a in total_count}

    gnn_embedder.train(G, labels=labels, epochs=200)
    all_scores = gnn_embedder.get_all_scores()

    risk_scores = {}
    for node, score in all_scores.items():
        risk_scores[node] = {
            "account_id":  node,
            "risk_score":  round(score, 4),
            "tier":        assign_tier(score),
            "fraud_ratio": round(labels.get(node, 0.0), 3),
        }

    communities     = detect_communities(G)
    scored_clusters = score_all_clusters(G, communities, all_scores, min_members=3)
    stats           = graph_summary(G)
    return G, risk_scores, scored_clusters, stats, gnn_embedder, labels

G, risk_scores, scored_clusters, stats, embedder, labels = load_pipeline()

@st.cache_data(show_spinner=False)
def get_explanation(account_id):
    return embedder.explain_node(account_id)

@st.cache_data(show_spinner=False)
def get_global_importance():
    from src.gnn.node_features import FEATURE_NAMES
    import torch
    data, model, scores_t = embedder._data, embedder._model, embedder._scores
    high_mask = scores_t >= 0.5
    if high_mask.sum() == 0:
        return {k: 0.0 for k in FEATURE_NAMES}
    model.eval()
    x_in = data.x.detach().clone().requires_grad_(True)
    probs = model(x_in, data.edge_index)
    probs[high_mask].mean().backward()
    imp = (x_in.grad.abs().mean(0) * data.x.abs().mean(0)).detach()
    m = imp.max()
    if m > 0: imp = imp / m
    return {k: round(float(imp[i].item()), 4) for i, k in enumerate(FEATURE_NAMES)}

def render_pyvis(net, height=500):
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        html = open(f.name, encoding="utf-8").read()
    st.components.v1.html(html, height=height, scrolling=False)

# ── Sidebar ────────────────────────────────────────────────────────────────────
n_critical = sum(1 for r in risk_scores.values() if r["tier"] == "CRITICAL")
n_high     = sum(1 for r in risk_scores.values() if r["tier"] == "HIGH")
n_mule     = sum(1 for c in scored_clusters if c["is_mule_ring"])

NAV_ICONS = {
    "Overview":          "◉",
    "Account Inspector": "⬡",
    "Clusters":          "⬡",
    "SAR Queue":         "⚑",
}

with st.sidebar:
    st.markdown("## 🔍 FraudGraph AI")
    st.caption("GNN-powered transaction intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Account Inspector", "Clusters", "SAR Queue"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("<div style='font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em;font-weight:600;margin-bottom:8px'>Live Stats</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='stat-pill'><span class='sp-label'>Accounts</span>
      <span class='sp-value blue'>{stats['nodes']:,}</span></div>
    <div class='stat-pill'><span class='sp-label'>Transactions</span>
      <span class='sp-value'>{stats['total_transactions']:,}</span></div>
    <div class='stat-pill'><span class='sp-label'>Critical</span>
      <span class='sp-value red'>{n_critical}</span></div>
    <div class='stat-pill'><span class='sp-label'>High Risk</span>
      <span class='sp-value orange'>{n_high}</span></div>
    <div class='stat-pill'><span class='sp-label'>Mule Rings</span>
      <span class='sp-value {"red" if n_mule else "green"}'>{n_mule}</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:auto;padding-top:24px;font-size:0.72rem;color:#3d444d;text-align:center'>GraphSAGE · Community Detection · AUC 100%</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("<h1>Overview</h1>", unsafe_allow_html=True)

    n_susp_clusters = sum(1 for c in scored_clusters if c["cluster_risk_score"] >= 0.6)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi-card blue'>
          <div class='kpi-label'>Total Accounts</div>
          <div class='kpi-value'>{stats['nodes']:,}</div>
          <div class='kpi-sub'>{stats['total_transactions']:,} transactions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pct = f"{n_critical/max(stats['nodes'],1)*100:.1f}%"
        st.markdown(f"""<div class='kpi-card red'>
          <div class='kpi-label'>Critical Accounts</div>
          <div class='kpi-value'>{n_critical}</div>
          <div class='kpi-sub'>{pct} of total</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card {"red" if n_mule else "green"}'>
          <div class='kpi-label'>Mule Rings</div>
          <div class='kpi-value'>{n_mule}</div>
          <div class='kpi-sub'>{"Active networks" if n_mule else "None detected"}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card orange'>
          <div class='kpi-label'>Suspicious Clusters</div>
          <div class='kpi-value'>{n_susp_clusters}</div>
          <div class='kpi-sub'>risk score ≥ 0.60</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    st.divider()

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown("<div class='section-title'>Risk Score Distribution</div>", unsafe_allow_html=True)
        scores = [r["risk_score"] for r in risk_scores.values()]
        fig = px.histogram(x=scores, nbins=40, color_discrete_sequence=["#1f6feb"],
                           labels={"x": "Risk Score", "y": "Accounts"})
        fig.update_traces(marker_line_width=0)
        fig = _chart(fig, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>Accounts by Risk Tier</div>", unsafe_allow_html=True)
        tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in risk_scores.values():
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
        fig2 = px.pie(names=list(tier_counts.keys()), values=list(tier_counts.values()),
                      color=list(tier_counts.keys()), color_discrete_map=TIER_COLOR, hole=0.55)
        fig2.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", family="Inter, sans-serif"),
            margin=dict(t=12, b=12, l=12, r=12),
            height=260,
            legend=dict(
                bgcolor="rgba(0,0,0,0)", font=dict(size=12),
                orientation="v", x=1.0,
            ),
        )
        fig2.update_traces(textfont_color="#e6edf3", textfont_size=12)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Top 20 Riskiest Accounts</div>", unsafe_allow_html=True)
    top20 = sorted(risk_scores.values(), key=lambda r: r["risk_score"], reverse=True)[:20]
    df_top = pd.DataFrame([{
        "Account":     r["account_id"],
        "Risk Score":  r["risk_score"],
        "Tier":        r["tier"],
        "Fraud Ratio": f"{r['fraud_ratio']:.0%}",
    } for r in top20])
    st.dataframe(style_scores(df_top, "Risk Score"), use_container_width=True, hide_index=True, height=340)

    st.markdown("<div class='section-title'>GNN Feature Importance</div>", unsafe_allow_html=True)
    global_imp = get_global_importance()
    sorted_imp = dict(sorted(global_imp.items(), key=lambda x: x[1]))
    fig3 = px.bar(
        x=list(sorted_imp.values()), y=list(sorted_imp.keys()),
        orientation="h", color=list(sorted_imp.values()),
        color_continuous_scale=[[0, "#1f6feb"], [0.5, "#e67e22"], [1, "#f85149"]],
        labels={"x": "Importance", "y": ""},
    )
    fig3.update_traces(marker_line_width=0)
    fig3 = _chart(fig3, height=310)
    fig3.update_layout(coloraxis_showscale=False)
    fig3.update_xaxes(range=[0, 1.05])
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Account Inspector":
    st.markdown("<h1>Account Inspector</h1>", unsafe_allow_html=True)

    query = st.text_input(
        "search", placeholder="🔎  Search by account ID…",
        label_visibility="collapsed",
    )
    all_accounts     = sorted(risk_scores.keys(), key=lambda a: risk_scores[a]["risk_score"], reverse=True)
    filtered_accounts = [a for a in all_accounts if query.lower() in a.lower()] if query else all_accounts

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("<div class='section-title'>Accounts</div>", unsafe_allow_html=True)
        df_all = pd.DataFrame([{
            "Account": a,
            "Score":   risk_scores[a]["risk_score"],
            "Tier":    risk_scores[a]["tier"],
        } for a in filtered_accounts])
        event = st.dataframe(
            style_scores(df_all, "Score"),
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            height=540,
        )

    with right:
        sel_rows   = event.selection.rows if event and event.selection else []
        account_id = df_all.iloc[sel_rows[0]]["Account"] if sel_rows else (
            df_all.iloc[0]["Account"] if not df_all.empty else None)

        if account_id:
            r     = risk_scores[account_id]
            tier  = r["tier"]
            color = TIER_COLOR.get(tier, "#888")
            bg    = TIER_BG.get(tier, "#161b22")
            bd    = TIER_BORDER.get(tier, "#30363d")

            st.markdown(f"""
            <div class='account-header' style='background:{bg};border:1px solid {bd}'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                  <div style='font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em;font-weight:600'>Account ID</div>
                  <div style='font-size:1.25rem;font-weight:700;color:#e6edf3;margin-top:2px;font-family:monospace'>{account_id}</div>
                </div>
                <div style='background:{color}22;border:1px solid {color}55;border-radius:20px;padding:4px 14px;font-size:0.75rem;font-weight:700;color:{color};letter-spacing:0.05em'>{tier}</div>
              </div>
              <div style='display:flex;gap:32px;margin-top:18px'>
                <div>
                  <div style='font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em'>Risk Score</div>
                  <div style='font-size:2rem;font-weight:700;color:{color};margin-top:2px'>{r['risk_score']:.3f}</div>
                </div>
                <div>
                  <div style='font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em'>Fraud Ratio</div>
                  <div style='font-size:2rem;font-weight:700;color:{"#f85149" if r["fraud_ratio"]>0.5 else "#d29922" if r["fraud_ratio"]>0.1 else "#3fb950"};margin-top:2px'>{r["fraud_ratio"]:.0%}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>GNN Feature Importances</div>", unsafe_allow_html=True)
            contrib = get_explanation(account_id)
            sorted_contrib = dict(sorted(contrib.items(), key=lambda x: x[1], reverse=True))
            bar_colors = ["#f85149" if v > 0.5 else "#e67e22" if v > 0.25 else "#1f6feb"
                          for v in sorted_contrib.values()]
            fig_feat = go.Figure(go.Bar(
                x=list(sorted_contrib.values()), y=list(sorted_contrib.keys()),
                orientation="h", marker_color=bar_colors, marker_line_width=0,
                text=[f"{v:.2f}" for v in sorted_contrib.values()],
                textposition="outside", textfont=dict(color="#8b949e", size=11),
            ))
            fig_feat.update_layout(
                **CHART_BASE,
                xaxis=dict(range=[0, 1.15], gridcolor="#161b22", title="Importance"),
                yaxis=dict(gridcolor="#161b22"),
                margin=dict(t=5, b=5, l=10, r=60), height=290,
            )
            st.plotly_chart(fig_feat, use_container_width=True)

            in_nb  = list(G.predecessors(account_id))
            out_nb = list(G.successors(account_id))
            nc1, nc2 = st.columns(2, gap="medium")
            with nc1:
                st.markdown(f"<div class='section-title'>Receives From &nbsp;<span style='color:#6e7681;font-weight:400'>({len(in_nb)})</span></div>", unsafe_allow_html=True)
                if in_nb:
                    for nb in in_nb[:8]:
                        nb_r = risk_scores.get(nb, {})
                        c    = TIER_COLOR.get(nb_r.get("tier", "LOW"), "#3fb950")
                        s    = nb_r.get("risk_score", 0)
                        st.markdown(
                            f'<div class="nb-row">'
                            f'<span style="color:{c};font-size:0.7rem">⬤</span>'
                            f'<code style="color:#c9d1d9;font-size:0.8rem;flex:1">{nb}</code>'
                            f'<span style="color:{c};font-size:0.82rem;font-weight:600">{s:.3f}</span>'
                            f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#6e7681;font-size:0.85rem;padding:8px 0'>No incoming connections</div>", unsafe_allow_html=True)
            with nc2:
                st.markdown(f"<div class='section-title'>Sends To &nbsp;<span style='color:#6e7681;font-weight:400'>({len(out_nb)})</span></div>", unsafe_allow_html=True)
                if out_nb:
                    for nb in out_nb[:8]:
                        nb_r = risk_scores.get(nb, {})
                        c    = TIER_COLOR.get(nb_r.get("tier", "LOW"), "#3fb950")
                        s    = nb_r.get("risk_score", 0)
                        st.markdown(
                            f'<div class="nb-row">'
                            f'<span style="color:{c};font-size:0.7rem">⬤</span>'
                            f'<code style="color:#c9d1d9;font-size:0.8rem;flex:1">{nb}</code>'
                            f'<span style="color:{c};font-size:0.82rem;font-weight:600">{s:.3f}</span>'
                            f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#6e7681;font-size:0.85rem;padding:8px 0'>No outgoing connections</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>32-dim GNN Embedding</div>", unsafe_allow_html=True)
            emb = embedder.get_embedding(account_id).reshape(4, 8)
            fig_emb = px.imshow(emb, color_continuous_scale="RdBu_r", aspect="auto")
            fig_emb.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", family="Inter, sans-serif"),
                margin=dict(t=5, b=5, l=5, r=5), height=130,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_emb, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Clusters":
    st.markdown("<h1>Community Clusters</h1>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='kpi-card blue'>
          <div class='kpi-label'>Total Clusters</div>
          <div class='kpi-value'>{len(scored_clusters)}</div>
          <div class='kpi-sub'>detected communities</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card {"red" if n_mule else "green"}'>
          <div class='kpi-label'>Mule Rings</div>
          <div class='kpi-value'>{n_mule}</div>
          <div class='kpi-sub'>{"confirmed rings" if n_mule else "none detected"}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        n_hi = sum(1 for c in scored_clusters if c["cluster_risk_score"] >= 0.6)
        st.markdown(f"""<div class='kpi-card orange'>
          <div class='kpi-label'>High-Risk Clusters</div>
          <div class='kpi-value'>{n_hi}</div>
          <div class='kpi-sub'>risk score ≥ 0.60</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    st.divider()

    fc1, fc2 = st.columns([1, 2])
    mule_only   = fc1.checkbox("Mule rings only", value=False)
    min_cl_risk = fc2.slider("Min cluster risk score", 0.0, 1.0, 0.0, 0.05,
                             help="Filter clusters by minimum risk score")

    to_show = sorted(
        [c for c in scored_clusters
         if c["cluster_risk_score"] >= min_cl_risk
         and (not mule_only or c["is_mule_ring"])],
        key=lambda c: c["cluster_risk_score"], reverse=True,
    )

    if not to_show:
        st.markdown("""<div class='empty-state'>
          <div class='icon'>🔍</div>
          <div>No clusters match the current filters.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-title'>Cluster Summary</div>", unsafe_allow_html=True)
        df_cl = pd.DataFrame([{
            "ID":           c.get("community_id", i),
            "Members":      c["size"],
            "Cluster Risk": round(c["cluster_risk_score"], 3),
            "Avg Risk":     round(c["avg_risk"], 3),
            "Max Risk":     round(c["max_risk"], 3),
            "Density":      round(c["density"], 3),
            "Cycle":        "✓" if c["has_cycle"] else "—",
            "Mule Ring":    "YES" if c["is_mule_ring"] else "—",
        } for i, c in enumerate(to_show)])
        st.dataframe(style_scores(df_cl, "Cluster Risk"), use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Cluster Detail View</div>", unsafe_allow_html=True)

        for i, c in enumerate(to_show[:15]):
            risk_val = c["cluster_risk_score"]
            risk_tier = (
                "CRITICAL" if risk_val >= 0.85 else
                "HIGH"     if risk_val >= 0.60 else
                "MEDIUM"   if risk_val >= 0.30 else "LOW"
            )
            risk_col  = TIER_COLOR[risk_tier]
            mule_tag  = "  🔴 MULE RING" if c["is_mule_ring"] else ""
            exp_label = (
                f"Cluster {c.get('community_id', i)}  ·  "
                f"{c['size']} members  ·  "
                f"risk {risk_val:.3f}{mule_tag}"
            )

            with st.expander(exp_label, expanded=(i == 0)):
                members    = c.get("members", [])
                member_set = set(members)

                st.markdown("<div class='section-title'>Transaction Subgraph</div>", unsafe_allow_html=True)
                from pyvis.network import Network
                net2 = Network(height="560px", width="100%", directed=True,
                               bgcolor="#080c12", font_color="#c9d1d9")
                net2.set_options("""{
                  "physics": {
                    "stabilization": {"iterations": 300, "fit": true},
                    "barnesHut": {
                      "gravitationalConstant": -20000,
                      "centralGravity": 0.1,
                      "springLength": 200,
                      "springConstant": 0.04,
                      "damping": 0.2
                    }
                  },
                  "edges": {
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                    "smooth": {"type": "dynamic"}
                  },
                  "nodes": {"borderWidth": 2}
                }""")
                for m in members:
                    score = risk_scores.get(m, {}).get("risk_score", 0)
                    tier  = risk_scores.get(m, {}).get("tier", "LOW")
                    col   = TIER_COLOR.get(tier, "#3fb950")
                    net2.add_node(m, label=m[:10],
                                  title=f"<b>{m}</b><br>Score: {score:.3f}<br>Tier: {tier}",
                                  color={"background": col, "border": "#0d1117",
                                         "highlight": {"background": col, "border": "#fff"}},
                                  size=14 + int(score * 28))
                for u, v, d in G.edges(data=True):
                    if u in member_set and v in member_set:
                        amt = d.get("total_amount", 0)
                        net2.add_edge(u, v, title=f"${amt:,.0f}",
                                      width=1.5 + min(amt / 4000, 4),
                                      color={"color": "#21262d", "highlight": "#58a6ff"})
                render_pyvis(net2, height=570)

                stats_col, mem_col = st.columns([1, 2], gap="large")

                with stats_col:
                    st.markdown("<div class='section-title'>Cluster Stats</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                      <div class='metric-card'><div class='label'>Cluster Risk</div>
                        <div class='value' style='color:{risk_col};font-size:1.4rem'>{c['cluster_risk_score']:.3f}</div></div>
                      <div class='metric-card'><div class='label'>Members</div>
                        <div class='value' style='font-size:1.4rem'>{c['size']}</div></div>
                      <div class='metric-card'><div class='label'>Avg Risk</div>
                        <div class='value' style='font-size:1.4rem'>{c['avg_risk']:.3f}</div></div>
                      <div class='metric-card'><div class='label'>Max Risk</div>
                        <div class='value' style='font-size:1.4rem'>{c['max_risk']:.3f}</div></div>
                      <div class='metric-card'><div class='label'>Density</div>
                        <div class='value' style='font-size:1.4rem'>{c['density']:.3f}</div></div>
                      <div class='metric-card'><div class='label'>Cycle</div>
                        <div class='value {"red" if c["has_cycle"] else "green"}' style='font-size:1.4rem'>{"YES" if c["has_cycle"] else "NO"}</div></div>
                      <div class='metric-card' style='grid-column:span 2'><div class='label'>Mule Ring Detection</div>
                        <div class='value {"red" if c["is_mule_ring"] else "green"}' style='font-size:1.3rem'>{"🔴 CONFIRMED" if c["is_mule_ring"] else "🟢 NOT DETECTED"}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                with mem_col:
                    st.markdown("<div class='section-title'>Member Accounts</div>", unsafe_allow_html=True)
                    mem_rows = sorted(
                        [risk_scores[m] for m in members if m in risk_scores],
                        key=lambda r: r["risk_score"], reverse=True,
                    )
                    df_mem = pd.DataFrame([{
                        "Account": r["account_id"],
                        "Score":   r["risk_score"],
                        "Tier":    r["tier"],
                        "Fraud %": f"{r['fraud_ratio']:.0%}",
                    } for r in mem_rows])
                    st.dataframe(style_scores(df_mem, "Score"),
                                 use_container_width=True, hide_index=True, height=340)


# ══════════════════════════════════════════════════════════════════════════════
# SAR QUEUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SAR Queue":
    import requests

    API_BASE = "http://localhost:8000"

    def api_get(path):
        try:
            r = requests.get(f"{API_BASE}{path}", timeout=3)
            return r.json() if r.ok else []
        except Exception:
            return []

    def api_post(path):
        try:
            r = requests.post(f"{API_BASE}{path}", timeout=3)
            return r.ok
        except Exception:
            return False

    st.markdown("<h1>SAR Queue</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#6e7681;font-size:0.88rem;margin-bottom:20px'>Suspicious Activity Report drafts — review and approve before regulatory filing.</div>", unsafe_allow_html=True)

    all_sars = api_get("/sar")

    n_pending   = sum(1 for s in all_sars if s["status"] == "pending")
    n_approved  = sum(1 for s in all_sars if s["status"] == "approved")
    n_dismissed = sum(1 for s in all_sars if s["status"] == "dismissed")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi-card blue'>
          <div class='kpi-label'>Total SARs</div>
          <div class='kpi-value'>{len(all_sars)}</div>
          <div class='kpi-sub'>all time</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card {"orange" if n_pending else "green"}'>
          <div class='kpi-label'>Pending Review</div>
          <div class='kpi-value'>{n_pending}</div>
          <div class='kpi-sub'>{"requires action" if n_pending else "all reviewed"}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card green'>
          <div class='kpi-label'>Approved</div>
          <div class='kpi-value'>{n_approved}</div>
          <div class='kpi-sub'>filed with regulator</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card blue'>
          <div class='kpi-label'>Dismissed</div>
          <div class='kpi-value'>{n_dismissed}</div>
          <div class='kpi-sub'>false positives</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    st.divider()

    if not all_sars:
        st.markdown("""
        <div class='empty-state'>
          <div class='icon'>📋</div>
          <div style='font-size:1rem;color:#c9d1d9;font-weight:600;margin-bottom:8px'>No SAR drafts yet</div>
          <div>The system auto-generates a SAR after a cluster is flagged <strong>3 times</strong>.<br>
          Run the pipeline with <code>--stream --retrain-every 20</code> to start detecting.</div>
        </div>""", unsafe_allow_html=True)
    else:
        status_filter = st.radio(
            "filter", ["All", "Pending", "Approved", "Dismissed"],
            horizontal=True, label_visibility="collapsed",
        )
        filtered_sars = all_sars if status_filter == "All" else [
            s for s in all_sars if s["status"] == status_filter.lower()
        ]

        if not filtered_sars:
            st.markdown(f"""<div class='empty-state'>
              <div class='icon'>🔍</div>
              <div>No {status_filter.lower()} SARs found.</div>
            </div>""", unsafe_allow_html=True)
        else:
            for sar in filtered_sars:
                status       = sar["status"]
                status_color = {"pending": "#e67e22", "approved": "#3fb950", "dismissed": "#6e7681"}.get(status, "#888")
                pattern_label = sar["pattern_type"].replace("_", " ").upper()

                with st.expander(
                    f"{'🟡' if status=='pending' else '🟢' if status=='approved' else '⚫'}  "
                    f"{sar['sar_id']}  ·  {pattern_label}  ·  "
                    f"{sar['member_count']} accounts  ·  ${sar['total_amount_moved']:,.0f}",
                    expanded=(status == "pending"),
                ):
                    # ── Status badge + meta row ──
                    badge_cls = status
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'>
                      <span class='sar-badge {badge_cls}'>{status}</span>
                      <span style='color:#6e7681;font-size:0.8rem'>
                        Generated {sar['created_at'][:10]}
                        {f" · Reviewed {(sar.get('reviewed_at') or '')[:10]} by {sar.get('reviewed_by','—')}" if sar.get('reviewed_at') else ""}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Metrics row ──
                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Cluster Risk",  f"{sar['cluster_risk_score']:.3f}")
                    h2.metric("Times Flagged", sar["times_flagged"])
                    h3.metric("Members",       sar["member_count"])
                    h4.metric("Amount Moved",  f"${sar['total_amount_moved']:,.0f}")

                    # ── Narrative ──
                    st.markdown("<div class='section-title'>Narrative</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='narrative-box'>{sar['narrative']}</div>",
                                unsafe_allow_html=True)

                    # ── Subject accounts ──
                    st.markdown("<div class='section-title'>Subject Accounts (Control Nodes)</div>", unsafe_allow_html=True)
                    df_subj = pd.DataFrame(sar["subject_accounts"])
                    if not df_subj.empty:
                        st.dataframe(df_subj, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No subject accounts identified.")

                    # ── Top transactions ──
                    st.markdown("<div class='section-title'>Top Transactions (Evidence)</div>", unsafe_allow_html=True)
                    top_txns = sar.get("evidence", {}).get("top_transactions", [])
                    if top_txns:
                        df_txn = pd.DataFrame(top_txns)
                        st.dataframe(df_txn, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No transactions in evidence.")

                    # ── Timeframe ──
                    tf = sar.get("timeframe", {})
                    st.markdown(f"""
                    <div class='timeline-caption'>
                      Active period: {tf.get('earliest_str', '—')} → {tf.get('latest_str', '—')}
                      &nbsp;·&nbsp; {tf.get('duration_days', 0):.1f} days
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Actions ──
                    if status == "pending":
                        st.markdown("<div style='margin-top:20px;padding-top:16px;border-top:1px solid #21262d'></div>", unsafe_allow_html=True)
                        b1, b2, _ = st.columns([1, 1, 4])
                        if b1.button(
                            "✅  Approve", key=f"approve_{sar['sar_id']}",
                            type="primary", use_container_width=True,
                        ):
                            api_post(f"/sar/{sar['sar_id']}/approve")
                            st.success(f"SAR {sar['sar_id']} approved and queued for filing.")
                            st.rerun()
                        if b2.button(
                            "✕  Dismiss", key=f"dismiss_{sar['sar_id']}",
                            use_container_width=True,
                        ):
                            api_post(f"/sar/{sar['sar_id']}/dismiss")
                            st.warning(f"SAR {sar['sar_id']} dismissed.")
                            st.rerun()
