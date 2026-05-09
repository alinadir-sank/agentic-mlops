"""
dashboard/pages/2_Metrics.py

Live metrics page — accuracy, drift, latency, error rate, prediction confidence histogram.
Auto-refreshes every 10s when polling is enabled.
"""

import time
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Metrics · MLOps", page_icon="⬡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0b0e;--surface:#111318;--surface2:#181b22;--border:#1f2330;--border2:#2a2f3d;
--text:#e8eaf0;--text2:#8b91a8;--text3:#555c72;--accent:#00d4ff;--green:#00e5a0;
--amber:#ffb800;--red:#ff4560;--mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="metric-container"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:16px!important;}
[data-testid="metric-container"] label{color:var(--text2)!important;font-size:0.7rem!important;letter-spacing:0.12em!important;text-transform:uppercase!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--accent)!important;font-size:1.6rem!important;font-weight:600!important;}
.stButton>button{background:transparent!important;border:1px solid #2a2f3d!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.8rem!important;border-radius:4px!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.stButton>button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#000!important;font-weight:600!important;}
[data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
[data-baseweb="tab"]{background:transparent!important;color:#555c72!important;font-family:var(--mono)!important;font-size:0.75rem!important;letter-spacing:0.08em!important;text-transform:uppercase!important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API          = st.session_state.get("api_url",   "http://localhost:8000")
MODEL_SERVER = st.session_state.get("model_url", "http://localhost:8080")

# ── init metrics history in session state ────────────────────────────────────
if "metrics_history" not in st.session_state:
    st.session_state["metrics_history"] = []

def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:24px 0 14px 0;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+subtitle+"</span>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def threshold_line(val, thresholds, metric):
    """Return color based on threshold breach."""
    if metric == "accuracy":
        if val < 0.65: return "#ff4560"
        if val < 0.72: return "#ffb800"
        if val < 0.80: return "#00d4ff"
        return "#00e5a0"
    elif metric == "drift_score":
        if val > 0.60: return "#ff4560"
        if val > 0.35: return "#ffb800"
        if val > 0.20: return "#00d4ff"
        return "#00e5a0"
    elif metric == "error_rate":
        if val > 0.10: return "#ff4560"
        if val > 0.05: return "#ffb800"
        return "#00e5a0"
    return "#00d4ff"

# ── page header ───────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Metrics</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
            LIVE MODEL PERFORMANCE · TREND HISTORY
        </div>
    </div>
    """, unsafe_allow_html=True)
with hc2:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    polling = st.toggle("Auto-poll", value=False, key="metrics_polling")

# ── fetch current metrics ─────────────────────────────────────────────────────
metrics = None
error   = None
try:
    r = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": "get_current_metrics", "params": {}},
        timeout=8,
    )
    r.raise_for_status()
    metrics = r.json()

    # append to history
    import datetime
    entry = {**metrics, "ts": datetime.datetime.now().strftime("%H:%M:%S")}
    st.session_state["metrics_history"].append(entry)
    if len(st.session_state["metrics_history"]) > 120:
        st.session_state["metrics_history"] = st.session_state["metrics_history"][-120:]

except Exception as e:
    error = str(e)

if error:
    st.error(f"Model server unreachable: {error}")
    st.stop()

# ── current metric cards ──────────────────────────────────────────────────────
section("Current Snapshot")

m = metrics
cols = st.columns(8)
card_data = [
    ("Accuracy",    f"{m.get('accuracy',0):.4f}",    "accuracy"),
    ("Drift Score", f"{m.get('drift_score',0):.4f}", "drift_score"),
    ("Latency p95", f"{m.get('latency_ms',0):.1f}ms","latency"),
    ("Error Rate",  f"{m.get('error_rate',0):.4f}",  "error_rate"),
    ("Precision",   f"{m.get('precision',0):.4f}",   None),
    ("Recall",      f"{m.get('recall',0):.4f}",      None),
    ("F1",          f"{m.get('f1',0):.4f}",          None),
    ("ROC-AUC",     f"{m.get('roc_auc',0):.4f}",     None),
]
for col, (label, value, metric_key) in zip(cols, card_data):
    val_f = m.get(label.lower().replace(" ", "_").replace("-","_"), 0) or 0
    color = threshold_line(val_f, None, metric_key) if metric_key else "#00d4ff"
    with col:
        st.markdown(f"""
        <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:14px 12px;text-align:center;">
            <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;text-transform:uppercase;">{label}</div>
            <div style="font-size:1.1rem;font-weight:600;color:{color};margin-top:6px;font-family:'JetBrains Mono',monospace;">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# drift active banner
if m.get("drift_active"):
    dt = m.get("drift_type","").replace("_"," ").upper()
    st.markdown(f"""
    <div style="margin:12px 0 4px;padding:10px 16px;background:rgba(255,184,0,0.08);
                border:1px solid rgba(255,184,0,0.3);border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;">
        ⚠ DRIFT INJECTION ACTIVE — type: <strong>{dt}</strong>
        · sample_size: {m.get('sample_size',0)} predictions
    </div>
    """, unsafe_allow_html=True)

# ── trend charts ──────────────────────────────────────────────────────────────
section("Trend", f"{len(st.session_state['metrics_history'])} snapshots")

history = st.session_state["metrics_history"]

if len(history) < 2:
    st.markdown("""
    <div style="color:#555c72;font-size:0.8rem;padding:16px 0;">
        Collecting data — need at least 2 snapshots. Enable Auto-poll or refresh manually.
    </div>
    """, unsafe_allow_html=True)
else:
    df = pd.DataFrame(history)

    tab1, tab2, tab3, tab4 = st.tabs([
        "ACCURACY / DRIFT", "LATENCY", "ERROR RATE", "FRAUD RATE"
    ])

    with tab1:
        chart_df = df[["ts","accuracy","drift_score"]].set_index("ts")
        st.line_chart(chart_df, color=["#00d4ff","#ff4560"], height=220)
        st.markdown("""
        <div style="font-size:0.7rem;color:#555c72;margin-top:-8px;">
            <span style="color:#00d4ff;">■</span> accuracy &nbsp;&nbsp;
            <span style="color:#ff4560;">■</span> drift_score
            &nbsp;·&nbsp; thresholds: accuracy &lt;0.72=major, drift &gt;0.35=major
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        chart_df = df[["ts","latency_ms"]].set_index("ts")
        st.line_chart(chart_df, color=["#00e5a0"], height=220)
        st.markdown("""
        <div style="font-size:0.7rem;color:#555c72;margin-top:-8px;">
            <span style="color:#00e5a0;">■</span> latency_ms (p95)
            &nbsp;·&nbsp; threshold: &gt;1000ms=major, &gt;2000ms=critical
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        chart_df = df[["ts","error_rate"]].set_index("ts")
        st.line_chart(chart_df, color=["#ffb800"], height=220)
        st.markdown("""
        <div style="font-size:0.7rem;color:#555c72;margin-top:-8px;">
            <span style="color:#ffb800;">■</span> error_rate (real — from serving exceptions)
            &nbsp;·&nbsp; threshold: &gt;0.05=major, &gt;0.10=critical
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        if "fraud_rate" in df.columns:
            chart_df = df[["ts","fraud_rate"]].set_index("ts")
            st.line_chart(chart_df, color=["#9b59ff"], height=220)
        else:
            st.info("fraud_rate available after 10+ predictions")

# ── prediction confidence histogram ──────────────────────────────────────────
section("Prediction Confidence Distribution", "last 200 predictions")

try:
    r2 = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": "get_prediction_history", "params": {"n": 200}},
        timeout=6,
    )
    r2.raise_for_status()
    preds = r2.json().get("predictions", [])

    if len(preds) < 5:
        st.markdown('<div style="color:#555c72;font-size:0.8rem;padding:8px 0;">Not enough predictions yet — send some requests to /predict.</div>',
                    unsafe_allow_html=True)
    else:
        df_p = pd.DataFrame(preds)
        probs = df_p["fraud_prob"].values

        # bin into 20 buckets
        hist, edges = np.histogram(probs, bins=20, range=(0,1))
        bin_labels   = [f"{e:.2f}" for e in edges[:-1]]
        hist_df      = pd.DataFrame({"bin": bin_labels, "count": hist}).set_index("bin")

        hc1, hc2 = st.columns([3, 1])
        with hc1:
            st.bar_chart(hist_df, color="#00d4ff", height=200)
            st.markdown("""
            <div style="font-size:0.7rem;color:#555c72;margin-top:-8px;">
                X-axis: fraud probability bins (0=certain legitimate, 1=certain fraud)
                · Bimodal = confident model · Clustered near 0.5 = uncertain (drift signal)
            </div>
            """, unsafe_allow_html=True)

        with hc2:
            fraud_count = int(df_p["prediction"].sum())
            legit_count = len(df_p) - fraud_count
            avg_conf    = float(df_p["fraud_prob"].apply(lambda p: max(p,1-p)).mean())
            avg_lat     = float(df_p["latency_ms"].mean())

            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:16px;">
                <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;margin-bottom:12px;">PREDICTION STATS</div>
                <div style="display:flex;flex-direction:column;gap:8px;font-size:0.78rem;">
                    <div><span style="color:#555c72;">total</span> <span style="color:#e8eaf0;float:right;">{len(df_p)}</span></div>
                    <div><span style="color:#555c72;">fraud flagged</span> <span style="color:#ff4560;float:right;">{fraud_count}</span></div>
                    <div><span style="color:#555c72;">legitimate</span> <span style="color:#00e5a0;float:right;">{legit_count}</span></div>
                    <div style="border-top:1px solid #1f2330;padding-top:8px;margin-top:4px;">
                        <span style="color:#555c72;">avg confidence</span>
                        <span style="color:#00d4ff;float:right;">{avg_conf:.3f}</span>
                    </div>
                    <div><span style="color:#555c72;">avg latency</span>
                        <span style="color:#00d4ff;float:right;">{avg_lat:.2f}ms</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.warning(f"Could not fetch prediction history: {e}")

# ── auto-refresh ──────────────────────────────────────────────────────────────
if polling:
    st.markdown("""
    <div style="position:fixed;bottom:16px;right:20px;font-family:'JetBrains Mono',monospace;
                font-size:0.65rem;color:#555c72;background:#0a0b0e;padding:4px 10px;
                border:1px solid #1f2330;border-radius:3px;">
        ● polling every 10s
    </div>
    """, unsafe_allow_html=True)
    time.sleep(10)
    st.rerun()
else:
    if st.button("↺  Refresh", use_container_width=False):
        st.rerun()
