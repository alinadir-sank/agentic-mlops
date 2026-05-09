"""
dashboard/pages/3_Incidents.py

Incident history — ChromaDB incident feed, severity filter, detail view.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Incidents · MLOps", page_icon="⬡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0b0e;--surface:#111318;--surface2:#181b22;--border:#1f2330;
--text:#e8eaf0;--text2:#8b91a8;--text3:#555c72;--accent:#00d4ff;--green:#00e5a0;--amber:#ffb800;--red:#ff4560;
--mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
.stButton>button{background:transparent!important;border:1px solid #2a2f3d!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.8rem!important;border-radius:4px!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
[data-baseweb="select"]>div{background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;}
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:6px!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url", "http://localhost:8000")

SEV_COLOR = {
    "none":     "#00e5a0",
    "minor":    "#00d4ff",
    "major":    "#ffb800",
    "critical": "#ff4560",
}

def sev_badge(sev):
    c = SEV_COLOR.get(sev, "#555c72")
    return f'<span style="background:rgba({_rgb(c)},0.15);color:{c};font-size:0.68rem;font-weight:600;letter-spacing:0.1em;padding:2px 8px;border-radius:3px;border:1px solid rgba({_rgb(c)},0.3);">{(sev or "—").upper()}</span>'

def _rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:24px 0 14px 0;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+subtitle+"</span>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

# ── page header ───────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns([4, 1, 1])
with fc1:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Incidents</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
            CHROMADB INCIDENT FEED · FULL AUDIT TRAIL
        </div>
    </div>
    """, unsafe_allow_html=True)
with fc2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    sev_filter = st.selectbox(
        "Severity",
        ["all", "critical", "major", "minor", "none"],
        key="inc_sev_filter",
        label_visibility="collapsed",
    )
with fc3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    limit = st.selectbox("Limit", [20, 50, 100], key="inc_limit", label_visibility="collapsed")

# ── fetch incidents ───────────────────────────────────────────────────────────
try:
    params = {"limit": limit}
    if sev_filter != "all":
        params["severity"] = sev_filter

    incidents = requests.get(f"{API}/incidents", params=params, timeout=8).json()
except Exception as e:
    st.error(f"Could not fetch incidents: {e}")
    st.stop()

# ── summary stats ─────────────────────────────────────────────────────────────
section("Summary", f"{len(incidents)} incidents")

if incidents:
    from collections import Counter
    sev_counts = Counter(i.get("severity","none") for i in incidents)
    act_counts = Counter(i.get("recommended_action","—") for i in incidents)
    success_n  = sum(1 for i in incidents if i.get("remediation_status") == "success")
    total_n    = len(incidents)

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    for col, sev in zip([sc1,sc2,sc3,sc4], ["critical","major","minor","none"]):
        c = SEV_COLOR.get(sev,"#555c72")
        cnt = sev_counts.get(sev, 0)
        with col:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid {c}33;border-radius:8px;
                        padding:14px 12px;text-align:center;">
                <div style="font-size:0.62rem;color:{c};letter-spacing:0.12em;">{sev.upper()}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{c};margin-top:4px;">{cnt}</div>
            </div>
            """, unsafe_allow_html=True)
    with sc5:
        rate = success_n / total_n if total_n else 0
        c = "#00e5a0" if rate > 0.8 else "#ffb800" if rate > 0.5 else "#ff4560"
        st.markdown(f"""
        <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;
                    padding:14px 12px;text-align:center;">
            <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">SUCCESS RATE</div>
            <div style="font-size:1.6rem;font-weight:700;color:{c};margin-top:4px;">{rate:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

# ── incident table ────────────────────────────────────────────────────────────
section("Incident Log")

if not incidents:
    st.markdown("""
    <div style="color:#555c72;font-size:0.8rem;padding:24px 0;text-align:center;">
        No incidents recorded yet.<br>
        <span style="font-size:0.72rem;">Run a monitoring cycle to populate this feed.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    # table header
    st.markdown("""
    <div style="display:grid;grid-template-columns:180px 80px 90px 110px 90px 90px 1fr;
                gap:8px;padding:8px 12px;
                font-size:0.63rem;color:#555c72;letter-spacing:0.12em;
                border-bottom:1px solid #1f2330;margin-bottom:2px;">
        <div>INCIDENT ID</div><div>SEV</div><div>MODEL</div>
        <div>ACTION</div><div>STATUS</div><div>CREATED</div><div>ACCURACY / DRIFT</div>
    </div>
    """, unsafe_allow_html=True)

    for inc in incidents:
        inc_id   = inc.get("incident_id", "—")
        sev      = inc.get("severity", "—")
        model    = inc.get("model_id", "—")
        action   = inc.get("recommended_action", "—")
        rem_stat = inc.get("remediation_status", "—")
        created  = (inc.get("created_at", "")[:16]).replace("T"," ")
        acc      = inc.get("accuracy", 0)
        drift    = inc.get("drift_score", 0)
        sev_c    = SEV_COLOR.get(sev, "#555c72")
        rem_c    = "#00e5a0" if rem_stat == "success" else "#ff4560" if rem_stat == "failed" else "#555c72"

        with st.expander(f"", expanded=False):
            # collapsed row inside expander label (workaround)
            pass

        # custom row
        row_html = f"""
        <div style="display:grid;grid-template-columns:180px 80px 90px 110px 90px 90px 1fr;
                    gap:8px;padding:10px 12px;border-bottom:1px solid #13161e;
                    font-size:0.76rem;cursor:pointer;"
             onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
            <div style="color:#8b91a8;font-size:0.7rem;">{inc_id[:20]}…</div>
            <div>{sev_badge(sev)}</div>
            <div style="color:#8b91a8;">{model[:12]}</div>
            <div style="color:#e8eaf0;">{action}</div>
            <div style="color:{rem_c};">{'● ' if rem_stat != '—' else ''}{rem_stat}</div>
            <div style="color:#555c72;">{created}</div>
            <div style="color:#555c72;font-size:0.72rem;">acc={acc:.3f} drift={drift:.3f}</div>
        </div>
        """
        st.markdown(row_html, unsafe_allow_html=True)

    # ── detail view via selectbox ────────────────────────────────────────────
    section("Incident Detail")
    inc_ids = [i.get("incident_id","") for i in incidents if i.get("incident_id")]
    if inc_ids:
        selected_id = st.selectbox(
            "Select incident",
            options=inc_ids,
            format_func=lambda x: x[:32] + "…" if len(x) > 32 else x,
            key="inc_detail_select",
            label_visibility="collapsed",
        )
        selected = next((i for i in incidents if i.get("incident_id") == selected_id), None)
        if selected:
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"""
                <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:20px;">
                    <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;margin-bottom:12px;">INCIDENT DETAILS</div>
                    <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
                        <tr><td style="color:#555c72;padding:5px 0;">ID</td>
                            <td style="color:#8b91a8;text-align:right;font-size:0.7rem;">{selected.get('incident_id','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Severity</td>
                            <td style="text-align:right;">{sev_badge(selected.get('severity'))}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Model</td>
                            <td style="color:#e8eaf0;text-align:right;">{selected.get('model_id','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Environment</td>
                            <td style="color:#e8eaf0;text-align:right;">{selected.get('environment','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Action</td>
                            <td style="color:#e8eaf0;text-align:right;">{selected.get('recommended_action','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Remediation</td>
                            <td style="color:#00e5a0;text-align:right;">{selected.get('remediation_status','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Human Approved</td>
                            <td style="color:#e8eaf0;text-align:right;">{'Yes' if selected.get('human_approved') else 'No'}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Created</td>
                            <td style="color:#555c72;text-align:right;font-size:0.7rem;">{(selected.get('created_at','')[:19]).replace('T',' ')}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with dc2:
                st.markdown(f"""
                <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:20px;">
                    <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;margin-bottom:12px;">METRICS AT DETECTION</div>
                    <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
                        <tr><td style="color:#555c72;padding:5px 0;">Accuracy</td>
                            <td style="color:#00d4ff;text-align:right;font-weight:600;">{selected.get('accuracy',0):.4f}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Drift Score</td>
                            <td style="color:#ff4560;text-align:right;font-weight:600;">{selected.get('drift_score',0):.4f}</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Latency p99</td>
                            <td style="color:#00d4ff;text-align:right;">{selected.get('latency_p99_ms',0):.1f}ms</td></tr>
                        <tr><td style="color:#555c72;padding:5px 0;">Error Rate</td>
                            <td style="color:#ffb800;text-align:right;">{selected.get('error_rate',0):.4f}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
