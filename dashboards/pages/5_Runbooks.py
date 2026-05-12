"""
dashboard/pages/5_Runbooks.py

Runbooks — ingest, browse, delete, and test-query the ChromaDB runbooks collection.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Runbooks · MLOps", page_icon="⬡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0b0e;--surface:#111318;--surface2:#181b22;--border:#1f2330;
--text:#e8eaf0;--text2:#8b91a8;--text3:#555c72;--accent:#00d4ff;--green:#00e5a0;
--mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
.stButton>button{background:transparent!important;border:1px solid #2a2f3d!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.8rem!important;border-radius:4px!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.stButton>button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#000!important;font-weight:600!important;}
.stTextArea textarea{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.82rem!important;}
.stTextInput input{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-baseweb="select"]>div{background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;}
[data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
[data-baseweb="tab"]{background:transparent!important;color:#555c72!important;font-family:var(--mono)!important;font-size:0.75rem!important;letter-spacing:0.08em!important;text-transform:uppercase!important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url", "http://localhost:8000")

def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:24px 0 14px 0;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+subtitle+"</span>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def get_rag():
    from mlops_agents.rag.store import RAGStore
    return RAGStore()

def _rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

# ── page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Runbooks</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
        INSTITUTIONAL KNOWLEDGE BASE · DIAGNOSIS RAG CONTEXT
    </div>
</div>
<div style="margin:12px 0 8px;padding:10px 16px;background:#111318;border:1px solid #1f2330;
            border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#8b91a8;line-height:1.7;">
    Runbooks are ingested into ChromaDB and retrieved by the Diagnosis Agent during root cause analysis.
    Every incident query pulls the top-3 most semantically similar runbooks as context for the LLM.
    The more specific and well-tagged your runbooks, the better the diagnosis quality.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["LIBRARY", "INGEST", "TEST QUERY"])

# ── tab 1: library ────────────────────────────────────────────────────────────
with tab1:
    section("Runbook Library")

    try:
        rag = get_rag()
        results = rag._runbooks.get(include=["metadatas", "documents"], limit=100)
        metas = results.get("metadatas") or []
        docs  = results.get("documents") or []
        ids   = results.get("ids") or []

        if not metas:
            st.markdown("""
            <div style="color:#555c72;font-size:0.8rem;padding:24px 0;text-align:center;">
                No runbooks ingested yet.<br>
                <span style="font-size:0.72rem;">Use the Ingest tab to add runbooks, post-mortems, or playbooks.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#555c72;
                        margin-bottom:12px;">{len(metas)} documents in knowledge base</div>
            """, unsafe_allow_html=True)

            for i, (meta, doc, doc_id) in enumerate(zip(metas, docs, ids)):
                title    = meta.get("title", "Untitled")
                dtype    = meta.get("doc_type", "note")
                tags     = meta.get("tags", "")
                author   = meta.get("author", "")
                updated  = (meta.get("updated_at", "")[:10])
                preview  = doc[:180] + "…" if len(doc) > 180 else doc

                dtype_c = {"runbook":"#00d4ff","post_mortem":"#ff4560",
                           "playbook":"#00e5a0","note":"#555c72"}.get(dtype,"#555c72")

                with st.expander(f"{title}", expanded=False):
                    ec1, ec2 = st.columns([3,1])
                    with ec1:
                        st.markdown(f"""
                        <div style="font-family:'JetBrains Mono',monospace;">
                            <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;">
                                <span style="background:rgba({_rgb(dtype_c)},0.15);color:{dtype_c};
                                             font-size:0.65rem;padding:2px 8px;border-radius:3px;
                                             border:1px solid rgba({_rgb(dtype_c)},0.3);">{dtype.upper()}</span>
                                {"<span style='color:#555c72;font-size:0.72rem;'>by "+author+"</span>" if author else ""}
                                {"<span style='color:#555c72;font-size:0.72rem;'>"+updated+"</span>" if updated else ""}
                            </div>
                            {"<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;'>"+"".join(f"<span style='background:#1a1d26;color:#555c72;font-size:0.65rem;padding:2px 8px;border-radius:3px;'>{t.strip()}</span>" for t in tags.split(",") if t.strip())+"</div>" if tags else ""}
                            <div style="color:#8b91a8;font-size:0.78rem;line-height:1.7;
                                        background:#0d0f14;padding:12px;border-radius:4px;">
                                {preview}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with ec2:
                        st.markdown(f"""
                        <div style="font-size:0.65rem;color:#555c72;word-break:break-all;
                                    font-family:'JetBrains Mono',monospace;">
                            ID:<br>{doc_id[:32]}…
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Delete", key=f"del_{doc_id}", use_container_width=True):
                            try:
                                rag._runbooks.delete(ids=[doc_id])
                                st.success("Deleted")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")

    except Exception as e:
        st.error(f"Could not connect to ChromaDB: {e}")

# ── tab 2: ingest ─────────────────────────────────────────────────────────────
with tab2:
    section("Ingest Runbook")

    ic1, ic2 = st.columns(2)
    with ic1:
        rb_title  = st.text_input("Title *", placeholder="High Drift Remediation Runbook", key="rb_title")
        rb_type   = st.selectbox("Type", ["runbook","post_mortem","playbook","note"], key="rb_type")
        rb_tags   = st.text_input("Tags (comma-separated)", placeholder="retrain,drift,accuracy", key="rb_tags")
    with ic2:
        rb_author = st.text_input("Author", placeholder="ml-platform-team", key="rb_author")
        rb_url    = st.text_input("Source URL", placeholder="https://...", key="rb_url")
        rb_file   = st.file_uploader("Or upload .md / .txt file", type=["md","txt"], key="rb_file")

    rb_content = st.text_area(
        "Content *",
        height=280,
        placeholder="Paste runbook content here…\n\nDescribe: symptoms, root causes, diagnosis steps, remediation actions, and post-remediation checks.",
        key="rb_content",
    )

    if rb_file:
        rb_content = rb_file.read().decode("utf-8")
        st.info(f"Loaded {rb_file.name} ({len(rb_content)} chars)")

    col_btn, _ = st.columns([1,4])
    with col_btn:
        if st.button("⊕  Ingest", type="primary", use_container_width=True):
            if not rb_title or not rb_content:
                st.error("Title and content are required.")
            else:
                try:
                    rag = get_rag()
                    doc_id = rag.ingest_runbook({
                        "title":      rb_title,
                        "content":    rb_content,
                        "doc_type":   rb_type,
                        "tags":       rb_tags,
                        "author":     rb_author,
                        "source_url": rb_url,
                    })
                    st.success(f"Ingested — doc_id: {doc_id[:32]}…")
                    st.info("The Diagnosis Agent will use this runbook in future incident diagnoses.")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

    st.divider()
    section("Fraud-Specific Starter Runbooks")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#8b91a8;
                line-height:1.7;margin-bottom:12px;">
        Click to auto-fill the form with a starter runbook for common fraud classifier scenarios.
    </div>
    """, unsafe_allow_html=True)

    starters = {
        "High Drift — Retrain Runbook": {
            "type": "runbook", "tags": "drift,retrain,data_drift,concept_drift",
            "content": """# High Drift Remediation Runbook

## Symptoms
- drift_score > 0.35 (major) or > 0.60 (critical)
- accuracy may still be acceptable but trending down
- prediction confidence distribution shifting toward 0.5

## Root Cause Categories
- **Data drift (covariate shift)**: input feature distributions changed — new merchant categories, geographic shift, seasonal patterns
- **Concept drift**: fraud patterns changed — fraudsters changed technique, previously-fraud features now look legitimate

## How to Distinguish
- Data drift: drift_score high, accuracy degrades gradually, model uncertain
- Concept drift: drift_score moderate, recall collapses suddenly, model confidently wrong

## Remediation Steps
1. Identify which features drifted (check per-feature PSI)
2. For data drift: retrain with recent_window strategy (last 14-30 days)
3. For concept drift: retrain with drift_period_only strategy, high drift_period_weight (2.0)
4. Set optimize_for=recall for fraud models (missing fraud is worse than false positives)
5. Deploy with canary strategy at 10% traffic — monitor for 2 hours before promoting
6. Validate: recall >= 0.80, roc_auc >= 0.88

## Post-Remediation
- Monitor for 48 hours after full promotion
- If drift recurs within 7 days, investigate upstream data pipeline for schema changes
"""},
        "Low Accuracy — Model Staleness": {
            "type": "runbook", "tags": "accuracy,staleness,retrain",
            "content": """# Low Accuracy — Model Staleness Runbook

## Symptoms
- accuracy < 0.72 (major) or < 0.65 (critical)
- drift_score may be low (model seen similar inputs but learned wrong associations)
- usually develops gradually over weeks

## Root Cause
Model was trained on stale data. Fraud patterns have evolved but model has not been updated.
Distinct from drift — the input distribution may look similar but label relationships have changed.

## Remediation
1. Check training date in model metadata — if > 30 days old, staleness is likely
2. Retrain with full_history or weighted_recent strategy
3. Use GradientBoostingClassifier (LogisticRegression cannot capture complex patterns)
4. Apply SMOTE for class imbalance
5. Threshold tuning: optimize_for=recall, target_recall=0.82

## Validation Gates
- roc_auc >= 0.88 (hard gate — do not promote if below)
- recall >= 0.80 (hard gate)
- compare against degraded model — must beat current on all metrics

## Deployment
- Use blue_green strategy for clean cutover
- Keep old model warm for 1 hour in case rollback needed
"""},
        "Latency Spike — Scale Runbook": {
            "type": "runbook", "tags": "latency,scale,infrastructure",
            "content": """# Latency Spike Remediation Runbook

## Symptoms
- latency_ms p99 > 1000ms (major) or > 2000ms (critical)
- accuracy and drift_score remain healthy
- error_rate may be rising due to timeouts

## Root Cause
Infrastructure capacity issue — not a model quality problem.
Common causes: traffic spike, pod OOMKill, noisy neighbour on node, GC pause in serving layer.

## Remediation
1. Check if accuracy is still healthy — if yes, this is infrastructure, not model
2. Scale horizontally: kubectl scale deployment fraud-classifier --replicas=N*2
3. Check pod resource limits — may need to increase memory if OOMKill
4. Do NOT retrain — this will waste time and not fix the root cause

## Action: scale
- Doubles current replica count (up to K8S_MAX_REPLICAS=20)
- Takes effect within 60 seconds
- Latency should recover within 2-3 minutes

## Post-Remediation
- If latency recurs within 1 hour, investigate node-level metrics
- Consider adding HPA (Horizontal Pod Autoscaler) to automate scaling
"""},
    }

    sc1, sc2, sc3 = st.columns(3)
    for col, (name, data) in zip([sc1,sc2,sc3], starters.items()):
        with col:
            if st.button(name, use_container_width=True, key=f"starter_{name}"):
                st.session_state["rb_title"]   = name
                st.session_state["rb_type"]    = data["type"]
                st.session_state["rb_tags"]    = data["tags"]
                st.session_state["rb_content"] = data["content"]
                st.rerun()

# ── tab 3: test query ─────────────────────────────────────────────────────────
with tab3:
    section("Test RAG Query")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#8b91a8;margin-bottom:16px;line-height:1.7;">
        Simulate what the Diagnosis Agent sees when it queries runbooks.
        Enter a scenario description and see which runbooks are retrieved and ranked.
    </div>
    """, unsafe_allow_html=True)

    query = st.text_area(
        "Scenario query",
        placeholder="Model fraud-classifier-v1 in production. Severity: critical. Accuracy: 0.58. Drift score: 0.62. Root cause: concept drift suspected.",
        height=100,
        key="rb_query",
    )
    n_results = st.slider("Number of results", 1, 10, 3, key="rb_n_results")

    if st.button("⊗  Query", type="primary", key="rb_query_btn"):
        if not query:
            st.error("Enter a scenario query.")
        else:
            try:
                rag = get_rag()
                results = rag.query_runbooks(query_text=query, n_results=n_results)
                if not results:
                    st.warning("No runbooks retrieved. Ingest some runbooks first.")
                else:
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                color:#555c72;margin-bottom:12px;">
                        Retrieved {len(results)} runbooks (ranked by cosine similarity)
                    </div>
                    """, unsafe_allow_html=True)
                    for i, r in enumerate(results):
                        meta = r.get("metadata",{})
                        dist = r.get("distance",1.0)
                        relevance = max(0, 1 - dist)
                        doc = r.get("document","")[:400]

                        rel_c = "#00e5a0" if relevance > 0.7 else "#ffb800" if relevance > 0.4 else "#ff4560"
                        st.markdown(f"""
                        <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;
                                    padding:16px;margin-bottom:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                <span style="font-size:0.85rem;font-weight:600;color:#e8eaf0;">
                                    #{i+1} — {meta.get('title','Untitled')}
                                </span>
                                <span style="font-size:0.72rem;color:{rel_c};">
                                    relevance: {relevance:.3f}
                                </span>
                            </div>
                            <div style="font-size:0.7rem;color:#555c72;margin-bottom:8px;">
                                {meta.get('doc_type','').upper()}
                                {"· "+meta.get('tags','') if meta.get('tags') else ""}
                            </div>
                            <div style="font-size:0.78rem;color:#8b91a8;line-height:1.7;
                                        background:#0d0f14;padding:12px;border-radius:4px;">
                                {doc}{"…" if len(r.get('document',''))>400 else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Query failed: {e}")
