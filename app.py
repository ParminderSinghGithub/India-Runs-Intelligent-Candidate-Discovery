"""app.py — RedrobAI Intelligent Candidate Discovery Demo

Streamlit application that acts as an internal recruiter tool.
Uses the existing pipeline (JobParser → Retriever → HybridRanker →
SubmissionGenerator) and never rebuilds embeddings or FAISS.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── page config must be the very first st call ───────────────────────────────
st.set_page_config(
    page_title="RedrobAI — Intelligent Candidate Discovery",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.express as px
import plotly.graph_objects as go

from src.config import FAISS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.parser.job_description_parser import JobDescriptionParser
from src.retrieval.retriever import Retriever
from src.scoring.hybrid_ranker import HybridRanker
from src.submission import CandidateResolver, SubmissionGenerator
from src.utils import setup_logging

setup_logging()

# ── constants ────────────────────────────────────────────────────────────────
REQUIRED_ARTIFACTS = [
    FAISS_DIR / "faiss.index",
    FAISS_DIR / "candidate_lookup.pkl",
    FAISS_DIR / "embedding_metadata.pkl",
]
CANDIDATES_JSONL = (
    PROJECT_ROOT
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)

COMPONENT_COLORS = {
    "Career":      "#6366f1",
    "Skill":       "#22d3ee",
    "Semantic":    "#f59e0b",
    "Behavior":    "#34d399",
    "Consistency": "#f87171",
}

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* dark gradient background */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }

    /* metric cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 16px;
        backdrop-filter: blur(10px);
    }

    /* dataframe */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }

    /* download buttons */
    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.9);
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 8px;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        border-color: #6366f1;
        color: #a5b4fc;
    }

    /* expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
        border: 1px solid #334155;
    }

    /* headers */
    h1, h2, h3 { color: #f1f5f9 !important; }
    p, li, label { color: #94a3b8 !important; }

    /* score badge */
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def artifacts_present() -> bool:
    return all(p.exists() for p in REQUIRED_ARTIFACTS)


def faiss_index_size() -> int:
    """Return cached FAISS index size without loading the full model."""
    try:
        import pickle
        with open(FAISS_DIR / "candidate_lookup.pkl", "rb") as fh:
            lookup = pickle.load(fh)
        return len(lookup)
    except Exception:
        return 0


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_retriever() -> Retriever:
    retriever = Retriever(force_rebuild=False)
    retriever.load_index()
    return retriever


def score_color(score: float) -> str:
    if score >= 0.80:
        return "#22d3ee"
    if score >= 0.65:
        return "#34d399"
    if score >= 0.50:
        return "#f59e0b"
    return "#f87171"


def render_score_bar(score: float, label: str) -> str:
    pct = int(score * 100)
    color = score_color(score)
    return (
        f"<div style='margin-bottom:6px'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
        f"<span style='color:#94a3b8;font-size:0.8rem'>{label}</span>"
        f"<span style='color:{color};font-size:0.8rem;font-weight:600'>{pct}%</span>"
        f"</div>"
        f"<div style='background:#1e293b;border-radius:4px;height:6px'>"
        f"<div style='background:{color};width:{pct}%;height:6px;border-radius:4px'></div>"
        f"</div></div>"
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:20px 0 10px'>
            <div style='font-size:2.5rem'>🤖</div>
            <h2 style='margin:8px 0 4px;color:#f1f5f9;font-size:1.3rem'>RedrobAI</h2>
            <p style='color:#64748b;font-size:0.8rem;margin:0'>Intelligent Candidate Discovery</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Challenge**")
    st.caption("India Runs Data & AI Challenge · Track 1")

    st.markdown("**Embedding Model**")
    st.caption("BAAI/bge-base-en-v1.5 · 768 dim")

    st.markdown("**Index**")
    if artifacts_present():
        n = faiss_index_size()
        st.success(f"FAISS loaded — {n:,} candidates indexed")
    else:
        st.error("FAISS artifacts missing")

    st.markdown("**Ranking Weights**")
    weights = {
        "Career": 38,
        "Skill": 34,
        "Semantic": 18,
        "Behavior": 5,
        "Consistency": 5,
    }
    for name, w in weights.items():
        color = COMPONENT_COLORS.get(name, "#94a3b8")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"background:rgba(30,41,59,0.5);border-radius:6px;padding:4px 10px;margin-bottom:4px'>"
            f"<span style='color:#94a3b8;font-size:0.8rem'>{name}</span>"
            f"<span style='color:{color};font-weight:600;font-size:0.85rem'>{w}%</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("© 2026 RedrobAI India Runs Version by Parminder Singh")


# ── main page ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style='padding:20px 0 10px'>
        <h1 style='font-size:2rem;margin:0;color:#f1f5f9'>
            Intelligent Candidate Discovery
        </h1>
        <p style='color:#64748b;margin-top:6px'>
            Retrieval-augmented ranking · Deterministic · No LLM calls
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── artifact guard ────────────────────────────────────────────────────────────
if not artifacts_present():
    st.error(
        "**FAISS artifacts not found.**\n\n"
        "Run the offline build once to create them:\n"
        "```\npython run_pipeline.py --rebuild-index\n```\n\n"
        "This is a one-time step (~15–40 min on CPU). "
        "After that, rankings complete in ~20 seconds."
    )
    st.stop()

# ── job description input ─────────────────────────────────────────────────────
st.markdown("### Job Description")
col_src, _ = st.columns([3, 1])
with col_src:
    jd_source = st.radio(
        "Source",
        ["Use existing job_description.json", "Upload a custom JSON"],
        horizontal=True,
        label_visibility="collapsed",
    )

jd_bytes: bytes | None = None

if jd_source == "Use existing job_description.json":
    default_jd = PROJECT_ROOT / "job_description.json"
    if default_jd.exists():
        jd_bytes = default_jd.read_bytes()
        with st.expander("Preview job_description.json"):
            st.json(json.loads(jd_bytes))
    else:
        st.warning("job_description.json not found in project root.")
else:
    uploaded = st.file_uploader("Upload job description JSON", type=["json"])
    if uploaded:
        jd_bytes = uploaded.read()
        with st.expander("Preview uploaded job"):
            st.json(json.loads(jd_bytes))

# ── top-k slider ──────────────────────────────────────────────────────────────
top_k = st.slider("Candidates to rank", min_value=10, max_value=100, value=100, step=10)

# ── generate button ───────────────────────────────────────────────────────────
run_col, _ = st.columns([2, 5])
with run_col:
    run_clicked = st.button("Generate Top Candidates", type="primary", use_container_width=True)

# ── ranking execution ─────────────────────────────────────────────────────────
if run_clicked:
    if jd_bytes is None:
        st.error("Please provide a job description before running.")
        st.stop()

    if not CANDIDATES_JSONL.exists():
        st.error(
            f"Candidates JSONL not found:\n`{CANDIDATES_JSONL}`\n\n"
            "Ensure the challenge dataset is extracted to the project root."
        )
        st.stop()

    progress = st.progress(0, text="Initialising pipeline...")

    # Parse job description ────────────────────────────────────────────────
    import tempfile, json as _json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
        tmp.write(jd_bytes)
        tmp_path = Path(tmp.name)

    try:
        progress.progress(10, text="Parsing job description...")
        jd_parser = JobDescriptionParser()
        parsed_job = jd_parser.parse_from_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Load FAISS retriever ─────────────────────────────────────────────────
    progress.progress(25, text="Loading FAISS index (cached)...")
    retriever = load_retriever()

    # Candidate resolver ───────────────────────────────────────────────────
    progress.progress(35, text="Initialising candidate resolver...")
    candidate_resolver = CandidateResolver(CANDIDATES_JSONL)

    # Retrieval ────────────────────────────────────────────────────────────
    progress.progress(45, text="Running FAISS retrieval...")
    t_retrieval = time.perf_counter()
    retrieval_results = retriever.search(parsed_job.search_query.combined_query, k=top_k)
    retrieval_latency = time.perf_counter() - t_retrieval

    # Re-ranking ───────────────────────────────────────────────────────────
    progress.progress(55, text=f"Re-ranking {len(retrieval_results)} candidates...")
    ranker = HybridRanker(retriever=retriever, candidate_resolver=candidate_resolver.resolve)
    ranker.last_retrieved_candidates = retrieval_results

    t_rank = time.perf_counter()
    ranked_results = ranker.rank_retrieval_results(parsed_job, retrieval_results)
    rank_latency = time.perf_counter() - t_rank

    # Submission generation ────────────────────────────────────────────────
    progress.progress(80, text="Generating submission artifacts...")
    generator = SubmissionGenerator()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    submission_result = generator.generate_submission(
        parsed_job=parsed_job,
        ranked_results=ranked_results,
        candidate_resolver=candidate_resolver.resolve,
        top_n=top_k,
        submission_csv_path=OUTPUTS_DIR / "submission.csv",
        ranking_json_path=OUTPUTS_DIR / "ranking.json",
        pipeline_report_path=OUTPUTS_DIR / "pipeline_report.json",
        candidate_exists=candidate_resolver.has,
        retrieval_results=retrieval_results,
        timings={
            "retrieval_latency_seconds": retrieval_latency,
            "reranking_latency_seconds": rank_latency,
        },
        artifact_paths={
            "faiss_index": str(FAISS_DIR / "faiss.index"),
            "candidate_lookup": str(FAISS_DIR / "candidate_lookup.pkl"),
            "embedding_metadata": str(FAISS_DIR / "embedding_metadata.pkl"),
        },
        configuration={"top_k": top_k},
        ai_tools_used=["GitHub Copilot"],
    )

    # XLSX export ──────────────────────────────────────────────────────────
    progress.progress(90, text="Generating XLSX export...")
    from src.submission.xlsx_exporter import XlsxExporter
    xlsx_path = OUTPUTS_DIR / "submission.xlsx"
    XlsxExporter().export(submission_result["rows"], xlsx_path)

    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()

    total_time = retrieval_latency + rank_latency
    st.success(f"Ranking complete — {len(ranked_results)} candidates in {total_time:.1f}s")

    # ── store in session state ─────────────────────────────────────────────
    st.session_state["ranked_results"] = ranked_results
    st.session_state["submission_result"] = submission_result
    st.session_state["candidate_resolver"] = candidate_resolver
    st.session_state["parsed_job"] = parsed_job
    st.session_state["retrieval_latency"] = retrieval_latency
    st.session_state["rank_latency"] = rank_latency
    st.session_state["xlsx_path"] = xlsx_path


# ── results display ───────────────────────────────────────────────────────────
if "ranked_results" in st.session_state:
    ranked_results = st.session_state["ranked_results"]
    submission_result = st.session_state["submission_result"]
    candidate_resolver = st.session_state["candidate_resolver"]
    parsed_job = st.session_state["parsed_job"]
    xlsx_path: Path = st.session_state.get("xlsx_path", OUTPUTS_DIR / "submission.xlsx")

    # ── summary metrics ────────────────────────────────────────────────────
    st.markdown("### Pipeline Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    scores = [r.weighted_final_score for r in ranked_results]
    m1.metric("Candidates Ranked", len(ranked_results))
    m2.metric("Top Score", f"{max(scores):.3f}")
    m3.metric("Avg Score", f"{sum(scores)/len(scores):.3f}")
    m4.metric("Retrieval", f"{st.session_state['retrieval_latency']:.2f}s")
    m5.metric("Re-ranking", f"{st.session_state['rank_latency']:.1f}s")

    # ── download buttons ───────────────────────────────────────────────────
    st.markdown("### Download Artifacts")
    d1, d2, d3, d4 = st.columns(4)

    csv_path = OUTPUTS_DIR / "submission.csv"
    if csv_path.exists():
        d1.download_button(
            "submission.csv",
            data=csv_path.read_bytes(),
            file_name="submission.csv",
            mime="text/csv",
        )

    if xlsx_path.exists():
        d2.download_button(
            "submission.xlsx",
            data=xlsx_path.read_bytes(),
            file_name="submission.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    ranking_path = OUTPUTS_DIR / "ranking.json"
    if ranking_path.exists():
        d3.download_button(
            "ranking.json",
            data=ranking_path.read_bytes(),
            file_name="ranking.json",
            mime="application/json",
        )

    report_path = OUTPUTS_DIR / "pipeline_report.json"
    if report_path.exists():
        d4.download_button(
            "pipeline_report.json",
            data=report_path.read_bytes(),
            file_name="pipeline_report.json",
            mime="application/json",
        )

    # ── charts ────────────────────────────────────────────────────────────
    st.markdown("### Analytics")
    ch1, ch2 = st.columns(2)
    ch3, ch4 = st.columns(2)

    # Score distribution
    with ch1:
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Score Distribution",
            labels={"x": "Weighted Final Score", "y": "Candidates"},
            color_discrete_sequence=["#6366f1"],
            template="plotly_dark",
        )
        fig.update_layout(
            plot_bgcolor="rgba(15,23,42,0.8)",
            paper_bgcolor="rgba(15,23,42,0.0)",
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(color="#94a3b8"),
            title_font_color="#f1f5f9",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Component contribution (radar / bar)
    with ch2:
        comp_names = ["Career", "Skill", "Semantic", "Behavior", "Consistency"]
        comp_vals = {
            "Career":      [r.career_score for r in ranked_results],
            "Skill":       [r.skill_score for r in ranked_results],
            "Semantic":    [r.semantic_score for r in ranked_results],
            "Behavior":    [r.behavior_score for r in ranked_results],
            "Consistency": [r.consistency_score for r in ranked_results],
        }
        avg_vals = [sum(comp_vals[c]) / len(comp_vals[c]) for c in comp_names]
        colors = [COMPONENT_COLORS[c] for c in comp_names]

        fig2 = go.Figure(
            go.Bar(
                x=comp_names,
                y=avg_vals,
                marker_color=colors,
                text=[f"{v:.2f}" for v in avg_vals],
                textposition="outside",
            )
        )
        fig2.update_layout(
            title="Avg Component Scores",
            plot_bgcolor="rgba(15,23,42,0.8)",
            paper_bgcolor="rgba(15,23,42,0.0)",
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(color="#94a3b8"),
            title_font_color="#f1f5f9",
            yaxis=dict(range=[0, 1.1], gridcolor="#1e293b"),
            xaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Top industries
    with ch3:
        industries: dict[str, int] = {}
        for result in ranked_results:
            cand = candidate_resolver.resolve(result.candidate_id)
            if cand:
                ind = cand.profile.current_industry or "Unknown"
                industries[ind] = industries.get(ind, 0) + 1
        sorted_ind = sorted(industries.items(), key=lambda x: x[1], reverse=True)[:10]
        fig3 = px.bar(
            x=[v for _, v in sorted_ind],
            y=[k for k, _ in sorted_ind],
            orientation="h",
            title="Top Industries (ranked candidates)",
            labels={"x": "Count", "y": "Industry"},
            color_discrete_sequence=["#22d3ee"],
            template="plotly_dark",
        )
        fig3.update_layout(
            plot_bgcolor="rgba(15,23,42,0.8)",
            paper_bgcolor="rgba(15,23,42,0.0)",
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(color="#94a3b8"),
            title_font_color="#f1f5f9",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Experience distribution
    with ch4:
        exp_vals = []
        for result in ranked_results:
            cand = candidate_resolver.resolve(result.candidate_id)
            if cand:
                exp_vals.append(float(cand.profile.years_of_experience or 0))
        fig4 = px.histogram(
            x=exp_vals,
            nbins=15,
            title="Experience Distribution (years)",
            labels={"x": "Years of Experience", "y": "Candidates"},
            color_discrete_sequence=["#34d399"],
            template="plotly_dark",
        )
        fig4.update_layout(
            plot_bgcolor="rgba(15,23,42,0.8)",
            paper_bgcolor="rgba(15,23,42,0.0)",
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(color="#94a3b8"),
            title_font_color="#f1f5f9",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── ranked table ───────────────────────────────────────────────────────
    st.markdown("### Top Candidates")

    # Build display dataframe
    import pandas as pd

    rows_for_df = []
    for result in ranked_results:
        cand = candidate_resolver.resolve(result.candidate_id)
        row_data = next(
            (r for r in submission_result["rows"] if r.candidate_id == result.candidate_id),
            None,
        )
        rows_for_df.append(
            {
                "Rank":         result.metadata.get("retrieval_rank") or ranked_results.index(result) + 1,
                "Candidate ID": result.candidate_id,
                "Score":        round(result.weighted_final_score, 4),
                "Current Title": cand.profile.current_title if cand else "—",
                "Experience":   f"{cand.profile.years_of_experience:.1f} yrs" if cand else "—",
                "Industry":     cand.profile.current_industry if cand else "—",
                "Explanation":  row_data.reasoning[:120] + "…" if row_data and len(row_data.reasoning) > 120 else (row_data.reasoning if row_data else ""),
            }
        )

    df = pd.DataFrame(rows_for_df)
    df.index = range(1, len(df) + 1)

    st.dataframe(
        df,
        use_container_width=True,
        height=420,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=1, format="%.4f"
            ),
        },
    )

    # ── candidate detail expanders ─────────────────────────────────────────
    st.markdown("### Candidate Details")
    st.caption("Expand any candidate to see full scoring breakdown.")

    for i, result in enumerate(ranked_results[:20], start=1):
        cand = candidate_resolver.resolve(result.candidate_id)
        label = f"#{i} — {result.candidate_id}"
        if cand:
            label += f" · {cand.profile.current_title} · {cand.profile.years_of_experience:.0f} yrs"

        with st.expander(label):
            dc1, dc2 = st.columns([1, 1])

            with dc1:
                st.markdown("**Score Breakdown**")
                html_bars = "".join([
                    render_score_bar(result.career_score,      "Career"),
                    render_score_bar(result.skill_score,       "Skill"),
                    render_score_bar(result.semantic_score,    "Semantic"),
                    render_score_bar(result.behavior_score,    "Behavior"),
                    render_score_bar(result.consistency_score, "Consistency"),
                ])
                st.markdown(html_bars, unsafe_allow_html=True)

                st.markdown(f"**Final Score:** `{result.weighted_final_score:.4f}`")
                st.markdown(f"**Confidence:** `{result.confidence:.3f}`")

            with dc2:
                row_data = next(
                    (r for r in submission_result["rows"] if r.candidate_id == result.candidate_id),
                    None,
                )
                if row_data:
                    st.markdown("**Full Reasoning**")
                    st.info(row_data.reasoning)

                if result.matched_items:
                    st.markdown("**Matched Evidence**")
                    st.markdown(
                        " ".join(
                            f"<span style='background:#064e3b;color:#6ee7b7;padding:2px 8px;"
                            f"border-radius:12px;font-size:0.75rem;margin:2px;display:inline-block'>"
                            f"{item}</span>"
                            for item in result.matched_items[:12]
                        ),
                        unsafe_allow_html=True,
                    )

                if result.missing_items:
                    st.markdown("**Missing / Unconfirmed**")
                    st.markdown(
                        " ".join(
                            f"<span style='background:#450a0a;color:#fca5a5;padding:2px 8px;"
                            f"border-radius:12px;font-size:0.75rem;margin:2px;display:inline-block'>"
                            f"{item}</span>"
                            for item in result.missing_items[:8]
                        ),
                        unsafe_allow_html=True,
                    )

    if len(ranked_results) > 20:
        st.caption(f"Showing detailed view for top 20. Full table above shows all {len(ranked_results)} candidates.")
