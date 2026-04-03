"""
RLM HotpotQA Benchmark Dashboard
==================================
A Streamlit app for running and visualizing the RLM system on the
HotpotQA multi-hop reasoning benchmark.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import time
import threading
from pathlib import Path

# Make repo root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RLM · HotpotQA Benchmark",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Page background */
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(124, 77, 255, 0.3);
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(124, 77, 255, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(124,77,255,0.7); }
.metric-value { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #7c4dff, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
.metric-delta-pos { color: #00e676; font-size: 0.9rem; margin-top: 4px; }
.metric-delta-neg { color: #ff5252; font-size: 0.9rem; margin-top: 4px; }

/* Result table row colours */
.correct { color: #00e676; }
.incorrect { color: #ff5252; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.9);
    border-bottom: 2px solid rgba(124,77,255,0.5); padding-bottom: 8px; margin-bottom: 16px;
}

/* Glassmorphism container */
.glass-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #7c4dff, #00e5ff) !important; }

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #7c4dff, #00bcd4) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.6rem 2rem !important; font-size: 1rem !important;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom-color: #00e5ff !important;
}

/* Input fields */
.stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    color: white !important;
    border: 1px solid rgba(124,77,255,0.4) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "baseline_results": None,
        "enhanced_results": None,
        "running": False,
        "log_messages": [],
        "progress": 0,
        "current_question": "",
        "api_key_set": False,
        "examples": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RLM Dashboard")
    st.markdown("---")

    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Your Google Gemini API key. Stored in session only.",
    )
    if api_key:
        os.environ["GENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✅ API key set")

    st.markdown("### ⚙️ Benchmark Settings")
    model = st.selectbox(
        "Root LLM Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    num_examples = st.slider("Number of Examples", min_value=5, max_value=100, value=20, step=5)
    question_type = st.selectbox("Question Type", ["all", "bridge", "comparison"])
    max_iterations = st.slider("Max REPL Iterations", min_value=3, max_value=20, value=8)

    st.markdown("### 🔬 Enhancement Modules")
    enable_acc = st.toggle("Adaptive Compute (ACC)", value=True,
                           help="Dynamically sets max iterations based on query complexity.")
    enable_memory = st.toggle("Episodic Memory", value=True,
                              help="Retrieves past QA pairs to inform current reasoning.")
    enable_engine = st.toggle("Reasoning Engine", value=True,
                              help="Injects deep_reason() 3-step pipeline into REPL globals.")

    st.markdown("### 🚀 Run Modes")
    run_baseline = st.checkbox("Run Baseline (REPL only)", value=True)
    run_enhanced = st.checkbox("Run Enhanced (REPL + modules)", value=True)

    st.markdown("---")
    run_button = st.button("▶ Run Benchmark", use_container_width=True,
                           disabled=not st.session_state.api_key_set)

    if not st.session_state.api_key_set:
        st.warning("Enter your Gemini API key to run.")

    # Load saved results
    st.markdown("### 💾 Load Saved Results")
    results_dir = ROOT / "benchmark" / "results"
    saved_files = list(results_dir.glob("*.json")) if results_dir.exists() else []
    if saved_files:
        selected_file = st.selectbox("Load file", [f.name for f in saved_files])
        if st.button("Load"):
            with open(results_dir / selected_file) as f:
                data = json.load(f)
            if data.get("mode") == "baseline":
                st.session_state.baseline_results = data
            else:
                st.session_state.enhanced_results = data
            st.success(f"Loaded {selected_file}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
    <h1 style="font-size:2.8rem; font-weight:700; background: linear-gradient(90deg, #7c4dff, #00e5ff);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        🧠 RLM · HotpotQA Benchmark
    </h1>
    <p style="color:rgba(255,255,255,0.5); margin-top:8px; font-size:1.05rem;">
        Recursive Language Model · Multi-hop Reasoning Evaluation
    </p>
</div>
""", unsafe_allow_html=True)


# ── Run Logic ─────────────────────────────────────────────────────────────────
def run_benchmark_session(mode: str, enable_acc: bool, enable_memory: bool, enable_engine: bool):
    """Run a benchmark mode, streaming results into session state."""
    from benchmark.hotpotqa_runner import load_hotpotqa, run_benchmark, save_results
    from RLM.integrated_repl import IntegratedRLM
    from RLM.rlm_repl import RLM_REPL

    qt = None if question_type == "all" else question_type

    if st.session_state.examples is None:
        st.session_state.examples = load_hotpotqa(num_examples, qt)

    examples = st.session_state.examples
    total = len(examples)

    # Progress placeholders
    prog_bar = st.progress(0, text=f"Running {mode}…")
    status_text = st.empty()

    results_so_far = []

    def on_result(r: dict):
        results_so_far.append(r)
        pct = len(results_so_far) / total
        prog_bar.progress(pct, text=f"[{mode}] {len(results_so_far)}/{total} — Q: {r['question'][:55]}…")
        em_so_far = sum(x["em"] for x in results_so_far) / len(results_so_far)
        f1_so_far = sum(x["f1"] for x in results_so_far) / len(results_so_far)
        status_text.markdown(
            f"**Running EM:** `{em_so_far:.3f}` &nbsp;|&nbsp; **Running F1:** `{f1_so_far:.3f}`"
        )

    is_enhanced = (mode == "enhanced")

    def factory():
        if is_enhanced:
            return IntegratedRLM(
                model=model,
                recursive_model="gemini-2.5-flash",
                max_iterations=max_iterations,
                enable_acc=enable_acc,
                enable_memory=enable_memory,
                enable_engine=enable_engine,
            )
        else:
            return RLM_REPL(model=model, max_iterations=max_iterations)

    output = run_benchmark(examples, factory, mode=mode, on_result=on_result)

    # Save results
    results_dir = ROOT / "benchmark" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    save_path = str(results_dir / f"{mode}_{ts}.json")
    save_results(output, save_path)

    prog_bar.progress(1.0, text=f"✅ {mode.capitalize()} complete!")
    status_text.empty()

    return output


if run_button and st.session_state.api_key_set:
    st.session_state.examples = None  # force fresh load

    if run_baseline:
        with st.spinner("Running **baseline** (REPL only)…"):
            st.session_state.baseline_results = run_benchmark_session(
                "baseline", False, False, False
            )

    if run_enhanced:
        with st.spinner("Running **enhanced** (REPL + ACC + Memory + Engine)…"):
            st.session_state.enhanced_results = run_benchmark_session(
                "enhanced", enable_acc, enable_memory, enable_engine
            )

    st.balloons()


# ── Main Tabs ─────────────────────────────────────────────────────────────────
has_baseline = st.session_state.baseline_results is not None
has_enhanced = st.session_state.enhanced_results is not None
has_any = has_baseline or has_enhanced

tab_overview, tab_table, tab_analysis, tab_inspector = st.tabs([
    "📊 Overview", "📋 Results Table", "📈 Analysis", "🔍 Example Inspector"
])


# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab_overview:
    if not has_any:
        st.markdown("""
        <div class="glass-box" style="text-align:center; padding: 3rem;">
            <h3 style="color:rgba(255,255,255,0.6);">No results yet</h3>
            <p style="color:rgba(255,255,255,0.4);">Configure your API key and click <b>▶ Run Benchmark</b> in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary metric cards
        cols = st.columns(4)
        modes = []
        if has_baseline:
            modes.append(("Baseline", st.session_state.baseline_results["aggregate"], "#7c4dff"))
        if has_enhanced:
            modes.append(("Enhanced", st.session_state.enhanced_results["aggregate"], "#00e5ff"))

        for col, (name, agg, color) in zip(cols[:len(modes)*2:2], modes):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="background: linear-gradient(90deg, {color}, #fff);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {agg['em']:.3f}
                    </div>
                    <div class="metric-label">{name} · Exact Match</div>
                    <div class="metric-label">{agg['correct_em']}/{agg['total']} correct</div>
                </div>
                """, unsafe_allow_html=True)

        for col, (name, agg, color) in zip(cols[1:len(modes)*2:2], modes):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="background: linear-gradient(90deg, {color}, #fff);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {agg['f1']:.3f}
                    </div>
                    <div class="metric-label">{name} · F1 Score</div>
                    <div class="metric-label">avg {agg['avg_time']}s / example</div>
                </div>
                """, unsafe_allow_html=True)

        # Delta card if both modes exist
        if has_baseline and has_enhanced:
            st.markdown("---")
            b = st.session_state.baseline_results["aggregate"]
            e = st.session_state.enhanced_results["aggregate"]
            dem = e["em"] - b["em"]
            df1 = e["f1"] - b["f1"]
            c1, c2, c3 = st.columns(3)
            with c1:
                sign = "+" if dem >= 0 else ""
                color = "#00e676" if dem >= 0 else "#ff5252"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{sign}{dem:.3f}</div>
                    <div class="metric-label">EM Improvement</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                sign = "+" if df1 >= 0 else ""
                color = "#00e676" if df1 >= 0 else "#ff5252"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{sign}{df1:.3f}</div>
                    <div class="metric-label">F1 Improvement</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                dt = e["avg_time"] - b["avg_time"]
                sign = "+" if dt >= 0 else ""
                color = "#ff5252" if dt > 0 else "#00e676"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{sign}{dt:.1f}s</div>
                    <div class="metric-label">Avg Time Delta</div>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 2: Results Table ───────────────────────────────────────────────────────
with tab_table:
    if not has_any:
        st.info("Run the benchmark to see per-example results.")
    else:
        all_rows = []
        if has_baseline:
            all_rows.extend(st.session_state.baseline_results["results"])
        if has_enhanced:
            all_rows.extend(st.session_state.enhanced_results["results"])

        df = pd.DataFrame(all_rows)
        if not df.empty:
            # Filter controls
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                mode_filter = st.multiselect("Mode", df["mode"].unique().tolist(),
                                             default=df["mode"].unique().tolist())
            with col_f2:
                type_filter = st.multiselect("Question Type", df["type"].unique().tolist(),
                                             default=df["type"].unique().tolist())
            with col_f3:
                em_filter = st.selectbox("EM Filter", ["All", "Correct only", "Incorrect only"])

            filtered = df[df["mode"].isin(mode_filter) & df["type"].isin(type_filter)]
            if em_filter == "Correct only":
                filtered = filtered[filtered["em"] == 1]
            elif em_filter == "Incorrect only":
                filtered = filtered[filtered["em"] == 0]

            display_cols = ["mode", "type", "question", "gold", "predicted", "em", "f1", "time_s"]
            display = filtered[display_cols].copy()
            display["question"] = display["question"].str[:70] + "…"
            display["gold"] = display["gold"].str[:40]
            display["predicted"] = display["predicted"].str[:60] + "…"

            st.dataframe(
                display.style.apply(
                    lambda row: ["" if col != "em" else
                                 "color: #00e676; font-weight: bold" if row["em"] == 1
                                 else "color: #ff5252;" for col in display.columns],
                    axis=1
                ),
                use_container_width=True,
                height=500,
            )


# ── Tab 3: Analysis ────────────────────────────────────────────────────────────
with tab_analysis:
    if not has_any:
        st.info("Run the benchmark to see analytics.")
    else:
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            c1, c2 = st.columns(2)

            # EM / F1 comparison bar chart
            with c1:
                st.markdown('<div class="section-header">EM & F1 Comparison</div>', unsafe_allow_html=True)
                rows = []
                if has_baseline:
                    agg = st.session_state.baseline_results["aggregate"]
                    rows += [{"Mode": "Baseline", "Metric": "Exact Match", "Score": agg["em"]},
                             {"Mode": "Baseline", "Metric": "F1", "Score": agg["f1"]}]
                if has_enhanced:
                    agg = st.session_state.enhanced_results["aggregate"]
                    rows += [{"Mode": "Enhanced", "Metric": "Exact Match", "Score": agg["em"]},
                             {"Mode": "Enhanced", "Metric": "F1", "Score": agg["f1"]}]

                fig = px.bar(
                    pd.DataFrame(rows), x="Metric", y="Score", color="Mode", barmode="group",
                    color_discrete_map={"Baseline": "#7c4dff", "Enhanced": "#00e5ff"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_range=[0, 1], height=350, font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Question type breakdown
            with c2:
                st.markdown('<div class="section-header">F1 by Question Type</div>', unsafe_allow_html=True)
                type_rows = []
                for mode_key, label in [("baseline_results", "Baseline"), ("enhanced_results", "Enhanced")]:
                    data = st.session_state.get(mode_key)
                    if data:
                        df_tmp = pd.DataFrame(data["results"])
                        for qtype, grp in df_tmp.groupby("type"):
                            type_rows.append({"Mode": label, "Type": qtype, "F1": grp["f1"].mean()})

                if type_rows:
                    fig2 = px.bar(
                        pd.DataFrame(type_rows), x="Type", y="F1", color="Mode", barmode="group",
                        color_discrete_map={"Baseline": "#7c4dff", "Enhanced": "#00e5ff"},
                        template="plotly_dark",
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis_range=[0, 1], height=350, font=dict(family="Inter"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # F1 over time (streaming improvement)
            st.markdown('<div class="section-header">Cumulative F1 Over Examples</div>', unsafe_allow_html=True)
            cum_df_rows = []
            for mode_key, label in [("baseline_results", "Baseline"), ("enhanced_results", "Enhanced")]:
                data = st.session_state.get(mode_key)
                if data:
                    f1s = [r["f1"] for r in data["results"]]
                    for i, f1 in enumerate(f1s):
                        cum_f1 = sum(f1s[:i+1]) / (i+1)
                        cum_df_rows.append({"Example": i+1, "Mode": label, "Cumulative F1": cum_f1})

            if cum_df_rows:
                fig3 = px.line(
                    pd.DataFrame(cum_df_rows), x="Example", y="Cumulative F1", color="Mode",
                    color_discrete_map={"Baseline": "#7c4dff", "Enhanced": "#00e5ff"},
                    template="plotly_dark",
                )
                fig3.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_range=[0, 1], height=300, font=dict(family="Inter"),
                )
                st.plotly_chart(fig3, use_container_width=True)

        except ImportError:
            st.warning("Install `plotly` for charts: `pip install plotly`")


# ── Tab 4: Example Inspector ──────────────────────────────────────────────────
with tab_inspector:
    if not has_any:
        st.info("Run the benchmark to inspect examples.")
    else:
        all_rows = []
        if has_baseline:
            all_rows.extend(st.session_state.baseline_results["results"])
        if has_enhanced:
            all_rows.extend(st.session_state.enhanced_results["results"])

        df_all = pd.DataFrame(all_rows)
        if not df_all.empty:
            questions = df_all["question"].unique().tolist()
            selected_q = st.selectbox("Select a question", questions, format_func=lambda q: q[:90] + "…")

            for_q = df_all[df_all["question"] == selected_q]
            gold = for_q.iloc[0]["gold"]

            st.markdown(f"""
            <div class="glass-box">
                <div class="section-header">❓ Question</div>
                <p style="color:white; font-size:1.1rem; margin:0;">{selected_q}</p>
                <br/>
                <div class="section-header">✅ Gold Answer</div>
                <p style="color:#00e676; font-size:1.2rem; font-weight:600; margin:0;">{gold}</p>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(len(for_q))
            for col, (_, row) in zip(cols, for_q.iterrows()):
                with col:
                    em_icon = "✅" if row["em"] == 1 else "❌"
                    color = "#00e676" if row["em"] == 1 else "#ff5252"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px;">
                            {row['mode']}
                        </div>
                        <div style="color:{color}; font-size:2rem; margin:12px 0;">{em_icon}</div>
                        <div style="color:white; font-size:0.95rem; margin-bottom:12px;">
                            <b>Prediction:</b><br/>{row['predicted'][:300]}
                        </div>
                        <div style="color:rgba(255,255,255,0.6); font-size:0.85rem;">
                            EM: <b style="color:{color};">{row['em']}</b> &nbsp;|&nbsp;
                            F1: <b>{row['f1']:.3f}</b> &nbsp;|&nbsp;
                            Time: <b>{row['time_s']}s</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color: rgba(255,255,255,0.25); font-size:0.8rem;">
    RLM · Recursive Language Model · HotpotQA Benchmark Dashboard
</div>
""", unsafe_allow_html=True)
