#!/usr/bin/env python
"""Streamlit app: clonotype expansion/contraction calling via downsampling
+ (beta-)binomial testing, with XY and volcano plots.

Run with:
    streamlit run streamlit_app.py
"""

pip install plotly

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from clonotype_stats import benjamini_hochberg_threshold, call_clonotypes, downsample_counts

st.set_page_config(page_title="Clonotype expansion/contraction caller", layout="wide")
st.title("Clonotype expansion/contraction caller")

CALL_COLORS = {"expansion": "crimson", "contraction": "steelblue", "ns": "lightgray"}
CALL_ORDER = ["ns", "expansion", "contraction"]


def guess_column(columns, exact_candidates, substr_candidates, fallback_idx=0):
    lower = {c.lower(): c for c in columns}
    for cand in exact_candidates:
        if cand in lower:
            return lower[cand]
    for cand in substr_candidates:
        for c in columns:
            if cand in c.lower():
                return c
    return columns[min(fallback_idx, len(columns) - 1)]


uploaded = st.file_uploader("Upload a CSV file", type="csv")

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df = pd.read_csv(uploaded)
st.write(f"Loaded **{len(df):,}** rows, **{len(df.columns)}** columns.")
with st.expander("Preview data"):
    st.dataframe(df.head(20))

cols = list(df.columns)
c1, c2, c3 = st.columns(3)
with c1:
    clono_col = st.selectbox(
        "Clonotype column", cols,
        index=cols.index(guess_column(cols, ["clono", "clonotype"], ["clon"])),
    )
with c2:
    t1_col = st.selectbox(
        "Abundance column, timepoint 1", cols,
        index=cols.index(guess_column(cols, ["counts_pre", "abundance_t1"], ["pre"])),
    )
with c3:
    t2_col = st.selectbox(
        "Abundance column, timepoint 2", cols,
        index=cols.index(guess_column(cols, ["counts_post", "abundance_t2"], ["post"])),
    )

raw0 = pd.to_numeric(df[t1_col], errors="coerce").fillna(0).clip(lower=0).values
raw1 = pd.to_numeric(df[t2_col], errors="coerce").fillna(0).clip(lower=0).values
total0, total1 = raw0.sum(), raw1.sum()
min_total = int(min(total0, total1))

st.caption(
    f"Timepoint 1 total = {total0:,.0f} | Timepoint 2 total = {total1:,.0f} | "
    f"min = {min_total:,}"
)

st.subheader("Downsampling & test parameters")
p1c, p2c, p3c = st.columns(3)
with p1c:
    ss = st.number_input(
        "Downsample sample size", min_value=1, value=max(min_total, 1), step=1,
        help="Both columns are multinomial-resampled to this total. "
             "Defaults to the smaller of the two column totals.",
    )
    if ss > min_total:
        st.warning("Chosen sample size exceeds the smaller library total; "
                   "this is upsampling, not downsampling.")
with p3c:
    fix_seed = st.checkbox("Fix random seed", value=True)
    seed = st.number_input("Seed", value=0, step=1, disabled=not fix_seed)

impute_col1, impute_col2 = st.columns([1, 1])
with impute_col1:
    impute_unobserved = st.checkbox(
        "Include clonotypes unobserved at one timepoint",
        value=False,
        help="By default, clonotypes with a downsampled count of 0 at either "
             "timepoint are dropped from testing (no proportion to test against). "
             "Checking this imputes a pseudocount for the unobserved side instead "
             "of dropping them, so 'appeared'/'disappeared' clonotypes get tested too.",
    )
with impute_col2:
    pseudocount_choice = st.radio(
        "Pseudocount for the unobserved side",
        ["Smallest observed clone size", "Half smallest observed clone size"],
        horizontal=True, disabled=not impute_unobserved,
        help="'Smallest observed clone size' assumes an unobserved clonotype sits "
             "right at this depth's detection limit (1 read after downsampling). "
             "'Half' is more conservative, treating it as somewhere below that limit.",
    )

RHO_KEY = "rho_select"
rho_options = [0, 8e-6, 1e-5, 2e-5, 3e-5, 5e-5, 0.0001, 0.0002, 0.0003]
# Read rho's current value before its own widget is defined (below, inside the
# Advanced expander) via its session_state key — Streamlit resolves the key
# to the same value the widget will use this run, so this isn't stale.
rho_preview = st.session_state.get(RHO_KEY, 1e-4)

n_persist_est = int((((raw0 > 0) & (raw1 > 0)) if not impute_unobserved
                      else ((raw0 > 0) | (raw1 > 0))).sum())
bonferroni_default = 0.05 / n_persist_est if n_persist_est > 0 else 0.05
bonferroni_neglog = float(-np.log10(bonferroni_default))

# Benjamini-Hochberg has no closed form like Bonferroni — it needs the actual
# p-value distribution — so it only becomes available (as another tick) once
# a downsampled run is cached, computed here using the live rho.
bh_p = None
bh_neglog = None
if "downsampled" in st.session_state:
    _d = st.session_state["downsampled"]
    _preview_pvals, _, _ = call_clonotypes(_d["c0_use"], _d["c1_use"], _d["ss"],
                                            rho=rho_preview, alpha=1.0)
    bh_p = benjamini_hochberg_threshold(_preview_pvals, q=0.05)
    if bh_p is not None and bh_p > 0:
        bh_neglog = min(float(-np.log10(bh_p)), 30.0)

special_ticks = {"bonferroni": bonferroni_neglog}
if bh_neglog is not None and abs(bh_neglog - bonferroni_neglog) > 0.15:
    special_ticks["bh"] = bh_neglog

max_tick = int(min(30, max(5, np.ceil(max(special_ticks.values())) + 3)))
int_ticks = [t for t in range(0, max_tick + 1)
             if all(abs(t - v) > 0.15 for v in special_ticks.values())]
tick_options = sorted(int_ticks + list(special_ticks.values()))

special_labels = {bonferroni_neglog: f"Bonferroni (p={bonferroni_default:.1e})"}
if "bh" in special_ticks:
    special_labels[bh_neglog] = f"BH FDR 5% (p={bh_p:.1e})"


def _format_tick(v):
    for tick_val, label in special_labels.items():
        if abs(v - tick_val) < 1e-9:
            return label
    iv = int(round(v))
    return "p=1" if iv == 0 else f"p=1e-{iv}"


neglog_alpha = st.select_slider(
    "Significance threshold (-log10 p)",
    options=tick_options,
    value=bonferroni_neglog,
    format_func=_format_tick,
    help="Drag between whole orders of magnitude of p (1, 0.1, 0.01, ...). "
         "Bonferroni (0.05 / # clonotypes tested) is always offered. A "
         "Benjamini-Hochberg (FDR 5%) tick appears too once you've run an "
         "analysis, since — unlike Bonferroni — it depends on the actual "
         "p-value distribution, not just the count of tests.",
)
alpha = 10 ** (-neglog_alpha)
st.caption(f"p-value threshold = {alpha:.3e}")

with st.expander("Advanced: overdispersion (beta-binomial rho)", expanded=True):

    def _format_rho(v):
        return "0 (exact binomial)" if v == 0 else f"{v:g}"

    col, _ = st.columns([1, 2])  # constrains the slider to ~1/3 width
    with col:
        rho = st.select_slider("rho (intraclass correlation)",
            options=rho_options, value=1e-4, format_func=_format_rho, key=RHO_KEY,
            help="rho = 0 uses an exact binomial null. rho > 0 uses a beta-binomial "
             "null, widening the expected variance to guard against overdispersed "
             "(clumpy) resampling noise. Drag while looking at the 5 SD envelope "
             "on the XY scatter below (once you've run an analysis) to tune it "
             "visually — moving this slider redraws the envelope immediately, "
             "without re-running the test.",
        )

run = st.button("Run analysis", type="primary")

if run:
    rng = np.random.RandomState(int(seed)) if fix_seed else np.random
    c0_ds = downsample_counts(raw0, ss, rng=rng)
    c1_ds = downsample_counts(raw1, ss, rng=rng)

    if impute_unobserved:
        factor = 1.0 if pseudocount_choice.startswith("Smallest") else 0.5
        min_c0 = c0_ds[c0_ds > 0].min() if (c0_ds > 0).any() else 1
        min_c1 = c1_ds[c1_ds > 0].min() if (c1_ds > 0).any() else 1
        imputed_t1 = c0_ds == 0
        imputed_t2 = c1_ds == 0
        c0_use = np.where(imputed_t1, min_c0 * factor, c0_ds).astype(float)
        c1_use = np.where(imputed_t2, min_c1 * factor, c1_ds).astype(float)
        mask = (c0_ds > 0) | (c1_ds > 0)  # keep everything but both-zero rows
    else:
        c0_use, c1_use = c0_ds.astype(float), c1_ds.astype(float)
        imputed_t1 = np.zeros(len(c0_ds), dtype=bool)
        imputed_t2 = np.zeros(len(c1_ds), dtype=bool)
        mask = (c0_ds > 0) & (c1_ds > 0)

    n_excluded = int((~mask).sum())

    # Cache only the downsampled counts here — downsampling is the expensive,
    # stochastic step (depends on ss/seed/imputation choice). The p-values,
    # fold changes, and calls depend only on rho/alpha, which are cheap to
    # recompute, so they're derived fresh below on every rerun (including
    # just moving a slider) rather than frozen at click time.
    st.session_state["downsampled"] = {
        "clonotype": df[clono_col].values[mask],
        "count_t1_raw": raw0[mask],
        "count_t2_raw": raw1[mask],
        "c0_use": c0_use[mask],
        "c1_use": c1_use[mask],
        "imputed_t1": imputed_t1[mask],
        "imputed_t2": imputed_t2[mask],
        "n_excluded": n_excluded,
        "ss": ss,
        "impute_unobserved": impute_unobserved,
    }
    # The significance slider's BH tick (built earlier in this same script
    # pass) reads st.session_state["downsampled"], which we just set — force
    # one more rerun so the tick reflects it immediately, instead of only
    # appearing after the next unrelated widget interaction.
    st.rerun()

if "downsampled" in st.session_state:
    d = st.session_state["downsampled"]
    ss_used = d["ss"]
    alpha_used = alpha
    impute_used = d["impute_unobserved"]
    n_excluded = d["n_excluded"]

    pvals, fc, call = call_clonotypes(d["c0_use"], d["c1_use"], ss_used, rho=rho, alpha=alpha)
    result = pd.DataFrame({
        "clonotype": d["clonotype"],
        "count_t1_raw": d["count_t1_raw"],
        "count_t2_raw": d["count_t2_raw"],
        "count_t1_downsampled": d["c0_use"],
        "count_t2_downsampled": d["c1_use"],
        "prop_t1": d["c0_use"] / ss_used,
        "prop_t2": d["c1_use"] / ss_used,
        "imputed_t1": d["imputed_t1"],
        "imputed_t2": d["imputed_t2"],
        "log2fc": fc,
        "pval": pvals,
        "neglog10p": -np.log10(pvals),
        "call": call,
    })
    result["neglog10p_capped"] = np.minimum(result["neglog10p"], 10)

    n_exp = (result["call"] == "expansion").sum()
    n_con = (result["call"] == "contraction").sum()
    n_imputed = int((result["imputed_t1"] | result["imputed_t2"]).sum())

    st.subheader("Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tested clonotypes", f"{len(result):,}")
    m2.metric("Expansions", f"{n_exp:,}")
    m3.metric("Contractions", f"{n_con:,}")
    excluded_label = "Excluded (0 in both downsamples)" if impute_used else "Excluded (0 in either downsample)"
    m4.metric(excluded_label, f"{n_excluded:,}")
    m5.metric("Imputed (pseudocount)", f"{n_imputed:,}" if impute_used else "—")

    n_tested = len(result)
    bonferroni_p = 0.05 / n_tested if n_tested > 0 else float("nan")

    st.caption("Hover a point in the XY scatter to highlight the matching clonotype in the volcano plot. "
               "The significance threshold and rho sliders above update the calls and plots immediately — "
               "only downsampling itself needs a re-click of Run analysis.")

    fig = go.Figure()
    lo = max(result[["prop_t1", "prop_t2"]].to_numpy().min(), 1 / ss_used) * 0.8
    hi = result[["prop_t1", "prop_t2"]].to_numpy().max() * 1.2
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(color="gray", dash="dash"),
                              showlegend=False, hoverinfo="skip"))

    # 5 SD envelope around the diagonal, from the beta-binomial variance of the
    # downsampled proportion: Var(p_hat) = p0*(1-p0)*(1 + (ss-1)*rho) / ss
    # (reduces to the plain-binomial variance when rho = 0). Built directly as
    # p0 +/- 5*SD, this is symmetric in linear space but not on a log-log plot,
    # and can drive the lower bound negative for rare clones (SD ~ sqrt(p0), so
    # 5*SD often exceeds p0 itself there) — undefined for a proportion, so it
    # gets clipped away, leaving a lopsided-looking band. Instead, apply the
    # delta method to get the SD on the log10 scale, SD(log10 p_hat) ~=
    # SD(p_hat) / (p0 * ln 10), and build the envelope there before
    # exponentiating back — symmetric on the plot, and always positive.
    p0_grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    var_factor = 1 + (ss_used - 1) * rho
    sd = np.sqrt(p0_grid * (1 - p0_grid) * var_factor / ss_used)
    sd_log10 = sd / (p0_grid * np.log(10))
    log10_p0 = np.log10(p0_grid)
    env_upper = np.minimum(10 ** (log10_p0 + 5 * sd_log10), 1.0)  # a proportion can't exceed 1
    env_lower = 10 ** (log10_p0 - 5 * sd_log10)
    fig.add_trace(go.Scatter(x=p0_grid, y=env_lower, mode="lines",
                              line=dict(color="gold", width=1, dash="dot"),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p0_grid, y=env_upper, mode="lines",
                              line=dict(color="gold", width=1, dash="dot"),
                              name=f"5 SD envelope (rho={rho:g})",
                              fill="tonexty", fillcolor="rgba(255,215,0,0.08)",
                              hoverinfo="skip"))

    for grp in CALL_ORDER:
        sub = result[result["call"] == grp]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["prop_t1"], y=sub["prop_t2"], mode="markers",
            name=f"{grp} ({len(sub)})",
            marker=dict(color=CALL_COLORS[grp], size=6,
                        opacity=0.85 if grp != "ns" else 0.35),
            text=sub["clonotype"],
            customdata=list(zip(sub["log2fc"], sub["neglog10p_capped"], [CALL_COLORS[grp]] * len(sub))),
            hovertemplate="%{text}<br>t1=%{x:.2e}<br>t2=%{y:.2e}<extra></extra>",
        ))
    # Lock the view to the data's own span — at high rho the envelope can
    # legitimately extend far beyond it (even to p=1), and letting the axes
    # autorange to fit it would squash the actual points into a sliver.
    axis_range = [float(np.log10(lo)), float(np.log10(hi))]
    fig.update_xaxes(type="log", range=axis_range, title="proportional abundance, t1",
                      gridcolor="#444", zerolinecolor="#444")
    fig.update_yaxes(type="log", range=axis_range, title="proportional abundance, t2",
                      gridcolor="#444", zerolinecolor="#444")
    fig.update_layout(title="XY scatter (downsampled proportions)", height=500,
                       legend=dict(title="call"), paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6e6e6"))

    fig2 = go.Figure()
    fig2.add_hline(y=min(-np.log10(alpha_used), 10), line_dash="dash", line_color="pink",
                    annotation_text=f"p-call={alpha_used:.2e}", annotation_position="top left")

    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    for grp in CALL_ORDER:
        sub = result[result["call"] == grp]
        if len(sub) == 0:
            continue
        fig2.add_trace(go.Scatter(
            x=sub["log2fc"], y=sub["neglog10p_capped"], mode="markers",
            name=f"{grp} ({len(sub)})",
            marker=dict(color=CALL_COLORS[grp], size=6,
                        opacity=0.85 if grp != "ns" else 0.35),
            text=sub["clonotype"],
            hovertemplate="%{text}<br>log2fc=%{x:.2f}<br>p=%{customdata:.2e}<extra></extra>",
            customdata=sub["pval"],
        ))
    highlight_idx = len(fig2.data)
    fig2.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(size=16, color="white", line=dict(width=2, color="black")),
        hoverinfo="skip", showlegend=False, name="highlight",
    ))
    fig2.update_xaxes(title="log2 fold change (t2 / t1)", gridcolor="#444", zerolinecolor="#444")
    fig2.update_yaxes(title="-log10(p)", range=[-0.3, 10.5],
                       tickvals=[0, 1.3, 2, 4, 6, 8, 10],
                       ticktext=["0", "1.3 (p=0.05)", "2", "4", "6", "8", "≥10"],
                       gridcolor="#444", zerolinecolor="#444")
    fig2.update_layout(title="Volcano plot", height=500, legend=dict(title="call"),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e6e6e6"))

    html = f"""
    <div style="display:flex; gap:1rem; background:#0e1117;">
      <div id="xy_plot" style="flex:1; min-width:0;"></div>
      <div id="volcano_plot" style="flex:1; min-width:0;"></div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script>
      var xyFig = {fig.to_json()};
      var volFig = {fig2.to_json()};
      Plotly.newPlot('xy_plot', xyFig.data, xyFig.layout, {{responsive: true}});
      Plotly.newPlot('volcano_plot', volFig.data, volFig.layout, {{responsive: true}});

      var xyDiv = document.getElementById('xy_plot');
      var volDiv = document.getElementById('volcano_plot');
      var highlightIdx = {highlight_idx};

      xyDiv.on('plotly_hover', function(evt) {{
        var pt = evt.points && evt.points[0];
        if (!pt || !pt.customdata) return;
        var cd = pt.customdata;
        Plotly.restyle(volDiv, {{x: [[cd[0]]], y: [[cd[1]]], 'marker.color': [[cd[2]]]}}, [highlightIdx]);
      }});
      xyDiv.on('plotly_unhover', function(evt) {{
        Plotly.restyle(volDiv, {{x: [[]], y: [[]]}}, [highlightIdx]);
      }});
    </script>
    """
    components.html(html, height=560, scrolling=False)

    st.subheader("Significant clonotypes")
    sig = result[result["call"] != "ns"].sort_values("pval").reset_index(drop=True)

    if len(sig) == 0:
        st.info("No clonotypes clear the current significance threshold.")
    else:
        def _row_color(row):
            color = "rgba(220,20,60,0.35)" if row["call"] == "expansion" else "rgba(70,130,180,0.35)"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(sig.style.apply(_row_color, axis=1), use_container_width=True)

    st.download_button(
        "Download significant clonotypes CSV",
        sig.to_csv(index=False).encode(),
        file_name="significant_clonotypes.csv",
        mime="text/csv",
    )
