"""Streamlit app: LendingClub application PD scorecard.

Reads saved artifacts only. Nothing is retrained at start-up.
Run: streamlit run app.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = Path(os.environ.get("LCPD_ART", ROOT / "artifacts"))   # LCPD_ART points the app at another artifact folder
FIGDIR = ART / "figures"
sys.path.insert(0, str(ART))
import lcpd_lib as L  # noqa: E402  (copy written by the notebook, identical to ./lcpd_lib.py)

st.set_page_config(page_title="LendingClub PD scorecard", layout="wide", page_icon=":bank:")

NAVY = "#1f3b73"
AMBER = "#d9822b"
RED = "#c0392b"
GREY = "#8a8f98"
LIGHT = "#c8ccd2"

CSS = """
<style>
  .block-container { padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1280px; }
  h1 { font-size: 2rem; font-weight: 650; letter-spacing: -0.015em; margin-bottom: 0.15rem; }
  h2 { font-size: 1.28rem; font-weight: 600; letter-spacing: -0.01em; }
  h3 { font-size: 1.02rem; font-weight: 600; }
  [data-testid="stMetricLabel"] p { font-size: 0.78rem; letter-spacing: 0.01em; color: #5a6472; }
  [data-testid="stMetricValue"] { font-size: 1.42rem; font-weight: 600; }
  [data-testid="stCaptionContainer"] p { color: #5a6472; }
  [data-testid="stSidebar"] { min-width: 268px; }
  [data-testid="stSidebar"] [role="radiogroup"] { gap: 0.05rem; }
  [data-testid="stSidebar"] [role="radiogroup"] label { padding: 0.1rem 0; }
  div[data-testid="stDataFrame"] { border: 1px solid #e3e6ea; border-radius: 6px; }
  hr { margin: 1.2rem 0; border-color: #e3e6ea; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
@st.cache_data
def jload(name):
    with open(ART / name, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def cload(name):
    try:
        return pd.read_csv(ART / name)
    except pd.errors.EmptyDataError:      # for example when no VIF iterations were needed
        return pd.DataFrame()


@st.cache_data
def captions():
    return jload("figure_captions.json")


@st.cache_resource
def scorecard():
    return L.Scorecard.from_dict(jload("scorecard.json"))


@st.cache_data
def oot():
    """The scored 2016 sample, 31.9 MB, read once per session."""
    return pd.read_csv(ART / "oot_scored.csv.gz")


def have(*names):
    return all((ART / n).exists() for n in names)


# ------------------------------------------------------------------ presentation helpers
def sec(title, lead=None):
    st.subheader(title)
    if lead:
        st.caption(lead)


def blk(title):
    st.markdown(f"**{title}**")


def show_fig(name, width=None):
    path = FIGDIR / name
    if path.exists():
        if width is None:
            st.image(str(path), caption=captions().get(name, ""))
        else:
            st.image(str(path), caption=captions().get(name, ""), width=width)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def table(df, height=None):
    if height is None:
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.dataframe(df, hide_index=True, width="stretch", height=height)


M = jload("metrics.json")
CH = M["champion"]
NAMES = {"plain_lr": "Plain logistic", "l2_lr": "L2 logistic", "l1_lr": "L1 logistic", "elasticnet_lr": "ElasticNet logistic",
         "woe_scorecard": "WoE scorecard", "xgb_tuned": "XGBoost (tuned)", "xgb_monotone": "XGBoost (monotone)",
         "ref_xgb_plus_LC_grade": "Reference: XGBoost with LC grade and rate", "ref_LC_subgrade_only": "Reference: LC sub-grade alone"}
nm = lambda k: NAMES.get(k, k)  # noqa: E731

MODEL_ORDER = ["woe_scorecard", "plain_lr", "l2_lr", "l1_lr", "elasticnet_lr", "xgb_tuned", "xgb_monotone"]
SECTIONS = ["Overview", "Metrics", "Data quality", "Features", "Multicollinearity", "Logistic ladder", "Challengers",
            "Model comparison", "Scorecard", "Applicant scorer", "Monitoring", "Calibration", "Economics and fairness", "Glossary"]

st.title("LendingClub application PD scorecard")
st.caption("12-month probability of default for 36-month personal loans  ·  application data only  ·  trained on 2013–2015, tested on 2016  ·  every figure is read from saved artifacts")

with st.sidebar:
    page = st.radio("Section", SECTIONS, key="nav")
    st.divider()
    MODEL = st.selectbox("Scoring model", MODEL_ORDER, index=0, format_func=nm, key="model",
                         help="Drives the headline metrics, the cut-off simulator and the applicant scorer. "
                              "Reference models that borrow LendingClub's own grade are not offered.")
    st.caption("Default is the weight-of-evidence scorecard because it is the explainable one. The frozen champion is still reported "
               "on the Overview page, and selecting another model here does not change it.")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    o = M["oot"][MODEL]
    st.write(
        "This is a probability-of-default model for unsecured 36-month personal loans. It uses only information that exists "
        "at application, is trained on 2013–2015 originations and tested on 2016 originations it never saw, and follows the "
        "sequence a bank credit-risk team would use: a selection rule frozen before the test, calibration, a master scale, "
        "a cut-off policy and a monitoring plan. Data: public LendingClub accepted loans.")
    c = st.columns(6)
    c[0].metric("Scoring model", nm(MODEL), help="Change it in the sidebar.")
    c[1].metric("OOT Gini", f"{o['gini']:.3f}")
    c[2].metric("OOT AUC", f"{o['auc']:.3f}", help=f"95% bootstrap interval {o['auc_ci'][0]:.3f} to {o['auc_ci'][1]:.3f}")
    c[3].metric("OOT KS", f"{o['ks']:.3f}")
    c[4].metric("PSI, train to OOT", f"{M['psi']['psi_train_vs_oot']:.3f}")
    c[5].metric("OOT bad rate", pct(M["data"]["bad_rate_oot"]))
    st.info(
        f"**How to read the headline.** On 2016 loans the selected model ranks borrowers with a Gini of {o['gini']:.2f}, where 0 is a coin toss "
        f"and 1 is perfect. That is a realistic level for public consumer-credit data once LendingClub's own grade is excluded. "
        f"Calibration is fitted on validation only. On 2016 the model predicts {pct(M['psi']['mean_pd_oot'])} on average against "
        f"{pct(M['data']['bad_rate_oot'])} observed, which is why calibration checks and monitoring are part of the design rather than an afterthought.")
    st.divider()
    l, r = st.columns(2)
    with l:
        blk("Default definition")
        st.markdown("- **Bad**: charged off or default, with the last payment within 9 months on book. This is 90+ days past due inside the 12-month window.\n"
                    "- **Good**: every other loan, including paid, current, late at the data cut, and charged off only after month 12.\n"
                    "- **Dropped**: none. Every 2013–2016 loan has a full 12-month window.\n"
                    "- **Horizon**: 12 months from issue. Observation point: the application date.")
        blk("Windows")
        d = M["data"]
        st.markdown(f"- Train: {d['train_window']} with {d['train_rows_used']:,} rows and a {pct(d['bad_rate_train'])} bad rate\n"
                    f"- Validation: {d['valid_window']} with {d['valid_rows']:,} rows and a {pct(d['bad_rate_valid'])} bad rate\n"
                    f"- **Out-of-time test**: {d['oot_window']} with {d['oot_rows']:,} rows and a {pct(d['bad_rate_oot'])} bad rate")
        st.caption(d["sampling_note"])
    with r:
        blk("Why this model is the frozen champion")
        st.caption("The selection rule was frozen before the 2016 test was scored.")
        rule = jload("champion_rule.json")
        st.write(rule["rule"])
        st.json({"validation_auc": rule["validation_auc"], "champion": rule["champion"], "gap_xgb_monotone_vs_best_interpretable": rule["xgb_monotone_gap_vs_best_interpretable"],
                 "gap_ci95": rule["gap_ci95"]}, expanded=False)
    st.divider()
    blk("Limitations")
    st.markdown(
        "- Only **accepted** loans exist in the data, so there is no reject inference. The model describes people LendingClub approved.\n"
        "- The default flag is a 12-month window proxied from the last-payment date, because the public file has no monthly delinquency history. "
        "Losses after month 12 are outside the label, and the Data quality tab shows what changes if the window is widened.\n"
        "- `grade`, `sub_grade` and `int_rate` are LendingClub's own risk outputs and are excluded by design.\n"
        "- Profit numbers use **assumed** economics, not observed ones. No claim of regulatory compliance under Basel or IFRS 9 is made.")
    st.divider()
    sec("Population waterfall")
    table(cload("population_waterfall.csv"))

# ------------------------------------------------------------------ Metrics
if page == "Metrics":
    sec("Model scoreboard", "Every specification is scored once on the same 2016 out-of-time sample. The model chosen in the sidebar is the reference point.")
    o = M["oot"][MODEL]
    v = M["valid"][MODEL]
    k = st.columns(6)
    k[0].metric("Model", nm(MODEL))
    k[1].metric("OOT AUC", f"{o['auc']:.3f}")
    k[2].metric("OOT Gini", f"{o['gini']:.3f}")
    k[3].metric("OOT KS", f"{o['ks']:.3f}")
    k[4].metric("OOT PR-AUC", f"{o['pr_auc']:.3f}")
    k[5].metric("OOT Brier", f"{o['brier']:.4f}")
    st.caption(f"The same model scores a validation AUC of {v['auc']:.3f} and an out-of-time AUC of {o['auc']:.3f}. "
               f"The gap is the realistic estimate of how it will behave on a later vintage, not the validation number.")

    cmp_ = cload("model_comparison.csv").set_index("model")
    r = cmp_.loc[MODEL]
    st.divider()
    sec("Selected model in detail", "Validation and out-of-time side by side, with a bootstrap interval for the out-of-time AUC.")
    sel = pd.DataFrame([{
        "model": nm(MODEL), "role": r["role"],
        "validation AUC": r["valid_auc"], "OOT AUC": r["oot_auc"],
        "OOT AUC 95% interval": f"{r['auc_ci_low']:.3f} to {r['auc_ci_high']:.3f}",
        "validation Gini": r["valid_gini"], "OOT Gini": r["oot_gini"], "Gini drop": r["gini_drop_valid_to_oot"],
        "OOT PR-AUC": r["oot_pr_auc"], "OOT Brier": r["oot_brier"], "OOT ECE": r["oot_ece"]}])
    table(sel.round(4))
    st.caption("A negative Gini drop means the model held up or improved on the later vintage. A drop of a few points is normal for credit models "
               "and is the reason the selection rule was frozen on validation only.")

    st.divider()
    sec("All models on the out-of-time sample", "Ranked by AUC. Reference rows are not eligible models.")
    t = cmp_.reset_index()[["model", "role", "oot_auc", "oot_gini", "oot_ks", "oot_pr_auc", "oot_brier", "oot_ece",
                            "valid_gini", "gini_drop_valid_to_oot", "auc_diff_vs_champion", "diff_ci_low", "diff_ci_high"]].copy()
    t["model"] = t.model.map(nm)
    table(t.round(4))
    st.caption("The last three columns give the paired-bootstrap difference in AUC against the frozen champion, with a 95% interval from 1000 resamples. "
               "An interval that contains zero means the difference is not established. Reference rows use LendingClub's own grade and interest rate "
               "and are shown only to size what the excluded variables would add.")

    st.divider()
    sec("Ranking power by model", "The selected model is highlighted in amber. Both charts use the same 2016 sample.")
    g = cmp_.sort_values("oot_gini")
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=g.oot_gini, y=[nm(m) for m in g.index], orientation="h",
                               marker_color=[AMBER if m == MODEL else NAVY for m in g.index],
                               text=[f"{x:.3f}" for x in g.oot_gini], textposition="outside"))
        fig.update_layout(title="Out-of-time Gini", xaxis_title="Gini", height=380, margin=dict(l=10, r=30))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = go.Figure(go.Bar(x=g.oot_pr_auc, y=[nm(m) for m in g.index], orientation="h",
                               marker_color=[AMBER if m == MODEL else GREY for m in g.index],
                               text=[f"{x:.3f}" for x in g.oot_pr_auc], textposition="outside"))
        fig.update_layout(title="Out-of-time PR-AUC", xaxis_title="PR-AUC", height=380, margin=dict(l=10, r=30))
        st.plotly_chart(fig, width="stretch")
    st.caption(f"The out-of-time bad rate is {pct(M['data']['bad_rate_oot'])}, so PR-AUC has to be read against that floor rather than against 0.5. "
               "The two charts order the models almost identically here, which is expected. PR-AUC is still worth reporting because it is the metric "
               "a cut-off policy acts on, and it is the one that moves when the bad rate moves.")

    st.divider()
    sec("What each metric means", "Each metric, what it measures, and the level that counts as good for a portfolio of this kind.")
    defs = pd.DataFrame([
        ("AUC", "Probability that a randomly chosen bad borrower scores riskier than a randomly chosen good one.", "0.5 is chance and 1.0 is perfect. Compare models on the same sample only."),
        ("Gini", "Twice AUC minus one, rescaled from 0 to 1.", "Portfolio scorecards for unsecured credit usually land between 0.3 and 0.6."),
        ("KS", "Largest gap between the cumulative share of bads and goods caught at any cut-off.", "Higher is better. Used to choose a cut-off, not to judge calibration."),
        ("PR-AUC", "Area under the precision and recall curve.", f"The base rate is {pct(M['data']['bad_rate_oot'])}, so read it against that floor, not against 0.5."),
        ("Brier", "Mean squared error of the predicted probabilities.", "Lower is better. It mixes discrimination and calibration, so it moves when either does."),
        ("ECE", "Average gap between predicted PD and observed default rate across risk deciles.", f"The selected model is at {o['ece']:.4f}. Below 0.01 is tight."),
        ("PSI", "Population stability index between the training scores and the 2016 scores.", f"Currently {M['psi']['psi_train_vs_oot']:.3f}. Below 0.10 is stable, 0.10 to 0.25 is watch, above 0.25 is act."),
        ("Bootstrap interval", "The same fitted model is rescored on 1000 resamples of the 2016 loans.", "If two intervals overlap heavily, the ranking difference is not established."),
    ], columns=["Metric", "What it measures", "How to read it here"])
    table(defs)
    st.caption("The champion rule, the calibration and the master scale are all fitted on validation. The 2016 sample is touched once, for the numbers above.")

# ------------------------------------------------------------------ Data quality
if page == "Data quality":
    sec("Bad rate by vintage", "Each point is an origination year. The grey bars show how much of the vintage was still open at the data cut.")
    v = cload("bad_rate_by_vintage.csv")
    fig = go.Figure()
    fig.add_bar(x=v.issue_year, y=v.share_open_at_cut * 100, name="Still open at data cut (kept, not dropped)", marker_color=LIGHT, yaxis="y2", opacity=0.6)
    fig.add_scatter(x=v.issue_year, y=v.bad_rate_12m * 100, name="12-month bad rate (used here)", mode="lines+markers", line=dict(color=NAVY))
    fig.add_scatter(x=v.issue_year, y=v.bad_rate_lifetime_resolved_only * 100, name="Lifetime bad rate, resolved loans only (rejected label)", mode="lines+markers", line=dict(color=AMBER, dash="dash"))
    fig.update_layout(yaxis=dict(title="Bad rate (%)"), yaxis2=dict(title="Open at data cut (%)", overlaying="y", side="right", range=[0, 100]), height=380, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    c2 = M["label_sensitivity"]
    st.caption(f"Every loan receives the same 12-month observation window, so no loan is dropped for being unfinished. "
               f"The 2016 12-month bad rate is {pct(c2['oot_bad_rate_main'])}. The older lifetime label on resolved loans only would give "
               f"{pct(c2['oot_bad_rate_old_lifetime_resolved_only'])}, because it drops {pct(c2['oot_share_open_at_cut'])} of 2016 loans that were still open. "
               f"Ranking power barely moves: champion AUC {c2['oot_auc_main']:.3f} against {c2['oot_auc_old_lifetime_resolved_only']:.3f} on the older label.")
    st.divider()
    blk("Rows dropped per split")
    table(cload("dropped_by_split.csv"))
    blk("Status transitions by vintage")
    st.caption("A monthly roll-rate matrix needs delinquency history the public file does not contain.")
    table(cload("status_transition_by_vintage.csv").round(4))
    table(v.round(4))
    st.divider()
    sec("Vintage curves", "Cumulative charge-off share by months on book, for each origination year.")
    vc = cload("vintage_curves.csv")
    fig = px.line(vc, x="months_on_book", y=vc.cum_bad_share * 100, color=vc.issue_year.astype(str), labels={"y": "Cumulative charged-off share of vintage (%)", "color": "Issued"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Months on book is taken from the month of last payment, which is a proxy for default timing. This field is post-origination and is never a model input.")
    st.divider()
    sec("Missing values", "Share of each field that is blank in the training data, for fields above 0.5%.")
    ms_ = cload("missingness.csv")
    ms_ = ms_[ms_.missing_train > 0.005].head(25)
    fig = px.bar(ms_.iloc[::-1], x=ms_.iloc[::-1].missing_train * 100, y="feature", orientation="h", labels={"x": "Missing in train (%)"}, height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("Most blanks mean the event never happened, such as no delinquency or no public record, so a blank is kept as its own category rather than imputed.")
    st.divider()
    a, b = st.columns(2)
    with a:
        blk("Sentinel codes handled")
        table(cload("sentinel_rules.csv"))
        blk("Train, validation and test split")
        table(cload("split_summary.csv").round(4))
    with b:
        blk("Outlier caps")
        st.caption("Winsorised at the 0.5th and 99.5th percentiles, with the cut points learned on train only.")
        table(cload("outlier_caps.csv").round(3), height=300)
    st.divider()
    sec("Leakage audit", "Every source column, the decision taken on it, and the reason.")
    au = cload("leakage_audit.csv")
    opts = ["all"] + sorted(au.decision.unique())
    sel = st.selectbox("Filter by decision", opts)
    table(au if sel == "all" else au[au.decision == sel], height=360)

# ------------------------------------------------------------------ Features
if page == "Features":
    sec("Information value by feature", "The top 30 fields, ranked by how strongly they separate borrowers who repay from those who do not.")
    iv = cload("iv_table.csv")
    top = iv.head(30).iloc[::-1]
    fig = px.bar(top, x="iv_coarse", y="feature", orientation="h", color="decision", height=620,
                 color_discrete_map={"keep": NAVY, "drop: IV < 0.02": "#b8bcc4", "flag: IV > 0.5 leakage suspect": RED})
    fig.add_vline(x=0.02, line_dash="dash", line_color=AMBER)
    st.plotly_chart(fig, width="stretch")
    st.caption("Below 0.02 a field adds almost nothing. Above 0.5 it usually signals a leak. Nothing here crosses that line.")
    table(iv.round(4), height=300)
    st.divider()
    sec("Binned bad rate and weight of evidence", "How one field behaves across its bins, and the weight of evidence each bin carries.")
    wb = cload("woe_bins.csv")
    var = st.selectbox("Feature", iv.feature.tolist(), index=0, format_func=lambda x: f"{x}: {L.LABELS.get(x, x)}")
    t = wb[wb.variable == var]
    fig = go.Figure()
    fig.add_bar(x=t.bin, y=t.bad_rate * 100, name="Bad rate (%)", marker_color=NAVY)
    fig.add_scatter(x=t.bin, y=t.woe, name="Weight of evidence", yaxis="y2", mode="lines+markers", line=dict(color=AMBER))
    fig.update_layout(yaxis=dict(title="Bad rate (%)"), yaxis2=dict(title="Weight of evidence", overlaying="y", side="right"), height=380)
    st.plotly_chart(fig, width="stretch")
    r_ = iv[iv.feature == var].iloc[0]
    trend = "yes" if bool(r_.monotone) else "no"
    st.markdown(f"Fine-classing IV **{r_.iv_fine:.3f}**  ·  coarse IV **{r_.iv_coarse:.3f}**  ·  {int(r_.n_bins)} bins  ·  monotone bad-rate trend **{trend}**  ·  decision **{r_.decision}**")
    table(t[["bin", "n", "share", "bad", "bad_rate", "woe", "iv_contrib"]].round(4))
    show_fig("woe_bins_top6.png")

# ------------------------------------------------------------------ Multicollinearity
if page == "Multicollinearity":
    sec("Correlation between model inputs", "Weight-of-evidence coded, so the scale is comparable across fields.")
    mc = jload("multicollinearity.json")
    cm = cload("corr_matrix_woe.csv").set_index("feature")
    fig = px.imshow(cm, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=640, aspect="auto")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"After removing the lower-IV member of each pair above |r| 0.7, {mc['n_features_after_corr']} features remain. "
               f"Variance inflation pruning above 5 leaves {mc['n_features_after_vif']}. The condition index is {mc['condition_index']:.1f}, "
               f"and anything below 30 is acceptable.")
    a, b = st.columns(2)
    with a:
        blk("Pairs with |r| above 0.7")
        table(cload("corr_pairs_gt_0_7.csv").round(3), height=300)
    with b:
        blk("Variance inflation iterations")
        st.caption("One feature is dropped per round until every remaining variance inflation factor is below 5.")
        vi = cload("vif_iterations.csv")
        if len(vi):
            table(vi.round(2), height=300)
        else:
            st.success("No feature needed dropping. Every variance inflation factor was already below 5 after the correlation step.")
    vf = cload("vif_final.csv").sort_values("vif")
    fig = px.bar(vf, x="vif", y="feature", orientation="h", height=max(300, 22 * len(vf)))
    fig.add_vline(x=5, line_dash="dash", line_color=RED)
    st.plotly_chart(fig, width="stretch")
    st.divider()
    sec("Coefficient sign check", "Each fitted sign against what a credit officer would expect.")
    sg = cload("sign_check.csv")
    table(sg.round(4))
    st.caption("The expected effect is the direction a credit officer would predict as the value rises. A consistent sign means the fitted "
               "coefficient agrees. Variables with no clear expectation are left unjudged.")

# ------------------------------------------------------------------ Logistic ladder
if page == "Logistic ladder":
    sec("Five specifications on validation", "Increasingly constrained linear models, compared on the same validation window.")
    lad = cload("ladder_metrics_valid.csv")
    lad["model"] = lad.model.map(nm)
    table(lad[["model", "auc", "gini", "ks", "brier_calibrated", "ece_calibrated", "hl_p_value_calibrated"]].round(4))
    st.caption("All five specifications rank borrowers almost identically, so the simplest and most explainable one is a defensible choice. "
               "Hosmer–Lemeshow p-values are tiny at this sample size because the test over-rejects. Read ECE and the calibration plot instead.")
    st.divider()
    sec("Odds ratios", "Plain logistic regression fitted with statsmodels, per one standard deviation of each numeric input.")
    cv_ = jload("statsmodels_convergence.json")
    st.markdown(f"Converged: **{cv_['converged']}**  ·  pseudo R-squared **{cv_['pseudo_r2']:.3f}**  ·  observations **{cv_['nobs']:,}**")
    pl = cload("statsmodels_logit_raw.csv")
    pl = pl[pl.feature != "const"]
    pl["absz"] = pl.z.abs()
    top = pl.sort_values("absz", ascending=False).head(25).iloc[::-1]
    fig = go.Figure(go.Scatter(x=top.odds_ratio, y=top.feature, mode="markers", marker=dict(color=NAVY),
                               error_x=dict(type="data", symmetric=False, array=top.or_ci_high - top.odds_ratio, arrayminus=top.odds_ratio - top.or_ci_low)))
    fig.add_vline(x=1, line_dash="dash", line_color=RED)
    fig.update_xaxes(type="log", title="Odds ratio per 1 SD (95% interval)")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")
    st.caption("Points to the right of the red line raise the odds of default. Points to the left lower them. Numeric inputs are standardised, so the bars are comparable.")
    table(pl.drop(columns="absz").round(4), height=300)
    st.divider()
    sec("Regularisation", "What each penalty does to the coefficients.")
    ct = cload("ladder_coefficients.csv")
    ct["a"] = ct.plain.abs()
    mt = ct.sort_values("a", ascending=False).head(20).melt(id_vars="feature", value_vars=["plain", "l2", "l1", "elasticnet"], var_name="model", value_name="coef")
    st.plotly_chart(px.bar(mt, x="coef", y="feature", color="model", barmode="group", orientation="h", height=700), width="stretch")
    st.markdown(f"Features zeroed out: L1 **{int(ct.l1_zeroed.sum())}**, ElasticNet **{int(ct.en_zeroed.sum())}** of {len(ct)}.")
    a, b = st.columns(2)
    with a:
        blk("Class imbalance")
        st.caption("No weighting, class weighting and SMOTE, each evaluated inside cross-validation.")
        table(cload("imbalance_cv.csv").round(4))
        show_fig("imbalance_comparison.png")
    with b:
        blk("Coefficient stability")
        stb = cload("coef_stability.csv")
        st.markdown(f"Sign flips across the five folds: **{int(stb.sign_flips.sum())}** of {len(stb)} coefficients.")
        show_fig("coef_stability.png")
    st.divider()
    blk("Hosmer–Lemeshow test on validation")
    st.json(jload("hosmer_lemeshow_valid.json"), expanded=False)
    show_fig("calibration_scorecard_valid.png", width=450)

# ------------------------------------------------------------------ Challengers
if page == "Challengers":
    sec("XGBoost, tuned and monotone constrained", "A flexible challenger, then the same model with business-expected directions imposed.")
    st.write("Hyper-parameters are tuned by randomised search with three-fold cross-validation on a training sub-sample and a modest budget, "
             "then refitted on the full training sample. The monotone version reuses those settings and forces a one-way effect wherever "
             "business expectation and the training data agree.")
    a, b = st.columns(2)
    with a:
        blk("Best hyper-parameters")
        st.json(jload("xgb_best_params.json"))
        blk("Search results")
        st.caption("Top eight candidates by cross-validated AUC.")
        table(cload("xgb_search.csv").head(8).round(4))
    with b:
        blk("Monotone constraints applied")
        mt_ = cload("monotone_constraints.csv")
        table(mt_[mt_.constraint_applied != 0].round(3), height=300)
    cmp_ = cload("model_comparison.csv").set_index("model")
    cost = cmp_.loc["xgb_tuned", "oot_auc"] - cmp_.loc["xgb_monotone", "oot_auc"]
    st.info(f"**What the constraints cost.** Out-of-time AUC falls by {cost:+.4f}, from {cmp_.loc['xgb_tuned', 'oot_auc']:.4f} for the tuned model "
            f"to {cmp_.loc['xgb_monotone', 'oot_auc']:.4f} for the monotone model. The best boosted model gains "
            f"{cmp_.loc['xgb_tuned', 'oot_auc'] - cmp_.loc['woe_scorecard', 'oot_auc']:+.4f} AUC over the weight-of-evidence scorecard.")
    st.divider()
    sec("SHAP", "Mean absolute SHAP value per feature, for the tuned and monotone models.")
    imp = cload("shap_importance.csv").head(15).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=imp.feature, x=imp.mean_abs_shap_tuned, name="Tuned", orientation="h", marker_color=GREY)
    fig.add_bar(y=imp.feature, x=imp.mean_abs_shap_monotone, name="Monotone", orientation="h", marker_color=NAVY)
    fig.update_layout(barmode="group", height=520, xaxis_title="Mean absolute SHAP value (log-odds)")
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        show_fig("shap_beeswarm.png")
    with b:
        show_fig("shap_waterfall.png")

# ------------------------------------------------------------------ Model comparison
if page == "Model comparison":
    sec("Curves, lift and calibration", "The same 2016 sample shown visually. The numbers behind these curves are in the Metrics section.")
    cmp_ = cload("model_comparison.csv")
    models = cmp_.model.tolist()
    default = list(dict.fromkeys([MODEL] + [m for m in ["woe_scorecard", "elasticnet_lr", "xgb_tuned", "xgb_monotone", "ref_xgb_plus_LC_grade"] if m in models]))
    pick = st.multiselect("Models shown on the curves", models, default=default, format_func=nm,
                          help="The sidebar selection is included by default. Reference models are available but are not eligible models.")
    roc = cload("roc_points.csv")
    pr = cload("pr_points.csv")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for m in pick:
            if m in roc.model.unique():
                s = roc[roc.model == m]
                fig.add_scatter(x=s.fpr, y=s.tpr, name=nm(m), mode="lines")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), showlegend=False)
        fig.update_layout(title="ROC, out of time", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420)
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = go.Figure()
        for m in pick:
            if m in pr.model.unique():
                s = pr[pr.model == m]
                fig.add_scatter(x=s.recall, y=s.precision, name=nm(m), mode="lines")
        fig.update_layout(title="Precision and recall, out of time", xaxis_title="Recall", yaxis_title="Precision", height=420)
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        ks = cload("ks_points.csv")
        fig = go.Figure()
        fig.add_scatter(x=ks.threshold, y=ks.tpr, name="Bads", line=dict(color=RED))
        fig.add_scatter(x=ks.threshold, y=ks.fpr, name="Goods", line=dict(color=NAVY))
        fig.update_xaxes(type="log", title="PD threshold")
        fig.update_layout(title=f"KS curve for the champion (KS {M['oot'][CH]['ks']:.3f})", height=400)
        st.plotly_chart(fig, width="stretch")
    with b:
        rel = cload("reliability_oot.csv")
        fig = go.Figure()
        for m, s in rel.groupby("model"):
            fig.add_scatter(x=s.mean_pred, y=s.obs_rate, name=nm(m), mode="lines+markers")
        mx = float(max(rel.mean_pred.max(), rel.obs_rate.max()))
        fig.add_scatter(x=[0, mx], y=[0, mx], line=dict(dash="dot", color="grey"), showlegend=False)
        fig.update_layout(title="Reliability by out-of-time decile", xaxis_title="Predicted PD", yaxis_title="Observed bad rate", height=400)
        st.plotly_chart(fig, width="stretch")
        st.caption("Points above the dotted line mean actual defaults exceeded the prediction.")
    st.divider()
    lift = cload("lift_deciles.csv")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.bar(lift, x=lift.index + 1, y="lift", labels={"x": "Risk decile, 1 is riskiest", "lift": "Bad rate divided by average"}, title="Decile lift, champion", height=380), width="stretch")
    with b:
        g = pd.concat([pd.DataFrame({"cum_pop": [0], "cum_bad_capture": [0]}), lift[["cum_pop", "cum_bad_capture"]]])
        fig = go.Figure()
        fig.add_scatter(x=g.cum_pop, y=g.cum_bad_capture, mode="lines+markers", name="Champion")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), showlegend=False)
        fig.update_layout(title="Cumulative gains", xaxis_title="Share of applicants, riskiest first", yaxis_title="Share of bads caught", height=380)
        st.plotly_chart(fig, width="stretch")
    table(lift.round(4))
    show_fig("score_distribution.png", width=600)

# ------------------------------------------------------------------ Scorecard
if page == "Scorecard":
    sec("Points table", "Each answer earns points. Total points map to a probability of default.")
    sm_ = jload("scorecard_meta.json")
    st.markdown(f"Scaling: **{sm_['base_score']:.0f} points at {sm_['base_odds_good_to_bad']:.0f}:1 good-to-bad odds**. Every "
                f"{sm_['pdo']:.0f} points doubles the odds of being good, with factor {sm_['factor']:.2f} and offset {sm_['offset']:.1f}. "
                f"A score is the sum of the points of the bin the applicant falls into, across {sm_['n_variables']} variables.")
    pt = cload("points_table.csv")
    v = st.selectbox("Variable", pt.variable.unique().tolist(), format_func=lambda x: f"{x}: {L.LABELS.get(x, x)}", key="ptvar")
    table(pt[pt.variable == v][["label", "bin", "woe", "points", "n_train", "bad_rate_train"]].round(3))
    rng_ = pt.groupby("variable").points.agg(lambda s: s.max() - s.min()).sort_values().reset_index()
    st.plotly_chart(px.bar(rng_, x="points", y="variable", orientation="h", title="Points range per variable", height=420), width="stretch")
    with st.expander("Full points table"):
        table(pt.round(3), height=400)
    st.divider()
    sec("Master scale", "Eight grades with geometrically spaced PD bands, each tested against its 2016 experience.")
    ms = cload("master_scale.csv")
    table(ms[["grade", "pd_lower", "pd_upper", "master_scale_pd", "score_lower", "score_upper", "share_valid", "share_oot", "observed_rate_oot", "binomial_p_two_sided", "verdict_oot"]].round(4))
    fig = go.Figure()
    fig.add_bar(x=ms.grade, y=ms.master_scale_pd * 100, name="Master-scale PD, validation", marker_color=NAVY)
    fig.add_bar(x=ms.grade, y=ms.observed_rate_oot * 100, name="Actual, 2016 out of time", marker_color=RED)
    fig.update_layout(barmode="group", yaxis_title="Default rate (%)", height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption("The binomial test asks whether the 2016 default count is consistent with the grade's promised PD. With hundreds of thousands of loans "
               "and correlated defaults inside a vintage it over-rejects, so read the size of the gap alongside the p-value.")
    st.divider()
    sec("Cut-off simulator", "Approve every applicant whose PD is at or below the cut-off. The economics below are assumptions, not observed data.")
    st.warning("Margin, loss given default and exposure at default are assumptions you control. LendingClub's risk-based pricing is ignored, so margin is flat across borrowers.")
    A = jload("cutoff.json")["assumptions"]
    c1, c2, c3 = st.columns(3)
    margin = c1.slider("Net margin on a good loan, percent of principal over 12 months", 1, 30, int(A["margin_pct_of_principal"] * 100)) / 100
    lgd = c2.slider("Loss given default", 30, 100, int(A["lgd"] * 100)) / 100
    ead = c3.slider("Exposure at default, percent of principal", 40, 100, int(A["ead_pct_of_principal"] * 100)) / 100
    scored = oot()
    curve = L.cutoff_curve(scored[f"pd_{MODEL}"].values, scored.y.values, scored.loan_amnt.values, margin, lgd, ead, 400)
    opt = curve.loc[curve.profit_per_application.idxmax()]
    cut = st.slider("Cut-off: approve while PD is at or below", float(curve.cutoff_pd.min()), float(curve.cutoff_pd.max()), float(opt.cutoff_pd), 0.005, format="%.3f")
    row = curve.iloc[(curve.cutoff_pd - cut).abs().argmin()]
    k = st.columns(4)
    k[0].metric("Approval rate", pct(row.approval_rate))
    k[1].metric("Bad rate among approved", pct(row.bad_rate_approved))
    k[2].metric("Profit per application", f"${row.profit_per_application:,.0f}")
    k[3].metric("Profit-maximising cut-off", f"{opt.cutoff_pd:.3f}")
    fig = go.Figure()
    fig.add_scatter(x=curve.cutoff_pd, y=curve.profit_per_application, name="Profit per application", line=dict(color=NAVY))
    fig.add_scatter(x=[cut], y=[row.profit_per_application], mode="markers", marker=dict(size=12, color=AMBER), name="Selected")
    fig.update_layout(xaxis_title="Approve while PD is at or below", yaxis_title="Profit per application ($)", height=360)
    st.plotly_chart(fig, width="stretch")
    ci = jload("cutoff.json")
    st.caption(f"Curve computed on all 2016 loans for {nm(MODEL)}. The slider starts at the cut-off that maximises profit on this sample, so treat that position as "
               f"illustrative rather than validated. The notebook froze {ci['cutoff_pd']:.3f} on validation for the champion. Out of time that gives an approval rate of "
               f"{pct(ci['oot_approval_rate_at_cutoff'])} and ${ci['oot_profit_per_application']:,.0f} per application, against "
               f"${ci['oot_profit_approve_all']:,.0f} when everyone is approved.")
    blk("Sensitivity")
    st.caption("Champion cut-off chosen on validation, then applied to 2016.")
    table(cload("cutoff_sensitivity.csv").round(3), height=300)

# ------------------------------------------------------------------ Applicant scorer
if page == "Applicant scorer":
    sec("Score an applicant", "Change any field and the score, grade and reason codes update. Nothing is stored.")
    schema = jload("feature_schema.json")
    sc = scorecard()
    examples = jload("applicant_examples.json")["examples"]
    labels = ["Typical applicant"] + [f"Real 2016 loan {i + 1}, actual outcome {'bad' if e['actual_bad'] else 'good'}" for i, e in enumerate(examples)]
    pickx = st.selectbox("Start from", labels)
    base = {} if pickx == labels[0] else examples[labels.index(pickx) - 1]["features"]
    if MODEL == "woe_scorecard":
        st.caption("PD, grade, points and reason codes all come from the weight-of-evidence scorecard, so the number and the explanation use the same model. "
                   "The form shows the variables the scorecard uses, and everything else is held at the training median. Reason codes measure points lost "
                   "against the best bin for that variable.")
    else:
        st.caption(f"PD and grade come from {nm(MODEL)}. Points and reason codes come from the weight-of-evidence scorecard, which is the explainable one. "
                   "The form shows the variables the scorecard uses, and everything else is held at the training median. Reason codes measure points lost "
                   "against the best bin for that variable.")
    vars_ = sc.vars
    cols = st.columns(3)
    raw = {}
    for c_name, spec in schema.items():           # defaults for all features
        raw[c_name] = base.get(c_name) if base.get(c_name) is not None else spec["default"]
    for i, vname in enumerate(vars_):
        spec = schema[vname]
        with cols[i % 3]:
            if spec["kind"] == "cat":
                lv = spec["levels"]
                dv = raw[vname] if raw[vname] in lv else lv[0]
                raw[vname] = st.selectbox(spec["label"], lv, index=lv.index(dv), key=f"f_{vname}_{pickx}")
            else:
                lo, hi = float(spec["min"]), float(spec["max"])
                dv = raw[vname]
                was_missing = base.get(vname) is None and bool(base)
                dv = float(dv) if dv is not None and not (isinstance(dv, float) and np.isnan(dv)) else float(spec["default"] or 0.0)
                lo, hi = min(lo, dv), max(hi, dv)
                val = st.number_input(spec["label"], min_value=lo, max_value=hi, value=dv, key=f"f_{vname}_{pickx}")
                if spec.get("missing_share", 0) > 0.01 and st.checkbox("Not available or not applicable", value=was_missing, key=f"m_{vname}_{pickx}"):
                    val = np.nan
                raw[vname] = val
    raw_df = pd.DataFrame([{k: raw[k] for k in L.ALL_FEATURES}])
    res = L.score_applicant(raw_df.iloc[0].to_dict(), ART, MODEL)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scorecard points", f"{res['score']:.0f}", help="Always from the weight-of-evidence scorecard, so the points and reasons stay explainable.")
    c2.metric("PD, selected model", pct(res["pd"], 2), help=f"Calibrated on validation, scored here by {nm(res['model'])}.")
    c3.metric("Grade, A best to H worst", res["grade"])
    c4.metric("Model", nm(res["model"]))
    if res["model"] != "woe_scorecard":
        st.caption(f"PD and grade come from {nm(res['model'])}. The scorecard PD for this applicant is {pct(res['pd_scorecard'], 2)}, "
                   f"and the points and reason codes above are the scorecard's. Grade bands are the master scale fitted on the champion's "
                   "validation PD, so with another model selected the grade is indicative rather than a policy decision.")
    st.divider()
    blk("Reasons this applicant scores lower")
    if res["reasons"]:
        for r in res["reasons"]:
            aa_ = f" Adverse-action wording: *{r['adverse_action']}*." if r.get("adverse_action") else ""
            st.markdown(f"- {r['text']}.{aa_}")
    else:
        st.write("No variable is materially below its best bin.")
    cut = jload("cutoff.json")["cutoff_pd"]
    champ = jload("champion_rule.json")["champion"]
    st.markdown(f"Decision at the frozen cut-off of {cut:.3f}, where PD is at or below the cut-off and the economics are assumed: "
                f"**{'approve' if res['pd'] <= cut else 'decline'}**.")
    if res["model"] != champ:
        st.caption(f"The cut-off was chosen on validation for the champion, {nm(champ)}. Comparing it with {nm(res['model'])} PDs is indicative, "
                   "because the two models are calibrated separately.")
    st.caption("Demonstration only. This is not a lending decision tool.")

# ------------------------------------------------------------------ Monitoring
if page == "Monitoring":
    sec("Stability", "Population stability on the score, and characteristic stability on each input.")
    p = M["psi"]
    k = st.columns(4)
    k[0].metric("Score PSI, train to OOT", f"{p['psi_train_vs_oot']:.3f}", help=p["label_train_vs_oot"])
    k[1].metric("Score PSI, train to validation", f"{p['psi_train_vs_valid']:.3f}")
    k[2].metric("Mean score, train", f"{p['mean_score_train']:.0f}")
    k[3].metric("Mean score, OOT", f"{p['mean_score_oot']:.0f}", delta=f"{p['mean_score_oot'] - p['mean_score_train']:+.1f}")
    csi = cload("csi.csv")
    top = csi.head(20).iloc[::-1]
    fig = px.bar(top, x="csi_train_vs_oot", y="feature", orientation="h", color="flag", height=520,
                 color_discrete_map={"stable (<0.10)": NAVY, "moderate shift (0.10-0.25)": AMBER, "significant shift (>0.25)": RED})
    fig.add_vline(x=0.1, line_dash="dot")
    fig.add_vline(x=0.25, line_dash="dash", line_color=RED)
    st.plotly_chart(fig, width="stretch")
    st.caption("Stability indices measure whether today's applicants resemble the population the model was built on. Below 0.10 is stable. Above 0.25 triggers an investigation.")
    table(csi.round(4), height=260)
    st.divider()
    sec("Predicted against actual by quarter", "A drift check inside the out-of-time year.")
    qt = cload("drift_by_quarter.csv")
    fig = go.Figure()
    fig.add_scatter(x=qt.quarter, y=qt.actual * 100, name="Actual", mode="lines+markers", line=dict(color=RED))
    fig.add_scatter(x=qt.quarter, y=qt.predicted * 100, name="Predicted", mode="lines+markers", line=dict(color=NAVY, dash="dash"))
    fig.update_layout(yaxis_title="Bad rate (%)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.divider()
    sec("Backtest by grade", "Expected against observed defaults in each grade, 2016 out of time.")
    ms = cload("master_scale.csv")
    table(ms[["grade", "master_scale_pd", "n_oot", "bads_oot", "expected_bads_oot", "observed_rate_oot", "binomial_p_two_sided"] + ([c for c in ["jeffreys_p_two_sided", "vasicek_p_two_rho0.03", "verdict_oot_vasicek"] if c in ms.columns]) + ["verdict_oot"]].round(4))
    fig = go.Figure()
    fig.add_bar(x=ms.grade, y=ms.expected_bads_oot, name="Expected bads", marker_color=NAVY)
    fig.add_bar(x=ms.grade, y=ms.bads_oot, name="Observed bads", marker_color=RED)
    fig.update_layout(barmode="group", height=340)
    st.plotly_chart(fig, width="stretch")
    st.divider()
    sec("Redevelopment triggers", "The conditions that would send the model back for redevelopment, and where each stands today.")
    mon = jload("monitoring_triggers.json")
    tr = pd.DataFrame(mon["triggers"]).astype(str)
    table(tr)
    if mon.get("reference_tests"):
        st.caption(mon.get("calibration_basis", ""))
        table(pd.DataFrame(mon["reference_tests"]).astype(str))
    st.write(mon["redevelopment_policy"])
    st.caption(f"Review cadence: {mon['review_frequency']}.")

# ------------------------------------------------------------------ Calibration
if page == "Calibration":
    sec("Calibration tests for correlated defaults", "Defaults inside one vintage share an economy, so tests that assume independence over-reject.")
    if have("calibration_tests.json", "calibration_longrun.json"):
        ct = jload("calibration_tests.json")
        msx = cload("master_scale.csv")
        st.write("The monitoring light uses the Vasicek single-factor test with asset correlation rho of 0.03. Binomial and Jeffreys results are shown as references, "
                 "because both treat borrowers as independent.")
        cols_ = ["grade", "master_scale_pd", "observed_rate_oot", "binomial_p_two_sided", "jeffreys_p_two_sided"] + [f"vasicek_p_two_rho{r:.2f}" for r in ct["rho_grid"]] + ["vasicek_status_rho0.03"]
        table(msx[cols_].round(4))
        k = st.columns(4)
        k[0].metric("Grades failing binomial, 5%", ct["grades_failing_binomial_5pct"])
        k[1].metric("Grades failing Jeffreys, 5%", ct["grades_failing_jeffreys_5pct"])
        k[2].metric("Grades failing Vasicek, 5%", ct["grades_failing_vasicek_5pct"])
        k[3].metric("Portfolio Vasicek p", f"{ct['portfolio_oot']['vasicek_p_two_rho0.03']:.3f}")
        table(cload("calibration_rho_sensitivity.csv"))
        ci_ = ct["calibration_intercept_slope"]
        table(pd.DataFrame([{"model": nm(m), "sample": s, "intercept (CITL)": v["intercept_citl"], "intercept 95% interval": f"{v['intercept_ci'][0]:.3f} to {v['intercept_ci'][1]:.3f}", "slope": v["slope"],
                             "slope 95% interval": f"{v['slope_ci'][0]:.3f} to {v['slope_ci'][1]:.3f}", "mean predicted": v["mean_pred"], "observed": v["obs_rate"]}
                            for m, d_ in ci_.items() for s, v in d_.items()]).round(4))
        st.caption("An intercept of 0 and a slope of 1 mean perfect calibration. A positive intercept means the model under-predicts risk on average.")
        st.divider()
        sec("Long-run calibration", "A 2013–2015 anchor with a stated margin of conservatism.")
        lr = jload("calibration_longrun.json")
        k = st.columns(4)
        k[0].metric("Long-run default rate, 2013–15", pct(lr["long_run_default_rate"], 2))
        k[1].metric("OOT predicted over actual, before", f"{lr['oot_pred_over_actual_before']:.2f}")
        k[2].metric("OOT predicted over actual, after", f"{lr['oot_pred_over_actual_after']:.2f}")
        k[3].metric("Stated margin of conservatism", f"+{lr['moc_rel_stated']:.0%}")
        st.write(lr["moc_note"])
        st.info(lr["reading"])
        table(cload("calibration_longrun_oot.csv")[["model", "variant", "oot_mean_pred", "oot_obs_rate", "oot_pred_over_actual", "oot_ece", "oot_vasicek_p_two_rho0.03", "oot_binomial_p_two"]].round(4))
        show_fig("calibration_longrun.png")
    else:
        st.info("The calibration upgrade artifacts are not in this folder yet. They are produced by the next notebook run.")
    st.divider()
    sec("Rolling-origin backtest", "Refit on successive windows and scored forward, the way a model is used in production.")
    if have("rolling_origin_backtest.csv"):
        table(cload("rolling_origin_backtest.csv").round(4))
        show_fig("rolling_origin.png")
        ri = jload("rolling_origin_summary.json")
        st.caption("Held fixed from the main fit: " + ", ".join(ri["fixed_from_main_fit"]) + ".")
    st.divider()
    sec("Discrimination with confidence intervals", "Somers' D and the cumulative accuracy profile.")
    if have("discrimination_ci.csv"):
        table(cload("discrimination_ci.csv").round(4))
        show_fig("cap_curve.png", width=520)
    st.divider()
    sec("Default-definition reconciliation", "How the 12-month label splits the charged-off population.")
    if have("default_reconciliation.json"):
        dr = jload("default_reconciliation.json")
        st.write(f"Of {dr['charged_off_loans']:,} charged-off loans, {pct(dr['charged_off_share_last_payment_month_le_9'])} were last paid by month 9 and are labelled bad. "
                 f"A further {pct(dr['charged_off_share_last_payment_month_10_12'])} were last paid in months 10 to 12, and {pct(dr['charged_off_share_last_payment_after_12'])} later. "
                 f"Loans currently late or defaulted whose last payment was by month 9 number {dr['loans_in_those_statuses_with_last_payment_le_9']:,}, of which "
                 f"{dr['of_which_not_in_bad_class']:,} ({pct(dr['share_not_in_bad_class'])}) are not in the bad class. Counting them as bad moves the out-of-time "
                 f"bad rate from {pct(dr['oot_bad_rate_main'], 2)} to {pct(dr['oot_bad_rate_if_added'], 2)}.")
        st.caption(dr["lag_note"])
        table(cload("default_definition_sensitivity.csv").round(4))
        table(cload("status_vs_bad_class.csv"))

# ------------------------------------------------------------------ Economics and fairness
if page == "Economics and fairness":
    sec("Model roles", "Which model is the decision model of record, which is the challenger, and why.")
    if have("model_roles.json"):
        rl = jload("model_roles.json")
        st.markdown(f"- **Decision model of record, explainable:** {nm(rl['decision_model_of_record'])}. {rl['decision_model_rationale']}.\n"
                    f"- **Challenger and benchmark:** {nm(rl['challenger_benchmark'])}. {rl['challenger_rationale']}.\n"
                    f"- **Frozen-rule champion for PD accuracy:** {nm(rl['frozen_rule_champion'])}.")
        st.markdown(f"Monotonicity evidence across the {rl['top_numeric_features_checked']} most important numeric features: "
                    f"{rl['constrained_pdp_exactly_monotone']} of {rl['constrained_among_checked']} constrained features have an exactly monotone partial-dependence profile, and "
                    f"{rl['unconstrained_ale_in_expected_direction']} of {rl['unconstrained_among_checked']} unconstrained features follow the expected sign in accumulated local effects. "
                    f"{rl['unconstrained_with_no_expectation']} have no expected sign.")
        table(cload("pdp_ale_monotonicity.csv").round(3))
        show_fig("pdp_ale_top_features.png")
    st.divider()
    sec("Realised economics", "The assumed 6% margin, 85% loss given default and 90% exposure, checked against what actually happened.")
    if have("realised_economics.json") and not jload("realised_economics.json").get("skipped"):
        re_ = jload("realised_economics.json")
        st.caption(re_["horizon_note"])
        k = st.columns(4)
        k[0].metric("Realised LGD on EAD, bad loans", pct(re_["realised_lgd_on_ead_bad_loans"]), help="Assumed 85%")
        k[1].metric("Realised EAD share, bad loans", pct(re_["realised_ead_share_bad_loans"]), help="Assumed 90%")
        k[2].metric("Realised best cut-off PD", f"{re_['realised_best_cutoff_pd_oot']:.3f}", help=f"Assumed-economics cut-off {re_['assumed_cutoff_pd']:.3f}")
        k[3].metric("Realised profit per application", f"${re_['realised_profit_per_app_at_assumed_cutoff_oot']:,.0f}")
        table(cload("realised_economics_by_grade.csv").round(4))
        show_fig("realised_economics.png")
    st.divider()
    sec("Accepted against rejected applications", "What the book would look like if declined applicants were included.")
    if have("reject_parcelling_sensitivity.csv"):
        rj = jload("metrics.json")["benchmark_fixes"]["reject_check"]
        st.warning(rj["risk_score_caveat"] + " The parcelling table is a sensitivity analysis, not reject inference.")
        st.write("Accept share by year: " + ", ".join(f"{y} at {pct(v)}" for y, v in rj["accept_share_by_year"].items()) + ".")
        table(cload("reject_vs_accepted_distributions.csv").round(3), height=300)
        table(cload("reject_parcelling_sensitivity.csv").round(4))
        show_fig("reject_vs_accepted.png")
    st.divider()
    sec("Fairness screen", "Default rates across income bands, and the adverse-action wording used for declined applicants.")
    if have("fairness.json"):
        fj = jload("fairness.json")
        st.caption(fj["caveat"])
        table(cload("fairness_income_bands.csv").round(3))
        show_fig("fairness_income.png")
        blk("Scorecard variable to adverse-action statement")
        table(cload("adverse_action_map.csv"))
        blk("Principal reasons among declined out-of-time applicants")
        table(cload("adverse_action_reasons_declined.csv").round(3))

# ------------------------------------------------------------------ Glossary
if page == "Glossary":
    sec("Glossary", "Plain-language definitions for every term used in the sections above.")
    G = {
        "PD (probability of default)": "The chance a borrower will fail to repay. Here it is the chance a 36-month loan goes 90+ days past due, and is later written off, within its first 12 months.",
        "LGD (loss given default)": "If a borrower does default, the share of the money that is lost after recoveries. Assumed at 85% here.",
        "EAD (exposure at default)": "How much is still owed at the moment of default. Loans amortise, so it is less than the original amount.",
        "Expected loss": "PD × LGD × EAD. The average loss a lender should price in.",
        "Bad and good": "Bad means charged off, in other words written off. Good means fully paid. Loans still open are left out because their outcome is unknown.",
        "Out-of-time (OOT)": "Testing on a later period, 2016, than the one the model learned from, 2013–2015. This is how the model will actually be used.",
        "AUC": "The chance that a randomly chosen bad borrower receives a riskier score than a randomly chosen good borrower. 0.5 is a coin toss and 1 is perfect.",
        "Gini": "Twice AUC minus one. The same information rescaled from 0, useless, to 1, perfect. Bank scorecards for unsecured credit often sit between 0.3 and 0.6.",
        "KS (Kolmogorov–Smirnov)": "The biggest gap between the share of bad and good borrowers caught at any score cut. Bigger means better separation.",
        "WoE (weight of evidence)": "For a group of borrowers, the log ratio of the share of all goods to the share of all bads. Positive is safer than average and negative is riskier.",
        "IV (information value)": "The sum of weight-of-evidence contributions across groups. Rule of thumb: below 0.02 useless, 0.1 to 0.3 medium, 0.3 to 0.5 strong, above 0.5 suspicious.",
        "Vasicek test": "A calibration test that lets all borrowers in a vintage share one economy, through an asset correlation rho of 0.03 here. Unlike the plain binomial test it does not treat 300,000 loans as 300,000 independent coin flips.",
        "Jeffreys test": "A Bayesian check of a promised default rate using a Beta posterior. It works well with few defaults but still assumes independence.",
        "Margin of conservatism (MoC)": "A deliberate uplift to a long-run PD to cover estimation error and data weakness. Here it is a stated +10%, not an estimate.",
        "CAP and accuracy ratio": "Cumulative Accuracy Profile: the share of bad loans caught against the share of applicants screened. The accuracy ratio equals the Gini for a yes or no outcome.",
        "Somers' D": "Rank correlation between score and outcome. For a binary outcome it equals the Gini.",
        "Parcelling": "Giving rejected applicants an assumed bad rate, here k times the accepted rate in the same band. Used as a sensitivity only, not as reject inference.",
        "PSI and CSI": "Population and characteristic stability indices: how much a distribution has shifted. Below 0.10 is stable, 0.10 to 0.25 is watch, above 0.25 is act.",
        "VIF": "Variance inflation factor: how well the other inputs predict this one. Above 5 the input is redundant.",
        "Calibration and ECE": "Whether predicted PDs match actual default rates. ECE is the average gap across risk deciles.",
        "Platt scaling": "A two-parameter logistic curve that maps a model's raw probability to a calibrated one. It is fitted on validation only, never on the 2016 test.",
        "Paired bootstrap": "Resampling the 2016 loans with replacement and recomputing the AUC difference between two models on each resample. Pairing removes the noise from which loans happen to be drawn.",
        "Frozen champion rule": "The selection rule written down before the 2016 sample was scored, so the choice of model cannot be reverse-engineered from the test result.",
        "Brier score": "Average squared error of the predicted probabilities. Lower is better.",
        "Master scale": "A fixed ladder of risk grades from A to H, each with an agreed PD, used across the bank for pricing and limits.",
        "Binomial test": "Checks whether the number of defaults seen in a grade is consistent with the grade's promised PD.",
        "Scorecard points": "A table where each answer earns points. Total points map to a PD. 600 points correspond to 50:1 good-to-bad odds, and every 20 points doubles the odds.",
        "SHAP": "A way to show how much each input pushed one borrower's risk up or down.",
        "Monotone constraint": "A rule telling a flexible model that more of a risk factor, such as higher utilisation, can never make a borrower look safer.",
        "Reject inference": "Guessing how declined applicants would have performed. Not possible here, because the data contains accepted loans only.",
        "Right-censoring and survivor bias": "A lifetime label cannot be known for loans still open at the data cut, and dropping them over-represents early defaulters in the latest year. The 12-month window used here avoids this because every loan has a full year of history.",
    }
    for k_, v_ in G.items():
        st.markdown(f"**{k_}.** {v_}")
