"""Tests: artifacts exist, metrics are sane, the scorer works, the app boots. Run: python -m pytest tests -q"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("LCPD_ART", ROOT / "artifacts"))   # LCPD_ART lets the suite run against a smoke-run artifact folder
sys.path.insert(0, str(ART))
import lcpd_lib as L  # noqa: E402

REQUIRED = [
    "metrics.json", "model_comparison.csv", "champion_rule.json", "calibrators.json", "scorecard.json", "points_table.csv", "master_scale.csv",
    "master_scale.json", "iv_table.csv", "woe_bins.csv", "vif_final.csv", "corr_matrix_woe.csv", "sign_check.csv", "ladder_metrics_valid.csv",
    "imbalance_cv.csv", "xgb_search.csv", "shap_importance.csv", "psi_score.json", "csi.csv", "monitoring_triggers.json", "cutoff.json",
    "cutoff_sensitivity.csv", "bad_rate_by_vintage.csv", "vintage_curves.csv", "leakage_audit.csv", "feature_schema.json", "oot_scored.csv.gz",
    "figure_captions.json", "lcpd_lib.py", "label_sensitivity.json", "dropped_by_split.csv", "status_transition_by_vintage.csv", "models/xgb_monotone.json", "models/xgb_tuned.json", "models/logistic_ladder.joblib",
]
FIGS = ["bad_rate_by_vintage.png", "missingness.png", "iv_bar.png", "woe_bins_top6.png", "corr_heatmap.png", "vif_bar.png", "odds_ratio_forest.png",
        "roc_pr_oot.png", "ks_curve.png", "reliability_oot.png", "lift_gains.png", "score_distribution.png", "shap_beeswarm.png", "shap_waterfall.png",
        "psi_bar.png", "grade_pd_vs_actual.png", "cutoff_profit.png"]


def J(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", REQUIRED)
def test_artifact_exists(name):
    assert (ART / name).exists() and (ART / name).stat().st_size > 0, name


@pytest.mark.parametrize("name", FIGS)
def test_figure_exists(name):
    assert (ART / "figures" / name).exists()


def test_metrics_sane():
    m = J("metrics.json")
    ch = m["champion"]
    for k, v in m["oot"].items():
        assert 0.5 < v["auc"] < 0.9, (k, v["auc"])            # nothing leaks, nothing is broken
        assert abs(v["gini"] - (2 * v["auc"] - 1)) < 1e-9
        assert 0 < v["ks"] < 0.8 and 0 < v["brier"] < 0.25 and 0 <= v["ece"] < 0.2
    assert m["oot"][ch]["gini"] > 0.2, "champion should beat random meaningfully"
    assert m["oot"][ch]["gini"] < 0.6, "implausibly high for application-time-only data: check leakage"
    # reference models using LC's own grade are not eligible
    assert not ch.startswith("ref_")
    assert m["data"]["oot_rows"] > 1000 and m["data"]["bad_rate_oot"] > m["data"]["bad_rate_train"] * 0.9


def test_label_12m_definition():
    st = ["Charged Off", "Charged Off", "Fully Paid", "Current", "Default", "Charged Off"]
    mob = [5, 11, 5, 30, 9, 9.0]
    assert L.label_12m(st, mob).tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]       # only charged off AND last payment <= month 9
    assert L.label_12m(st, mob, 12).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    mo = L.months_on_book_last_payment(pd.Series(pd.to_datetime(["2015-01-01", "2015-06-01"])), pd.Series(pd.to_datetime(["2015-10-01", None])))
    assert mo.tolist() == [9.0, 0.0]


def test_no_rows_dropped_for_label_and_window_complete():
    d = pd.read_csv(ART / "dropped_by_split.csv")
    assert (d.dropped_for_unresolved_status_now == 0).all() and set(d.set) == {"train", "validation", "OOT test"}
    # the old lifetime label would have dropped a real share of the latest vintage (the reason for the change)
    assert d.set_index("set").loc["OOT test", "old_label_drop_share"] > 0.05
    ls = J("label_sensitivity.json")
    assert ls["oot_loans_dropped_by_label"] == 0 and "90+ DPD" in ls["definition"]
    assert 0.5 < ls["oot_auc_main"] < 0.9 and 0.5 < ls["oot_auc_old_lifetime_resolved_only"] < 0.9
    m = J("metrics.json")
    assert m["label_sensitivity"]["oot_bad_rate_main"] == pytest.approx(m["data"]["bad_rate_oot"])
    v = pd.read_csv(ART / "bad_rate_by_vintage.csv")
    assert v.loc[v.issue_year == 2016, "share_open_at_cut"].iloc[0] > 0.05                 # censoring exists in the file...
    assert v.bad_rate_12m.between(0.005, 0.15).all()                                        # ...but the 12-month label is on a sane scale


def test_champion_rule_frozen_before_oot():
    r = J("champion_rule.json")
    assert "OOT" in r["scored_on"] and r["champion"] in r["candidates"]
    assert r["champion"] == J("metrics.json")["champion"]
    assert len(r["sha256_of_rule_body"]) == 64


def test_no_leaky_or_lc_variables_in_features():
    banned = {"grade", "sub_grade", "int_rate", "installment", "total_pymnt", "recoveries", "last_pymnt_d", "last_fico_range_high", "out_prncp",
              "funded_amnt", "loan_status", "issue_d"}
    assert not banned & set(L.ALL_FEATURES)
    sc = J("scorecard.json")
    assert not banned & set(sc["variables"])
    iv = pd.read_csv(ART / "iv_table.csv")
    assert not (iv.iv_coarse > 0.5).any() or (iv[iv.iv_coarse > 0.5].decision == "flag: IV > 0.5 leakage suspect").all()


def test_scorecard_points_reproduce_score():
    sc = L.Scorecard.from_dict(J("scorecard.json"))
    ex = J("applicant_examples.json")["examples"][0]["features"]
    df = L.engineer(pd.DataFrame([ex]))
    pts = sc.points_table()
    total = 0.0
    for v in sc.vars:
        val = df[v].values
        bi = int(sc.woe.bin_index(v, val if sc.woe.vars[v]["kind"] == "num" else df[v].astype("object").values)[0])
        lab = sc.woe.bin_label(v, bi)
        total += float(pts[(pts.variable == v) & (pts.bin == lab)].points.iloc[0])
    assert abs(total - sc.score(df)[0]) < 1e-6
    assert abs(L.score_to_pd(sc.score(df))[0] - sc.raw_pd(df)[0]) < 1e-9


def test_scaling_pdo():
    assert abs(L.pd_to_score(1 / 51) - 600) < 1e-9          # 50:1 good:bad -> 600
    assert abs((L.pd_to_score(1 / 101) - L.pd_to_score(1 / 51)) - 0) > 0
    odds = lambda s: np.exp((s - L.OFFSET) / L.FACTOR)      # noqa: E731
    assert abs(odds(620) / odds(600) - 2) < 1e-9             # +20 points doubles the odds


def test_scorer_function():
    ex = J("applicant_examples.json")["examples"]
    out = [L.score_applicant(e["features"], ART) for e in ex[:5]]
    for o in out:
        assert 0 < o["pd"] < 1 and 300 < o["score"] < 900 and o["grade"] in "ABCDEFGH"
    # better FICO must not raise risk
    base = dict(ex[0]["features"])
    lo, hi = dict(base, fico=640.0), dict(base, fico=800.0)
    assert L.score_applicant(hi, ART)["score"] > L.score_applicant(lo, ART)["score"]
    assert L.score_applicant(hi, ART)["pd"] <= L.score_applicant(lo, ART)["pd"]
    # every scorable model must produce a probability and keep the FICO direction
    for m in L.SCORABLE_MODELS:
        assert 0 < L.score_applicant(ex[0]["features"], ART, m)["pd"] < 1
        assert L.score_applicant(hi, ART, m)["pd"] <= L.score_applicant(lo, ART, m)["pd"]
    # the saved booster must reproduce the stored champion PD exactly
    assert L.score_applicant(ex[0]["features"], ART, "xgb_monotone")["pd"] == pytest.approx(ex[0]["pd_champion"], abs=1e-9)
    with pytest.raises(ValueError):
        L.model_pd(L.engineer(pd.DataFrame([ex[0]["features"]])), ART, "ref_LC_subgrade_only")


def test_master_scale_ordered_and_covers():
    ms = pd.read_csv(ART / "master_scale.csv")
    assert len(ms) >= 7 and ms.master_scale_pd.is_monotonic_increasing
    assert abs(ms.share_oot.sum() - 1) < 1e-6 and abs(ms.share_valid.sum() - 1) < 1e-6
    assert ms.observed_rate_oot.is_monotonic_increasing


def test_psi_and_ece_helpers():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert L.psi_numeric(a, rng.normal(size=20000)) < 0.02
    assert L.psi_numeric(a, rng.normal(1.0, 1, size=20000)) > 0.25
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, 50000) < p).astype(int)
    assert L.ece(y, p) < 0.02
    assert L.ece(y, np.clip(p * 0.5, 0, 1)) > 0.1


def test_cutoff_curve_monotone_approval():
    s = pd.read_csv(ART / "oot_scored.csv.gz")
    c = L.cutoff_curve(s.pd_champion.values, s.y.values, s.loan_amnt.values, 0.15, 0.85, 0.7, 100)
    assert c.approval_rate.is_monotonic_increasing
    assert c.bad_rate_approved.iloc[-1] > c.bad_rate_approved.iloc[10]      # the safest 10% are cleaner than everyone


def test_app_boots_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    nav = at.sidebar.radio(key="nav")
    assert len(nav.options) == 14 and nav.options[0] == "Overview" and "Metrics" in nav.options
    assert at.sidebar.selectbox(key="model").value == "woe_scorecard"
    for section in nav.options:                     # every section must render on its own
        at.sidebar.radio(key="nav").set_value(section).run()
        assert not at.exception, (section, [e.value for e in at.exception])
    at.sidebar.radio(key="nav").set_value("Metrics").run()
    at.sidebar.selectbox(key="model").set_value("xgb_monotone").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "XGBoost (monotone)" in [m.value for m in at.metric]


# ======================================================================================
# Benchmark-driven additions: calibration tests, long-run calibration, rejects, backtest, economics, fairness
# ======================================================================================
NEW = pytest.mark.skipif(not (ART / "calibration_longrun.json").exists(), reason="benchmark-fix artifacts appear after the next Kaggle run is pulled")


def test_vasicek_reduces_to_binomial_and_widens_with_rho():
    k, n, p = 1900, 40000, 0.04
    b = L.binomial_pvalues(k, n, p)
    v0 = L.vasicek_test(k, n, p, rho=1e-6)
    assert abs(v0["p_under"] - b["p_under"]) < 5e-3
    v = [L.vasicek_test(k, n, p, rho=r)["p_under"] for r in L.RHO_GRID]
    assert v == sorted(v) and v[0] > b["p_under"]            # more correlation -> a given excess is less surprising
    assert L.vasicek_test(3200, n, p, 0.03)["p_under"] < 0.05  # a doubling of the default rate is still rejected
    assert 0 <= L.jeffreys_test(k, n, p)["p_two"] <= 1
    assert L.traffic_light(0.2) == "GREEN" and L.traffic_light(0.03) == "AMBER" and L.traffic_light(0.001) == "RED"


def test_calibration_intercept_slope_detects_miscalibration():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 20, 60000)
    y = (rng.uniform(size=60000) < p).astype(int)
    ok = L.calibration_intercept_slope(y, p)
    assert abs(ok["intercept_citl"]) < 0.1 and abs(ok["slope"] - 1) < 0.1
    low = L.calibration_intercept_slope(y, p * 0.7)
    assert low["intercept_citl"] > 0.2


def test_anchor_shift_hits_target():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 25, 5000)
    w = rng.uniform(1, 3, 5000)
    cal = {"a": 0.05, "b": 0.95}
    d = L.anchor_shift(p, cal, 0.031, w)
    assert abs(np.average(L.sigmoid(cal["a"] + d + cal["b"] * L.logit(p)), weights=w) - 0.031) < 1e-9


def test_delong_cap_and_accuracy_ratio_agree_with_gini():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    p = rng.beta(1, 15, 30000)
    y = (rng.uniform(size=30000) < p).astype(int)
    dl = L.delong_auc_ci(y, p)
    assert abs(dl["auc"] - roc_auc_score(y, p)) < 1e-12
    assert dl["somers_d_ci"][0] < dl["somers_d"] < dl["somers_d_ci"][1] and 0 < dl["somers_d_se"] < 0.05
    ar, _ = L.accuracy_ratio(y, p)
    assert abs(ar - dl["somers_d"]) < 1e-3
    x, c = L.cap_curve(y, p)
    assert x[0] == 0 and c[0] == 0 and abs(x[-1] - 1) < 1e-9 and abs(c[-1] - 1) < 1e-9 and np.all(np.diff(c) >= 0)


def test_ale_pdp_and_monotone_summary():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 3))
    f = lambda M: 1.5 * M[:, 0] - 0.5 * M[:, 1]            # noqa: E731
    xs, ale = L.ale_1d(f, X, 0)
    assert L.monotone_summary(ale, +1)["consistent"] is True and L.monotone_summary(ale, -1)["consistent"] is False
    pdp = L.pdp_1d(f, X, 1, np.linspace(-2, 2, 9))
    assert L.monotone_summary(pdp, -1)["consistent"] is True
    assert L.monotone_summary([0, 1, 0.5, 2, 3], 1)["consistent"] is False        # a dip that is >5% of the movement breaks monotonicity


def test_eval_only_columns_are_never_features():
    assert not set(L.EVAL_ONLY_COLS) & set(L.ALL_FEATURES)
    assert set(L.EVAL_ONLY_COLS) <= set(L.RAW_READ_COLS)


def test_adverse_action_mapping_covers_every_model_input():
    sc = J("scorecard.json")
    missing = [v for v in sc["variables"] if v not in L.ADVERSE_ACTION]
    assert not missing, missing
    assert set(L.ALL_FEATURES) <= set(L.ADVERSE_ACTION)             # every input the model could ever use has wording


@NEW
def test_calibration_tests_artifacts():
    t = J("calibration_tests.json")
    assert t["base_rho"] == 0.03 and 0.01 in t["rho_grid"] and 0.04 in t["rho_grid"]
    ms = pd.read_csv(ART / "master_scale.csv")
    for c in ["binomial_p_two_sided", "jeffreys_p_two_sided", "vasicek_p_two_rho0.03", "vasicek_p_two_rho0.01", "vasicek_p_two_rho0.04"]:
        assert ms[c].between(0, 1).all(), c
    tail = ms.binomial_p_two_sided < 0.3                                                   # in the tails more correlation can only make a gap less surprising
    assert (ms.loc[tail, "vasicek_p_two_rho0.04"] >= ms.loc[tail, "vasicek_p_two_rho0.01"] - 1e-9).all()
    assert (ms.loc[tail, "vasicek_p_two_rho0.03"] >= ms.loc[tail, "binomial_p_two_sided"] - 1e-9).all()
    ci = t["calibration_intercept_slope"][J("metrics.json")["champion"]]["oot"]
    assert -2 < ci["intercept_citl"] < 2 and 0.3 < ci["slope"] < 2
    mon = J("monitoring_triggers.json")
    names = [x["metric"] for x in mon["triggers"]]
    assert any("Vasicek" in n for n in names) and not any(n.startswith("Grades failing binomial") for n in names)
    assert any(r["status"] == "REFERENCE" and "binomial" in r["metric"] for r in mon["reference_tests"])
    fails = [x for x in mon["triggers"] if x["metric"].startswith("Grades failing correlation-adjusted")][0]["current"]
    assert fails == t["grades_failing_vasicek_5pct"] == int((ms["vasicek_p_two_rho0.03"] < 0.05).sum())


@NEW
def test_long_run_calibration_never_uses_2016():
    c = J("calibration_longrun.json")
    assert "2013-2015" in c["long_run_window"] and "never used" in c["long_run_window"]
    assert c["moc_rel_stated"] > 0 and "STATED" in c["moc_note"]
    vint = pd.read_csv(ART / "bad_rate_by_vintage.csv").set_index("issue_year")
    assert 0.02 < c["long_run_default_rate"] < 0.06
    assert min(vint.loc[2013:2015, "bad_rate_12m"]) - 0.01 <= c["long_run_default_rate"] <= max(vint.loc[2013:2015, "bad_rate_12m"]) + 0.01
    assert abs(c["target_pd_with_moc"] - c["long_run_default_rate"] * (1 + c["moc_rel_stated"])) < 1e-12
    tab = pd.read_csv(ART / "calibration_longrun_oot.csv")
    ch = c["champion"]
    sub = tab[(tab.model == ch) & tab.moc_rel.notna()].set_index("moc_rel")
    assert sub.oot_mean_pred.is_monotonic_increasing                                   # more conservatism -> higher PD
    assert c["oot_pred_over_actual_before"] != c["oot_pred_over_actual_after"] and "oot_ece_after" in c
    cal = J("calibrators_longrun.json")["calibrators_longrun_moc10"][ch]
    base = J("calibrators.json")[ch]
    assert abs(cal["b"] - base["b"]) < 1e-12 and abs(cal["a"] - base["a"] - cal["shift_vs_validation_platt"]) < 1e-9   # only the intercept moves


@NEW
def test_rejected_applications_check():
    r = J("metrics.json")["benchmark_fixes"]["reject_check"]
    assert not r.get("skipped"), "rejected_2007_to_2018Q4 must be attached"
    sh = r["accept_share_by_year"]
    assert all(0.01 < v < 0.3 for v in sh.values()) and set(sh) >= {"2013", "2016"}
    pc_ = r["parcelling"]
    assert "SENSITIVITY" in pc_["label"] and "not reject inference" in pc_["label"]
    assert "Vantage" in r["risk_score_caveat"]
    s = pd.read_csv(ART / "reject_parcelling_sensitivity.csv")
    assert list(s.k.dropna()) == [1.5, 2.0, 3.0] and s.gini_accepted_oot.between(0.1, 0.6).all()
    assert (s.implied_through_the_door_bad_rate.dropna().diff().dropna() > 0).all()    # harsher multiplier -> higher implied bad rate
    d = pd.read_csv(ART / "reject_vs_accepted_distributions.csv")
    assert {"DTI (%)", "Loan amount ($)", "Employment length (years)"} <= set(d.variable)
    assert d.year.min() == 2013 and d.year.max() == 2016
    share = pd.read_csv(ART / "accept_share_by_band.csv")
    assert share.accept_share.between(0, 1).all() and {"DTI", "Loan amount", "Employment"} <= set(share.variable)


@NEW
def test_default_reconciliation():
    r = J("default_reconciliation.json")
    assert r["charge_off_date_available"] is False
    assert abs(r["charged_off_share_last_payment_month_le_9"] + r["charged_off_share_last_payment_month_10_12"] + r["charged_off_share_last_payment_after_12"] - 1) < 1e-9
    assert r["of_which_not_in_bad_class"] <= r["loans_in_those_statuses_with_last_payment_le_9"]
    assert r["oot_bad_rate_if_added"] >= r["oot_bad_rate_main"]
    d = pd.read_csv(ART / "default_definition_sensitivity.csv")
    assert d.oot_bad_rate.iloc[:6].is_monotonic_increasing                              # a later cut-off month can only add bads


@NEW
def test_rolling_origin_backtest():
    r = pd.read_csv(ART / "rolling_origin_backtest.csv")
    assert set(r.model) == {"woe_scorecard", "xgb_monotone"} and set(r.test_year) == {2014, 2015, 2016}
    assert r.gini.between(0.15, 0.6).all() and r.ks.between(0.1, 0.5).all() and r.ece.between(0, 0.1).all()
    assert (r.train_years.str[:4] == "2013").all()
    m = J("metrics.json")
    final = r[(r.test_year == 2016) & (r.model == m["champion"])]
    if len(final):                                                                      # fold 3 re-creates the main design: Gini within a few points
        assert abs(final.gini.iloc[0] - m["oot"][m["champion"]]["gini"]) < 0.06


@NEW
def test_realised_economics_evaluation_only():
    r = J("realised_economics.json")
    if r.get("skipped"):
        pytest.skip("cash-flow columns not in this run")
    assert r["assumed"]["lgd"] == 0.85 and 0 < r["realised_lgd_on_ead_bad_loans"] < 1.0 and 0 < r["realised_ead_share_bad_loans"] <= 1.0
    assert r["realised_valid"]["loss_on_bad_loans"] > 0 and r["realised_valid"]["net_return_good_loans"] > 0
    t = pd.read_csv(ART / "realised_economics_by_grade.csv")
    assert t.loans.sum() > 1000 and t.bad_rate_12m.between(0, 1).all()
    assert not set(J("feature_schema.json")) & set(L.EVAL_ONLY_COLS)
    assert not set(J("scorecard.json")["variables"]) & set(L.EVAL_ONLY_COLS)


@NEW
def test_discrimination_ci_and_cap():
    d = pd.read_csv(ART / "discrimination_ci.csv")
    assert (d.somers_d_ci_low < d.gini_somers_d).all() and (d.gini_somers_d < d.somers_d_ci_high).all()
    assert (d.ar_minus_gini.abs() < 2e-3).all()
    m = J("metrics.json")
    o = d[(d["sample"] == "OOT 2016") & (d.model == m["champion"])].iloc[0]
    assert abs(o.gini_somers_d - m["oot"][m["champion"]]["gini"]) < 1e-6
    assert (o.somers_d_ci_high - o.somers_d_ci_low) < (0.05 if o.n >= 200_000 else 0.12)   # ~300k loans: a narrow interval (smoke runs use 40k)
    cap = pd.read_csv(ART / "cap_points.csv")
    assert cap.groupby("model").bad_share_captured.apply(lambda s: s.is_monotonic_increasing).all()


@NEW
def test_fairness_and_adverse_action():
    f = J("fairness.json")
    assert f["protected_attributes_in_data"] is False and "not a disparate-impact test" in f["caveat"]
    assert f["scorecard_variables_mapped_to_adverse_action"] == f["scorecard_variables"]
    t = pd.read_csv(ART / "fairness_income_bands.csv")
    assert t.approval_rate.between(0, 1).all() and abs(t.share_of_applicants.sum() - 1) < 1e-6
    assert abs(t.adverse_impact_ratio_vs_best_band.max() - 1) < 1e-9
    a = pd.read_csv(ART / "adverse_action_reasons_declined.csv")
    assert len(a) > 0 and a.share_of_declined_citing.between(0, 1).all()
    assert set(a.adverse_action_reason) <= set(L.ADVERSE_ACTION.values()) | {"Credit bureau information"}      # the generic fallback


@NEW
def test_model_roles_and_monotonicity_evidence():
    r = J("model_roles.json")
    assert r["decision_model_of_record"] == "woe_scorecard" and r["challenger_benchmark"] == "xgb_monotone"
    e = pd.read_csv(ART / "pdp_ale_monotonicity.csv")
    assert len(e) >= 5
    con = e[e.constraint_applied != 0]
    assert len(con) > 0 and (con.pdp_against_share < 1e-6).all(), "monotone-constrained features must have exactly monotone PDPs"
    assert (e.expected_sign.isin([-1, 0, 1])).all()
    assert (ART / "pdp_ale_curves.csv").exists()


@NEW
@pytest.mark.parametrize("name", ["calibration_longrun.png", "rolling_origin.png", "realised_economics.png", "cap_curve.png", "fairness_income.png",
                                  "reject_vs_accepted.png", "pdp_ale_top_features.png", "chargeoff_lastpay_month.png"])
def test_new_figures_exist(name):
    assert (ART / "figures" / name).exists()


def test_artifact_lib_copy_matches_source():
    if "LCPD_ART" in os.environ:
        pytest.skip("running against an external artifact folder")
    assert (ART / "lcpd_lib.py").read_text(encoding="utf-8") == (ROOT / "lcpd_lib.py").read_text(encoding="utf-8")
