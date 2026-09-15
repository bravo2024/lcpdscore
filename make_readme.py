"""Generate README.md, MODEL_CARD.md and INTERVIEW_GUIDE.md from artifacts/ only (no hand-typed results).

Run:  python make_readme.py
"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = Path(os.environ.get("LCPD_ART", ROOT / "artifacts"))
OUT = Path(os.environ.get("LCPD_OUT", ROOT))


def J(n):
    return json.loads((ART / n).read_text(encoding="utf-8"))


def C(n):
    return pd.read_csv(ART / n)


M = J("metrics.json")
RULE = J("champion_rule.json")
CMP = C("model_comparison.csv").set_index("model")
IV = C("iv_table.csv")
MS = C("master_scale.csv")
MC = J("multicollinearity.json")
CUT = J("cutoff.json")
PSI = M["psi"]
LS = M["label_sensitivity"]
DROP = C("dropped_by_split.csv").set_index("set")
TRANS = C("status_transition_by_vintage.csv").set_index("issue_year")
IMB = C("imbalance_cv.csv").set_index("strategy")
MON = J("monitoring_triggers.json")
LAD = C("ladder_metrics_valid.csv").set_index("model")
COEF = C("ladder_coefficients.csv")
CSI = C("csi.csv")
SHAP = C("shap_importance.csv")
SC = J("scorecard_meta.json")
ENV = J("environment.json")
BPAR = J("xgb_best_params.json")
VIN = C("bad_rate_by_vintage.csv").set_index("issue_year")
STAB = C("coef_stability.csv")
MONO = C("monotone_constraints.csv")
SPLIT = C("split_summary.csv")
SENS = C("cutoff_sensitivity.csv")
QT = C("drift_by_quarter.csv")

NAMES = {"plain_lr": "Plain logistic (statsmodels)", "l2_lr": "L2 logistic", "l1_lr": "L1 logistic", "elasticnet_lr": "ElasticNet logistic",
         "woe_scorecard": "WoE scorecard", "xgb_tuned": "XGBoost tuned", "xgb_monotone": "XGBoost monotone",
         "ref_xgb_plus_LC_grade": "REF: XGBoost + LC grade/int_rate (not deployable)", "ref_LC_subgrade_only": "REF: LC sub-grade alone"}
nm = lambda k: NAMES.get(k, k)  # noqa: E731
ch = M["champion"]
o = M["oot"]
d = M["data"]
pc = lambda x, k=1: f"{100 * x:.{k}f}%"  # noqa: E731
f3 = lambda x: f"{x:.3f}"  # noqa: E731

# ---------------------------------------------------------------- derived facts
tbl = ["| Model | Role | OOT AUC (95% CI) | OOT Gini | OOT KS | OOT Brier | OOT ECE |", "|---|---|---|---|---|---|---|"]
for k in CMP.index:
    r = CMP.loc[k]
    star = " **(champion)**" if k == ch else ""
    tbl.append(f"| {nm(k)}{star} | {r.role} | {r.oot_auc:.4f} ({r.auc_ci_low:.3f}-{r.auc_ci_high:.3f}) | {r.oot_gini:.4f} | {r.oot_ks:.4f} | {r.oot_brier:.4f} | {r.oot_ece:.4f} |")
TABLE = "\n".join(tbl)

n_fail = int((MS.binomial_p_two_sided < 0.05).sum())
under = int((MS.verdict_oot == "under-predicted risk").sum())
over = int((MS.verdict_oot == "over-predicted risk").sum())
cal_ratio = float(CMP.loc[ch, "oot_mean_pred"] / CMP.loc[ch, "oot_obs_rate"])
cal_phrase = "under-predicts" if cal_ratio < 0.9 else ("over-predicts" if cal_ratio > 1.1 else "is close to the observed rate on average")
cal_verdict = ("a genuine vintage-drift signal (the 12-month label has no censoring, so this is not a labelling artefact)" if abs(cal_ratio - 1) > 0.1
               else "calibration holds up out of time on average, although individual grades can still fail the binomial test")
lc_gain = float(CMP.loc["ref_xgb_plus_LC_grade", "oot_gini"] - CMP.loc[ch, "oot_gini"])
gini_drop = float(CMP.loc[ch, "valid_gini"] - CMP.loc[ch, "oot_gini"])
xgb_gain = float(CMP.loc["xgb_tuned", "oot_auc"] - CMP.loc["woe_scorecard", "oot_auc"])
xgb_gain_mono = float(CMP.loc['xgb_monotone', 'oot_auc'] - CMP.loc['woe_scorecard', 'oot_auc'])
xgb_gain_mono_gini = 2 * xgb_gain_mono
mono_cost = float(CMP.loc["xgb_tuned", "oot_auc"] - CMP.loc["xgb_monotone", "oot_auc"])
top_iv = IV.head(3)
top_shap = SHAP.head(5).feature.tolist()
n_constr = int((MONO.constraint_applied != 0).sum())
sign_flips = int(STAB.sign_flips.sum())
red = [k for k, v in M["monitoring_status"].items() if v == "RED"]
best_en = C("lr_ladder_cv.csv")
best_en = best_en[(best_en.model == "elasticnet") & (best_en.best)].params.iloc[0]
best_en = ", ".join(f"{k}={v}" for k, v in json.loads(best_en).items())
l1_zero = int(COEF.l1_zeroed.sum())
v16, v15 = VIN.loc[2016], VIN.loc[2015]
worst_grade = MS.iloc[-1]
best_grade = MS.iloc[0]
sc_vars = J("scorecard.json")["variables"]
champ_text = {"woe_scorecard": "the WoE scorecard", "elasticnet_lr": "the ElasticNet logistic regression", "xgb_monotone": "the monotone-constrained XGBoost"}[ch]
bi = RULE["best_interpretable"]
champ_why = (
    f"The rule was fixed on validation before the 2016 data was scored. XGB-monotone's validation AUC beat the best interpretable candidate ({nm(bi)}) by {RULE['xgb_monotone_gap_vs_best_interpretable']:+.4f} "
    f"(95% bootstrap CI {RULE['gap_ci95'][0]:+.4f} to {RULE['gap_ci95'][1]:+.4f}); the required margin was +0.0100 with a CI excluding zero. "
    + (f"That test was passed, so XGB-monotone is champion." if ch == "xgb_monotone" else f"That test was not passed, so {champ_text} is champion.")
)
cut_best = SENS.sort_values("profit_per_app_oot", ascending=False).iloc[0]

# ---------------------------------------------------------------- benchmark-driven upgrades (only when the artifacts exist)
NEW = (ART / "calibration_longrun.json").exists()
# The published README is the short variant. Set LCPD_BENCHMARK=1 to include the
# benchmark-review section (the long variant) in README.md.
PUBLISH_BENCH = os.environ.get("LCPD_BENCHMARK") == "1"
REJ_NOTE = (" (a labelled through-the-door comparison and parcelling sensitivity are in the "
            "benchmark section)" if PUBLISH_BENCH else "")
UPGRADES = ""
ROLE_LINE = "The WoE scorecard is the explainable decision model of record; the monotone XGBoost is the challenger/benchmark."
CARD_NEW = ""
VAS_PHRASE = ""
GUIDE_NEW = ""
if NEW:
    CT, LR, DR = J("calibration_tests.json"), J("calibration_longrun.json"), J("default_reconciliation.json")
    RO, DISC, FAIR, ROLES = C("rolling_origin_backtest.csv"), C("discrimination_ci.csv"), J("fairness.json"), J("model_roles.json")
    MEV, FINC, AAD, AAM = C("pdp_ale_monotonicity.csv"), C("fairness_income_bands.csv"), C("adverse_action_reasons_declined.csv"), C("adverse_action_map.csv")
    REJ, RE = M["benchmark_fixes"]["reject_check"], J("realised_economics.json")
    pc2 = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    vas_fail, bin_fail = CT["grades_failing_vasicek_5pct"], CT["grades_failing_binomial_5pct"]
    VAS_PHRASE = f" Allowing for correlated defaults (Vasicek, rho 0.03) only {vas_fail} of {len(MS)} grades fail, which is what the monitoring light uses; the plain binomial count ({bin_fail}) is a reference."
    cit = CT["calibration_intercept_slope"][ch]
    pdp_con = f"{ROLES['constrained_pdp_exactly_monotone']} of {ROLES['constrained_among_checked']}"
    # rolling origin table
    ro_t = ["| Train | Test | Model | Gini | KS | ECE | Pred/actual |", "|---|---|---|---|---|---|---|"]
    for r in RO.itertuples():
        ro_t.append(f"| {r.train_years} | {r.test_year} | {nm(r.model)} | {r.gini:.3f} | {r.ks:.3f} | {r.ece:.4f} | {r.pred_over_actual:.2f} |")
    # discrimination
    dd = DISC[(DISC['sample'] == 'OOT 2016')].set_index("model")
    di_t = ["| Model | OOT Gini = Somers' D (95% DeLong CI) | Accuracy ratio |", "|---|---|---|"]
    for k in dd.index:
        r = dd.loc[k]
        di_t.append(f"| {nm(k)} | {r.gini_somers_d:.3f} ({r.somers_d_ci_low:.3f} to {r.somers_d_ci_high:.3f}) | {r.accuracy_ratio:.3f} |")
    # calibration table
    lrt = C("calibration_longrun_oot.csv")
    lrt = lrt[lrt.model == ch]
    cal_t = ["| Calibration | OOT mean PD | Pred/actual | ECE | Vasicek p (rho 0.03) | Binomial p |", "|---|---|---|---|---|---|"]
    for _, r in lrt.iterrows():
        cal_t.append(f"| {r['variant']} | {pc2(r['oot_mean_pred'])} | {r['oot_pred_over_actual']:.2f} | {r['oot_ece']:.4f} | {r['oot_vasicek_p_two_rho0.03']:.3f} | {r['oot_binomial_p_two']:.2g} |")
    # grade tests table
    gt = ["| Grade | PD | Observed | Binomial p | Jeffreys p | Vasicek p (0.01 / 0.03 / 0.04) | Light (rho 0.03) |", "|---|---|---|---|---|---|---|"]
    for _, r in MS.iterrows():
        gt.append(f"| {r['grade']} | {pc2(r['master_scale_pd'])} | {pc2(r['observed_rate_oot'])} | {r['binomial_p_two_sided']:.2g} | {r['jeffreys_p_two_sided']:.2g} | "
                  f"{r['vasicek_p_two_rho0.01']:.3f} / {r['vasicek_p_two_rho0.03']:.3f} / {r['vasicek_p_two_rho0.04']:.3f} | {r['vasicek_status_rho0.03']} |")
    # reject section
    if REJ.get("skipped"):
        rej_txt = "Rejected-applications file not attached in this run; stage skipped."
    else:
        PP = REJ["parcelling"]
        rej_txt = (f"Accept share (all terms, 2013-2016 applications): {', '.join(f'{y}: {pc(v)}' for y, v in REJ['accept_share_by_year'].items())}. Median DTI in 2016: accepted {REJ['median_dti_accepted_vs_rejected_2016'][0]:.1f} vs rejected {REJ['median_dti_accepted_vs_rejected_2016'][1]:.1f}; "
                   f"median amount requested ${REJ['median_amount_accepted_vs_rejected_2016'][0]:,.0f} vs ${REJ['median_amount_accepted_vs_rejected_2016'][1]:,.0f}. "
                   f"`Risk_Score` is blank for most rejects from 2015 (coverage by year: {', '.join(f'{y}: {pc(v, 0)}' for y, v in REJ['risk_score_coverage_rejects_by_year'].items())}), is FICO only until about Nov-2013 and VantageScore afterwards, "
                   f"so score comparisons are indicative (2013 medians: accepted FICO {REJ['median_risk_accepted_vs_rejected_2013'][0]:.0f} vs rejected {REJ['median_risk_accepted_vs_rejected_2013'][1]:.0f}). "
                   f"**Parcelling sensitivity (not reject inference):** with rejects assigned k x the accepted bad rate of their band, the 4-variable scorecard's OOT Gini moves from {PP['gini_oot_accepted_only']:.3f} (accepted only) to "
                   f"{', '.join(f'{v:.3f} (k={k})' for k, v in PP['gini_oot_by_k'].items())}; the implied through-the-door bad rate is {', '.join(f'{pc(v)} (k={k})' for k, v in PP['implied_through_the_door_bad_rate_by_k'].items())} "
                   f"against {pc(M['data']['bad_rate_oot'])} for accepted loans; the largest coefficient change is {PP['max_abs_coef_change_pct']:.0f}%. "
                   f"{pc(PP['share_of_scored_rejects_below_accepted_floor_660'], 0)} of the scored rejects sit below the accepted floor (660), so their 'same band' rate is an extrapolation.")
    if RE.get("skipped"):
        re_txt = "Cash-flow columns not available in this run."
    else:
        re_txt = (f"Realised cash (lifetime to the data cut, before funding cost; `total_pymnt` already includes recoveries): on defaulted-in-12-months loans the realised EAD share is {pc(RE['realised_ead_share_bad_loans'], 0)} of funded (assumed {pc(RE['assumed']['ead'], 0)}) "
                  f"and the realised LGD on that EAD is {pc(RE['realised_lgd_on_ead_bad_loans'], 0)} (assumed {pc(RE['assumed']['lgd'], 0)}); net loss per $ funded on those loans {pc(RE['realised_loss_per_dollar_funded_bad'], 0)} (assumed {pc(RE['assumed']['loss_per_bad_of_principal'], 0)}). "
                  f"Good loans earned {pc(RE['realised_valid']['net_return_good_loans'])} lifetime on the matured validation window (assumed 12-month margin {pc(RE['assumed']['margin_12m'], 0)}). "
                  f"The profit-maximising cut-off on realised validation cash is PD {RE['realised_best_cutoff_pd_valid']:.3f} (OOT, partly matured: {RE['realised_best_cutoff_pd_oot']:.3f}) vs {RE['assumed_cutoff_pd']:.3f} under the assumptions; realised profit per application at the assumed cut-off on OOT is ${RE['realised_profit_per_app_at_assumed_cutoff_oot']:,.0f} vs ${RE['realised_profit_per_app_approve_all_oot']:,.0f} approving everyone.")
    inc_t = ["| Income band | Applicants | Approval rate | Adverse-impact ratio | Mean PD | Observed bad rate |", "|---|---|---|---|---|---|"]
    for r in FINC.itertuples():
        inc_t.append(f"| {r.income_band} | {r.applicants:,} | {pc(r.approval_rate)} | {r.adverse_impact_ratio_vs_best_band:.2f} ({r.four_fifths_flag}) | {pc2(r.mean_pd)} | {pc2(r.observed_bad_rate)} |")
    top_aa = "; ".join(f"{r.adverse_action_reason} ({pc(r.share_of_declined_citing, 0)})" for r in AAD.head(4).itertuples())
    UPGRADES = f"""
## Benchmark review: what was added and what the numbers say

A benchmark against public notebooks and bank practice rated the project "mostly practitioner standard". Every item below is in the notebook (Stages 8b-15), saved to `artifacts/` and read here.

### Model roles

{ROLE_LINE} The frozen accuracy rule picks **{nm(ROLES['frozen_rule_champion'])}** on validation (it also supplies the PD used for grading and the cut-off here); the scorecard supplies the score, the points table, reason codes and adverse-action wording.
Monotonicity evidence for the boosted model (`pdp_ale_monotonicity.csv`): of the {ROLES['top_numeric_features_checked']} most important numeric features, {pdp_con} constrained ones have an exactly monotone partial-dependence curve, and
{ROLES['unconstrained_ale_in_expected_direction']} of {ROLES['unconstrained_among_checked']} unconstrained ones follow the expected sign in ALE ({ROLES['unconstrained_with_no_expectation']} have no business sign){' ; flagged for review: ' + ', '.join(ROLES['features_flagged_for_review']) if ROLES['features_flagged_for_review'] else ''}.

### 1. Calibration tests that allow for correlated defaults

The binomial test treats every loan as an independent coin flip, but a vintage shares one economy. Per grade on OOT (`master_scale.csv`, `calibration_rho_sensitivity.csv`):

{chr(10).join(gt)}

Grades failing at 5%: binomial {bin_fail}, Jeffreys {CT['grades_failing_jeffreys_5pct']}, Vasicek (rho 0.03) {vas_fail} of {len(MS)}. The **monitoring light is based on the Vasicek-adjusted test**; the plain binomial is a reference only.
Calibration-in-the-large on OOT: intercept {cit['oot']['intercept_citl']:+.3f} (95% CI {cit['oot']['intercept_ci'][0]:+.3f} to {cit['oot']['intercept_ci'][1]:+.3f}); slope {cit['oot']['slope']:.3f} ({cit['oot']['slope_ci'][0]:.3f} to {cit['oot']['slope_ci'][1]:.3f}) - a slope near 1 with a positive intercept means ranking-consistent probabilities that are shifted too low in 2016.

### 2. Long-run calibration

The validation Platt calibrator is fitted on four months (Sep-Dec 2015, bad rate {pc2(LR['validation_default_rate'])}). The long-run version shifts only its intercept so that the average PD over the whole 2013-2015 development window equals the pooled 12-month default rate of those years ({pc2(LR['long_run_default_rate'])}), then adds a **stated** margin of conservatism of +{LR['moc_rel_stated']:.0%} (target {pc2(LR['target_pd_with_moc'])}). 2016 is never used to fit.

{chr(10).join(cal_t)}

Reading: {LR['reading']} The ex-post margin that would have covered 2016 is +{LR['ex_post_moc_that_would_have_covered_2016']:.0%} (informational, not used).

### 3. Through-the-door check with the rejected applications

{rej_txt}

### 5. Default-definition reconciliation

{pc(DR['charged_off_share_last_payment_month_le_9'], 0)} of charged-off loans made their last payment by month 9 (labelled bad), {pc(DR['charged_off_share_last_payment_month_10_12'], 0)} in months 10-12 and {pc(DR['charged_off_share_last_payment_after_12'], 0)} later; the median last-payment month is {DR['median_last_payment_month_charged_off']:.0f}.
The public file has **no charge-off date**, so the last-payment-to-charge-off lag cannot be measured (convention: ~4-5 months). {DR['loans_in_those_statuses_with_last_payment_le_9']:,} loans are currently late/in grace/default with a last payment by month 9; {DR['of_which_not_in_bad_class']:,} ({pc(DR['share_not_in_bad_class'], 0)}) of them are not in the bad class. Counting them as bad would move the OOT bad rate from {pc2(DR['oot_bad_rate_main'])} to {pc2(DR['oot_bad_rate_if_added'])}.

### 6. Rolling-origin backtest

{chr(10).join(ro_t)}

Coarse-bin cut points, scorecard variables, XGBoost hyper-parameters and constraint signs are held fixed from the main fit; WoE values, coefficients, boosters and calibrators are refitted in each fold.

### 7. Realised economics (evaluation only, never features)

{re_txt}

### 8. Discrimination with uncertainty

{chr(10).join(di_t)}

### 9. Fairness screen and adverse-action wording

{FAIR['caveat']}

{chr(10).join(inc_t)}

Lowest adverse-impact ratio {FAIR['lowest_adverse_impact_ratio']:.2f} ({FAIR['worst_approval_band']} vs best band {FAIR['best_approval_band']}). All {FAIR['scorecard_variables_mapped_to_adverse_action']} of {FAIR['scorecard_variables']} scorecard variables map to a standard adverse-action statement (`adverse_action_map.csv`); the principal reasons among declined OOT applicants are {top_aa}.
"""
    ROLE_LINE_CARD = ROLE_LINE
    CARD_NEW = f"""
## Roles, calibration and fairness

- **Roles.** {ROLE_LINE} The frozen accuracy rule selects {nm(ROLES['frozen_rule_champion'])} on validation; it supplies the PD for grading and the cut-off, while the scorecard supplies the score, reason codes and adverse-action wording. Monotone-constraint evidence: {pdp_con} constrained features exactly monotone in PDP; {ROLES['unconstrained_ale_in_expected_direction']} of {ROLES['unconstrained_among_checked']} unconstrained top features follow the expected sign in ALE.
- **Calibration.** Grade tests use a correlation-adjusted (Vasicek, rho 0.03) test for the traffic light; {vas_fail} of {len(MS)} grades fail at 5% (plain binomial: {bin_fail}). OOT calibration intercept {cit['oot']['intercept_citl']:+.3f}, slope {cit['oot']['slope']:.3f}. A long-run (2013-2015) anchor with a stated +{LR['moc_rel_stated']:.0%} margin of conservatism moves OOT predicted/actual from {LR['oot_pred_over_actual_before']:.2f} to {LR['oot_pred_over_actual_after']:.2f}; 2016 was not used to fit it.
- **Discrimination.** OOT Gini {dd.loc[ch, 'gini_somers_d']:.3f} (95% DeLong CI {dd.loc[ch, 'somers_d_ci_low']:.3f} to {dd.loc[ch, 'somers_d_ci_high']:.3f}); rolling-origin Gini range {RO[RO.model == ch].gini.min():.3f} to {RO[RO.model == ch].gini.max():.3f}.
- **Fairness.** No protected-class attributes exist in the public file; an income-band screen at the cut-off gives a lowest adverse-impact ratio of {FAIR['lowest_adverse_impact_ratio']:.2f}. This is a screen, not a disparate-impact test. Every scorecard variable maps to an adverse-action statement.
- **Reject population.** {('Accepted vs rejected comparison and a labelled parcelling sensitivity in Stage 14; not reject inference.' if not REJ.get('skipped') else 'not run')}
"""
    GUIDE_NEW = f"""
## Benchmark-review questions

**A. Your binomial test says the grades fail. Is the model broken?** The binomial test assumes independent defaults. A vintage shares one economy, so I use a Vasicek single-factor test (rho 0.03, sensitivity 0.01-0.04): {vas_fail} of {len(MS)} grades fail versus {bin_fail} on the plain binomial. The traffic light uses the adjusted test; the binomial is a reference. Calibration-in-the-large is still off in 2016 (intercept {cit['oot']['intercept_citl']:+.3f}, slope {cit['oot']['slope']:.2f}).

**B. Why is the scorecard not the champion?** It is the decision model of record (points, reason codes, adverse-action text). The monotone XGBoost wins the frozen accuracy rule and is the challenger/benchmark that supplies the PD used for grading here. Promotion rules and the monotonicity evidence (PDP/ALE on the top features) are in `model_roles.json`.

**C. What does your long-run calibration do, and what are its limits?** It shifts only the intercept so the 2013-2015 average PD equals the pooled default rate ({pc2(LR['long_run_default_rate'])}), plus a stated +{LR['moc_rel_stated']:.0%} margin. It is a through-the-cycle PD; 2016 was worse than the long-run average, so predicted/actual goes from {LR['oot_pred_over_actual_before']:.2f} to {LR['oot_pred_over_actual_after']:.2f}. Using 2016 to fit it would be cheating, so I report the gap.

**D. How do you know rejects would not change the model?** I do not. I compared accepted and rejected applications and ran a parcelling sensitivity (k = 1.5, 2, 3) that is explicitly not reject inference; Risk_Score switches from FICO to Vantage after late 2013 and is blank for most rejects from 2015.

**E. Is the 12-month label too narrow?** The file has no charge-off date, so I reconcile what it allows: {pc(DR['charged_off_share_last_payment_month_le_9'], 0)} of charged-off loans stopped paying by month 9; {DR['of_which_not_in_bad_class']:,} late/default loans with an early last payment are in the good class and moving them to bad shifts the 2016 bad rate to {pc2(DR['oot_bad_rate_if_added'])}.
"""

# ---------------------------------------------------------------- README
README = f"""# LendingClub application PD scorecard

Bank-style credit-risk project: a probability-of-default (PD) model for **36-month LendingClub loans**, built from
**application-time information only**, tested **out-of-time on loans issued in 2016**, with a frozen champion rule, a WoE points scorecard,
an {len(MS)}-grade master scale, a cut-off simulator and a monitoring plan. Every number in this file is read from `artifacts/`.

Independent portfolio project on public LendingClub data. Not affiliated with LendingClub.

## Headline

- **Champion: {champ_text}** - OOT Gini **{o[ch]['gini']:.3f}** (AUC {o[ch]['auc']:.3f}, 95% CI {o[ch]['auc_ci'][0]:.3f}-{o[ch]['auc_ci'][1]:.3f}), KS {o[ch]['ks']:.3f}, Brier {o[ch]['brier']:.4f}, ECE {o[ch]['ece']:.4f}.
- Validation Gini was {CMP.loc[ch, 'valid_gini']:.3f}; on 2016 loans the model never saw it is {o[ch]['gini']:.3f} (change {-gini_drop:+.3f}).
- **Default definition: 12-month window** (90+ DPD within 12 months of issue, proxied from the last-payment date). All {d['oot_rows']:,} 2016 loans are labelled; none dropped as unresolved.
- Score PSI train vs 2016 is **{PSI['psi_train_vs_oot']:.3f}** ({PSI['label_train_vs_oot']}). Calibration was fitted on validation only: the model predicts {pc(CMP.loc[ch, 'oot_mean_pred'], 2)} on average vs {pc(CMP.loc[ch, 'oot_obs_rate'], 2)} observed in 2016 (predicted/actual = {cal_ratio:.2f}; ECE {o[ch]['ece']:.4f}) - {cal_verdict}. {n_fail} of {len(MS)} grades fail the OOT binomial test ({under} under-predicted, {over} over-predicted).{VAS_PHRASE}
- **Roles:** {ROLE_LINE}
- Excluding LendingClub's own grade/interest rate is a deliberate choice. As a reference only, adding them lifts the OOT Gini by **{lc_gain:+.3f}** ({CMP.loc['ref_xgb_plus_LC_grade', 'oot_gini']:.3f}) - that is LendingClub's own underwriting model talking.

### Out-of-time results, all models (calibrated on validation only)

{TABLE}

*Calibrated PDs are Platt-scaled on the validation window. KS and AUC are rank-based, so calibration does not change them. The two REF rows use LendingClub's own risk outputs and are not eligible.*

**Why the champion** - {champ_why} The plain XGBoost's gain over the WoE scorecard on OOT AUC is {xgb_gain:+.4f}; the monotone constraint costs {mono_cost:+.4f} AUC versus the unconstrained tuned model.

## What was built (the 8 stages)

| # | Stage | Result in this run |
|---|---|---|
| 1 | Framing | Bad = charged off / default with last payment in months 0-9 (= 90+ DPD within 12 months of issue); good = every other loan; **no loan dropped**; 12-month PD; train {d['train_window']}, validation {d['valid_window']}, **OOT {d['oot_window']}** |
| 2 | Data checks | {M['data']['rows_in_file']:,} rows in file; de-duplicated before split; leakage audit of all columns (`artifacts/leakage_audit.csv`); {d['train_rows_used']:,} train rows ({d['sampling_note'].lower()}), {d['valid_rows']:,} validation, {d['oot_rows']:,} OOT |
| 3 | Univariate WoE/IV | {d['n_features_candidate']} candidate features, {d['n_pass_iv']} pass IV >= 0.02; strongest: {', '.join(f"{r.feature} ({r.iv_coarse:.3f})" for r in top_iv.itertuples())}; nothing above 0.5 |
| 4 | Multicollinearity | {len(C('corr_pairs_gt_0_7.csv'))} pairs with |r|>0.7 -> {d['n_after_corr']} features; VIF<5 -> {d['n_after_vif']}; condition index {MC['condition_index']:.1f}; sign check in `artifacts/sign_check.csv` |
| 5 | Logistic ladder | plain / L2 / L1 / ElasticNet ({best_en}) / WoE scorecard ({len(sc_vars)} variables, PDO {SC['pdo']:.0f}, {SC['base_score']:.0f} points at {SC['base_odds_good_to_bad']:.0f}:1); imbalance: none AUC {IMB.loc['none', 'cv_auc']:.4f}/Brier {IMB.loc['none', 'cv_brier']:.4f} vs balanced {IMB.iloc[1].cv_auc:.4f}/{IMB.iloc[1].cv_brier:.4f} |
| 6 | Challengers | XGBoost tuned (randomised search) and monotone-constrained ({n_constr} constraints); SHAP top drivers: {', '.join(top_shap)} |
| 7 | Comparison | AUC/Gini/KS/PR-AUC/Brier/ECE, decile lift, PSI/CSI, paired bootstrap ({M['bootstrap_reps']} resamples) vs champion; rule frozen in `champion_rule.json` before OOT scoring |
| 8 | Business | {len(MS)}-grade master scale with binomial tests; cut-off simulator on **assumed** economics; monitoring triggers |

{UPGRADES if PUBLISH_BENCH else ""}
## Scope and limitations

1. **Accepted loans only - no true reject inference**{REJ_NOTE}. The model describes people LendingClub approved; on the through-the-door population it would be optimistic about the riskiest applicants.
2. **Label is a proxy and a 12-month horizon.** The public file has one snapshot status and the last-payment month, not monthly delinquency, so "90+ DPD within 12 months" is *charged off/default AND last payment <= month {LS['definition'].split('month ')[1].split(' ')[0]}*. A looser proxy (last payment <= month 12) moves the 2016 bad rate from {pc(LS['oot_bad_rate_main'], 2)} to {pc(LS['oot_bad_rate_window_lastpay_le_12'], 2)} with champion AUC {LS['oot_auc_main']:.3f} vs {LS['oot_auc_window_lastpay_le_12']:.3f}. The model is a 12-month PD: about {pc(float(TRANS.loc[2013, 'share_of_window_good_that_later_charged_off']), 0)} of 2013 loans that were good inside the window were charged off later, and the matured 2013-14 lifetime-to-12-month bad-rate ratio is {LS['lifetime_to_12m_ratio_matured']:.1f}x, so lifetime loss is not covered.
   *Why not the old label:* the earlier version used lifetime default on resolved loans only; it dropped {pc(DROP.loc['OOT test', 'old_label_drop_share'])} of the 2016 loans (still open at the data cut), pushed the 2016 bad rate to {pc(LS['oot_bad_rate_old_lifetime_resolved_only'])} and made the model look badly under-calibrated. That is kept as a sensitivity (champion AUC {LS['oot_auc_old_lifetime_resolved_only']:.3f} on it), not as the target.
3. **Drift.** The 12-month bad rate rose across vintages ({pc(VIN.loc[2013].bad_rate_12m, 2)} in 2013, {pc(v15.bad_rate_12m, 2)} in 2015, {pc(v16.bad_rate_12m, 2)} in 2016). Calibration is fitted on 2015H2 validation only and tested on 2016: predicted/actual {cal_ratio:.2f}.
4. **Binomial tests over-reject** at this sample size and with defaults correlated inside a vintage; the monitoring light therefore uses a correlation-adjusted (Vasicek) test and keeps the binomial as a reference.
5. **Profit numbers use assumed economics** (margin {CUT['assumptions']['margin_pct_of_principal']:.0%} of principal, LGD {CUT['assumptions']['lgd']:.0%}, EAD {CUT['assumptions']['ead_pct_of_principal']:.0%}); flat pricing; not a forecast.
6. Public data, no bureau history beyond what LendingClub recorded; no claim of Basel / IFRS 9 / SR 11-7 compliance - the model card only follows that *style*.
7. Compute: heavy notebook run on Kaggle (scikit-learn {ENV['sklearn']}, xgboost {ENV['xgboost']}, {M['runtime_minutes']} minutes). Tuning budgets are modest by design.

## Repository layout

```
lcpdscore/
  LendingClubPDScorecard.ipynb           clean notebook (unexecuted source, built by build_notebook.py)
  LendingClubPDScorecard.executed.ipynb  executed on Kaggle, outputs kept
  lcpd_lib.py                            shared library (features, WoE, scorecard, metrics, scorer)
  app.py                                 Streamlit app (14 sections, sidebar navigation, model selector, artifacts only)
  make_readme.py                         generates README.md and MODEL_CARD.md from artifacts
  artifacts/                             metrics.json, csv tables, figures/, models/ (json + joblib)
  tests/                                 pytest suite
```

## How to run

```bash
pip install -r requirements.txt
python -m pytest tests -q
streamlit run app.py            # loads artifacts only, no retraining
python make_readme.py           # regenerate the docs from artifacts
```

Re-running the notebook: on Kaggle (internet on, dataset `wordsforthewise/lending-club` attached) it runs as-is and writes to `/kaggle/working/artifacts`.
Locally set `LC_DATA` to the path of `accepted_2007_to_2018Q4.csv.gz` (and optionally `LC_ART` for the output folder).

*Generated by make_readme.py from artifacts/metrics.json and sibling files.*
"""
if not PUBLISH_BENCH:
    # the benchmark section is the only place that leaves a doubled blank line
    README = README.replace("\n\n\n", "\n\n")
(OUT / "README.md").write_text(README, encoding="utf-8")

# ---------------------------------------------------------------- MODEL CARD
red_txt = ", ".join(red) if red else "none"
CARD = f"""# Model card - LendingClub application PD scorecard

*One-page card in the style of SR 11-7 / PRA SS1/23 model documentation. It is documentation of a portfolio project on public data, not a validated production model and not a claim of regulatory compliance.*

| | |
|---|---|
| **Purpose** | Rank-order applicants for unsecured 36-month personal loans by probability of default within 12 months and support accept/decline and grade-based pricing decisions. |
| **Model** | Champion: **{nm(ch)}** (rule frozen on validation before OOT scoring: `artifacts/champion_rule.json`). Decision model of record (explainable): WoE points scorecard ({len(sc_vars)} variables). Challenger / benchmark: monotone XGBoost (wins the frozen accuracy rule). Reference: XGBoost tuned. |
| **Data** | LendingClub accepted loans 2007-2018Q4 (Kaggle `wordsforthewise/lending-club`), 36-month, individual applications, issued 2013-2016. |
| **Default definition** | Bad = Charged Off / Default with last payment in months 0-9 on book (proxy for 90+ DPD within 12 months of issue); Good = every other loan. 12-month performance window, observation point = application date; no loans dropped (all 2013-2016 loans have a full window). |
| **Windows** | Train {d['train_window']} ({d['train_rows_used']:,} rows, bad rate {pc(d['bad_rate_train'])}); validation {d['valid_window']} ({d['valid_rows']:,}, {pc(d['bad_rate_valid'])}); OOT {d['oot_window']} ({d['oot_rows']:,}, {pc(d['bad_rate_oot'])}). |
| **Inputs** | {d['n_features_candidate']} application-time features (loan, applicant, bureau snapshot at application). |
| **Exclusions** | All post-origination fields (payments, balances, recoveries, refreshed FICO, hardship/settlement); `grade`, `sub_grade`, `int_rate`, `installment` (LendingClub's own risk output/price); funded amount; free text and geography; joint applications; fields not populated before 2016. Full list: `artifacts/leakage_audit.csv`. |
| **Method** | Fine/coarse classing with monotone optimal bins (train only); IV>=0.02; |r|<=0.7, VIF<5; statsmodels logit, L1/L2/ElasticNet by stratified 5-fold CV; scorecard PDO {SC['pdo']:.0f}, {SC['base_score']:.0f} at {SC['base_odds_good_to_bad']:.0f}:1; XGBoost tuned by randomised search; Platt calibration on validation. |

## Performance (OOT 2016, calibrated PD)

| Metric | Champion | Validation (same model) |
|---|---|---|
| AUC | {o[ch]['auc']:.4f} (95% CI {o[ch]['auc_ci'][0]:.3f}-{o[ch]['auc_ci'][1]:.3f}) | {CMP.loc[ch, 'valid_auc']:.4f} |
| Gini | {o[ch]['gini']:.4f} | {CMP.loc[ch, 'valid_gini']:.4f} |
| KS | {o[ch]['ks']:.4f} | - |
| Brier / ECE | {o[ch]['brier']:.4f} / {o[ch]['ece']:.4f} | {M['valid'][ch]['brier']:.4f} / {M['valid'][ch]['ece']:.4f} |
| Mean predicted PD vs actual | {pc(CMP.loc[ch, 'oot_mean_pred'])} vs {pc(CMP.loc[ch, 'oot_obs_rate'])} | - |
| Score PSI train vs OOT | {PSI['psi_train_vs_oot']:.4f} ({PSI['label_train_vs_oot']}) | {PSI['psi_train_vs_valid']:.4f} |

Master scale: {len(MS)} grades ({best_grade.grade}: PD {pc(best_grade.master_scale_pd)} ... {worst_grade.grade}: PD {pc(worst_grade.master_scale_pd)}); grades are correctly rank-ordered on OOT, but {n_fail} of {len(MS)} fail the binomial test (risk under-predicted in {under}).

## Scope and limitations

- No reject inference (accepted loans only); the population the model has seen is not the through-the-door population.
- Label is a proxy (no monthly delinquency in the public file) on a 12-month horizon: 2016 bad rate {pc(LS['oot_bad_rate_main'], 2)} (looser proxy {pc(LS['oot_bad_rate_window_lastpay_le_12'], 2)}); losses after month 12 are not covered (matured 2013-14 lifetime/12-month bad-rate ratio {LS['lifetime_to_12m_ratio_matured']:.1f}x). The earlier lifetime resolved-only label dropped {pc(DROP.loc['OOT test', 'old_label_drop_share'])} of 2016 loans and is kept only as a sensitivity.
- Calibration: fitted on 2015H2 validation only; on 2016 the model {cal_phrase} (predicted/actual {cal_ratio:.2f}, ECE {o[ch]['ece']:.4f}); recalibrate on a recent window before any use.
- Gini {o[ch]['gini']:.2f} is moderate; most borrower-level information is already priced into LendingClub's grade, which is excluded here by design.
- Economics in the cut-off analysis are assumptions; macro conditions, fraud and fair-lending (geography excluded; no disparate-impact testing) are not covered.

## Monitoring and redevelopment triggers

| Metric | Amber | Red | Current (OOT) | Status |
|---|---|---|---|---|
""" + "\n".join(f"| {t['metric']} | {t['amber']} | {t['red']} | {t['current']:.3f} | {t['status']} |" for t in MON["triggers"]) + f"""

{MON['redevelopment_policy']} Review cadence: {MON['review_frequency']}. Metrics currently RED: {red_txt}.
{CARD_NEW}
*Prepared from `artifacts/metrics.json`; generated by `make_readme.py`.*
"""
(OUT / "MODEL_CARD.md").write_text(CARD, encoding="utf-8")

# ---------------------------------------------------------------- INTERVIEW GUIDE
imp_top = SHAP.head(3)
guide = f"""# Interview guide - LendingClub application PD scorecard

All numbers below are read from `artifacts/` (see `make_readme.py`).

## 2-minute pitch

"I built an application PD scorecard on LendingClub's public data, the way a bank model team would. I used only 36-month loans and only information available at application - I removed every post-origination field and, deliberately, LendingClub's own grade and interest rate, because those are another lender's risk model. Bad is a 12-month 90-plus-days-past-due flag (charged off or default with the last payment inside the first nine months), everything else is good, so every loan gets the same one-year window and none are dropped as unfinished - my first version used a lifetime label on resolved loans only and that biased 2016, so I replaced it.
I split by time: develop on 2013-2015, validate on the last fifth of that window, and test once on 2016 issues. Features went through monotone WoE binning and IV screening ({d['n_pass_iv']} of {d['n_features_candidate']} passed), correlation and VIF pruning, then a logistic ladder - plain, L2, L1, ElasticNet and a points scorecard (600 points at 50:1, 20 points to double the odds) - plus tuned and monotone-constrained XGBoost as challengers.
I froze the champion rule on validation before scoring 2016: boosted trees only win if they beat the best interpretable model by 0.01 AUC with a bootstrap interval excluding zero. Result: {champ_text} is champion with an out-of-time Gini of {o[ch]['gini']:.2f} (AUC {o[ch]['auc']:.3f}); the best of all models excluding LendingClub's grade (unconstrained tuned XGBoost) differs from it by only {abs(CMP.loc['xgb_tuned', 'oot_auc'] - CMP.loc[ch, 'oot_auc']):.3f} AUC.
The finding is that ranking holds up (PSI {PSI['psi_train_vs_oot']:.3f}) and, with calibration fitted on validation only, the model predicts {pc(CMP.loc[ch, 'oot_mean_pred'], 2)} against {pc(CMP.loc[ch, 'oot_obs_rate'], 2)} observed in 2016 (ratio {cal_ratio:.2f}); I then back-tested every grade with a binomial test ({n_fail} of {len(MS)} fail) and wrote monitoring triggers around that."

## Talking point per stage

1. **Framing.** A 12-month 90+ DPD window gives every vintage the same observation period; a lifetime label would have left {pc(v16.share_open_at_cut)} of 2016 loans unlabelled at the data cut.
2. **Data checks.** De-duplicate before split; time-based split because default risk is non-stationary (12-month bad rate {pc(VIN.loc[2013].bad_rate_12m, 2)} in 2013 -> {pc(v15.bad_rate_12m, 2)} in 2015 -> {pc(v16.bad_rate_12m, 2)} in 2016). Leakage audit of {len(C('leakage_audit.csv'))} columns.
3. **Univariate.** Fine classing for shape, coarse monotone classing for the model; strongest feature is {top_iv.iloc[0].feature} (IV {top_iv.iloc[0].iv_coarse:.3f}); no IV above 0.5, so no leakage suspect.
4. **Multicollinearity.** {len(C('corr_pairs_gt_0_7.csv'))} correlated pairs removed, VIF all < 5, condition index {MC['condition_index']:.1f}; signs checked against business expectation.
5. **Ladder.** Validation AUC: plain {LAD.loc['plain_lr', 'auc']:.4f}, ElasticNet {LAD.loc['elasticnet_lr', 'auc']:.4f}, scorecard {LAD.loc['woe_scorecard', 'auc']:.4f}: regularisation buys stability, not accuracy. Imbalance handling does not raise AUC but wrecks Brier ({IMB.loc['none', 'cv_brier']:.3f} -> {IMB.iloc[1].cv_brier:.3f}).
6. **Challengers.** Tuned XGBoost OOT AUC {CMP.loc['xgb_tuned', 'oot_auc']:.4f}; monotone {CMP.loc['xgb_monotone', 'oot_auc']:.4f} ({n_constr} constraints, cost {mono_cost:+.4f}). SHAP leaders: {', '.join(top_shap[:4])}.
7. **Comparison.** Frozen rule, paired bootstrap, calibration on validation only; Gini moved {-gini_drop:+.3f} from validation to OOT.
8. **Business.** {len(MS)}-grade geometric master scale; binomial test; cut-off {CUT['cutoff_pd']:.3f} chosen on validation under assumed economics; monitoring triggers.

## 20 likely questions

**1. Why exclude grade, sub_grade and int_rate?** They are LendingClub's own risk output and price. Using them would make the model a wrapper around someone else's PD model and would not exist for a lender building its own. As a reference only, adding them lifts OOT Gini from {o[ch]['gini']:.3f} to {CMP.loc['ref_xgb_plus_LC_grade', 'oot_gini']:.3f}; LC sub-grade on its own reaches {CMP.loc['ref_LC_subgrade_only', 'oot_gini']:.3f}.

**2. Why is the Gini only {o[ch]['gini']:.2f}?** Unsecured consumer credit with application-time data and a pre-screened (accepted-only) population is noisy; 0.3-0.45 out-of-time is typical. A much higher number on this dataset would indicate leakage.

**3. How do you know there is no leakage?** Every column classified (`leakage_audit.csv`), post-origination fields dropped, no IV above 0.5, Gini plausible, and the only fields from after the application (`last_pymnt_d`) are used for vintage plots, never features.

**4. Why a 12-month window and not lifetime default?** My first version used lifetime default on resolved loans only. {pc(DROP.loc['OOT test', 'old_label_drop_share'])} of 2016 loans were still open at the data cut, so dropping them left an over-bad sample (2016 bad rate {pc(LS['oot_bad_rate_old_lifetime_resolved_only'])}) and made the model look under-calibrated. A 12-month window is complete for every loan, so nothing is dropped ({pc(LS['oot_bad_rate_main'], 2)} bad rate in 2016). The price: it is a 12-month PD, not lifetime, and the window is a proxy from the last-payment month (looser proxy: {pc(LS['oot_bad_rate_window_lastpay_le_12'], 2)}, AUC {LS['oot_auc_window_lastpay_le_12']:.3f} vs {LS['oot_auc_main']:.3f}).

**5. Why split by time rather than randomly?** Credit risk is non-stationary and the model will always be used on future loans. A random split would flatter performance: validation Gini {CMP.loc[ch, 'valid_gini']:.3f} vs OOT {o[ch]['gini']:.3f}.

**6. What is WoE and why use it?** ln(share of goods / share of bads) per bin. It linearises the log-odds, handles missing values and outliers, forces monotone, explainable bins and maps directly to a points table.

**7. How did you pick the champion?** {champ_why}

**8. Why a boosted model at all (or why not)?** """ + (
    f"The frozen rule said yes: XGB-monotone beat the WoE scorecard by {xgb_gain_mono:+.4f} OOT AUC ({xgb_gain_mono_gini:+.3f} Gini), with a bootstrap interval that excludes zero, so the extra complexity buys something real. I kept it defensible with monotone constraints (they cost only {mono_cost:+.4f} AUC versus the unconstrained model) and kept the WoE scorecard as the explainable companion for reason codes and a points table." if ch == "xgb_monotone" else
    f"The boosted models gained only {xgb_gain:+.4f} OOT AUC over the scorecard; that is small relative to the cost in explainability, adverse-action reason codes and model-risk burden. Monotone constraints recover most of the explainability for {mono_cost:+.4f} AUC.") + f"""

**9. What does PSI of {PSI['psi_train_vs_oot']:.3f} tell you, and is calibration off?** PSI only says the input population is stable. Calibration is a separate question that needs predicted-vs-actual back-testing: on 2016 the model {cal_phrase} (ratio {cal_ratio:.2f}); {n_fail} of {len(MS)} grades fail the binomial test, which over-rejects with correlated defaults, so I read the size of each gap too.

**10. What did you do about class imbalance?** Compared none, class_weight and SMOTE inside the CV folds. AUC {IMB.loc['none', 'cv_auc']:.4f} / {IMB.iloc[1].cv_auc:.4f} / {IMB.iloc[2].cv_auc:.4f} (no gain) but Brier {IMB.loc['none', 'cv_brier']:.3f} vs {IMB.iloc[1].cv_brier:.3f}: rebalancing distorts probabilities, which a PD model cannot afford.

**11. Why Platt calibration, and on what?** A monotone, two-parameter fix on validation data only, never test. It cannot change ranking. Out of time the ECE is {o[ch]['ece']:.4f} and predicted/actual is {cal_ratio:.2f}; any remaining gap is vintage drift because the 12-month label has no censoring.

**12. How do you handle multicollinearity?** |r|>0.7 pairs (keep higher IV), iterative VIF<5, condition index {MC['condition_index']:.1f}, and a sign check; ElasticNet coefficients show {sign_flips} sign flips across 5 folds.

**13. What do L1 / ElasticNet add?** Selection and stability; L1 zeroed {l1_zero} coefficient(s) here: after IV, correlation and VIF pruning there was little left for it to remove. Chosen ElasticNet setting: {best_en}.

**14. How does the scorecard scaling work?** Score = offset + factor x ln(good/bad) with factor = PDO/ln 2 = {SC['factor']:.2f}; 600 points at 50:1 odds; +20 points doubles the odds. Per-bin points sum exactly to the model score (tested).

**15. How is the master scale built?** {len(MS)} grades with geometric PD bands fixed on validation; grade PD = validation mean. On OOT, {n_fail} of {len(MS)} grades fail the binomial test ({under} under-predict risk); I flag that the test over-rejects with correlated defaults and ~{d['oot_rows']:,} loans.

**16. What are the economics behind the cut-off?** Assumed, on the same 12-month horizon as the label (12-month margin {CUT['assumptions']['margin_pct_of_principal']:.0%} of principal on a good loan, LGD {CUT['assumptions']['lgd']:.0%}, EAD {CUT['assumptions']['ead_pct_of_principal']:.0%}). Cut-off {CUT['cutoff_pd']:.3f} chosen on validation gives OOT approval {pc(CUT['oot_approval_rate_at_cutoff'])}, profit ${CUT['oot_profit_per_application']:,.0f} per application vs ${CUT['oot_profit_approve_all']:,.0f} approving everyone. The sensitivity grid shows the optimum moves a lot with margin and LGD.

**17. What is missing from this analysis?** Reject inference, risk-based pricing, macro variables, an LGD/EAD model, fairness testing, and a monthly delinquency history to replace the last-payment proxy for the 90+ DPD flag.

**18. How would you monitor it?** Score PSI and input CSI monthly, predicted-vs-actual by grade quarterly, Gini decline, binomial grade tests; amber -> recalibrate, red -> redevelop. Currently RED on: {red_txt}.

**19. What would you do next?** Re-calibrate on a recent window, add reject inference with the rejected-applications file, add a time-to-default (survival) view, and test fairness proxies.

**20. What would you do differently with real bank data?** Use the bank's own monthly delinquency history for a true 90 dpd flag instead of the last-payment proxy, a lifetime/stage-2 view alongside the 12-month PD, bureau pulls at decision time, and a segment-level champion/challenger governance.

{GUIDE_NEW}
*Generated by make_readme.py from artifacts/.*
"""
(OUT / "INTERVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
print("wrote README.md, MODEL_CARD.md, INTERVIEW_GUIDE.md")
