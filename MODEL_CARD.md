# Model card - LendingClub application PD scorecard

*One-page card in the style of SR 11-7 / PRA SS1/23 model documentation. It is documentation of a portfolio project on public data, not a validated production model and not a claim of regulatory compliance.*

| | |
|---|---|
| **Purpose** | Rank-order applicants for unsecured 36-month personal loans by probability of default within 12 months and support accept/decline and grade-based pricing decisions. |
| **Model** | Champion: **XGBoost monotone** (rule frozen on validation before OOT scoring: `artifacts/champion_rule.json`). Decision model of record (explainable): WoE points scorecard (18 variables). Challenger / benchmark: monotone XGBoost (wins the frozen accuracy rule). Reference: XGBoost tuned. |
| **Data** | LendingClub accepted loans 2007-2018Q4 (Kaggle `wordsforthewise/lending-club`), 36-month, individual applications, issued 2013-2016. |
| **Default definition** | Bad = Charged Off / Default with last payment in months 0-9 on book (proxy for 90+ DPD within 12 months of issue); Good = every other loan. 12-month performance window, observation point = application date; no loans dropped (all 2013-2016 loans have a full window). |
| **Windows** | Train Jan-2013 to Aug-2015 (400,000 rows, bad rate 3.1%); validation Sep-2015 to Dec-2015 (107,338, 3.7%); OOT Jan-2016 to Dec-2016 (318,277, 4.3%). |
| **Inputs** | 62 application-time features (loan, applicant, bureau snapshot at application). |
| **Exclusions** | All post-origination fields (payments, balances, recoveries, refreshed FICO, hardship/settlement); `grade`, `sub_grade`, `int_rate`, `installment` (LendingClub's own risk output/price); funded amount; free text and geography; joint applications; fields not populated before 2016. Full list: `artifacts/leakage_audit.csv`. |
| **Method** | Fine/coarse classing with monotone optimal bins (train only); IV>=0.02; |r|<=0.7, VIF<5; statsmodels logit, L1/L2/ElasticNet by stratified 5-fold CV; scorecard PDO 20, 600 at 50:1; XGBoost tuned by randomised search; Platt calibration on validation. |

## Performance (OOT 2016, calibrated PD)

| Metric | Champion | Validation (same model) |
|---|---|---|
| AUC | 0.6959 (95% CI 0.692-0.700) | 0.7179 |
| Gini | 0.3918 | 0.4359 |
| KS | 0.2823 | - |
| Brier / ECE | 0.0406 / 0.0068 | 0.0345 / 0.0015 |
| Mean predicted PD vs actual | 3.7% vs 4.3% | - |
| Score PSI train vs OOT | 0.0012 (stable (<0.10)) | 0.0052 |

Master scale: 8 grades (A: PD 0.4% ... H: PD 14.7%); grades are correctly rank-ordered on OOT, but 5 of 8 fail the binomial test (risk under-predicted in 5).

## Scope and limitations

- No reject inference (accepted loans only); the population the model has seen is not the through-the-door population.
- Label is a proxy (no monthly delinquency in the public file) on a 12-month horizon: 2016 bad rate 4.34% (looser proxy 6.32%); losses after month 12 are not covered (matured 2013-14 lifetime/12-month bad-rate ratio 4.4x). The earlier lifetime resolved-only label dropped 28.1% of 2016 loans and is kept only as a sensitivity.
- Calibration: fitted on 2015H2 validation only; on 2016 the model under-predicts (predicted/actual 0.84, ECE 0.0068); recalibrate on a recent window before any use.
- Gini 0.39 is moderate; most borrower-level information is already priced into LendingClub's grade, which is excluded here by design.
- Economics in the cut-off analysis are assumptions; macro conditions, fraud and fair-lending (geography excluded; no disparate-impact testing) are not covered.

## Monitoring and redevelopment triggers

| Metric | Amber | Red | Current (OOT) | Status |
|---|---|---|---|---|
| Score PSI (train vs current) | 0.1 | 0.25 | 0.001 | GREEN |
| Max input CSI | 0.1 | 0.25 | 0.047 | GREEN |
| Gini decline vs validation (relative) | 0.1 | 0.2 | 0.101 | AMBER |
| Calibration: mean predicted PD / actual | outside 0.9-1.1 | outside 0.8-1.25 | 0.843 | AMBER |
| Grades failing correlation-adjusted (Vasicek rho=0.03) test at 5% (of 8) | 2 | 4 | 0.000 | GREEN |
| Portfolio calibration: Vasicek two-sided p (rho=0.03) | < 0.05 | < 0.01 | 0.545 | GREEN |

Recalibrate on a fresh validation window when any metric is AMBER; redevelop and revalidate when any is RED or two are AMBER for two consecutive quarters. Review quarterly; full revalidation at least annually. Review cadence: monthly PSI/CSI, quarterly performance, annual validation. Metrics currently RED: none.

## Roles, calibration and fairness

- **Roles.** The WoE scorecard is the explainable decision model of record; the monotone XGBoost is the challenger/benchmark. The frozen accuracy rule selects XGBoost monotone on validation; it supplies the PD for grading and the cut-off, while the scorecard supplies the score, reason codes and adverse-action wording. Monotone-constraint evidence: 10 of 10 constrained features exactly monotone in PDP; 0 of 2 unconstrained top features follow the expected sign in ALE.
- **Calibration.** Grade tests use a correlation-adjusted (Vasicek, rho 0.03) test for the traffic light; 0 of 8 grades fail at 5% (plain binomial: 5). OOT calibration intercept +0.182, slope 0.931. A long-run (2013-2015) anchor with a stated +10% margin of conservatism moves OOT predicted/actual from 0.84 to 0.83; 2016 was not used to fit it.
- **Discrimination.** OOT Gini 0.392 (95% DeLong CI 0.383 to 0.400); rolling-origin Gini range 0.386 to 0.417.
- **Fairness.** No protected-class attributes exist in the public file; an income-band screen at the cut-off gives a lowest adverse-impact ratio of 0.80. This is a screen, not a disparate-impact test. Every scorecard variable maps to an adverse-action statement.
- **Reject population.** Accepted vs rejected comparison and a labelled parcelling sensitivity in Stage 14; not reject inference.

*Prepared from `artifacts/metrics.json`; generated by `make_readme.py`.*
