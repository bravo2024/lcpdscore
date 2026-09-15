# LendingClub application PD scorecard

Bank-style credit-risk project: a probability-of-default (PD) model for **36-month LendingClub loans**, built from
**application-time information only**, tested **out-of-time on loans issued in 2016**, with a frozen champion rule, a WoE points scorecard,
an 8-grade master scale, a cut-off simulator and a monitoring plan. Every number in this file is read from `artifacts/`.

Independent portfolio project on public LendingClub data. Not affiliated with LendingClub.

## Headline

- **Champion: the monotone-constrained XGBoost** - OOT Gini **0.392** (AUC 0.696, 95% CI 0.692-0.700), KS 0.282, Brier 0.0406, ECE 0.0068.
- Validation Gini was 0.436; on 2016 loans the model never saw it is 0.392 (change -0.044).
- **Default definition: 12-month window** (90+ DPD within 12 months of issue, proxied from the last-payment date). All 318,277 2016 loans are labelled; none dropped as unresolved.
- Score PSI train vs 2016 is **0.001** (stable (<0.10)). Calibration was fitted on validation only: the model predicts 3.66% on average vs 4.34% observed in 2016 (predicted/actual = 0.84; ECE 0.0068) - a genuine vintage-drift signal (the 12-month label has no censoring, so this is not a labelling artefact). 5 of 8 grades fail the OOT binomial test (5 under-predicted, 0 over-predicted). Allowing for correlated defaults (Vasicek, rho 0.03) only 0 of 8 grades fail, which is what the monitoring light uses; the plain binomial count (5) is a reference.
- **Roles:** The WoE scorecard is the explainable decision model of record; the monotone XGBoost is the challenger/benchmark.
- Excluding LendingClub's own grade/interest rate is a deliberate choice. As a reference only, adding them lifts the OOT Gini by **+0.058** (0.450) - that is LendingClub's own underwriting model talking.

### Out-of-time results, all models (calibrated on validation only)

| Model | Role | OOT AUC (95% CI) | OOT Gini | OOT KS | OOT Brier | OOT ECE |
|---|---|---|---|---|---|---|
| REF: XGBoost + LC grade/int_rate (not deployable) | reference | 0.7249 (0.721-0.729) | 0.4498 | 0.3279 | 0.0401 | 0.0044 |
| REF: LC sub-grade alone | reference | 0.7061 (0.702-0.711) | 0.4123 | 0.3011 | 0.0405 | 0.0048 |
| XGBoost tuned | ladder / challenger | 0.6983 (0.694-0.703) | 0.3965 | 0.2872 | 0.0406 | 0.0067 |
| XGBoost monotone **(champion)** | champion candidate | 0.6959 (0.692-0.700) | 0.3918 | 0.2823 | 0.0406 | 0.0068 |
| L2 logistic | ladder / challenger | 0.6727 (0.668-0.677) | 0.3454 | 0.2476 | 0.0409 | 0.0069 |
| Plain logistic (statsmodels) | ladder / challenger | 0.6726 (0.668-0.677) | 0.3453 | 0.2481 | 0.0409 | 0.0068 |
| ElasticNet logistic | champion candidate | 0.6726 (0.668-0.677) | 0.3453 | 0.2477 | 0.0409 | 0.0069 |
| L1 logistic | ladder / challenger | 0.6724 (0.668-0.677) | 0.3449 | 0.2470 | 0.0409 | 0.0069 |
| WoE scorecard | champion candidate | 0.6706 (0.666-0.675) | 0.3412 | 0.2456 | 0.0409 | 0.0070 |

*Calibrated PDs are Platt-scaled on the validation window. KS and AUC are rank-based, so calibration does not change them. The two REF rows use LendingClub's own risk outputs and are not eligible.*

**Why the champion** - The rule was fixed on validation before the 2016 data was scored. XGB-monotone's validation AUC beat the best interpretable candidate (WoE scorecard) by +0.0267 (95% bootstrap CI +0.0226 to +0.0309); the required margin was +0.0100 with a CI excluding zero. That test was passed, so XGB-monotone is champion. The plain XGBoost's gain over the WoE scorecard on OOT AUC is +0.0277; the monotone constraint costs +0.0024 AUC versus the unconstrained tuned model.

## What was built (the 8 stages)

| # | Stage | Result in this run |
|---|---|---|
| 1 | Framing | Bad = charged off / default with last payment in months 0-9 (= 90+ DPD within 12 months of issue); good = every other loan; **no loan dropped**; 12-month PD; train Jan-2013 to Aug-2015, validation Sep-2015 to Dec-2015, **OOT Jan-2016 to Dec-2016** |
| 2 | Data checks | 2,260,701 rows in file; de-duplicated before split; leakage audit of all columns (`artifacts/leakage_audit.csv`); 400,000 train rows (train was stratified-downsampled from 438,588 to 400,000 rows), 107,338 validation, 318,277 OOT |
| 3 | Univariate WoE/IV | 62 candidate features, 27 pass IV >= 0.02; strongest: fico (0.112), acc_open_past_24mths (0.091), num_tl_op_past_12m (0.084); nothing above 0.5 |
| 4 | Multicollinearity | 9 pairs with |r|>0.7 -> 20 features; VIF<5 -> 20; condition index 3.6; sign check in `artifacts/sign_check.csv` |
| 5 | Logistic ladder | plain / L2 / L1 / ElasticNet (C=0.3, l1_ratio=0.3) / WoE scorecard (18 variables, PDO 20, 600 points at 50:1); imbalance: none AUC 0.6842/Brier 0.0295 vs balanced 0.6846/0.2238 |
| 6 | Challengers | XGBoost tuned (randomised search) and monotone-constrained (41 constraints); SHAP top drivers: dti, fico, acc_open_past_24mths, purpose=credit_card, mths_since_recent_inq |
| 7 | Comparison | AUC/Gini/KS/PR-AUC/Brier/ECE, decile lift, PSI/CSI, paired bootstrap (1000 resamples) vs champion; rule frozen in `champion_rule.json` before OOT scoring |
| 8 | Business | 8-grade master scale with binomial tests; cut-off simulator on **assumed** economics; monitoring triggers |

## Scope and limitations

1. **Accepted loans only - no true reject inference**. The model describes people LendingClub approved; on the through-the-door population it would be optimistic about the riskiest applicants.
2. **Label is a proxy and a 12-month horizon.** The public file has one snapshot status and the last-payment month, not monthly delinquency, so "90+ DPD within 12 months" is *charged off/default AND last payment <= month 9*. A looser proxy (last payment <= month 12) moves the 2016 bad rate from 4.34% to 6.32% with champion AUC 0.696 vs 0.690. The model is a 12-month PD: about 10% of 2013 loans that were good inside the window were charged off later, and the matured 2013-14 lifetime-to-12-month bad-rate ratio is 4.4x, so lifetime loss is not covered.
   *Why not the old label:* the earlier version used lifetime default on resolved loans only; it dropped 28.1% of the 2016 loans (still open at the data cut), pushed the 2016 bad rate to 19.8% and made the model look badly under-calibrated. That is kept as a sensitivity (champion AUC 0.680 on it), not as the target.
3. **Drift.** The 12-month bad rate rose across vintages (2.86% in 2013, 3.48% in 2015, 4.34% in 2016). Calibration is fitted on 2015H2 validation only and tested on 2016: predicted/actual 0.84.
4. **Binomial tests over-reject** at this sample size and with defaults correlated inside a vintage; the monitoring light therefore uses a correlation-adjusted (Vasicek) test and keeps the binomial as a reference.
5. **Profit numbers use assumed economics** (margin 6% of principal, LGD 85%, EAD 90%); flat pricing; not a forecast.
6. Public data, no bureau history beyond what LendingClub recorded; no claim of Basel / IFRS 9 / SR 11-7 compliance - the model card only follows that *style*.
7. Compute: heavy notebook run on Kaggle (scikit-learn 1.6.1, xgboost 3.4.1, 22.3 minutes). Tuning budgets are modest by design.

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
