"""Builds LendingClubPDScorecard.ipynb (unexecuted) from the cell definitions below.

Run:  python build_notebook.py   -> writes LendingClubPDScorecard.ipynb and kaggle/LendingClubPDScorecard.ipynb
The shared library lcpd_lib.py is embedded in cell 2 (%%writefile) so the Kaggle kernel is self-contained.
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ======================================================================================
md(r"""
# LendingClub application PD scorecard (36-month loans, out-of-time 2016)

**What this notebook does.** It builds a probability-of-default (PD) model that a bank could run *at the moment a
borrower applies*, using only information that exists at application time, and then evaluates it on a later
time period the model never saw. It follows the usual 8-stage credit-risk development path:
framing, data checks, univariate WoE/IV, multicollinearity, a logistic ladder (plain, L1, L2, ElasticNet, WoE scorecard),
challenger models (XGBoost tuned and monotone-constrained), comparison with a frozen champion rule, and the business
layer (master scale, cut-off, monitoring). Stages 8b-15 then add correlation-adjusted calibration tests and a long-run calibration, default-definition reconciliation, a rolling-origin backtest, realised-economics validation, Somers' D/CAP discrimination, a fairness screen with adverse-action wording, a rejected-applications check and monotonicity evidence for the boosted model.

**Plain-English summary.** We look at about 3 years of past loans, learn which application details go with people not
paying back *within their first year*, then check on loans from the *following year* whether the ranking still works. The headline number is the
out-of-time Gini (0 = no better than a coin toss, 1 = perfect). On public LendingClub data with application-time
information only, a Gini of roughly 0.3 to 0.45 is realistic, and that is what we report.

**Scope notes up front.** (1) Only *accepted* loans are in the data, so there is no true reject inference (Stage 14 adds a through-the-door comparison and a parcelling sensitivity, clearly labelled as such).
(2) The target is a **12-month performance window** (90+ days past due within 12 months of issue, proxied from the last-payment date; see Stage 1).
Every 2013-2016 loan has a complete 12-month window by the Mar-2019 data cut, so **no loan is dropped as "unresolved" and nothing is right-censored**. An earlier
version of this project used a lifetime label on resolved loans only; that dropped every 2016 loan still open at the data cut (a large share), which inflated the 2016 bad rate and made the model look
under-calibrated. We keep that old label as a sensitivity check rather than hiding it. (3) Grade, sub-grade and interest rate are
LendingClub's own risk outputs and are deliberately **excluded** from the main models.
""")

code(r"""
%%writefile lcpd_lib.py
__LIB_SOURCE__
""")

code(r"""
# ---- Step 0: environment, paths, switches -------------------------------------------------
import os, sys, glob, json, time, math, warnings, platform, subprocess, hashlib, shutil
from pathlib import Path

ON_KAGGLE = os.path.exists("/kaggle/input")
FAST = os.environ.get("LC_FAST") == "1"          # tiny smoke-test mode for local debugging only
if not ON_KAGGLE:
    os.environ.setdefault("SCIPY_ARRAY_API", "1")  # lets imbalanced-learn import with this local scikit-learn
warnings.filterwarnings("ignore")
try:
    import optbinning
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optbinning"], check=True)
    import optbinning

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import sklearn, xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.getcwd())
import lcpd_lib as L

ART = Path("/kaggle/working/artifacts") if ON_KAGGLE else Path(os.environ.get("LC_ART", "artifacts"))
FIG, MOD = ART / "figures", ART / "models"
for d in (ART, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
T0 = time.time()
CAPS = dict(train=40_000, tune=15_000, smote=12_000, valid=20_000, oot=40_000) if FAST else \
       dict(train=400_000, tune=120_000, smote=60_000, valid=None, oot=None)
BOOT = 100 if FAST else 1000

plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titlesize": 11})
NAVY, ORANGE, GREY, TEAL, RED = "#1f3b73", "#d9822b", "#8a8f98", "#2a9d8f", "#c0392b"
CAPTIONS = {}

def tick(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}")

def save_csv(df, name, index=False):
    df.to_csv(ART / name, index=index)
    return df

def save_json(obj, name):
    with open(ART / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

def savefig(fig, name, caption):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    CAPTIONS[name] = caption
    plt.show()

print("kaggle:", ON_KAGGLE, "| FAST:", FAST, "| artifacts:", ART.resolve())
print("sklearn", sklearn.__version__, "| xgboost", xgboost.__version__, "| optbinning", optbinning.__version__)
""")

# ======================================================================================
md(r"""
## Stage 1 - Framing: what is a "bad", over what window, for what decision

| Item | Choice |
|---|---|
| Population | LendingClub **accepted** loans, **36-month term**, individual applications |
| Unit / decision | One loan application; decision = approve or decline at the point of application |
| Bad (1) | Loan ended `Charged Off` / `Default` (or its credit-policy twin) **and** its last payment fell within the first 9 months on book, i.e. 3 missed instalments = **90+ days past due by month 12** |
| Good (0) | every other loan: paid off, still current, late/in grace at the data cut, or charged off only *after* month 12 (not bad inside the window) |
| Dropped | **none**: all 2013-2016 loans have a full 12-month window by the Mar-2019 snapshot (asserted in the load cell) |
| Performance window | **12 months from issue** (the IFRS 9 stage-1 horizon); a lifetime view is shown as a sensitivity |
| Observation point | application date (`issue_d`); only information known then may be a feature |
| Development window | issue years **2013-2015** (train = first ~80% of the months, validation = last ~20%) |
| Out-of-time (OOT) test | issue year **2016**, never used to bin, fit, tune, calibrate or choose |
| Business use | accept/decline cut-off and risk-grade pricing tiers for an unsecured personal-loan book |

**Banker's note.** A fixed 12-month observation window with a 90+ DPD definition is the usual bank application-scorecard target (and the IFRS 9 12-month PD horizon).
It fixes the problem of the earlier lifetime/resolved-only label: by the data cut (Mar-2019) many 36-month loans issued in 2016 are still open
(see the share in Stage 2), so a lifetime label must either drop those loans (survivor bias: early defaulters resolve first, so the kept sample is over-weighted with bads)
or call them good (too optimistic). A 12-month window is complete for every loan issued up to Mar-2018, so all vintages are labelled on the same footing.
*Proxy caveat:* the public file has no monthly delinquency history, so "90+ DPD within 12 months" is approximated by *final status = charged off/default AND last payment in months 0-9*.
Loans that stop paying in months 10-12 (30-90 DPD at month 12) and loans that go bad after month 12 count as good; Stage 7c shows how much that choice moves the results.
The model therefore estimates a **12-month PD**, not a lifetime PD; losses after month 12 are not in the label.

**Plain-English note.** "Bad" means the borrower stopped paying within the first year and the loan was eventually written off. Every loan gets the same
one-year look, so loans that are still being repaid today are no longer a problem: we know they did not go bad in their first year.
""")

# ======================================================================================
md(r"""
## Stage 2 - Data checks: load, filter, leakage audit, vintages, splits

**What.** Read the LendingClub accepted-loans file, keep 36-month individual loans, remove duplicates *before* any
split, label good/bad, and classify every one of the ~150 columns as feature / target / dropped (with a reason).

**Why.** In credit scoring the most expensive mistake is leakage: fields such as `total_pymnt`, `recoveries` or
`last_fico_range_*` are recorded *after* the loan starts and encode the outcome. A model using them looks brilliant and is
useless at decision time.

**Banker's note.** `grade`, `sub_grade` and `int_rate` are LendingClub's own PD model output and price. Using them
re-uses someone else's risk model, hides what *our* variables add, and would not exist for a lender building its own
scorecard. They are excluded from every model that is a candidate; one clearly-labelled reference variant at the end
shows how much they would add.

**Plain-English note.** We only let the model see what a loan officer could know on day one, and we write down every
column we refused to use and why.
""")

code(r"""
# ---- Load -------------------------------------------------------------------------------
def find_data():
    env = os.environ.get("LC_DATA")
    if env and os.path.isfile(env):
        return env
    hits = [p for p in glob.glob("/kaggle/input/**/*accepted*2018*", recursive=True) if os.path.isfile(p)]
    hits = sorted(hits, key=lambda p: (not p.endswith(".gz"), p))      # prefer the .csv.gz
    if not hits:
        raise FileNotFoundError("accepted_2007_to_2018Q4 file not found; set LC_DATA for local runs")
    return hits[0]

CACHE = os.environ.get("LC_CACHE")
DATA = find_data() if not (CACHE and os.path.exists(CACHE)) else "(local parquet cache)"
print("data file:", DATA)
if CACHE and os.path.exists(CACHE):
    raw = pd.read_parquet(CACHE)
    N_FILE = json.load(open(CACHE + ".meta.json"))["n_file"]
    ALL_COLS = json.load(open(CACHE + ".meta.json"))["all_cols"]
else:
    want = set(L.RAW_READ_COLS)
    ALL_COLS = list(pd.read_csv(DATA, nrows=0).columns)
    raw = pd.read_csv(DATA, usecols=lambda c: c in want, low_memory=False)
    N_FILE = len(raw)
    raw = raw[raw["term"].astype(str).str.strip() == "36 months"].copy()
    raw["issue_d"] = pd.to_datetime(raw["issue_d"], format="%b-%Y")
    raw["last_pymnt_d"] = pd.to_datetime(raw["last_pymnt_d"], format="%b-%Y")
    if CACHE:
        raw.to_parquet(CACHE)
        json.dump({"n_file": N_FILE, "all_cols": ALL_COLS}, open(CACHE + ".meta.json", "w"))
raw["year"] = raw["issue_d"].dt.year
SNAPSHOT = raw["last_pymnt_d"].max()
tick(f"loaded: {N_FILE:,} rows in file, {len(raw):,} are 36-month loans; data snapshot ~ {SNAPSHOT:%b-%Y}")

wf = [("All accepted loans in file (2007-2018Q4)", N_FILE), ("36-month term", len(raw))]
raw = raw[raw["application_type"] == "Individual"]
wf.append(("Individual applications (joint apps excluded: income/DTI semantics differ, only 1.6% of 2016)", len(raw)))
n_before = len(raw)
raw = raw.drop_duplicates(subset="id")
wf.append((f"After de-duplicating on loan id (removed {n_before - len(raw):,}) - BEFORE any split", len(raw)))
base = raw[raw["year"].between(2012, 2016)].copy()
wf.append(("Issued 2012-2016 (vintage analysis window)", len(base)))
# 12-month performance window: bad = charged off / default AND last payment within 9 months on book (90+ DPD by month 12)
base["mob_last_pay"] = L.months_on_book_last_payment(base["issue_d"], base["last_pymnt_d"])
base["y"] = L.label_12m(base["loan_status"], base["mob_last_pay"])
# the window must be fully observable for every modelled loan: last issue month + 12 months <= data snapshot
_last_issue = base.loc[base["year"].between(2013, 2016), "issue_d"].max()
assert _last_issue + pd.DateOffset(months=L.PERF_WINDOW_MONTHS) <= SNAPSHOT, "12-month window not fully observed"
D = base[base["year"].between(2013, 2016)].copy()
wf.append(("Issued 2013-2016 = modelling population (12-month window fully observed; 0 loans dropped as unresolved)", len(D)))
save_csv(pd.DataFrame(wf, columns=["step", "rows"]), "population_waterfall.csv")
pd.DataFrame(wf, columns=["step", "rows"])
""")

code(r"""
# ---- Leakage audit: classify every column in the source file ------------------------------
FEATURE_SRC = set(L.ALL_FEATURES) | {"fico_range_low", "fico_range_high", "earliest_cr_line"}
POST = {"out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
        "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "pymnt_plan", "debt_settlement_flag",
        "debt_settlement_flag_date"}
LC_OUT = {"grade": "LendingClub's own risk grade (its PD model output)",
          "sub_grade": "LendingClub's own risk sub-grade (its PD model output)",
          "int_rate": "LendingClub's price, set from its own risk grade",
          "installment": "payment amount is a function of int_rate and term: leaks the LC risk price"}
FUNDING = {"funded_amnt": "amount funded after investor take-up; decided after application",
           "funded_amnt_inv": "amount funded by investors; decided after application"}
ID_TEXT_GEO = {"id": "identifier", "member_id": "identifier (empty)", "url": "link to listing", "emp_title": "free text, high cardinality",
               "desc": "free text (empty after 2014)", "title": "free text", "zip_code": "geography, high cardinality / fair-lending proxy",
               "addr_state": "geography, fair-lending proxy; excluded for compliance caution"}
NEW_BUREAU = {"open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util",
              "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", "inq_last_12m"}
def classify(c):
    if c in FEATURE_SRC: return "FEATURE (application-time)", "available at application; cleaned in L.engineer()"
    if c == "loan_status": return "TARGET", "defines good/bad; never a feature"
    if c == "issue_d": return "TIME INDEX", "used only to split in time; never a feature"
    if c == "term": return "FILTER", "36-month loans only"
    if c == "application_type": return "FILTER", "individual applications only"
    if c == "last_pymnt_d": return "ANALYSIS ONLY", "used for vintage months-on-book curves; post-origination, never a feature"
    if c in LC_OUT: return "DROPPED - LC risk output", LC_OUT[c]
    if c in FUNDING: return "DROPPED - post-application", FUNDING[c]
    if c in ID_TEXT_GEO: return "DROPPED - id/text/geo", ID_TEXT_GEO[c]
    if c in POST: return "DROPPED - post-origination", "recorded after the loan started (payments, balances, refreshed FICO, plan flags)"
    if c.startswith("hardship") or c in {"deferral_term", "payment_plan_start_date", "orig_projected_additional_accrued_interest"}:
        return "DROPPED - post-origination", "hardship programme: entered after distress"
    if c.startswith("settlement"): return "DROPPED - post-origination", "debt settlement: after default"
    if c.startswith("sec_app") or c in {"annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint"}:
        return "DROPPED - joint applicants", "joint/co-borrower fields; individual applications only"
    if c in NEW_BUREAU: return "DROPPED - not populated", "bureau fields LendingClub only began recording in late 2015/2016 (missing for ~all of 2013-2015)"
    if c in {"initial_list_status", "policy_code", "disbursement_method"}: return "DROPPED - not a borrower attribute", "platform/process field, no application-time risk meaning"
    return "DROPPED - other", "not used"
audit = pd.DataFrame([(c, *classify(c)) for c in ALL_COLS], columns=["column", "decision", "reason"])
save_csv(audit, "leakage_audit.csv")
print(audit["decision"].value_counts().to_string())
fd = pd.DataFrame([(k, v[0], v[1], {1: "risk up", -1: "risk down", 0: "unclear"}[v[2]]) for k, v in L.FEATURE_META.items()]
                  + [(c, L.CAT_LABELS[c], "Applicant", "n/a") for c in L.CATEGORICAL_FEATURES],
                  columns=["feature", "label", "group", "expected_effect_on_PD"])
save_csv(fd, "feature_dictionary.csv")
audit[audit.decision.str.startswith("DROPPED")].head(12)
""")

code(r"""
# ---- Outcome mix: 12-month label vs the old lifetime/resolved-only label, by vintage -------------------
st = pd.crosstab(base["year"], base["loan_status"])
bad_cols = [c for c in st.columns if c in L.BAD_STATUS]; good_cols = [c for c in st.columns if c in L.GOOD_STATUS]
vint = pd.DataFrame({"loans": st.sum(axis=1), "bad_12m": base.groupby("year")["y"].sum().astype(int),
                     "resolved_bad": st[bad_cols].sum(axis=1), "resolved_good": st[good_cols].sum(axis=1)})
vint["bad_rate_12m"] = vint["bad_12m"] / vint["loans"]
vint["open_at_cut"] = vint["loans"] - vint["resolved_bad"] - vint["resolved_good"]
vint["share_open_at_cut"] = vint["open_at_cut"] / vint["loans"]
vint["bad_rate_lifetime_resolved_only"] = vint["resolved_bad"] / (vint["resolved_bad"] + vint["resolved_good"])
vint["bad_rate_lifetime_open_as_good"] = vint["resolved_bad"] / vint["loans"]
vint = vint.reset_index().rename(columns={"year": "issue_year"})
save_csv(vint, "bad_rate_by_vintage.csv"); save_csv(st.reset_index(), "loan_status_by_year.csv")
print(vint.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(vint.issue_year, vint.bad_rate_12m * 100, "o-", color=NAVY, label="12-month bad rate (our label, all loans)")
ax[0].plot(vint.issue_year, vint.bad_rate_lifetime_resolved_only * 100, "s--", color=ORANGE, label="old label: lifetime, resolved loans only")
ax[0].set_ylabel("bad rate %"); ax[0].set_title("Bad rate by issue year"); ax[0].legend(fontsize=8); ax[0].set_xticks(vint.issue_year)
ax[1].bar(vint.issue_year, vint.share_open_at_cut * 100, color=GREY); ax[1].set_title("Share of loans still open at the data cut (kept, not dropped)")
ax[1].set_ylabel("%"); ax[1].set_xticks(vint.issue_year)
savefig(fig, "bad_rate_by_vintage.png", "The 12-month bad rate (navy) is measured on every loan over the same one-year window, so vintages are comparable. The old lifetime resolved-only label (orange) is distorted where many loans were still open at the data cut (right panel).")

# ---- Status-transition view (what the snapshot fields allow; NOT a monthly roll-rate matrix) --------------
bad_ls = base["loan_status"].isin(L.BAD_STATUS); mb = base["mob_last_pay"]
trans = pd.DataFrame({
    "A_bad_in_window_charged_off_lastpay_le_9": (bad_ls & (mb <= L.BAD_MAX_MOB)).groupby(base["year"]).sum(),
    "B_grey_zone_charged_off_lastpay_10_12": (bad_ls & (mb > L.BAD_MAX_MOB) & (mb <= 12)).groupby(base["year"]).sum(),
    "C_charged_off_after_window_lastpay_gt_12": (bad_ls & (mb > 12)).groupby(base["year"]).sum(),
    "D_late_or_grace_at_cut": base["loan_status"].isin(["In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]).groupby(base["year"]).sum(),
    "E_current_at_cut": (base["loan_status"] == "Current").groupby(base["year"]).sum(),
    "F_fully_paid": base["loan_status"].isin(L.GOOD_STATUS).groupby(base["year"]).sum()}).astype(int)
trans.insert(0, "loans", trans.sum(axis=1))
trans["share_of_window_good_that_later_charged_off"] = (trans["B_grey_zone_charged_off_lastpay_10_12"] + trans["C_charged_off_after_window_lastpay_gt_12"]) / (trans["loans"] - trans["A_bad_in_window_charged_off_lastpay_le_9"])
trans = trans.reset_index().rename(columns={"year": "issue_year"}); save_csv(trans, "status_transition_by_vintage.csv")
print(trans.round(4).to_string(index=False))

# ---- Months-on-book curves (analysis only; last_pymnt_d is NEVER a model input) --------------
mob = ((base["last_pymnt_d"].dt.year - base["issue_d"].dt.year) * 12 + (base["last_pymnt_d"].dt.month - base["issue_d"].dt.month)).fillna(0).clip(0, 36)
is_bad = base["loan_status"].isin(L.BAD_STATUS)
rows = []
for yr in range(2012, 2017):
    m = base["year"] == yr
    horizon = min(36, (SNAPSHOT.year - yr) * 12 + SNAPSHOT.month - 12)     # months observable for the LAST month of that year
    for k in range(0, 37):
        if k <= horizon:
            rows.append({"issue_year": yr, "months_on_book": k, "cum_bad_share": float(((mob[m] <= k) & is_bad[m]).sum() / m.sum()), "fully_observed_to": horizon})
vc = pd.DataFrame(rows); save_csv(vc, "vintage_curves.csv")
fig, ax = plt.subplots(figsize=(7.5, 4))
for yr, c in zip(range(2012, 2017), [GREY, TEAL, NAVY, ORANGE, RED]):
    s = vc[vc.issue_year == yr]; ax.plot(s.months_on_book, s.cum_bad_share * 100, color=c, label=str(yr))
ax.set_xlabel("months on book (month of last payment, a proxy for default timing)"); ax.set_ylabel("cumulative charged-off % of vintage")
ax.set_title("Vintage curves (all loans in the vintage in the denominator)"); ax.legend(title="issued")
savefig(fig, "vintage_curves.png", "Each line is a year of lending. Newer vintages default faster and higher. 2016 is only fully observable for roughly two years, so its curve stops early.")
""")

md(r"""
**Roll-rate note.** A true roll-rate matrix (current -> 30 -> 60 -> 90 DPD, month by month) needs monthly delinquency history, which the public LendingClub file does not
contain: it has one snapshot status and the last-payment month per loan. The status-transition table above is the closest the fields allow: how each vintage splits into bad-inside-window,
the grey zone (stopped paying in months 10-12), charged off after the window, still late/current at the cut, and paid. Its last column is the share of window-good loans that
still end up charged off later, i.e. the part of lifetime loss a 12-month PD does not see.
""")

code(r"""
# ---- Features, time split, sampling ------------------------------------------------------------
D = D.sort_values("issue_d").reset_index(drop=True)
Xall = L.engineer(D)
yall = D["y"].astype(int).values
dev = D["year"].between(2013, 2015).values
oot = (D["year"] == 2016).values
months = D.loc[dev, "issue_d"]
cum = months.value_counts().sort_index().cumsum() / dev.sum()
CUT_MONTH = cum.index[np.argmax(cum.values >= 0.80)]
is_train = dev & (D["issue_d"] <= CUT_MONTH).values
is_valid = dev & (D["issue_d"] > CUT_MONTH).values

def strat_sample(idx_bool, cap, seed):
    idx = np.where(idx_bool)[0]
    if cap is None or len(idx) <= cap:
        return idx
    r = np.random.default_rng(seed)
    y_ = yall[idx]
    keep = np.concatenate([r.choice(idx[y_ == c], int(round(cap * (y_ == c).mean())), replace=False) for c in (0, 1)])
    return np.sort(keep)

n_train_full = int(is_train.sum())
tr_idx = strat_sample(is_train, CAPS["train"], SEED)
va_idx = strat_sample(is_valid, CAPS["valid"], SEED + 1)
oot_idx = strat_sample(oot, CAPS["oot"], SEED + 2)
Xtr, ytr = Xall.iloc[tr_idx], yall[tr_idx]
Xva, yva = Xall.iloc[va_idx], yall[va_idx]
split = pd.DataFrame([
    {"set": "train", "issue_window": f"{D.issue_d[tr_idx].min():%b-%Y} to {CUT_MONTH:%b-%Y}", "rows_before_sampling": n_train_full, "rows_used": len(tr_idx), "bad_rate": ytr.mean()},
    {"set": "validation", "issue_window": f"{D.issue_d[va_idx].min():%b-%Y} to {D.issue_d[va_idx].max():%b-%Y}", "rows_before_sampling": int(is_valid.sum()), "rows_used": len(va_idx), "bad_rate": yva.mean()},
    {"set": "OOT test", "issue_window": "Jan-2016 to Dec-2016", "rows_before_sampling": int(oot.sum()), "rows_used": len(oot_idx), "bad_rate": yall[oot_idx].mean()}])
save_csv(split, "split_summary.csv")
# rows dropped per split because of the label definition (the old lifetime label dropped every loan not Fully Paid / Charged Off)
_resolved = D["loan_status"].isin(L.BAD_STATUS + L.GOOD_STATUS).values
drop_tab = pd.DataFrame([{"set": nm, "loans_in_split": int(m.sum()), "dropped_for_unresolved_status_now": 0,
                          "would_have_been_dropped_by_old_lifetime_label": int((m & ~_resolved).sum()),
                          "old_label_drop_share": float((m & ~_resolved).sum() / m.sum())}
                         for nm, m in [("train", is_train), ("validation", is_valid), ("OOT test", oot)]])
save_csv(drop_tab, "dropped_by_split.csv"); print(drop_tab.to_string(index=False))
SAMPLING_NOTE = (f"Train was stratified-downsampled from {n_train_full:,} to {len(tr_idx):,} rows" if len(tr_idx) < n_train_full else "No downsampling needed for train")
print(SAMPLING_NOTE); tick("split ready")
split
""")

md(r"""
### Data quality: missing values, sentinel codes, outlier caps

**Banker's note.** Missingness here is mostly *structural* (e.g. "months since last delinquency" is blank when there never was
a delinquency), so it is information, not noise: WoE binning gives the missing group its own bin and the logistic
models add a missing-flag. Sentinels are set to missing (`dti` of -1 or 999, `annual_inc` of 0). Extreme values are
winsorised at the 0.5th/99.5th percentile learned on **train only**.

**Plain-English note.** Blank cells often mean "nothing bad happened", so we keep them as a signal. Absurd entries (an
income of zero, a debt ratio of 999) are treated as unknown rather than believed.
""")

code(r"""
miss = pd.DataFrame({"feature": Xtr.columns,
                     "label": [L.LABELS[c] for c in Xtr.columns],
                     "missing_train": Xtr.isna().mean().values,
                     "missing_valid": Xva.isna().mean().values,
                     "missing_oot": Xall.iloc[oot_idx].isna().mean().values}).sort_values("missing_train", ascending=False)
save_csv(miss, "missingness.csv")
top = miss[miss.missing_train > 0.005].head(20)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(top.feature[::-1], top.missing_train[::-1] * 100, color=NAVY); ax.set_xlabel("% missing in train")
ax.set_title("Features with missing values (train)")
savefig(fig, "missingness.png", "Most blanks are 'never happened' cases (no delinquency, no public record). We keep them as their own category rather than guessing a value.")

caps = []
for c in L.NUMERIC_FEATURES:
    x = Xtr[c]
    caps.append({"feature": c, "p0.5": x.quantile(.005), "median": x.median(), "p99.5": x.quantile(.995), "max": x.max(),
                 "share_beyond_caps": float(((x < x.quantile(.005)) | (x > x.quantile(.995))).mean())})
caps = pd.DataFrame(caps); save_csv(caps, "outlier_caps.csv")
sent = pd.DataFrame({"rule": ["dti < 0 or dti >= 999 set to missing", "annual_inc <= 0 set to missing"],
                     "rows_affected_train": [int(((D.loc[Xtr.index, 'dti'] < 0) | (D.loc[Xtr.index, 'dti'] >= 999)).sum()),
                                             int((D.loc[Xtr.index, 'annual_inc'] <= 0).sum())]})
save_csv(sent, "sentinel_rules.csv"); print(sent.to_string(index=False))
caps.sort_values("max", ascending=False).head(6)
""")

# ======================================================================================
md(r"""
## Stage 3 - Univariate analysis: fine classing, coarse classing, WoE and IV

**What.** For each feature: (a) *fine classing* into ~20 equal-sized bins to see the raw shape and IV; (b) *coarse classing*
with `optbinning` into at most 6 bins with a **monotone** bad-rate trend (so each variable tells one clean story);
(c) compute Weight of Evidence per bin, WoE = ln(share of goods / share of bads), and Information Value
IV = sum over bins of (share of goods - share of bads) x WoE. All bins are learned on **train only**.

**Rules.** IV < 0.02: useless, dropped. 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong. IV > 0.5: *suspiciously strong*,
flagged as a possible leakage and not used automatically.

**Banker's note.** WoE coding makes the logistic model additive and readable as a points table, handles missing values and
outliers without imputation, and linearises the log-odds. Monotonic bins are required so every point-table row can be
explained to a credit committee and a regulator.

**Plain-English note.** We group each detail (e.g. income) into a few bands, check how much riskier or safer each band is,
and score each detail by how well it separates people who repay from those who don't.
""")

code(r"""
from optbinning import OptimalBinning
woe_fine, woe = L.WoEMap(), L.WoEMap()
ytr_s = pd.Series(ytr)
bin_idx = np.random.default_rng(SEED).choice(len(Xtr), min(len(Xtr), 150_000 if not FAST else 40_000), replace=False)
iv_rows, t_bin = [], time.time()
for c in L.ALL_FEATURES:
    is_cat = c in L.CATEGORICAL_FEATURES
    x_full = Xtr[c].astype("object").values if is_cat else Xtr[c].values.astype(float)
    # (a) fine classing
    if is_cat:
        levels = [[str(v)] for v in pd.Series(x_full).dropna().astype(str).unique()]
        sf = woe_fine.fit_var(c, x_full, ytr, "cat", groups=levels)
    else:
        q = np.unique(np.nanquantile(x_full, np.linspace(0.05, 0.95, 19)))
        q = q[q > np.nanmin(x_full)]
        sf = woe_fine.fit_var(c, x_full, ytr, "num", splits=q)
    # (b) coarse classing: optbinning proposes monotone cut points on a sample, WoE/IV recomputed on all train
    try:
        ob = OptimalBinning(name=c, dtype="categorical" if is_cat else "numerical", solver="cp",
                            monotonic_trend=None if is_cat else "auto", max_n_bins=6, min_bin_size=0.05,
                            max_n_prebins=20, cat_cutoff=0.01 if is_cat else None, time_limit=20)
        ob.fit(x_full[bin_idx], ytr[bin_idx])
        spl = ob.splits
        ok = ob.status in ("OPTIMAL", "FEASIBLE") and len(spl) > 0
    except Exception as e:
        ok, spl = False, []
    if not ok:
        spl = [[str(v)] for v in pd.Series(x_full).dropna().astype(str).unique()] if is_cat else np.unique(np.nanquantile(x_full, [1/3, 2/3]))
    sc = woe.fit_var(c, x_full, ytr, "cat" if is_cat else "num",
                     **({"groups": [list(g) for g in spl]} if is_cat else {"splits": list(spl)}))
    iv_rows.append({"feature": c, "label": L.LABELS[c], "group": L.GROUPS[c], "iv_fine": sf["iv"], "iv_coarse": sc["iv"],
                    "n_bins": len(sc["woe"]), "has_missing_bin": any(r["bin_index"] == -1 for r in sc["table"]),
                    "monotone": L.monotone_flag(sc["table"]) if not is_cat else True,
                    "optbinning_ok": bool(ok)})
iv = pd.DataFrame(iv_rows).sort_values("iv_coarse", ascending=False).reset_index(drop=True)
iv["strength"] = pd.cut(iv.iv_coarse, [-1, .02, .1, .3, .5, 99], labels=["useless (<0.02)", "weak", "medium", "strong", "SUSPICIOUS (>0.5)"]).astype(str)
iv["decision"] = np.where(iv.iv_coarse < 0.02, "drop: IV < 0.02", np.where(iv.iv_coarse > 0.5, "flag: IV > 0.5 leakage suspect", "keep"))
save_csv(iv, "iv_table.csv"); save_csv(woe.bins_table(), "woe_bins.csv")
tick(f"binning done in {(time.time() - t_bin) / 60:.1f} min; {int((iv.decision == 'keep').sum())} of {len(iv)} features pass IV >= 0.02")
iv.head(15)[["feature", "group", "iv_fine", "iv_coarse", "n_bins", "monotone", "strength", "decision"]]
""")

code(r"""
top = iv.head(25)
fig, ax = plt.subplots(figsize=(7.5, 6))
cols = [NAVY if d == "keep" else GREY for d in top.decision]
ax.barh(top.feature[::-1], top.iv_coarse[::-1], color=cols[::-1])
ax.axvline(0.02, ls="--", color=ORANGE, lw=1); ax.axvline(0.1, ls=":", color=GREY, lw=1)
ax.set_xlabel("Information Value (coarse classing, train)"); ax.set_title("Top 25 features by IV")
savefig(fig, "iv_bar.png", "Longer bar = better at separating good from bad borrowers. FICO and utilisation-type variables lead; nothing exceeds 0.5, so no variable looks like leakage.")

show = [f for f in iv.feature.head(6)]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, f in zip(axes.ravel(), show):
    t = woe.bins_table(); t = t[t.variable == f]
    ax.bar(range(len(t)), t.bad_rate * 100, color=NAVY)
    ax.set_xticks(range(len(t))); ax.set_xticklabels([b[:18] for b in t.bin], rotation=60, ha="right", fontsize=7)
    ax2 = ax.twinx(); ax2.plot(range(len(t)), t.woe, "o-", color=ORANGE); ax2.grid(False); ax2.set_ylabel("WoE", color=ORANGE)
    ax.set_title(f"{f}  (IV {iv.set_index('feature').loc[f, 'iv_coarse']:.3f})"); ax.set_ylabel("bad rate %")
fig.tight_layout()
savefig(fig, "woe_bins_top6.png", "Bars are the bad rate in each band, the orange line is its WoE. Bands are monotone: as the variable moves, risk moves one way only.")
""")

# ======================================================================================
md(r"""
## Stage 4 - Multicollinearity: correlation, VIF, condition index, sign check

**What.** Among features that passed IV, (1) list pairs with |correlation| > 0.7 (on the WoE-coded matrix, i.e. the actual
model inputs) and keep the higher-IV member; (2) iteratively drop the feature with the largest Variance Inflation
Factor until all VIF < 5; (3) report the condition index; (4) later check every coefficient sign against business
expectation.

**Banker's note.** Collinearity does not hurt prediction much but it makes coefficients unstable and unexplainable
(a "sign flip" on income because loan-to-income is also in the model). VIF is computed from the inverse correlation
matrix, which is algebraically identical to regressing each variable on the others; the final set is cross-checked
with `statsmodels.variance_inflation_factor`.

**Plain-English note.** If two inputs say almost the same thing, the model can't tell which deserves the credit.
We keep the better one so each remaining variable tells its own story.
""")

code(r"""
from statsmodels.stats.outliers_influence import variance_inflation_factor
keep0 = iv.loc[iv.decision == "keep", "feature"].tolist()
ivmap = iv.set_index("feature").iv_coarse
W = woe.transform(Xtr, keep0)
C = W.corr()
pairs = [(a, b, C.loc[a, b]) for i, a in enumerate(keep0) for b in keep0[i + 1:] if abs(C.loc[a, b]) > 0.7]
pairs = pd.DataFrame(pairs, columns=["var_a", "var_b", "corr"])
pairs["drop"] = [a if ivmap[a] < ivmap[b] else b for a, b, _ in pairs.itertuples(index=False, name=None)] if len(pairs) else []
pairs["keep"] = [b if d == a else a for a, b, d in pairs[["var_a", "var_b", "drop"]].itertuples(index=False, name=None)] if len(pairs) else []
save_csv(pairs, "corr_pairs_gt_0_7.csv")
# greedy: process strongest pairs first, skip pairs whose member is already gone
dropped_corr, alive = [], set(keep0)
for a, b, r in sorted(pairs[["var_a", "var_b", "corr"]].itertuples(index=False, name=None), key=lambda t: -abs(t[2])):
    if a in alive and b in alive:
        d = a if ivmap[a] < ivmap[b] else b
        alive.discard(d); dropped_corr.append((d, a if d == b else b, r))
print(f"{len(pairs)} pairs with |r|>0.7; dropped {len(dropped_corr)} features (lower IV of each pair)")
keep1 = [v for v in keep0 if v in alive]

def vif_series(Wm):
    Cm = np.corrcoef(Wm.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.inv(Cm)), index=Wm.columns)

vif_log, cur = [], list(keep1)
while True:
    vs = vif_series(W[cur])
    worst = vs.idxmax()
    if vs.max() <= 5 or len(cur) <= 3:
        break
    vif_log.append({"iteration": len(vif_log) + 1, "dropped": worst, "vif_at_drop": vs.max(), "iv": ivmap[worst]})
    cur.remove(worst)
keep2 = cur
vif_final = vif_series(W[keep2]).rename("vif").reset_index().rename(columns={"index": "feature"}).sort_values("vif", ascending=False)
chk = [variance_inflation_factor(sm.add_constant(W[keep2].iloc[:60000]).values, i + 1) for i in range(len(keep2))]
vif_final["vif_statsmodels_check_60k_rows"] = pd.Series(chk, index=keep2).reindex(vif_final.feature).values
save_csv(vif_final, "vif_final.csv"); save_csv(pd.DataFrame(vif_log, columns=["iteration", "dropped", "vif_at_drop", "iv"]), "vif_iterations.csv")
ev = np.linalg.eigvalsh(np.corrcoef(W[keep2].values, rowvar=False))
cond = float(np.sqrt(ev.max() / ev.min()))
save_json({"condition_index": cond, "n_features_after_corr": len(keep1), "n_features_after_vif": len(keep2),
           "dropped_by_corr": [(d, k, float(r)) for d, k, r in dropped_corr], "dropped_by_vif": [r["dropped"] for r in vif_log]}, "multicollinearity.json")
Cs = W[keep1].corr(); save_csv(Cs.reset_index().rename(columns={"index": "feature"}), "corr_matrix_woe.csv")
tick(f"after corr: {len(keep1)} | after VIF<5: {len(keep2)} | condition index {cond:.1f}")

fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(Cs.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(Cs))); ax.set_xticklabels(Cs.columns, rotation=90, fontsize=6); ax.set_yticks(range(len(Cs))); ax.set_yticklabels(Cs.columns, fontsize=6)
fig.colorbar(im, shrink=.7); ax.set_title("Correlation of WoE-coded features (after dropping |r|>0.7 duplicates)")
savefig(fig, "corr_heatmap.png", "Red/blue squares mark features that move together. After pruning, no pair exceeds 0.7, so each variable adds distinct information.")
fig, ax = plt.subplots(figsize=(7, 4.5)); v = vif_final.sort_values("vif")
ax.barh(v.feature, v.vif, color=NAVY); ax.axvline(5, ls="--", color=RED); ax.set_xlabel("VIF"); ax.set_title("Variance Inflation Factor, final feature set")
savefig(fig, "vif_bar.png", "VIF measures how well other inputs can predict a given input. Everything is under the red line (5), so coefficients will be stable.")
vif_final.head(8)
""")

# ======================================================================================
md(r"""
## Stage 5 - The logistic ladder

Five rungs, all fitted on **train**, tuned only with stratified 5-fold CV on train, compared on **validation**:

1. **Plain logistic regression** (statsmodels, no penalty): p-values, odds ratios, convergence check.
2. **L2 (ridge)**, 3. **L1 (lasso)**, 4. **ElasticNet**: scikit-learn, regularisation strength (and L1 share) tuned by CV AUC.
5. **WoE scorecard**: logistic regression on WoE-coded inputs, converted to points (PDO = 20, 600 points at 50:1 good:bad odds).

Plus: the class-imbalance comparison (none vs `class_weight` vs SMOTE *inside* the CV loop), Hosmer-Lemeshow calibration test,
a calibration curve and coefficient stability across folds.

**Banker's note.** Logistic regression remains the regulatory default because every driver has a sign, a size and a p-value.
Regularisation trades a little in-sample fit for stability; ElasticNet is the sensible compromise when features are correlated.
The scorecard form is what a credit committee signs off: a points table anyone can recompute by hand.

**Plain-English note.** We start with the simplest, most explainable model and add safeguards one at a time. The scorecard
turns the final model into a table of points, like a quiz where a higher score means a safer borrower.
""")

code(r"""
# ---- Rung 1: plain logistic regression (statsmodels) on the VIF-pruned raw features --------------
sel_vars = keep2
num_sel = [v for v in sel_vars if v in L.NUMERIC_FEATURES]; cat_sel = [v for v in sel_vars if v in L.CATEGORICAL_FEATURES]
prep = L.RawPrep(num_sel, cat_sel).fit(Xtr)
A_tr, A_va = prep.transform(Xtr), prep.transform(Xva)
names_lr = prep.names()
tune_idx = np.random.default_rng(SEED).choice(len(Xtr), min(len(Xtr), CAPS["tune"]), replace=False)

sm_fit = sm.Logit(ytr, sm.add_constant(A_tr, prepend=True)).fit(method="newton", maxiter=100, disp=0)
conv = {"converged": bool(sm_fit.mle_retvals["converged"]), "iterations": int(sm_fit.mle_retvals.get("iterations", -1)), "llf": float(sm_fit.llf), "pseudo_r2": float(sm_fit.prsquared), "aic": float(sm_fit.aic), "nobs": int(sm_fit.nobs)}
ci = sm_fit.conf_int()
plain = pd.DataFrame({"feature": ["const"] + names_lr, "coef": sm_fit.params, "std_err": sm_fit.bse, "z": sm_fit.tvalues, "p_value": sm_fit.pvalues,
                      "odds_ratio": np.exp(sm_fit.params), "or_ci_low": np.exp(ci[:, 0]), "or_ci_high": np.exp(ci[:, 1])})
save_csv(plain, "statsmodels_logit_raw.csv"); save_json(conv, "statsmodels_convergence.json")
print(conv)
raw_feats_pd = plain[plain.feature != "const"].copy()
fig, ax = plt.subplots(figsize=(7.5, 8))
rf = raw_feats_pd.assign(absz=lambda d: d.z.abs()).sort_values("absz", ascending=False).head(25).iloc[::-1]
ax.errorbar(rf.odds_ratio, range(len(rf)), xerr=[rf.odds_ratio - rf.or_ci_low, rf.or_ci_high - rf.odds_ratio], fmt="o", color=NAVY, ecolor=GREY, capsize=2)
ax.axvline(1, color=RED, ls="--"); ax.set_yticks(range(len(rf))); ax.set_yticklabels(rf.feature, fontsize=7); ax.set_xscale("log")
ax.set_xlabel("odds ratio of default per 1 SD increase (log scale); numeric features standardised"); ax.set_title("Plain logistic regression: top 25 drivers (95% CI)")
savefig(fig, "odds_ratio_forest.png", "Right of the red line = raises the odds of default, left = lowers them. Intervals that avoid the line are statistically clear effects.")
raw_feats_pd.assign(absz=lambda d: d.z.abs()).sort_values("absz", ascending=False).head(8)[["feature", "coef", "p_value", "odds_ratio"]]
""")

code(r"""
# ---- Rungs 2-4: L2 / L1 / ElasticNet tuned by stratified 5-fold CV on a train sub-sample --------
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
A_t, y_t = A_tr[tune_idx], ytr[tune_idx]
grids = {
 "l2": (LogisticRegression(penalty="l2", solver="lbfgs", max_iter=300), {"C": [0.001, 0.01, 0.1, 1.0]}),
 "l1": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=300), {"C": [0.001, 0.003, 0.01, 0.1]}),
 "elasticnet": (LogisticRegression(penalty="elasticnet", solver="saga", max_iter=300, tol=1e-3),
                {"C": [0.003, 0.03, 0.3], "l1_ratio": [0.3, 0.7]}),
}
lr_models, lr_cv = {}, []
for k, (est, g) in grids.items():
    t = time.time()
    gs = GridSearchCV(est, g, cv=skf, scoring="roc_auc", n_jobs=-1 if k != "elasticnet" else 4, refit=False).fit(A_t, y_t)
    best = gs.cv_results_["params"][int(np.argmax(gs.cv_results_["mean_test_score"]))]
    for p, m, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"]):
        lr_cv.append({"model": k, "params": json.dumps(p), "cv_auc_mean": m, "cv_auc_std": s, "best": p == best})
    final = est.set_params(**best).fit(A_tr, ytr)         # refit on the FULL train sample with the chosen setting
    lr_models[k] = final
    tick(f"{k}: best {best}, cv AUC {max(gs.cv_results_['mean_test_score']):.4f}, {(time.time() - t) / 60:.1f} min")
save_csv(pd.DataFrame(lr_cv), "lr_ladder_cv.csv")
best_en = json.loads([r for r in lr_cv if r["model"] == "elasticnet" and r["best"]][0]["params"])
coef_tab = pd.DataFrame({"feature": names_lr, "plain": sm_fit.params[1:], **{k: m.coef_[0] for k, m in lr_models.items()}})
coef_tab["l1_zeroed"] = coef_tab["l1"] == 0; coef_tab["en_zeroed"] = coef_tab["elasticnet"] == 0
save_csv(coef_tab, "ladder_coefficients.csv")
print("features zeroed by L1:", int(coef_tab.l1_zeroed.sum()), "| by ElasticNet:", int(coef_tab.en_zeroed.sum()), "of", len(coef_tab))
fig, ax = plt.subplots(figsize=(8, 7.5))
ct = coef_tab.assign(a=lambda d: d.plain.abs()).sort_values("a", ascending=False).head(22).iloc[::-1]
yy = np.arange(len(ct))
for off, col, nm, c in [(-.27, "plain", "plain", NAVY), (-.09, "l2", "L2", TEAL), (.09, "l1", "L1", ORANGE), (.27, "elasticnet", "ElasticNet", RED)]:
    ax.barh(yy + off, ct[col], height=.17, label=nm, color=c)
ax.set_yticks(yy); ax.set_yticklabels(ct.feature, fontsize=7); ax.axvline(0, color="k", lw=.6); ax.legend(fontsize=8)
ax.set_title("Coefficients across the ladder (standardised inputs)")
savefig(fig, "ladder_coefficients.png", "Plain coefficients are the largest; ridge shrinks them, lasso and ElasticNet push weak ones to exactly zero. Strong drivers keep their sign on every rung.")
""")

code(r"""
# ---- Imbalance: none vs class_weight vs SMOTE (SMOTE inside each CV fold), AUC and Brier -----------
res = []
sm_idx = np.random.default_rng(SEED + 7).choice(len(Xtr), min(len(Xtr), CAPS["smote"]), replace=False)
A_s, y_s = A_tr[sm_idx], ytr[sm_idx]
bestC = json.loads([r for r in lr_cv if r["model"] == "l2" and r["best"]][0]["params"])["C"]
strategies = {"none": LogisticRegression(C=bestC, max_iter=300), "class_weight=balanced": LogisticRegression(C=bestC, max_iter=300, class_weight="balanced")}
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as IPipe
    strategies["SMOTE (inside CV)"] = IPipe([("smote", SMOTE(random_state=SEED)), ("lr", LogisticRegression(C=bestC, max_iter=300))])
except Exception as e:
    print("imblearn unavailable, SMOTE skipped:", repr(e)[:100])
for nm, est in strategies.items():
    cvr = cross_validate(est, A_s, y_s, cv=skf, scoring={"auc": "roc_auc", "brier": "neg_brier_score"}, n_jobs=1)
    res.append({"strategy": nm, "cv_auc": cvr["test_auc"].mean(), "cv_auc_std": cvr["test_auc"].std(), "cv_brier": -cvr["test_brier"].mean()})
imb = pd.DataFrame(res); save_csv(imb, "imbalance_cv.csv"); print(imb.round(4).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(imb.strategy, imb.cv_auc, color=NAVY); ax[0].set_ylim(imb.cv_auc.min() - .01, imb.cv_auc.max() + .01); ax[0].set_title("CV AUC (ranking)")
ax[1].bar(imb.strategy, imb.cv_brier, color=ORANGE); ax[1].set_title("CV Brier (probability accuracy, lower = better)")
for a in ax: a.tick_params(axis="x", labelrotation=15, labelsize=8)
savefig(fig, "imbalance_comparison.png", "Rebalancing does not improve ranking (AUC) but it distorts the probabilities (worse Brier). A PD model needs well-calibrated probabilities, so we keep the natural class mix.")
""")

code(r"""
# ---- Rung 5: WoE scorecard = logistic on WoE inputs, sign-pruned, converted to points -------------
Wtr = woe.transform(Xtr, keep2); Wva = woe.transform(Xva, keep2)
cur = list(keep2)
while True:
    fit = sm.Logit(ytr, sm.add_constant(Wtr[cur])).fit(method="newton", maxiter=100, disp=0)
    pos = [v for v in cur if fit.params[v] > 0]                       # WoE = ln(good/bad): a sensible coefficient is NEGATIVE
    if pos:
        worst = max(pos, key=lambda v: fit.params[v]); cur.remove(worst); print("dropped (wrong sign):", worst); continue
    insig = [v for v in cur if fit.pvalues[v] > 0.01]
    if insig:
        worst = max(insig, key=lambda v: fit.pvalues[v]); cur.remove(worst); print("dropped (p>0.01):", worst); continue
    break
sc_vars = cur
sc = L.Scorecard(woe, sc_vars, {v: float(fit.params[v]) for v in sc_vars}, float(fit.params["const"]))
sc_tab = pd.DataFrame({"feature": ["const"] + sc_vars, "coef": fit.params.values, "std_err": fit.bse.values, "p_value": fit.pvalues.values,
                       "odds_ratio_per_unit_woe": np.exp(fit.params.values), "iv": [np.nan] + [ivmap[v] for v in sc_vars]})
save_csv(sc_tab, "scorecard_logit.csv")
save_json({"converged": bool(fit.mle_retvals["converged"]), "pseudo_r2": float(fit.prsquared), "variables": sc_vars, "n_variables": len(sc_vars),
           "pdo": L.PDO, "base_score": L.BASE_SCORE, "base_odds_good_to_bad": L.BASE_ODDS, "factor": L.FACTOR, "offset": L.OFFSET}, "scorecard_meta.json")
pts = sc.points_table(); save_csv(pts, "points_table.csv")
chk = np.abs(sc.score(Xva) - (L.OFFSET - L.FACTOR * L.logit(sc.raw_pd(Xva)))).max()
tick(f"scorecard: {len(sc_vars)} variables, {len(pts)} point rows; max |score - formula| = {chk:.2e}")
pts.groupby("variable").points.agg(["min", "max"]).assign(range=lambda d: d["max"] - d["min"]).sort_values("range", ascending=False).head(8)
""")

code(r"""
# ---- Raw PDs on validation for every model, Platt calibration (validation only), HL test --------------
P_tr, P_va = {}, {}
P_va["plain_lr"] = sm_fit.predict(sm.add_constant(A_va, prepend=True, has_constant="add"))
for k in ("l2", "l1", "elasticnet"):
    P_va[k + "_lr"] = lr_models[k].predict_proba(A_va)[:, 1]
P_va["woe_scorecard"] = sc.raw_pd(Xva)
P_tr["woe_scorecard"] = sc.raw_pd(Xtr)

def fit_platt(p_raw, y):
    m = LogisticRegression(C=1e6, max_iter=500).fit(L.logit(p_raw).reshape(-1, 1), y)
    return {"a": float(m.intercept_[0]), "b": float(m.coef_[0, 0])}
CAL = {k: fit_platt(v, yva) for k, v in P_va.items()}
sc.cal = CAL["woe_scorecard"]
hl = {k: L.hosmer_lemeshow(yva, L.platt_apply(v, CAL[k])) for k, v in P_va.items()}
hl["woe_scorecard_UNcalibrated"] = L.hosmer_lemeshow(yva, P_va["woe_scorecard"])
save_json(hl, "hosmer_lemeshow_valid.json")
lad = pd.DataFrame([{"model": k, **{m: L.all_metrics(yva, v)[m] for m in ("auc", "gini", "ks", "brier", "ece")},
                     "brier_calibrated": L.all_metrics(yva, L.platt_apply(v, CAL[k]))["brier"], "ece_calibrated": L.ece(yva, L.platt_apply(v, CAL[k])),
                     "hl_p_value_calibrated": hl[k]["p_value"]} for k, v in P_va.items()])
save_csv(lad, "ladder_metrics_valid.csv"); print(lad.round(4).to_string(index=False))

# coefficient stability: refit the chosen ElasticNet on each of 5 folds
stab = []
for i, (a, b) in enumerate(skf.split(A_t, y_t)):
    m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=best_en["l1_ratio"], C=best_en["C"], max_iter=300, tol=1e-3).fit(A_t[a], y_t[a])
    stab.append(pd.Series(m.coef_[0], index=names_lr, name=f"fold{i + 1}"))
stab = pd.concat(stab, axis=1)
stab_s = pd.DataFrame({"mean": stab.mean(axis=1), "std": stab.std(axis=1), "min": stab.min(axis=1), "max": stab.max(axis=1),
                       "sign_flips": (np.sign(stab.min(axis=1)) != np.sign(stab.max(axis=1))) & (stab.min(axis=1) != 0) & (stab.max(axis=1) != 0)})
stab_s["cv_ratio"] = stab_s["std"] / stab_s["mean"].abs().replace(0, np.nan)
save_csv(stab_s.reset_index().rename(columns={"index": "feature"}), "coef_stability.csv")
top = stab_s.reindex(stab_s["mean"].abs().sort_values(ascending=False).index).head(18).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.errorbar(top["mean"], range(len(top)), xerr=top["std"], fmt="o", color=NAVY, ecolor=ORANGE, capsize=2)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=7); ax.axvline(0, color="k", lw=.6)
ax.set_title("ElasticNet coefficients: mean +/- 1 SD over 5 folds")
savefig(fig, "coef_stability.png", "Tiny whiskers mean the coefficient barely moves when the training data is reshuffled, which is what a stable model looks like.")
fig, ax = plt.subplots(figsize=(5.5, 5)); r = L.reliability_table(yva, L.platt_apply(P_va["woe_scorecard"], CAL["woe_scorecard"]))
ax.plot([0, r.mean_pred.max() * 1.05], [0, r.mean_pred.max() * 1.05], "--", color=GREY); ax.plot(r.mean_pred, r.obs_rate, "o-", color=NAVY)
ax.set_xlabel("predicted PD"); ax.set_ylabel("observed bad rate"); ax.set_title("WoE scorecard calibration, validation (deciles)")
savefig(fig, "calibration_scorecard_valid.png", "Points on the dashed line mean predicted risk equals actual risk. Calibration was fitted on validation, so this is an in-sample check; the real test is OOT.")
sc_tab.head(12)
""")

# ======================================================================================
md(r"""
## Stage 6 - Challenger models: XGBoost tuned, and XGBoost with monotone constraints

**What.** Gradient-boosted trees on all (unpruned) application-time features, hyper-parameters tuned by randomised search with
3-fold CV on a train sub-sample (modest budget), then refit on the full train sample. The second challenger re-uses the
tuned settings but forces a **monotone** relationship with PD for features where business expectation and the train data
agree (e.g. higher FICO never raises PD). SHAP gives global importance and per-applicant reason codes.

**Banker's note.** Trees capture interactions that a scorecard cannot, but they are harder to explain and can learn
counter-intuitive wiggles. Monotone constraints are the standard compromise banks use to make a boosted model defensible
(each variable's effect has the same sign as its scorecard counterpart). The price is a small loss of AUC, which we measure.

**Plain-English note.** We let a more flexible model try to beat the simple one, then tell it "a better credit score may never
make a borrower look riskier" and see how much accuracy that rule costs.
""")

code(r"""
from xgboost import XGBClassifier
import xgboost as xgb
prep_all = L.RawPrep(L.NUMERIC_FEATURES, L.CATEGORICAL_FEATURES).fit(Xtr)
T_tr, tree_names = prep_all.transform_trees(Xtr); T_va, _ = prep_all.transform_trees(Xva)

# constraint rule: business expectation sign, kept only if train univariate direction agrees
direction = {}
for c in L.NUMERIC_FEATURES:
    xs = Xtr[c]; m = xs.notna()
    direction[c] = int(np.sign(stats.spearmanr(xs[m], ytr[m.values])[0])) if m.sum() > 100 else 0
constraint = {c: (L.EXPECTED_SIGN[c] if L.EXPECTED_SIGN[c] == direction[c] else 0) for c in L.NUMERIC_FEATURES}
mono_vec = tuple(int(constraint.get(n, 0)) for n in tree_names)
mono_tab = pd.DataFrame({"feature": L.NUMERIC_FEATURES, "expected_sign": [L.EXPECTED_SIGN[c] for c in L.NUMERIC_FEATURES],
                         "train_spearman_direction": [direction[c] for c in L.NUMERIC_FEATURES], "constraint_applied": [constraint[c] for c in L.NUMERIC_FEATURES]})
save_csv(mono_tab, "monotone_constraints.csv")
# coefficient sign check vs business expectation (numeric features)
plc = plain.set_index("feature").coef
sg = pd.DataFrame({"feature": L.NUMERIC_FEATURES, "label": [L.LABELS[c] for c in L.NUMERIC_FEATURES],
                   "expected_effect_on_pd": [L.EXPECTED_SIGN[c] for c in L.NUMERIC_FEATURES], "train_univariate_direction": [direction[c] for c in L.NUMERIC_FEATURES]})
sg["plain_lr_coef"] = sg.feature.map(plc); sg["scorecard_woe_coef"] = sg.feature.map(sc.coefs)
sg["in_plain_lr"] = sg.plain_lr_coef.notna(); sg["in_scorecard"] = sg.scorecard_woe_coef.notna()
sg["plain_lr_consistent"] = np.where(sg.expected_effect_on_pd == 0, "n/a (unclear expectation)", np.where(~sg.in_plain_lr, "not in model",
                              np.where(np.sign(sg.plain_lr_coef) == sg.expected_effect_on_pd, "yes", "NO")))
sg["scorecard_consistent"] = np.where(~sg.in_scorecard, "not in scorecard", np.where(sg.scorecard_woe_coef < 0, "yes (WoE coef negative)", "NO"))
save_csv(sg, "sign_check.csv"); print(sg.plain_lr_consistent.value_counts().to_string())
print("constrained features:", int((mono_tab.constraint_applied != 0).sum()), "of", len(mono_tab))

space = {"max_depth": [3, 4, 5, 6], "learning_rate": [0.03, 0.05, 0.08], "n_estimators": [200, 350, 500],
         "min_child_weight": [20, 50, 100], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.5, 0.7, 0.9],
         "reg_lambda": [1, 5, 20], "gamma": [0, 1]}
base_xgb = XGBClassifier(tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=SEED)
rs = RandomizedSearchCV(base_xgb, space, n_iter=3 if FAST else 12, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                        scoring="roc_auc", random_state=SEED, n_jobs=1, refit=False)
t = time.time(); rs.fit(T_tr[tune_idx], ytr[tune_idx])
bp = rs.best_params_
srch = pd.DataFrame(rs.cv_results_["params"]); srch["cv_auc_mean"] = rs.cv_results_["mean_test_score"]; srch["cv_auc_std"] = rs.cv_results_["std_test_score"]
save_csv(srch.sort_values("cv_auc_mean", ascending=False), "xgb_search.csv"); save_json(bp, "xgb_best_params.json")
tick(f"xgb search done in {(time.time() - t) / 60:.1f} min; best CV AUC {rs.best_score_:.4f}; params {bp}")

xgb_tuned = XGBClassifier(**bp, tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=SEED).fit(T_tr, ytr)
xgb_mono = XGBClassifier(**bp, tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=SEED, monotone_constraints=mono_vec).fit(T_tr, ytr)
P_va["xgb_tuned"] = xgb_tuned.predict_proba(T_va)[:, 1]
P_va["xgb_monotone"] = xgb_mono.predict_proba(T_va)[:, 1]

# reference only: the same booster WITH LendingClub's own grade / sub-grade / int_rate (not deployable by design)
def lc_extra(idx):
    d = D.iloc[idx]
    return np.column_stack([d["int_rate"].astype(float).values, d["grade"].map({g: i for i, g in enumerate("ABCDEFG", 1)}).values.astype(float),
                            (d["grade"].map({g: i for i, g in enumerate("ABCDEFG")}).values * 5 + d["sub_grade"].str[1].astype(float).values)])
E_tr, E_va = lc_extra(tr_idx), lc_extra(va_idx)
xgb_lc = XGBClassifier(**bp, tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=SEED).fit(np.hstack([T_tr, E_tr]), ytr)
P_va["ref_xgb_plus_LC_grade"] = xgb_lc.predict_proba(np.hstack([T_va, E_va]))[:, 1]
P_va["ref_LC_subgrade_only"] = 1 / (1 + np.exp(-(E_va[:, 2] - 15) / 6))      # monotone transform of sub-grade rank, used for ranking metrics only
for k in ("xgb_tuned", "xgb_monotone", "ref_xgb_plus_LC_grade", "ref_LC_subgrade_only"):
    CAL[k] = fit_platt(P_va[k], yva)
print({k: round(roc_auc_score(yva, v), 4) for k, v in P_va.items()})
""")

code(r"""
# ---- SHAP via XGBoost's native TreeSHAP (exact, no extra dependency issues) ----------------------
import shap
sh_idx = np.random.default_rng(SEED).choice(len(T_va), min(len(T_va), 4000 if not FAST else 1000), replace=False)
def contribs(model, M):
    return model.get_booster().predict(xgb.DMatrix(M, feature_names=None), pred_contribs=True)
S_mono = contribs(xgb_mono, T_va[sh_idx]); S_tuned = contribs(xgb_tuned, T_va[sh_idx])
imp = pd.DataFrame({"feature": tree_names, "mean_abs_shap_tuned": np.abs(S_tuned[:, :-1]).mean(0), "mean_abs_shap_monotone": np.abs(S_mono[:, :-1]).mean(0)})
imp = imp.sort_values("mean_abs_shap_monotone", ascending=False); save_csv(imp, "shap_importance.csv")
plt.figure(figsize=(8, 6.5))
shap.summary_plot(S_mono[:, :-1], pd.DataFrame(T_va[sh_idx], columns=tree_names), max_display=15, show=False, plot_size=None)
plt.title("SHAP summary: monotone XGBoost (validation sample)")
savefig(plt.gcf(), "shap_beeswarm.png", "Each dot is a borrower. Right = pushes predicted default risk up, left = down; red = high value of the feature. Higher FICO (red) sits on the left, as it should.")
# local explanation for one OOT-like high-risk borrower from validation
j = int(np.argsort(P_va["xgb_monotone"][sh_idx])[-len(sh_idx) // 50])
expl = shap.Explanation(values=S_mono[j, :-1], base_values=S_mono[j, -1], data=T_va[sh_idx][j], feature_names=tree_names)
plt.figure(); shap.plots.waterfall(expl, max_display=10, show=False)
savefig(plt.gcf(), "shap_waterfall.png", "How one applicant's risk is built up: each bar is a feature pushing the log-odds of default up (red) or down (blue) from the average borrower. These become the reason codes.")
imp.head(10)
""")

# ======================================================================================
md(r"""
## Stage 7 - Comparison and the frozen champion rule

**Order of operations (this is the point of an out-of-time test).**
1. Score **validation only**: AUC, Gini, KS, PR-AUC, Brier, ECE, with Platt calibration fitted on validation.
2. **Freeze the champion rule** and the chosen champion, and write it to `champion_rule.json` with a hash, *before the 2016 OOT file is touched*.
3. Only then score OOT, report every model, compute PSI/CSI and paired-bootstrap confidence intervals (1000 resamples) for the AUC difference against the champion.

**Rule.** Candidates: WoE scorecard, ElasticNet LR, XGB-monotone. The XGB-monotone model is champion only if its validation AUC beats the
best interpretable candidate by at least 0.010 **and** the 95% bootstrap interval of that gap excludes zero. Otherwise the best interpretable candidate wins,
with the WoE scorecard preferred when it is within 0.005 AUC of ElasticNet (because a points table is directly auditable).
The plain XGBoost and the LendingClub-grade variant are shown for reference but are not eligible.

**Banker's note.** A model risk reviewer (SR 11-7 / PRA SS1/23 style) wants the selection logic fixed ahead of the test so the test
is a genuine out-of-sample, out-of-time check and not another round of model shopping.

**Plain-English note.** Before looking at the final exam, we write down exactly how we will pick the winner, seal it, then grade.
""")

code(r"""
def paired_boot(y, P, ref, B, seed=0, names=None):
    names = names or list(P)
    r = np.random.default_rng(seed); n = len(y); y = np.asarray(y)
    out = {k: [] for k in names}; ref_a = []
    for _ in range(B):
        ix = r.integers(0, n, n); yb = y[ix]
        if yb.sum() == 0 or yb.sum() == n: continue
        ref_a.append(L.auc_fast(yb, P[ref][ix]))
        for k in names:
            out[k].append(L.auc_fast(yb, P[k][ix]))
    ref_a = np.array(ref_a)
    rows = []
    for k in names:
        d = np.array(out[k]) - ref_a
        rows.append({"model": k, "auc_diff_vs_" + ref: float(np.mean(out[k]) - ref_a.mean()), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                     "auc_ci_low": float(np.percentile(out[k], 2.5)), "auc_ci_high": float(np.percentile(out[k], 97.5)), "boot_reps": len(d)})
    return pd.DataFrame(rows)

CANDS = ["woe_scorecard", "elasticnet_lr", "xgb_monotone"]
val_cmp = pd.DataFrame([{"model": k, **L.all_metrics(yva, L.platt_apply(v, CAL[k])), "auc_raw": roc_auc_score(yva, v)} for k, v in P_va.items()])
save_csv(val_cmp, "model_metrics_valid.csv")
t = time.time(); bv = paired_boot(yva, P_va, "woe_scorecard", 200 if not FAST else 50, names=CANDS)
save_csv(bv, "bootstrap_valid_candidates.csv"); tick(f"validation bootstrap {(time.time() - t):.0f}s")
vm = val_cmp.set_index("model")
interp = ["woe_scorecard", "elasticnet_lr"]
best_interp = max(interp, key=lambda k: vm.loc[k, "auc"])
if abs(vm.loc["woe_scorecard", "auc"] - vm.loc["elasticnet_lr", "auc"]) <= 0.005: best_interp = "woe_scorecard"
gap = vm.loc["xgb_monotone", "auc"] - vm.loc[best_interp, "auc"]
bg = paired_boot(yva, P_va, best_interp, 200 if not FAST else 50, names=["xgb_monotone"]).iloc[0]
champion = "xgb_monotone" if (gap >= 0.010 and bg["ci_low"] > 0) else best_interp
rule = {"frozen_at_utc": pd.Timestamp.now("UTC").isoformat(), "scored_on": "validation only (OOT 2016 not yet scored)",
        "candidates": CANDS, "not_eligible": ["xgb_tuned", "ref_xgb_plus_LC_grade", "ref_LC_subgrade_only", "plain_lr", "l1_lr", "l2_lr"],
        "rule": ("XGB-monotone wins only if validation AUC exceeds the best interpretable candidate by >= 0.010 AND the paired-bootstrap 95% CI of the gap excludes 0; "
                 "otherwise the best interpretable candidate wins; WoE scorecard preferred if within 0.005 AUC of ElasticNet."),
        "validation_auc": {k: float(vm.loc[k, "auc"]) for k in CANDS}, "best_interpretable": best_interp,
        "xgb_monotone_gap_vs_best_interpretable": float(gap), "gap_ci95": [float(bg["ci_low"]), float(bg["ci_high"])], "champion": champion}
rule["sha256_of_rule_body"] = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
save_json(rule, "champion_rule.json")
save_json(CAL, "calibrators.json")
print(json.dumps(rule, indent=1))
""")

# ======================================================================================
md(r"""
## Stage 7b - Out-of-time scoring (2016 issues): the out-of-time test

Everything below is computed on loans issued in **2016**, using only objects fixed above: bins, coefficients, hyper-parameters,
calibrators, and the champion rule.

**Banker's note.** Expect a lower Gini than validation, because the economy and LendingClub's own underwriting shifted. Calibration (Platt) is fitted on
**validation only** and never on OOT; because the 12-month label is complete for every vintage (no right-censoring), any remaining gap between predicted and
observed PD in 2016 is genuine vintage drift, not a labelling artefact. It is reported as it is: that is exactly what monitoring and recalibration are for.

**Plain-English note.** This is the real-world test: the model has never seen these loans. A drop in accuracy versus the practice run is normal and we show it as-is.
""")

code(r"""
Xoot, yoot = Xall.iloc[oot_idx], yall[oot_idx]
A_oot = prep.transform(Xoot); T_oot, _ = prep_all.transform_trees(Xoot); E_oot = lc_extra(oot_idx)
P_oot = {"plain_lr": sm_fit.predict(sm.add_constant(A_oot, prepend=True, has_constant="add")),
         "l2_lr": lr_models["l2"].predict_proba(A_oot)[:, 1], "l1_lr": lr_models["l1"].predict_proba(A_oot)[:, 1],
         "elasticnet_lr": lr_models["elasticnet"].predict_proba(A_oot)[:, 1], "woe_scorecard": sc.raw_pd(Xoot),
         "xgb_tuned": xgb_tuned.predict_proba(T_oot)[:, 1], "xgb_monotone": xgb_mono.predict_proba(T_oot)[:, 1],
         "ref_xgb_plus_LC_grade": xgb_lc.predict_proba(np.hstack([T_oot, E_oot]))[:, 1],
         "ref_LC_subgrade_only": 1 / (1 + np.exp(-(E_oot[:, 2] - 15) / 6))}
PC_oot = {k: L.platt_apply(v, CAL[k]) for k, v in P_oot.items()}
PC_va = {k: L.platt_apply(v, CAL[k]) for k, v in P_va.items()}
rows = []
for k in P_oot:
    mo = L.all_metrics(yoot, PC_oot[k]); mo["auc"] = roc_auc_score(yoot, P_oot[k]); mo["gini"] = 2 * mo["auc"] - 1; mo["ks"] = L.ks_stat(yoot, P_oot[k])
    mv = L.all_metrics(yva, PC_va[k]); mv["auc"] = roc_auc_score(yva, P_va[k]); mv["gini"] = 2 * mv["auc"] - 1
    kind = "reference" if k.startswith("ref_") else ("champion candidate" if k in CANDS else "ladder / challenger")
    rows.append({"model": k, "role": kind, "is_champion": k == champion, **{f"oot_{m}": mo[m] for m in ("auc", "gini", "ks", "pr_auc", "brier", "ece", "log_loss", "mean_pred", "obs_rate")},
                 "valid_auc": mv["auc"], "valid_gini": mv["gini"], "gini_drop_valid_to_oot": mv["gini"] - mo["gini"]})
cmp_ = pd.DataFrame(rows).sort_values("oot_auc", ascending=False).reset_index(drop=True)
t = time.time(); bo = paired_boot(yoot, P_oot, champion, BOOT, names=list(P_oot)); tick(f"OOT bootstrap {BOOT} reps in {time.time() - t:.0f}s")
cmp_ = cmp_.merge(bo.rename(columns={"auc_diff_vs_" + champion: "auc_diff_vs_champion", "ci_low": "diff_ci_low", "ci_high": "diff_ci_high"}), on="model")
save_csv(cmp_, "model_comparison.csv")
print(cmp_[["model", "role", "is_champion", "oot_auc", "oot_gini", "oot_ks", "oot_brier", "oot_ece", "auc_diff_vs_champion", "diff_ci_low", "diff_ci_high"]].round(4).to_string(index=False))
""")

code(r"""
# ---- Curves: ROC, KS, PR, reliability, lift/gains, score distribution --------------------------------
show = ["woe_scorecard", "elasticnet_lr", "xgb_tuned", "xgb_monotone", "l2_lr", "ref_xgb_plus_LC_grade"]
colors = dict(zip(show, [NAVY, TEAL, ORANGE, RED, GREY, "#9b59b6"]))
roc_rows, pr_rows = [], []
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for k in show:
    f, t_, _ = roc_curve(yoot, PC_oot[k]); ax[0].plot(f, t_, color=colors[k], label=f"{k} ({roc_auc_score(yoot, PC_oot[k]):.3f})", lw=2 if k == champion else 1.2, ls="--" if k.startswith("ref_") else "-")
    sel = np.linspace(0, len(f) - 1, 200).astype(int); roc_rows += [{"model": k, "fpr": a, "tpr": b} for a, b in zip(f[sel], t_[sel])]
    p_, r_, _ = precision_recall_curve(yoot, PC_oot[k]); sel = np.linspace(0, len(p_) - 1, 200).astype(int); pr_rows += [{"model": k, "recall": a, "precision": b} for a, b in zip(r_[sel], p_[sel])]
    ax[1].plot(r_, p_, color=colors[k], label=k, lw=2 if k == champion else 1.2)
ax[0].plot([0, 1], [0, 1], ":", color=GREY); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].set_title("ROC, OOT 2016"); ax[0].legend(fontsize=7)
ax[1].axhline(yoot.mean(), ls=":", color=GREY); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision-recall, OOT 2016")
save_csv(pd.DataFrame(roc_rows), "roc_points.csv"); save_csv(pd.DataFrame(pr_rows), "pr_points.csv")
savefig(fig, "roc_pr_oot.png", "ROC: further toward the top-left is better. Models are close to each other; the dashed purple line uses LendingClub's own grade as an input and shows how much of the gap is just LC's risk model.")

ch = PC_oot[champion]; order = np.argsort(ch); f, t_, thr = roc_curve(yoot, ch)
ksx = np.linspace(0, 1, len(f)); fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thr[1:], t_[1:], label="cumulative bads", color=RED); ax.plot(thr[1:], f[1:], label="cumulative goods", color=NAVY)
kk = np.argmax(t_ - f); ax.axvline(thr[kk], ls=":", color=GREY); ax.set_xscale("log"); ax.set_xlabel("PD threshold (champion, calibrated)"); ax.set_ylabel("share at or above threshold")
ax.set_title(f"KS curve, OOT (KS = {np.max(t_ - f):.3f})"); ax.legend()
ks_pts = pd.DataFrame({"threshold": thr[1:], "tpr": t_[1:], "fpr": f[1:]}); save_csv(ks_pts.iloc[np.linspace(0, len(ks_pts) - 1, 300).astype(int)], "ks_points.csv")
savefig(fig, "ks_curve.png", "KS is the widest vertical gap between the share of bad and good borrowers caught at a threshold. A bigger gap means better separation.")

rel_rows = []
fig, ax = plt.subplots(figsize=(5.8, 5.2)); mx = 0
for k in ["woe_scorecard", "elasticnet_lr", "xgb_monotone"]:
    r = L.reliability_table(yoot, PC_oot[k]); r["model"] = k; rel_rows.append(r); ax.plot(r.mean_pred, r.obs_rate, "o-", color=colors[k], label=k); mx = max(mx, r.mean_pred.max(), r.obs_rate.max())
ax.plot([0, mx], [0, mx], "--", color=GREY); ax.set_xlabel("predicted PD (calibrated on validation)"); ax.set_ylabel("observed bad rate OOT"); ax.legend(); ax.set_title("Reliability, OOT 2016 (deciles)")
save_csv(pd.concat(rel_rows), "reliability_oot.csv")
savefig(fig, "reliability_oot.png", "Points on the dashed line mean predicted PD equals the observed 2016 bad rate; above it the model under-predicted risk, below it over-predicted. Calibration was fitted on validation only.")

dec = pd.DataFrame({"pd": ch, "y": yoot}); dec["decile"] = pd.qcut(dec.pd.rank(method="first"), 10, labels=range(1, 11)).astype(int)
lift = dec.groupby("decile").agg(n=("y", "size"), mean_pd=("pd", "mean"), bad_rate=("y", "mean"), bads=("y", "sum")).reset_index()
lift["lift"] = lift.bad_rate / yoot.mean(); lift = lift.sort_values("decile", ascending=False).reset_index(drop=True)
lift["cum_bad_capture"] = lift.bads.cumsum() / lift.bads.sum(); lift["cum_pop"] = lift.n.cumsum() / lift.n.sum()
save_csv(lift, "lift_deciles.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
ax[0].bar(lift.index + 1, lift.lift, color=NAVY); ax[0].axhline(1, ls=":", color=GREY); ax[0].set_xlabel("risk decile (1 = riskiest)"); ax[0].set_ylabel("bad rate / average"); ax[0].set_title("Decile lift")
ax[1].plot([0] + lift.cum_pop.tolist(), [0] + lift.cum_bad_capture.tolist(), "o-", color=RED); ax[1].plot([0, 1], [0, 1], ":", color=GREY)
ax[1].set_xlabel("share of applicants (riskiest first)"); ax[1].set_ylabel("share of all bads caught"); ax[1].set_title("Cumulative gains")
savefig(fig, "lift_gains.png", "The riskiest 10% of applicants default at a multiple of the average rate, and the riskiest 30% contain a disproportionate share of all bad loans.")

scr = L.pd_to_score(ch)
fig, ax = plt.subplots(figsize=(7, 4)); bins = np.linspace(np.percentile(scr, .5), np.percentile(scr, 99.5), 50)
ax.hist(scr[yoot == 0], bins, alpha=.6, color=NAVY, density=True, label="good (repaid)"); ax.hist(scr[yoot == 1], bins, alpha=.6, color=RED, density=True, label="bad (90+ DPD within 12m)")
ax.set_xlabel("score (600 = 50:1 good:bad, +20 points = odds double)"); ax.set_title("Score distribution, OOT"); ax.legend()
savefig(fig, "score_distribution.png", "Good borrowers (blue) sit to the right of bad borrowers (red) but the two overlap a lot, which is normal for unsecured consumer credit and explains a Gini well below 0.8.")
lift.round(3)
""")

# ======================================================================================
md(r"""
## Stage 7c - Drift: PSI / CSI, score drift, and the label-definition sensitivity

**What.** Population Stability Index compares the distribution of the score (and of each input, called CSI) in train against OOT:
PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 significant shift. We also (a) trace predicted vs actual bad rate by issue quarter, and
(b) re-label the same 2016 loans under alternative default definitions (looser 12-month proxy; the old lifetime resolved-only label) to see how much the label choice matters.

**Banker's note.** PSI says the *population* moved; the quarterly predicted-vs-actual shows the *relationship* moved. A model can have a
stable PSI and still be miscalibrated (and vice versa), so monitoring needs both.

**Plain-English note.** "Are today's applicants the same kind of people as the ones we trained on, and is the model still right about how risky they are?"
""")

code(r"""
# champion-agnostic score PSI: use the champion's calibrated PD on train too
def champ_raw(Xdf, A, T):
    return {"woe_scorecard": lambda: sc.raw_pd(Xdf), "elasticnet_lr": lambda: lr_models["elasticnet"].predict_proba(A)[:, 1],
            "xgb_monotone": lambda: xgb_mono.predict_proba(T)[:, 1]}[champion]()
ch_tr = L.platt_apply(champ_raw(Xtr, A_tr, T_tr), CAL[champion]); ch_va = PC_va[champion]
S_tr, S_va, S_oot = L.pd_to_score(ch_tr), L.pd_to_score(ch_va), L.pd_to_score(ch)
psi_score = {"champion": champion, "psi_train_vs_valid": L.psi_numeric(S_tr, S_va), "psi_train_vs_oot": L.psi_numeric(S_tr, S_oot), "psi_valid_vs_oot": L.psi_numeric(S_va, S_oot),
             "mean_score_train": float(S_tr.mean()), "mean_score_valid": float(S_va.mean()), "mean_score_oot": float(S_oot.mean()),
             "mean_pd_train": float(ch_tr.mean()), "mean_pd_oot": float(ch.mean())}
psi_score["label_train_vs_oot"] = L.psi_label(psi_score["psi_train_vs_oot"])
save_json(psi_score, "psi_score.json"); print(psi_score)
csi = []
for c in L.ALL_FEATURES:
    v = L.psi_categorical(Xtr[c], Xoot[c]) if c in L.CATEGORICAL_FEATURES else L.psi_numeric(Xtr[c], Xoot[c])
    csi.append({"feature": c, "label": L.LABELS[c], "csi_train_vs_oot": v, "flag": L.psi_label(v), "in_scorecard": c in sc_vars, "iv": ivmap.get(c, np.nan)})
csi = pd.DataFrame(csi).sort_values("csi_train_vs_oot", ascending=False); save_csv(csi, "csi.csv")
top = csi.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top.feature, top.csi_train_vs_oot, color=[RED if v > .25 else ORANGE if v > .1 else NAVY for v in top.csi_train_vs_oot])
ax.axvline(.1, ls=":", color=GREY); ax.axvline(.25, ls="--", color=RED); ax.set_xlabel("CSI, train vs OOT"); ax.set_title(f"Input drift (score PSI train vs OOT = {psi_score['psi_train_vs_oot']:.3f})")
savefig(fig, "psi_bar.png", "Bars show how far each input's distribution moved between the training years and 2016. Under 0.10 is stable; over 0.25 would trigger investigation.")

# predicted vs actual by quarter (development + OOT), champion
q_all = pd.DataFrame({"issue_d": D.issue_d.values, "y": yall})
sub = np.concatenate([tr_idx, va_idx, oot_idx]); Xq = Xall.iloc[sub]
Aq, Tq = prep.transform(Xq), prep_all.transform_trees(Xq)[0]
pq = L.platt_apply(champ_raw(Xq, Aq, Tq), CAL[champion])
qd = pd.DataFrame({"quarter": pd.PeriodIndex(D.issue_d.iloc[sub], freq="Q").astype(str), "y": yall[sub], "pd": pq, "set": np.where(np.isin(sub, tr_idx), "train", np.where(np.isin(sub, va_idx), "valid", "OOT"))})
qt = qd.groupby("quarter").agg(n=("y", "size"), actual=("y", "mean"), predicted=("pd", "mean"), set=("set", "first")).reset_index(); qt["score"] = L.pd_to_score(qt.predicted)
save_csv(qt, "drift_by_quarter.csv")
fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(qt.quarter, qt.actual * 100, "o-", color=RED, label="actual bad rate"); ax.plot(qt.quarter, qt.predicted * 100, "s--", color=NAVY, label="model-predicted (calibrated on 2015H2 validation)")
ax.set_xticks(range(len(qt))); ax.set_xticklabels(qt.quarter, rotation=60, fontsize=7); ax.set_ylabel("%"); ax.legend(); ax.set_title("Predicted vs actual bad rate by issue quarter")
savefig(fig, "drift_by_quarter.png", "Predicted versus actual 12-month bad rate by issue quarter. Calibration was fitted on 2015H2 only; the 2016 quarters are out of time.")

# label-definition sensitivity on the SAME 2016 loans scored by the champion (the model is not refitted)
o_ = D.iloc[oot_idx]; st_o = o_["loan_status"]; mob_o = o_["mob_last_pay"].values
y_loose = L.label_12m(st_o, mob_o, 12)                                    # also counts stop-paying in months 10-12
res_mask = st_o.isin(L.BAD_STATUS + L.GOOD_STATUS).values                 # OLD label: lifetime, resolved loans only
y_life = st_o.isin(L.BAD_STATUS).values[res_mask].astype(int)
mat = base[base.year.isin([2013, 2014])]                                  # vintages fully matured (36m + data cut)
life_mat = float(mat["loan_status"].isin(L.BAD_STATUS).mean()); w12_mat = float(mat["y"].mean())
cens = {"champion": champion, "definition": "bad = Charged Off/Default AND last payment <= month %d on book (90+ DPD within %d months)" % (L.BAD_MAX_MOB, L.PERF_WINDOW_MONTHS),
        "oot_loans_total": int(len(o_)), "oot_loans_dropped_by_label": 0,
        "oot_bad_rate_main": float(yoot.mean()), "oot_auc_main": float(roc_auc_score(yoot, ch)),
        "oot_bad_rate_window_lastpay_le_12": float(y_loose.mean()), "oot_auc_window_lastpay_le_12": float(roc_auc_score(y_loose, ch)),
        "oot_share_open_at_cut": float(1 - res_mask.mean()), "oot_n_old_label_resolved": int(res_mask.sum()),
        "oot_bad_rate_old_lifetime_resolved_only": float(y_life.mean()), "oot_auc_old_lifetime_resolved_only": float(roc_auc_score(y_life, ch[res_mask])),
        "oot_mean_pred_champion": float(ch.mean()),
        "matured_2013_14_lifetime_bad_rate": life_mat, "matured_2013_14_12m_bad_rate": w12_mat, "lifetime_to_12m_ratio_matured": life_mat / w12_mat,
        "note": "Same champion, same 2016 loans, three label definitions. Ranking power (AUC) is similar; only the bad-rate LEVEL depends on the definition. The old label dropped the loans still open at the data cut."}
save_json(cens, "label_sensitivity.json"); cens
""")

# ======================================================================================
md(r"""
## Stage 8 - Business layer: master scale, grade backtest, cut-off economics, monitoring

**Master scale.** Eight grades (A = safest ... H = riskiest) from **geometric** PD bands, cut-points fixed on validation from the champion's calibrated PD. The
grade PD is the validation mean PD of the grade. On OOT we run a **binomial test** per grade (H0: true default rate = grade PD).

**Cut-off.** Approve if PD <= cut-off. Economics are **assumed, not observed** (public data has no pricing/cost data) and sit on the same **12-month horizon as the label**: a good loan earns a net margin of
a fixed % of principal over the 12 months; a bad loan loses EAD x LGD x principal. Losses that arrive after month 12 are outside the label, so the cut-off is optimistic about the riskiest borrowers
(see the lifetime-to-12-month ratio in Stage 7c); read the profit curve as a trade-off illustration. Cut-off is chosen on validation and then evaluated on OOT.

**Calibration tests (upgraded).** Each grade is tested three ways on OOT: the plain **binomial** (reported as a reference only), the **Jeffreys** posterior test, and a **Vasicek single-factor test**
that lets defaults share one economy (asset correlation rho = 0.03 for the traffic light, sensitivity 0.01-0.04). The monitoring traffic light below uses the Vasicek-adjusted test,
because a vintage is one draw of the economy and the binomial test treats 300,000 loans as 300,000 independent coin flips.

**Banker's note.** With hundreds of thousands of loans, the binomial test rejects almost any small gap, and defaults within a vintage are correlated (same economy), so the
test over-rejects. The correlation-adjusted test asks the right question: is the gap bigger than what one year's economy alone can explain?

**Plain-English note (calibration tests).** If every borrower in a year faces the same recession or boom, their defaults move together, so a year can miss the promised rate by a fair margin without the model being wrong.
The plain test ignores that and cries wolf; the adjusted one allows for it.

**Reject inference.** LendingClub's file contains only *accepted* loans. Applicants who were declined have no outcome, so the model describes
"people LendingClub approved". Applied to the full through-the-door population it would be biased (typically optimistic for the riskiest applicants).
A production build would add true reject inference (bureau performance of rejects, or an accept/reject frame with the lender's own cut-off history); that is **not done here**. Stage 14 instead compares accepted and rejected applications and runs a labelled parcelling *sensitivity*.

**Plain-English note.** We bucket borrowers into eight risk bands, check that each band's real default rate matches what we promised, and show how the
profit changes as the lender gets pickier. The profit numbers use stated assumptions - they illustrate the trade-off, they are not a forecast.
""")

code(r"""
# ---- Master scale (edges fixed on validation) ----------------------------------------------------
K = 8
lo, hi = np.quantile(ch_va, [0.03, 0.97])
edges = np.exp(np.linspace(np.log(lo), np.log(hi), K - 1)).tolist()
g_va, g_oot = L.assign_grade(ch_va, edges), L.assign_grade(ch, edges)
rows = []
for g in range(1, K + 1):
    mv, mo = g_va == g, g_oot == g
    pd_g = float(ch_va[mv].mean()); n = int(mo.sum()); k = int(yoot[mo].sum())
    bt = stats.binomtest(k, n, pd_g) if n else None
    jf = L.jeffreys_test(k, n, pd_g) if n else {"p_two": np.nan}
    vs = {r_: (L.vasicek_test(k, n, pd_g, r_) if n else {"p_two": np.nan}) for r_ in L.RHO_GRID}
    rows.append({"grade": L.grade_letter(g), "grade_index": g, "pd_lower": 0.0 if g == 1 else edges[g - 2], "pd_upper": 1.0 if g == K else edges[g - 1],
                 "master_scale_pd": pd_g, "score_lower": float(L.pd_to_score(1.0 if g == K else edges[g - 1])), "score_upper": float(L.pd_to_score(1e-6 if g == 1 else edges[g - 2])),
                 "share_valid": float(mv.mean()), "share_oot": float(mo.mean()), "n_oot": n, "bads_oot": k, "expected_bads_oot": pd_g * n,
                 "observed_rate_oot": k / n if n else np.nan, "binomial_p_two_sided": bt.pvalue if bt else np.nan,
                 "binomial_p_valid_in_sample": stats.binomtest(int(yva[mv].sum()), int(mv.sum()), pd_g).pvalue,
                 "jeffreys_p_two_sided": jf["p_two"], **{f"vasicek_p_two_rho{r_:.2f}": vs[r_]["p_two"] for r_ in L.RHO_GRID},
                 "vasicek_status_rho0.03": L.traffic_light(vs[L.RHO_BASE]["p_two"]) if n else "n/a",
                 "verdict_oot": ("under-predicted risk" if k / n > pd_g else "over-predicted risk") if (bt and bt.pvalue < 0.05) else "consistent (p>=0.05)",
                 "verdict_oot_vasicek": ("under-predicted risk" if k / n > pd_g else "over-predicted risk") if (n and vs[L.RHO_BASE]["p_two"] < 0.05) else "consistent with correlated defaults (p>=0.05)"})
ms = pd.DataFrame(rows); save_csv(ms, "master_scale.csv")
save_json({"champion": champion, "n_grades": K, "edges": edges, "grade_pd": ms.master_scale_pd.tolist(), "grades": ms.grade.tolist()}, "master_scale.json")
print(ms[["grade", "master_scale_pd", "share_oot", "observed_rate_oot", "binomial_p_two_sided", "jeffreys_p_two_sided", "vasicek_p_two_rho0.03", "vasicek_status_rho0.03"]].round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(K)
ax.bar(x - .2, ms.master_scale_pd * 100, .4, color=NAVY, label="master-scale PD (validation)"); ax.bar(x + .2, ms.observed_rate_oot * 100, .4, color=RED, label="actual OOT 2016")
ax.set_xticks(x); ax.set_xticklabels(ms.grade); ax.set_ylabel("default rate %"); ax.legend(); ax.set_title("Grade PD vs actual (OOT)")
savefig(fig, "grade_pd_vs_actual.png", "Blue is what each grade promised (validation), red is what happened in 2016. Grades should be ordered; gaps between the bars show where recalibration would be triggered.")
""")

code(r"""
# ---- Cut-off economics (ASSUMED) -----------------------------------------------------------------
ASSUME = {"margin_pct_of_principal": 0.06, "lgd": 0.85, "ead_pct_of_principal": 0.90,
          "note": "ASSUMED, not observed, on the 12-month horizon of the label. Margin = 12-month net interest margin after funding and servicing cost on a GOOD loan, as % of principal; a BAD loan (90+ DPD inside 12 months, so most principal still outstanding) loses EAD x LGD x principal. Flat pricing: LendingClub risk-based pricing is ignored; losses after month 12 are not in the label."}
amt_va, amt_oot = D.loan_amnt.values[va_idx].astype(float), D.loan_amnt.values[oot_idx].astype(float)
cv = L.cutoff_curve(ch_va, yva, amt_va, ASSUME["margin_pct_of_principal"], ASSUME["lgd"], ASSUME["ead_pct_of_principal"], 300)
co = L.cutoff_curve(ch, yoot, amt_oot, ASSUME["margin_pct_of_principal"], ASSUME["lgd"], ASSUME["ead_pct_of_principal"], 300)
best = cv.loc[cv.profit_per_application.idxmax()]
cut = float(best.cutoff_pd)
o_at = co.iloc[(co.cutoff_pd - cut).abs().argmin()]
cutinfo = {"assumptions": ASSUME, "chosen_on": "validation", "cutoff_pd": cut, "valid_approval_rate": float(best.approval_rate), "valid_bad_rate_approved": float(best.bad_rate_approved),
           "valid_profit_per_application": float(best.profit_per_application), "oot_approval_rate_at_cutoff": float(o_at.approval_rate), "oot_bad_rate_approved": float(o_at.bad_rate_approved),
           "oot_profit_per_application": float(o_at.profit_per_application), "oot_profit_approve_all": float(co.iloc[-1].profit_per_application),
           "oot_best_possible_cutoff_pd": float(co.loc[co.profit_per_application.idxmax()].cutoff_pd), "mean_loan_amount": float(amt_oot.mean())}
save_json(cutinfo, "cutoff.json"); save_csv(co, "cutoff_curve_oot.csv"); save_csv(cv, "cutoff_curve_valid.csv")
sens = []
for mg in (0.03, 0.06, 0.09, 0.12):
    for lg in (0.60, 0.75, 0.85, 0.95):
        c_ = L.cutoff_curve(ch_va, yva, amt_va, mg, lg, ASSUME["ead_pct_of_principal"], 300); b_ = c_.loc[c_.profit_per_application.idxmax()]
        o_ = L.cutoff_curve(ch, yoot, amt_oot, mg, lg, ASSUME["ead_pct_of_principal"], 300); oa = o_.iloc[(o_.cutoff_pd - b_.cutoff_pd).abs().argmin()]
        sens.append({"margin": mg, "lgd": lg, "optimal_cutoff_pd_valid": b_.cutoff_pd, "approval_rate_oot": oa.approval_rate, "bad_rate_approved_oot": oa.bad_rate_approved,
                     "profit_per_app_oot": oa.profit_per_application, "profit_per_app_oot_approve_all": o_.iloc[-1].profit_per_application})
sens = pd.DataFrame(sens); save_csv(sens, "cutoff_sensitivity.csv")
fig, ax = plt.subplots(figsize=(7.5, 4)); ax.plot(co.cutoff_pd, co.profit_per_application, color=NAVY, label="OOT 2016"); ax.plot(cv.cutoff_pd, cv.profit_per_application, color=GREY, ls="--", label="validation")
ax.axvline(cut, color=ORANGE, ls=":", label=f"cut-off chosen on validation ({cut:.3f})"); ax.axhline(0, color="k", lw=.5); ax.set_xlabel("approve if PD <= cut-off"); ax.set_ylabel("profit per application ($, ASSUMED economics)"); ax.legend(fontsize=8)
ax.set_title("Profit vs cut-off"); savefig(fig, "cutoff_profit.png", "Approving everyone is not best; profit peaks where the next-riskiest borrowers cost more in expected loss than they earn. The peak depends heavily on the assumed margin and LGD (see sensitivity table).")
print(json.dumps(cutinfo, indent=1)); sens.round(3).head(8)
""")

md(r"""
## Stage 8b - Calibration: intercept/slope, correlation-adjusted tests and a long-run anchor

**What.** (1) Calibration-in-the-large (intercept) and calibration slope on validation and OOT. (2) The per-grade tests of Stage 8 repeated at the portfolio level, plus the sensitivity of
the Vasicek verdict to the asset correlation rho. (3) **Long-run calibration**: the validation Platt calibrator is kept for ranking-consistent *point-in-time* PDs; for a *through-the-cycle* view its
intercept is shifted so that the average PD over the whole 2013-2015 development window equals the pooled 12-month default rate of those years, then a stated **margin of conservatism** is added.
2016 is never used for fitting; it only scores the result.

**Banker's note.** The validation window is only four months (Sep-Dec 2015), so a calibrator fitted on it carries that window's default level (a point-in-time view). Regulatory-style PDs are anchored to a
long-run average default rate and carry a margin of conservatism (MoC) that covers estimation error and data weakness. The MoC here is a **stated judgement (+10% relative)**, not an estimate; the table shows 0%, 10% and 20%.
Because 2016 is a worse vintage than 2013-2015, the long-run anchor lowers the average PD *further* below the 2016 outcome: that is the real cost of a through-the-cycle PD in a deteriorating year, and exactly why the
point-in-time calibrator is kept for pricing and cut-off work.

**Plain-English note.** The first calibration was learned from four recent months. The long-run version re-centres the model on what happened across three whole years and adds a deliberate safety cushion, then we check it on 2016.
""")

code(r"""
# ---- 8b.1 calibration intercept / slope (champion and scorecard) ------------------------------------------------------
EXTRA = {}
cal_diag = {}
for nm_ in dict.fromkeys([champion, "woe_scorecard"]):
    cal_diag[nm_] = {"valid": L.calibration_intercept_slope(yva, PC_va[nm_]), "oot": L.calibration_intercept_slope(yoot, PC_oot[nm_])}
# portfolio-level tests on OOT (champion, point-in-time calibration)
k_all, n_all, pd_all = int(yoot.sum()), len(yoot), float(ch.mean())
port = {"n": n_all, "bads": k_all, "predicted_pd": pd_all, "observed_rate": k_all / n_all,
        "binomial_p_two": L.binomial_pvalues(k_all, n_all, pd_all)["p_two"], "jeffreys_p_two": L.jeffreys_test(k_all, n_all, pd_all)["p_two"],
        **{f"vasicek_p_two_rho{r_:.2f}": L.vasicek_test(k_all, n_all, pd_all, r_)["p_two"] for r_ in L.RHO_GRID}}
# grade-level verdict as rho varies (how many of the 8 grades fail at 5%)
rho_sens = pd.DataFrame([{"rho": r_, "grades_failing_5pct": int((ms[f"vasicek_p_two_rho{r_:.2f}"] < 0.05).sum()), "grades_failing_1pct": int((ms[f"vasicek_p_two_rho{r_:.2f}"] < 0.01).sum())} for r_ in L.RHO_GRID] +
                        [{"rho": 0.0, "grades_failing_5pct": int((ms.binomial_p_two_sided < 0.05).sum()), "grades_failing_1pct": int((ms.binomial_p_two_sided < 0.01).sum())}]).sort_values("rho")
save_csv(rho_sens, "calibration_rho_sensitivity.csv")
tests = {"base_rho": L.RHO_BASE, "rho_grid": list(L.RHO_GRID), "calibration_intercept_slope": cal_diag, "portfolio_oot": port,
         "grades_failing_binomial_5pct": int((ms.binomial_p_two_sided < 0.05).sum()), "grades_failing_jeffreys_5pct": int((ms.jeffreys_p_two_sided < 0.05).sum()),
         "grades_failing_vasicek_5pct": int((ms["vasicek_p_two_rho0.03"] < 0.05).sum()), "grades_failing_vasicek_1pct": int((ms["vasicek_p_two_rho0.03"] < 0.01).sum()),
         "method_note": "Binomial assumes independent defaults (reference only). Jeffreys is a Bayesian posterior test, still independent. Vasicek integrates the binomial over one systematic factor with asset correlation rho."}
save_json(tests, "calibration_tests.json")
print(rho_sens.to_string(index=False))
print({k: (round(v["oot"]["intercept_citl"], 3), round(v["oot"]["slope"], 3)) for k, v in cal_diag.items()}, "portfolio:", {k: round(v, 4) for k, v in port.items() if "p_two" in k})
EXTRA["calibration_tests"] = tests
""")

code(r"""
# ---- 8b.2 long-run calibration (development window 2013-2015 only; 2016 is never used) ------------------------------
MOC_REL = 0.10                                          # STATED margin of conservatism: +10% relative on the long-run PD
dev_all = D["year"].between(2013, 2015).values
assert D.loc[dev_all, "issue_d"].max() < pd.Timestamp("2016-01-01") and not np.isin(oot_idx, np.where(dev_all)[0]).any()
LR_DR = float(yall[dev_all].mean())                    # pooled 12-month default rate of ALL 2013-2015 loans (not the sample)
by_year = D.loc[dev_all].groupby("year").apply(lambda g: float(yall[g.index].mean())).to_dict()
w_tr, w_va = n_train_full / len(tr_idx), int(is_valid.sum()) / len(va_idx)
raw_dev = {k: np.concatenate([(champ_raw(Xtr, A_tr, T_tr) if k == champion else P_tr[k]), P_va[k]]) for k in dict.fromkeys([champion, "woe_scorecard"])}
w_dev = np.concatenate([np.full(len(tr_idx), w_tr), np.full(len(va_idx), w_va)])
CAL_LR, lr_rows = {}, []
for k_, rv in raw_dev.items():
    for moc in (0.0, 0.10, 0.20):
        tgt = LR_DR * (1 + moc)
        dlt = L.anchor_shift(rv, CAL[k_], tgt, w_dev)
        cal_k = {"a": CAL[k_]["a"] + dlt, "b": CAL[k_]["b"], "shift_vs_validation_platt": dlt, "target_pd": tgt, "moc_rel": moc}
        pr_ = L.platt_apply(P_oot[k_], cal_k)
        lr_rows.append({"model": k_, "moc_rel": moc, "target_long_run_pd": tgt, "intercept_shift": dlt, "oot_mean_pred": float(pr_.mean()), "oot_obs_rate": float(yoot.mean()),
                        "oot_pred_over_actual": float(pr_.mean() / yoot.mean()), "oot_ece": L.ece(yoot, pr_), "oot_brier": L.all_metrics(yoot, pr_)["brier"],
                        "oot_vasicek_p_two_rho0.03": L.vasicek_test(int(yoot.sum()), len(yoot), float(pr_.mean()), L.RHO_BASE)["p_two"],
                        "oot_binomial_p_two": L.binomial_pvalues(int(yoot.sum()), len(yoot), float(pr_.mean()))["p_two"]})
        if moc == MOC_REL:
            CAL_LR[k_] = cal_k
    pit = L.platt_apply(P_oot[k_], CAL[k_])
    lr_rows.append({"model": k_, "moc_rel": np.nan, "target_long_run_pd": np.nan, "intercept_shift": 0.0, "oot_mean_pred": float(pit.mean()), "oot_obs_rate": float(yoot.mean()),
                    "oot_pred_over_actual": float(pit.mean() / yoot.mean()), "oot_ece": L.ece(yoot, pit), "oot_brier": L.all_metrics(yoot, pit)["brier"],
                    "oot_vasicek_p_two_rho0.03": L.vasicek_test(int(yoot.sum()), len(yoot), float(pit.mean()), L.RHO_BASE)["p_two"],
                    "oot_binomial_p_two": L.binomial_pvalues(int(yoot.sum()), len(yoot), float(pit.mean()))["p_two"], "variant": "BEFORE: validation-only Platt (point-in-time)"})
lr_tab = pd.DataFrame(lr_rows); lr_tab["variant"] = lr_tab["variant"].fillna("AFTER: long-run anchored, MoC " + (lr_tab.moc_rel * 100).round(0).astype("string") + "%")
save_csv(lr_tab, "calibration_longrun_oot.csv")
save_json({"champion": champion, "calibrators_longrun_moc10": CAL_LR, "moc_rel": MOC_REL}, "calibrators_longrun.json")
ch_lr = L.platt_apply(P_oot[champion], CAL_LR[champion])
need_moc = float(yoot.mean() / LR_DR - 1)                # informational, ex post: the MoC that would have covered 2016
lrinfo = {"champion": champion, "long_run_window": "issue years 2013-2015 (development window only; 2016 never used for fitting)", "long_run_default_rate": LR_DR,
          "default_rate_by_year": by_year, "validation_default_rate": float(yva.mean()), "moc_rel_stated": MOC_REL,
          "moc_note": "STATED judgement, not estimated: +10% relative to the long-run PD (table also shows 0% and 20%). Real MoC is sized from estimation error, data deficiency and model weakness.",
          "target_pd_with_moc": LR_DR * (1 + MOC_REL), "oot_obs_rate": float(yoot.mean()),
          "oot_pred_over_actual_before": float(ch.mean() / yoot.mean()), "oot_pred_over_actual_after": float(ch_lr.mean() / yoot.mean()),
          "oot_mean_pred_before": float(ch.mean()), "oot_mean_pred_after": float(ch_lr.mean()),
          "oot_ece_before": L.ece(yoot, ch), "oot_ece_after": L.ece(yoot, ch_lr),
          "ex_post_moc_that_would_have_covered_2016": need_moc, "intercept_shift_champion": CAL_LR[champion]["shift_vs_validation_platt"],
          "reading": "The long-run anchor is a through-the-cycle PD (capital / provisioning style). 2016 is a worse vintage than 2013-2015, so it sits further below the 2016 outcome than the point-in-time calibrator; the stated MoC closes only part of the gap."}
save_json(lrinfo, "calibration_longrun.json")
EXTRA["calibration_longrun"] = lrinfo
print(lr_tab[["model", "variant", "oot_mean_pred", "oot_pred_over_actual", "oot_ece", "oot_vasicek_p_two_rho0.03", "oot_binomial_p_two"]].round(4).to_string(index=False))

# by grade: before vs after vs observed (grades are the point-in-time master scale)
gl = []
for g in range(1, K + 1):
    m_ = g_oot == g
    n_, k_ = int(m_.sum()), int(yoot[m_].sum())
    p_lr = float(ch_lr[m_].mean())
    gl.append({"grade": L.grade_letter(g), "n_oot": n_, "observed_rate": k_ / n_, "pd_point_in_time": float(ch[m_].mean()), "pd_long_run_moc": p_lr,
               "vasicek_p_two_pit": L.vasicek_test(k_, n_, float(ch[m_].mean()), L.RHO_BASE)["p_two"], "vasicek_p_two_longrun": L.vasicek_test(k_, n_, p_lr, L.RHO_BASE)["p_two"]})
gl = pd.DataFrame(gl); save_csv(gl, "calibration_longrun_by_grade.csv")
qo = pd.DataFrame({"quarter": pd.PeriodIndex(D.issue_d.iloc[oot_idx], freq="Q").astype(str), "actual": yoot, "pit": ch, "lr": ch_lr}).groupby("quarter").mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x_ = np.arange(len(qo)); ax[0].bar(x_ - .27, qo.actual * 100, .27, color=RED, label="actual"); ax[0].bar(x_, qo.pit * 100, .27, color=NAVY, label="point-in-time (validation Platt)"); ax[0].bar(x_ + .27, qo.lr * 100, .27, color=TEAL, label="long-run anchored + MoC")
ax[0].set_xticks(x_); ax[0].set_xticklabels(qo.quarter); ax[0].set_ylabel("12-month bad rate %"); ax[0].legend(fontsize=7); ax[0].set_title("OOT 2016 by quarter: before vs after")
xg = np.arange(K); ax[1].bar(xg - .27, gl.observed_rate * 100, .27, color=RED, label="actual"); ax[1].bar(xg, gl.pd_point_in_time * 100, .27, color=NAVY, label="point-in-time"); ax[1].bar(xg + .27, gl.pd_long_run_moc * 100, .27, color=TEAL, label="long-run + MoC")
ax[1].set_xticks(xg); ax[1].set_xticklabels(gl.grade); ax[1].set_title("By grade"); ax[1].legend(fontsize=7)
savefig(fig, "calibration_longrun.png", "Red is what happened in 2016. Navy is the calibration fitted on four validation months, teal is the long-run (2013-2015) anchored version with a stated safety margin. 2016 was a worse year than the long-run average, so the through-the-cycle PD sits below it.")
pd.DataFrame([lrinfo]).T.head(14)
""")

code(r"""
# ---- Monitoring plan: thresholds and current status -----------------------------------------------
# The calibration traffic light is based on the CORRELATION-ADJUSTED (Vasicek, rho = 0.03) grade test; the plain binomial count is reported as a reference only.
gini_v = float(vm.loc[champion, "gini"]); row = cmp_.set_index("model").loc[champion]
fails_vas = int((ms["vasicek_p_two_rho0.03"] < 0.05).sum()); fails_bin = int((ms.binomial_p_two_sided < 0.05).sum()); fails = fails_vas
trig = [
 {"metric": "Score PSI (train vs current)", "amber": 0.10, "red": 0.25, "current": psi_score["psi_train_vs_oot"]},
 {"metric": "Max input CSI", "amber": 0.10, "red": 0.25, "current": float(csi.csi_train_vs_oot.max())},
 {"metric": "Gini decline vs validation (relative)", "amber": 0.10, "red": 0.20, "current": float(1 - row.oot_gini / gini_v)},
 {"metric": "Calibration: mean predicted PD / actual", "amber": "outside 0.9-1.1", "red": "outside 0.8-1.25", "current": float(row.oot_mean_pred / row.oot_obs_rate)},
 {"metric": "Grades failing correlation-adjusted (Vasicek rho=0.03) test at 5% (of 8)", "amber": 2, "red": 4, "current": fails_vas},
 {"metric": "Portfolio calibration: Vasicek two-sided p (rho=0.03)", "amber": "< 0.05", "red": "< 0.01", "current": tests["portfolio_oot"]["vasicek_p_two_rho0.03"]},
]
def status(t):
    c = t["current"]
    if t["metric"].startswith("Calibration"): return "RED" if (c < .8 or c > 1.25) else ("AMBER" if (c < .9 or c > 1.1) else "GREEN")
    if t["metric"].startswith("Portfolio calibration"): return L.traffic_light(c)
    return "RED" if c >= t["red"] else ("AMBER" if c >= t["amber"] else "GREEN")
for t in trig: t["status"] = status(t)
reference = [{"metric": "REFERENCE (not used for the status): grades failing plain binomial test at 5% (of 8)", "current": fails_bin, "status": "REFERENCE",
              "note": "Assumes independent defaults; over-rejects when a whole vintage shares one economy."},
             {"metric": "REFERENCE: grades failing Jeffreys test at 5% (of 8)", "current": int((ms.jeffreys_p_two_sided < 0.05).sum()), "status": "REFERENCE", "note": "Bayesian, still independent."}]
mon = {"triggers": trig, "reference_tests": reference, "calibration_basis": "correlation-adjusted (Vasicek single-factor, rho = 0.03; sensitivity 0.01-0.04 in calibration_rho_sensitivity.csv). The plain binomial count is a reference only.",
       "redevelopment_policy": "Recalibrate on a fresh validation window when any metric is AMBER; redevelop and revalidate when any is RED or two are AMBER for two consecutive quarters. Review quarterly; full revalidation at least annually.", "review_frequency": "monthly PSI/CSI, quarterly performance, annual validation"}
save_json(mon, "monitoring_triggers.json"); pd.DataFrame(trig)
""")

# ======================================================================================
md(r"""
## Stage 9 - Default-definition reconciliation

**What.** The label is a proxy (final status AND last-payment month <= 9). This stage reconciles it with what the file can show: (a) the distribution of the last-payment month for charged-off loans,
(b) how many loans that are late or defaulted *now* stopped paying in months 0-9 yet sit in the good class, and (c) how the bad rate and OOT ranking move if the cut-off month changes.

**Data limit.** The public file has **no charge-off date**, so "months from last payment to charge-off" cannot be measured directly. What it does give is the month on book of the last payment for every charged-off loan; with the usual
charge-off convention of roughly 120-150 days past due (about 4-5 missed instalments), a loan whose last payment fell in month k is charged off around month k+4 to k+5. That convention, not a measured lag, is why month 9 is used (90+ DPD by month 12).

**Banker's note.** A loan that is 31-120 days late at the 2018 snapshot but stopped paying in month 6 is a 12-month default under a pure 90+ DPD rule yet is labelled good here, because it has not (yet) been charged off. The count below is the size of that leakage.

**Plain-English note.** Some borrowers stopped paying early but have not been formally written off, so our label calls them good. We count them and show how much the answer changes if we call them bad instead.
""")

code(r"""
# ---- Stage 9: default-definition reconciliation ------------------------------------------------------------------
Dm = D[["issue_d", "year", "loan_status", "mob_last_pay"]].copy(); Dm["y"] = yall
co_mask = Dm.loan_status.isin(L.BAD_STATUS)
co_tab = (Dm[co_mask].assign(mob=lambda d: d.mob_last_pay.clip(upper=36).astype(int)).groupby("mob").size().rename("charged_off_loans").reset_index())
co_tab["share"] = co_tab.charged_off_loans / co_tab.charged_off_loans.sum(); co_tab["cum_share"] = co_tab.share.cumsum()
co_tab["in_bad_class"] = co_tab["mob"] <= L.BAD_MAX_MOB
save_csv(co_tab, "charged_off_lastpay_month_distribution.csv")
st_tab = []
for st_, g in Dm.groupby("loan_status"):
    st_tab.append({"loan_status": st_, "loans": len(g), "last_payment_month_0_9": int((g.mob_last_pay <= L.BAD_MAX_MOB).sum()), "last_payment_month_10_12": int(g.mob_last_pay.between(L.BAD_MAX_MOB + 1, 12).sum()),
                   "last_payment_after_12": int((g.mob_last_pay > 12).sum()), "share_labelled_bad": float(g.y.mean())})
st_tab = pd.DataFrame(st_tab).sort_values("loans", ascending=False); save_csv(st_tab, "status_vs_bad_class.csv")
late_set = [s_ for s_ in Dm.loan_status.dropna().unique() if ("Late" in s_ or "Grace" in s_ or s_ == "Default")]
late_early = Dm[Dm.loan_status.isin(late_set) & (Dm.mob_last_pay <= L.BAD_MAX_MOB)]
late_not_bad = late_early[~late_early.loan_status.isin(L.BAD_STATUS)]
# sensitivity: also call those loans bad, and move the cut-off month
y_alt = yall.copy(); y_alt[late_not_bad.index.values] = 1
rec_sens = []
for mx in (6, 7, 8, 9, 10, 12):
    yo_ = L.label_12m(D.iloc[oot_idx]["loan_status"], D.iloc[oot_idx]["mob_last_pay"].values, mx)
    rec_sens.append({"definition": f"charged off / default, last payment <= month {mx}" + (" (MAIN)" if mx == L.BAD_MAX_MOB else ""), "oot_bad_rate": float(yo_.mean()), "oot_champion_auc": float(roc_auc_score(yo_, ch))})
yo_ = y_alt[oot_idx]; rec_sens.append({"definition": "MAIN + late/grace statuses with last payment <= month 9 also bad", "oot_bad_rate": float(yo_.mean()), "oot_champion_auc": float(roc_auc_score(yo_, ch))})
rec_sens = pd.DataFrame(rec_sens); save_csv(rec_sens, "default_definition_sensitivity.csv")
chg_cum = co_tab.set_index("mob").cum_share
recon = {"definition": "bad = Charged Off / Default AND last payment <= month %d" % L.BAD_MAX_MOB, "population": "36-month loans issued 2013-2016",
         "charged_off_loans": int(co_mask.sum()), "charged_off_share_last_payment_month_le_9": float(chg_cum.loc[:L.BAD_MAX_MOB].iloc[-1]),
         "charged_off_share_last_payment_month_10_12": float(chg_cum.loc[:12].iloc[-1] - chg_cum.loc[:L.BAD_MAX_MOB].iloc[-1]), "charged_off_share_last_payment_after_12": float(1 - chg_cum.loc[:12].iloc[-1]),
         "charged_off_never_paid_month0_share": float(co_tab.set_index("mob").share.get(0, 0.0)),
         "median_last_payment_month_charged_off": float(Dm[co_mask].mob_last_pay.median()),
         "late_or_default_statuses_considered": late_set, "loans_in_those_statuses_with_last_payment_le_9": int(len(late_early)),
         "of_which_not_in_bad_class": int(len(late_not_bad)), "share_not_in_bad_class": float(len(late_not_bad) / max(len(late_early), 1)),
         "not_bad_as_share_of_labelled_bads": float(len(late_not_bad) / max(int(yall.sum()), 1)),
         "oot_bad_rate_main": float(yoot.mean()), "oot_bad_rate_if_added": float(y_alt[oot_idx].mean()),
         "charge_off_date_available": False,
         "lag_note": "No charge-off date in the public file: the last-payment-to-charge-off lag is not measured. Typical convention is charge-off at ~120-150 DPD, i.e. ~4-5 months after the last payment, which supports month 9 as the 90+ DPD-by-month-12 boundary."}
save_json(recon, "default_reconciliation.json"); EXTRA["default_reconciliation"] = recon
print(st_tab.to_string(index=False)); print(rec_sens.round(4).to_string(index=False)); print({k: v for k, v in recon.items() if "share" in k or "loans" in k})
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(co_tab["mob"], co_tab.charged_off_loans, color=[NAVY if b else GREY for b in co_tab.in_bad_class])
ax.axvline(L.BAD_MAX_MOB + .5, ls="--", color=ORANGE); ax.set_xlabel("month on book of the last payment (charged-off loans; 36 = 36 or later)"); ax.set_ylabel("charged-off loans"); ax.set_title("Charged-off loans by last-payment month: blue = labelled bad")
savefig(fig, "chargeoff_lastpay_month.png", "Blue bars are charged-off loans whose last payment fell by month 9 and are labelled bad; grey bars stopped paying later and count as good inside the 12-month window.")
rec_sens.round(4)
""")

# ======================================================================================
md(r"""
## Stage 10 - Rolling-origin backtest

**What.** The headline test is one split (train 2013-Aug 2015, test 2016). Here the same pipeline is re-run on three expanding windows: train 2013 -> test 2014, train 2013-14 -> test 2015, train 2013-15 -> test 2016.
In each fold the last ~20% of the training months is held out for Platt calibration (as in the main design), the WoE values and scorecard coefficients are refitted, and the monotone XGBoost is refitted. **Fixed from the main fit:** the
coarse-bin cut points, the scorecard variable list, the XGBoost hyper-parameters and the monotone constraint signs (selecting them again per fold would need the full binning pass). That choice lets fold 1-2 see a little look-ahead through those choices;
the test-year labels are never used.

**Banker's note.** A single out-of-time year can be lucky or unlucky. Stable Gini/KS across origins is the evidence that the ranking is not a one-off; the ECE and the predicted/actual ratio show how calibration behaves vintage by vintage.

**Plain-English note.** We repeat the exam three times, each time training only on earlier years, to see whether the model is consistently as good or just got lucky once.
""")

code(r"""
# ---- Stage 10: rolling-origin backtest ----------------------------------------------------------------------------
def fold_run(train_years, test_year):
    msk = D["year"].isin(train_years).values
    mth = D.loc[msk, "issue_d"]; cm = mth.value_counts().sort_index().cumsum() / msk.sum(); cut_m = cm.index[np.argmax(cm.values >= 0.80)]
    f_tr = strat_sample(msk & (D["issue_d"] <= cut_m).values, CAPS["train"], SEED)
    f_va = strat_sample(msk & (D["issue_d"] > cut_m).values, CAPS["valid"], SEED + 1)
    f_te = strat_sample((D["year"] == test_year).values, CAPS["oot"], SEED + 2)
    assert D.issue_d.iloc[f_tr].max() < D.issue_d.iloc[f_te].min() and D.issue_d.iloc[f_va].max() < D.issue_d.iloc[f_te].min()
    Xf_tr, Xf_va, Xf_te = Xall.iloc[f_tr], Xall.iloc[f_va], Xall.iloc[f_te]; yf_tr, yf_va, yf_te = yall[f_tr], yall[f_va], yall[f_te]
    out = {}
    # scorecard: same cut points / variables, WoE and coefficients refitted
    wf = L.WoEMap()
    for v in sc_vars:
        spec = woe.vars[v]; is_c = spec["kind"] == "cat"
        wf.fit_var(v, Xf_tr[v].astype("object").values if is_c else Xf_tr[v].values.astype(float), yf_tr, spec["kind"], splits=spec.get("splits"), groups=spec.get("groups"))
    Wf = wf.transform(Xf_tr, sc_vars); fit_f = sm.Logit(yf_tr, sm.add_constant(Wf[sc_vars])).fit(method="newton", maxiter=100, disp=0)
    scf = L.Scorecard(wf, sc_vars, {v: float(fit_f.params[v]) for v in sc_vars}, float(fit_f.params["const"]))
    # champion family: monotone XGB with the main hyper-parameters
    prep_f = L.RawPrep(L.NUMERIC_FEATURES, L.CATEGORICAL_FEATURES).fit(Xf_tr)
    names_f = prep_f.transform_trees(Xf_tr.iloc[:5])[1]
    mono_f = tuple(int(constraint.get(n, 0)) for n in names_f)
    xf = XGBClassifier(**bp, tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=SEED, monotone_constraints=mono_f).fit(prep_f.transform_trees(Xf_tr)[0], yf_tr)
    raw_f = {"woe_scorecard": (scf.raw_pd(Xf_va), scf.raw_pd(Xf_te)),
             "xgb_monotone": (xf.predict_proba(prep_f.transform_trees(Xf_va)[0])[:, 1], xf.predict_proba(prep_f.transform_trees(Xf_te)[0])[:, 1])}
    rows = []
    for k_, (pv, pt) in raw_f.items():
        cal_f = fit_platt(pv, yf_va); pc_ = L.platt_apply(pt, cal_f); m_ = L.all_metrics(yf_te, pc_)
        rows.append({"train_years": f"{min(train_years)}-{max(train_years)}", "test_year": test_year, "model": k_, "n_train": len(f_tr), "n_test": len(f_te),
                     "gini": 2 * roc_auc_score(yf_te, pt) - 1, "auc": roc_auc_score(yf_te, pt), "ks": L.ks_stat(yf_te, pt), "ece": m_["ece"], "brier": m_["brier"],
                     "mean_pred": m_["mean_pred"], "obs_rate": m_["obs_rate"], "pred_over_actual": m_["mean_pred"] / m_["obs_rate"]})
    return rows
t = time.time(); fold_rows = []
for tyrs, ty in [((2013,), 2014), ((2013, 2014), 2015), ((2013, 2014, 2015), 2016)]:
    fold_rows += fold_run(tyrs, ty); tick(f"fold train {tyrs[0]}-{tyrs[-1]} -> test {ty} done")
roll = pd.DataFrame(fold_rows); save_csv(roll, "rolling_origin_backtest.csv"); print(roll.round(4).to_string(index=False))
piv = roll.pivot(index="test_year", columns="model", values="gini")
roll_info = {"folds": roll.to_dict("records"), "gini_range_scorecard": [float(piv.woe_scorecard.min()), float(piv.woe_scorecard.max())], "gini_range_xgb_monotone": [float(piv.xgb_monotone.min()), float(piv.xgb_monotone.max())],
             "xgb_minus_scorecard_gini_by_year": {int(k): float(v) for k, v in (piv.xgb_monotone - piv.woe_scorecard).items()},
             "fixed_from_main_fit": ["coarse-bin cut points", "scorecard variable list", "XGBoost hyper-parameters", "monotone constraint signs"], "runtime_s": round(time.time() - t, 1)}
save_json(roll_info, "rolling_origin_summary.json"); EXTRA["rolling_origin"] = roll_info
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); w_ = .35; xt = np.arange(len(piv))
ax[0].bar(xt - w_ / 2, piv.woe_scorecard, w_, color=NAVY, label="WoE scorecard"); ax[0].bar(xt + w_ / 2, piv.xgb_monotone, w_, color=ORANGE, label="XGB monotone"); ax[0].set_xticks(xt); ax[0].set_xticklabels([f"test {y_}" for y_ in piv.index]); ax[0].set_ylabel("Gini"); ax[0].legend(); ax[0].set_title("Gini by origin")
pe_ = roll.pivot(index="test_year", columns="model", values="pred_over_actual"); ax[1].plot(xt, pe_.woe_scorecard, "o-", color=NAVY, label="WoE scorecard"); ax[1].plot(xt, pe_.xgb_monotone, "s-", color=ORANGE, label="XGB monotone"); ax[1].axhline(1, ls=":", color=GREY)
ax[1].set_xticks(xt); ax[1].set_xticklabels([f"test {y_}" for y_ in pe_.index]); ax[1].set_ylabel("mean predicted / actual"); ax[1].set_title("Calibration by origin"); ax[1].legend()
savefig(fig, "rolling_origin.png", "Each pair of bars is a model trained only on earlier years and tested on the next year. Similar heights mean the ranking power is stable; the right panel shows whether the calibration drifts as the economy changes.")
roll.round(4)
""")

# ======================================================================================
md(r"""
## Stage 11 - Realised economics (evaluation only)

**What.** The cut-off analysis uses *assumed* economics (6% margin, 85% LGD, 90% EAD on a 12-month horizon). LendingClub's file records what was actually collected: `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `funded_amnt`.
Here they are used **only to evaluate** the already-fitted model by score band and to redraw the cut-off curve with realised cash flows. They are post-origination fields: they are never features, never enter binning, tuning, calibration or the champion rule (a test asserts the feature list is disjoint).

**Caveats.** Realised cash is **lifetime to the data cut**, not 12-month, and before funding and servicing costs. The validation window (Sep-Dec 2015) is fully matured by Dec-2018; **2016 loans are not** (a Dec-2016 loan is only ~24 months old at the cut), so 2016 realised returns are partial and
understate lifetime interest on good loans. Realised LGD uses net recoveries (recoveries minus collection fees) on principal outstanding at default.

**Banker's note.** This is the back-test of the economic assumptions that a model validator would ask for: do the riskiest grades still earn more than they lose, and is the assumed LGD near the realised one?

**Plain-English note.** We replace our guessed profit numbers with what LendingClub actually collected, for the same loans, and see where the profit peak really is.
""")

code(r"""
# ---- Stage 11: realised economics, by grade and as a cut-off curve ------------------------------------------------------
need_cols = list(L.EVAL_ONLY_COLS)
if not all(c_ in D.columns for c_ in need_cols):
    print("evaluation-only cash-flow columns are not in this data cache; stage skipped"); real_info = {"skipped": True}
else:
    def realised_frame(idx, pd_, grade):
        d_ = D.iloc[idx]; fund = d_.funded_amnt.astype(float).values
        net_cash = (d_.total_pymnt.astype(float).fillna(0) - d_.collection_recovery_fee.astype(float).fillna(0)).values       # total_pymnt already includes recoveries
        rec_p = d_.total_rec_prncp.astype(float).values; net_rec = (d_.recoveries.astype(float).fillna(0) - d_.collection_recovery_fee.astype(float).fillna(0)).values
        return pd.DataFrame({"pd": pd_, "grade": grade, "y": yall[idx], "ever_co": d_.loan_status.isin(L.BAD_STATUS).astype(int).values, "fund": fund, "net_cash": net_cash,
                             "profit": net_cash - fund, "ead_at_default": fund - rec_p, "net_rec": net_rec})
    def by_grade(rf):
        rows = []
        for g in sorted(rf.grade.unique()):
            r_ = rf[rf.grade == g]; b_ = r_[r_.y == 1]
            rows.append({"grade": g, "loans": len(r_), "mean_pd": r_.pd.mean(), "bad_rate_12m": r_.y.mean(), "ever_charged_off_to_date": r_.ever_co.mean(),
                         "realised_net_return_on_funded": r_.profit.sum() / r_.fund.sum(), "realised_net_return_good_loans": r_[r_.y == 0].profit.sum() / r_[r_.y == 0].fund.sum(),
                         "realised_loss_on_bad_loans": -b_.profit.sum() / b_.fund.sum() if len(b_) else np.nan,
                         "realised_ead_share_bad": b_.ead_at_default.sum() / b_.fund.sum() if len(b_) else np.nan,
                         "realised_lgd_on_ead_bad": 1 - b_.net_rec.sum() / b_.ead_at_default.sum() if len(b_) else np.nan})
        return pd.DataFrame(rows)
    rf_va = realised_frame(va_idx, ch_va, [L.grade_letter(int(g)) for g in g_va]); rf_oo = realised_frame(oot_idx, ch, [L.grade_letter(int(g)) for g in g_oot])
    bg_va, bg_oo = by_grade(rf_va).assign(window="validation Sep-Dec 2015 (matured)"), by_grade(rf_oo).assign(window="OOT 2016 (partly matured)")
    save_csv(pd.concat([bg_va, bg_oo]), "realised_economics_by_grade.csv")
    chk = (D.iloc[oot_idx].total_pymnt - D.iloc[oot_idx].total_rec_prncp - D.iloc[oot_idx].total_rec_int - D.iloc[oot_idx].recoveries)
    def realised_curve(rf):
        o_ = rf.sort_values("pd"); cp = np.cumsum(o_.profit.values); n_ = len(o_); pos = np.unique(np.linspace(0, n_ - 1, 300).astype(int))
        return pd.DataFrame({"cutoff_pd": o_.pd.values[pos], "approval_rate": (pos + 1) / n_, "profit_per_application": cp[pos] / n_, "return_on_funded_approved": cp[pos] / np.cumsum(o_.fund.values)[pos]})
    rc_va, rc_oo = realised_curve(rf_va), realised_curve(rf_oo)
    save_csv(rc_va, "realised_cutoff_curve_valid.csv"); save_csv(rc_oo, "realised_cutoff_curve_oot.csv")
    def at(rc, c_): return rc.iloc[(rc.cutoff_pd - c_).abs().argmin()]
    bad_all = pd.concat([rf_va[rf_va.y == 1], rf_oo[rf_oo.y == 1]])
    real_info = {"horizon_note": "realised = lifetime cash to the data cut, before funding/servicing cost; 2016 loans only partly matured; total_pymnt includes recoveries",
                 "total_pymnt_minus_components_median_abs_oot": float(chk.abs().median()),
                 "assumed": {"margin_12m": ASSUME["margin_pct_of_principal"], "lgd": ASSUME["lgd"], "ead": ASSUME["ead_pct_of_principal"], "loss_per_bad_of_principal": ASSUME["lgd"] * ASSUME["ead_pct_of_principal"]},
                 "realised_valid": {"net_return_all": float(rf_va.profit.sum() / rf_va.fund.sum()), "net_return_good_loans": float(rf_va[rf_va.y == 0].profit.sum() / rf_va[rf_va.y == 0].fund.sum()),
                                    "loss_on_bad_loans": float(-rf_va[rf_va.y == 1].profit.sum() / rf_va[rf_va.y == 1].fund.sum())},
                 "realised_oot": {"net_return_all": float(rf_oo.profit.sum() / rf_oo.fund.sum()), "net_return_good_loans": float(rf_oo[rf_oo.y == 0].profit.sum() / rf_oo[rf_oo.y == 0].fund.sum()),
                                  "loss_on_bad_loans": float(-rf_oo[rf_oo.y == 1].profit.sum() / rf_oo[rf_oo.y == 1].fund.sum())},
                 "realised_ead_share_bad_loans": float(bad_all.ead_at_default.sum() / bad_all.fund.sum()), "realised_lgd_on_ead_bad_loans": float(1 - bad_all.net_rec.sum() / bad_all.ead_at_default.sum()),
                 "realised_loss_per_dollar_funded_bad": float(-bad_all.profit.sum() / bad_all.fund.sum()),
                 "assumed_cutoff_pd": cut, "realised_best_cutoff_pd_valid": float(rc_va.loc[rc_va.profit_per_application.idxmax(), "cutoff_pd"]), "realised_best_cutoff_pd_oot": float(rc_oo.loc[rc_oo.profit_per_application.idxmax(), "cutoff_pd"]),
                 "approval_at_assumed_cutoff_oot": float(at(rc_oo, cut).approval_rate), "realised_profit_per_app_at_assumed_cutoff_oot": float(at(rc_oo, cut).profit_per_application),
                 "realised_profit_per_app_approve_all_oot": float(rc_oo.iloc[-1].profit_per_application), "realised_best_profit_per_app_oot": float(rc_oo.profit_per_application.max()),
                 "realised_profit_per_app_at_assumed_cutoff_valid": float(at(rc_va, cut).profit_per_application), "realised_profit_per_app_approve_all_valid": float(rc_va.iloc[-1].profit_per_application)}
    save_json(real_info, "realised_economics.json"); EXTRA["realised_economics"] = real_info
    print(bg_va.round(3).to_string(index=False)); print(json.dumps({k: v for k, v in real_info.items() if k not in ("horizon_note",)}, indent=1, default=float)[:1800])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].plot(co.cutoff_pd, co.profit_per_application, color=GREY, ls="--", label="ASSUMED economics (OOT)"); ax[0].plot(rc_oo.cutoff_pd, rc_oo.profit_per_application, color=NAVY, label="REALISED cash (OOT, partly matured)")
    ax[0].plot(rc_va.cutoff_pd, rc_va.profit_per_application, color=TEAL, label="REALISED cash (validation, matured)"); ax[0].axvline(cut, color=ORANGE, ls=":"); ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_xlabel("approve if PD <= cut-off"); ax[0].set_ylabel("profit per application ($)"); ax[0].legend(fontsize=7); ax[0].set_title("Cut-off curve: assumed vs realised")
    ax[1].bar(np.arange(len(bg_va)) - .2, bg_va.realised_net_return_on_funded * 100, .4, color=TEAL, label="validation"); ax[1].bar(np.arange(len(bg_oo)) + .2, bg_oo.realised_net_return_on_funded * 100, .4, color=NAVY, label="OOT 2016")
    ax[1].set_xticks(np.arange(len(bg_va))); ax[1].set_xticklabels(bg_va.grade); ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("realised net return on funded %"); ax[1].set_title("Realised return by grade"); ax[1].legend(fontsize=7)
    savefig(fig, "realised_economics.png", "Left: profit per application against the cut-off using assumed economics (grey) and what was actually collected (colours). Right: realised net return by risk grade. Realised cash is lifetime-to-date and before funding cost, so levels are not comparable to the 12-month assumption, but the shape shows where the riskiest grades stop paying for themselves.")
real_info if real_info.get("skipped") else bg_va.round(3)
""")
# ======================================================================================
md(r"""
## Stage 12 - Discrimination with uncertainty: Somers' D confidence interval, CAP curve and accuracy ratio

**What.** For a binary outcome Somers' D equals the Gini coefficient (2 x AUC - 1). We report it with an analytic **DeLong** 95% interval (no resampling), and draw the **Cumulative Accuracy Profile (CAP)** with its
**accuracy ratio** (AR), the area-based discrimination measure used in Basel validation. For a binary outcome AR equals Gini; the table shows that they agree.

**Banker's note.** A point Gini without an interval cannot support a "model has deteriorated" claim. With ~300,000 OOT loans the interval is narrow (about +/-0.01), so the validation-to-OOT drop is real, not noise.

**Plain-English note.** The Gini is a single score of how well the model puts bad borrowers near the top. The interval says how much that score could wobble by chance; the CAP curve shows the share of all bad loans caught if we look at the riskiest X% of applicants.
""")

code(r"""
# ---- Stage 12: Somers' D CI and CAP / accuracy ratio -----------------------------------------------------------------------
disc_rows, cap_rows = [], []
for k_ in dict.fromkeys(["woe_scorecard", champion, "elasticnet_lr", "xgb_tuned"]):
    for nm_, yy, pp in (("validation", yva, PC_va[k_]), ("OOT 2016", yoot, PC_oot[k_])):
        dl = L.delong_auc_ci(yy, pp); ar, area = L.accuracy_ratio(yy, pp)
        disc_rows.append({"model": k_, "sample": nm_, "n": len(yy), "gini_somers_d": dl["somers_d"], "somers_d_ci_low": dl["somers_d_ci"][0], "somers_d_ci_high": dl["somers_d_ci"][1],
                          "auc": dl["auc"], "auc_ci_low": dl["auc_ci"][0], "auc_ci_high": dl["auc_ci"][1], "accuracy_ratio": ar, "cap_area": area, "ar_minus_gini": ar - dl["somers_d"]})
    xx, cc = L.cap_curve(yoot, PC_oot[k_]); cap_rows += [{"model": k_, "population_share": a, "bad_share_captured": b} for a, b in zip(xx, cc)]
disc = pd.DataFrame(disc_rows); save_csv(disc, "discrimination_ci.csv"); save_csv(pd.DataFrame(cap_rows), "cap_points.csv")
print(disc.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(5.8, 5)); cc_ = {"woe_scorecard": NAVY, champion: ORANGE, "elasticnet_lr": TEAL, "xgb_tuned": GREY}
for k_ in dict.fromkeys(["woe_scorecard", champion, "elasticnet_lr", "xgb_tuned"]):
    r_ = disc[(disc.model == k_) & (disc["sample"] == "OOT 2016")].iloc[0]; s_ = pd.DataFrame(cap_rows); s_ = s_[s_.model == k_]
    ax.plot(s_.population_share, s_.bad_share_captured, color=cc_[k_], label=f"{k_} (AR {r_.accuracy_ratio:.3f})")
pv_ = float(yoot.mean()); ax.plot([0, pv_, 1], [0, 1, 1], color="k", ls=":", lw=1, label="perfect model"); ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="random")
ax.set_xlabel("share of applicants (riskiest first)"); ax.set_ylabel("share of all bad loans captured"); ax.set_title("CAP curve, OOT 2016"); ax.legend(fontsize=7)
savefig(fig, "cap_curve.png", "The further a curve bows above the dashed diagonal, the better the model concentrates bad loans at the top of its risk ranking. The accuracy ratio (AR) summarises the area and equals the Gini for a yes/no outcome.")
EXTRA["discrimination"] = {"oot": disc[disc["sample"] == "OOT 2016"].set_index("model").to_dict("index"), "validation": disc[disc["sample"] == "validation"].set_index("model").to_dict("index")}
disc.round(4)
""")

# ======================================================================================
md(r"""
## Stage 13 - Fairness screen and adverse-action reasons

**What.** (1) **Approval-rate impact by income band** at the chosen cut-off on OOT 2016: approval rate per band, the adverse-impact ratio against the best-treated band (the "four-fifths" screen), and whether the model's PD is equally well calibrated in each band.
(2) **Reason codes mapped to adverse-action wording**: for declined OOT applicants, the top scorecard reasons are translated into ECOA / Regulation B style "principal reason" statements and counted.

**Limits.** The public file contains **no protected-class attributes** (race, sex, age, marital status) and geography is excluded from the features, so a true disparate-impact test is impossible here. Income is not itself a protected class; the band analysis is a *screen* for
unintended economic exclusion and a proxy check, not a legal conclusion. A four-fifths ratio below 0.80 is a flag for review, not proof of discrimination, and bands differ legitimately in risk.

**Banker's note.** Under Regulation B a declined applicant must receive the principal reasons for the decision; mapping each model input to a standard statement in advance is part of model documentation, and the mapping must be complete for every variable the model can use.

**Plain-English note.** We check that the approval rule does not shut out low-income applicants by more than their actual risk justifies, and that every "no" can be explained in plain words.
""")

code(r"""
# ---- Stage 13: fairness by income band + adverse-action reasons ---------------------------------------------------------------------
inc = D.annual_inc.values[oot_idx].astype(float)
bands = [(0, 40_000, "< $40k"), (40_000, 60_000, "$40-60k"), (60_000, 80_000, "$60-80k"), (80_000, 120_000, "$80-120k"), (120_000, 1e12, ">= $120k")]
appr = ch <= cut
pd_sc_oot = sc.calibrated_pd(Xoot)
fr = []
for lo_, hi_, lab in bands:
    m_ = (inc >= lo_) & (inc < hi_)
    if m_.sum() == 0: continue
    fr.append({"income_band": lab, "applicants": int(m_.sum()), "share_of_applicants": float(m_.mean()), "approval_rate": float(appr[m_].mean()), "mean_pd": float(ch[m_].mean()),
               "observed_bad_rate": float(yoot[m_].mean()), "pred_over_actual": float(ch[m_].mean() / yoot[m_].mean()), "bad_rate_among_approved": float(yoot[m_ & appr].mean()) if (m_ & appr).any() else np.nan,
               "approval_rate_if_scorecard_decides": float((pd_sc_oot[m_] <= np.quantile(pd_sc_oot, appr.mean())).mean())})
fr = pd.DataFrame(fr); ref_ap = fr.approval_rate.max(); fr["adverse_impact_ratio_vs_best_band"] = fr.approval_rate / ref_ap
fr["four_fifths_flag"] = np.where(fr.adverse_impact_ratio_vs_best_band < 0.8, "REVIEW (<0.80)", "ok (>=0.80)")
# risk-adjusted view: would the gap exist if every band had the same PD? approval within the same master-scale grade
gr_ap = pd.DataFrame({"band": pd.cut(inc, [b[0] for b in bands] + [1e12], right=False, labels=[b[2] for b in bands]), "grade": g_oot, "approved": appr}).groupby(["band", "grade"], observed=True).approved.mean().unstack()
save_csv(fr, "fairness_income_bands.csv"); save_csv(gr_ap.reset_index(), "fairness_income_by_grade.csv")
print(fr.round(3).to_string(index=False))
# adverse-action reasons for declined applicants (vectorised reason codes from the scorecard points table)
ptab = sc.points_table(); P_ = np.zeros((len(Xoot), len(sc_vars)))
for j_, v_ in enumerate(sc_vars):
    sub_ = ptab[ptab.variable == v_]; spec_ = woe.vars[v_]
    bi_ = woe.bin_index(v_, Xoot[v_].values.astype(float) if spec_["kind"] == "num" else Xoot[v_].astype("object").values)
    pts_by_bin = {i_: float(sub_[sub_.bin == woe.bin_label(v_, i_)].points.iloc[0]) if (sub_.bin == woe.bin_label(v_, i_)).any() else float(sub_.points.mean()) for i_ in np.unique(bi_)}
    P_[:, j_] = [pts_by_bin[i_] for i_ in bi_]
best_pts = np.array([ptab[ptab.variable == v_].points.max() for v_ in sc_vars]); short = best_pts[None, :] - P_
dec_mask = ~appr; top_idx = np.argsort(-short[dec_mask], axis=1)[:, :4]
vlist = np.array(sc_vars); aa_counts, top1 = {}, {}
for row_i, row_short in zip(top_idx, short[dec_mask]):
    seen = []
    for j_ in row_i:
        if row_short[j_] <= 0.5: continue
        t_ = L.adverse_action_text(vlist[j_])
        if t_ not in seen: seen.append(t_)
    for t_ in seen[:4]: aa_counts[t_] = aa_counts.get(t_, 0) + 1
    if seen: top1[seen[0]] = top1.get(seen[0], 0) + 1
n_dec = int(dec_mask.sum())
aa_tab = pd.DataFrame([{"adverse_action_reason": k_, "declined_applicants_citing": v_, "share_of_declined_citing": v_ / n_dec, "share_as_first_reason": top1.get(k_, 0) / n_dec} for k_, v_ in aa_counts.items()]).sort_values("declined_applicants_citing", ascending=False)
save_csv(aa_tab, "adverse_action_reasons_declined.csv")
aa_map = pd.DataFrame({"variable": sc_vars, "label": [L.LABELS[v_] for v_ in sc_vars], "plain_english_phrase": [L.reason_phrase(v_) for v_ in sc_vars], "adverse_action_statement": [L.adverse_action_text(v_) for v_ in sc_vars],
                       "mapped": [v_ in L.ADVERSE_ACTION for v_ in sc_vars]})
save_csv(aa_map, "adverse_action_map.csv"); print(aa_tab.head(8).round(3).to_string(index=False))
fair = {"cutoff_pd": cut, "protected_attributes_in_data": False, "geography_in_model": False, "income_in_model": "annual_inc" in sc_vars or "loan_to_income" in sc_vars,
        "lowest_adverse_impact_ratio": float(fr.adverse_impact_ratio_vs_best_band.min()), "bands_flagged_four_fifths": int((fr.adverse_impact_ratio_vs_best_band < 0.8).sum()),
        "best_approval_band": fr.loc[fr.approval_rate.idxmax(), "income_band"], "worst_approval_band": fr.loc[fr.approval_rate.idxmin(), "income_band"],
        "approval_rate_overall": float(appr.mean()), "declined_applicants": n_dec, "scorecard_variables_mapped_to_adverse_action": int(aa_map.mapped.sum()), "scorecard_variables": len(sc_vars),
        "caveat": "No protected-class data in the public file: this is an income-band screen, not a disparate-impact test; four-fifths < 0.80 is a review flag, not a finding."}
save_json(fair, "fairness.json"); EXTRA["fairness"] = fair
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8)); ax[0].bar(fr.income_band, fr.approval_rate * 100, color=NAVY); ax[0].axhline(0.8 * ref_ap * 100, ls="--", color=RED, label="80% of best band"); ax[0].set_ylabel("approval rate %"); ax[0].set_title("Approval by income band at the cut-off"); ax[0].legend(fontsize=8)
w_ = .38; xb = np.arange(len(fr)); ax[1].bar(xb - w_ / 2, fr.mean_pd * 100, w_, color=NAVY, label="mean predicted PD"); ax[1].bar(xb + w_ / 2, fr.observed_bad_rate * 100, w_, color=RED, label="observed bad rate"); ax[1].set_xticks(xb); ax[1].set_xticklabels(fr.income_band); ax[1].set_ylabel("%"); ax[1].legend(fontsize=8); ax[1].set_title("Calibration by income band (OOT)")
savefig(fig, "fairness_income.png", "Left: share of applicants approved in each income band at the cut-off, with the four-fifths screen line. Right: predicted vs observed default rate per band; similar bars mean the model is not systematically over- or under-rating a band.")
fr.round(3)
""")

# ======================================================================================
md(r"""
## Stage 14 - Through-the-door check with the rejected applications

**What.** The model is trained on accepted loans only. `rejected_2007_to_2018Q4` (same Kaggle dataset) lists the applications LendingClub declined, with only five usable fields: amount requested, DTI, employment length, `Risk_Score` and date. For 2013-2016 this stage
(1) compares accepted vs rejected distributions on those fields, (2) shows the accept share by band, and (3) runs a **parcelling sensitivity**.

**Caveats that matter.** (a) `Risk_Score` is a FICO score until roughly Nov-2013 and a **VantageScore afterwards**, whereas accepted `fico` stays FICO, so levels are not like-for-like after late 2013. (b) `Risk_Score` is **blank for most rejects from 2015** (see the coverage column), so any
score-based comparison is dominated by 2013-2014. (c) The rejected file has no term: accept share compares all-term accepted loans with rejects. (d) Rejects below ~660 lie outside the accepted score support, so a "same band" bad rate for them is an extrapolation from the lowest accepted band.

**Parcelling sensitivity (not reject inference).** Each reject is assigned a bad rate = k x the accepted bad rate in its band (Risk_Score band x DTI band, development window 2013-2015), for k = 1.5, 2, 3, as fractional bad/good rows. A small 4-variable scorecard (Risk_Score/FICO, DTI, log amount, employment length) is refitted on
accepted-only and on accepted + parcelled rejects and scored on accepted validation and OOT. This shows how fragile coefficients and Gini are to the unknown reject outcomes; it does **not** recover the true through-the-door default rate.

**Banker's note.** Real reject inference needs a bureau performance sample of declined applicants or an accept/reject modelling frame with the bank's own cut-off history; parcelling with an assumed multiplier is the simplest bracket on the answer.

**Plain-English note.** We look at the people who were turned away, see how different they look, and test how much our model would change if those people turned out to be riskier than similar accepted borrowers by 1.5x, 2x or 3x. It is a what-if, not a measurement.
""")

code(r"""
# ---- Stage 14: rejects (through-the-door) ---------------------------------------------------------------------------------------
rej_path = None
for pat in ("/kaggle/input/**/*rejected*2018*", ):
    hits_ = sorted([p_ for p_ in glob.glob(pat, recursive=True) if os.path.isfile(p_)], key=lambda p_: (not p_.endswith(".gz"), p_))
    rej_path = hits_[0] if hits_ else None
rej_path = os.environ.get("LC_REJ_DATA") or rej_path
reject_info = {"skipped": True, "reason": "rejected_2007_to_2018Q4 not found"}
if rej_path and os.path.isfile(rej_path):
    t = time.time()
    rej = L.load_rejects(rej_path, (2013, 2016), cache=os.environ.get("LC_REJ_CACHE")); tick(f"rejects loaded: {len(rej):,} applications 2013-2016 ({time.time() - t:.0f}s)")
    acc_cols = ["issue_d", "loan_amnt", "dti", "emp_length", "fico_range_low", "fico_range_high", "term"]
    acc_src = find_data()
    acc = pd.read_csv(acc_src, usecols=lambda c: c in acc_cols, low_memory=False)
    acc["issue_d"] = pd.to_datetime(acc["issue_d"], format="%b-%Y"); acc = acc[acc.issue_d.dt.year.between(2013, 2016)].copy()
    acc = pd.DataFrame({"year": acc.issue_d.dt.year.astype("int16").values, "month": acc.issue_d.dt.month.astype("int8").values, "amt": acc.loan_amnt.astype("float32").values,
                        "risk": ((acc.fico_range_low + acc.fico_range_high) / 2).astype("float32").values, "dti": acc.dti.astype("float32").values, "emp": L.parse_emp_length(acc.emp_length).astype("float32").values})
    rej.loc[(rej.dti < 0) | (rej.dti >= 999), "dti"] = np.nan; acc.loc[(acc.dti < 0) | (acc.dti >= 999), "dti"] = np.nan
    # (1) distributions
    dist = []
    for yr in (2013, 2014, 2015, 2016):
        for var in ("dti", "amt", "emp", "risk"):
            a_, r_ = acc.loc[acc.year == yr, var].dropna(), rej.loc[rej.year == yr, var].dropna()
            dist.append({"year": yr, "variable": {"dti": "DTI (%)", "amt": "Loan amount ($)", "emp": "Employment length (years)", "risk": "Risk_Score (reject) / FICO (accepted)"}[var],
                         "accepted_n": len(a_), "rejected_n": len(r_), "accepted_coverage": len(a_) / (acc.year == yr).sum(), "rejected_coverage": len(r_) / (rej.year == yr).sum(),
                         "accepted_p10": a_.quantile(.1), "accepted_median": a_.median(), "accepted_p90": a_.quantile(.9), "rejected_p10": r_.quantile(.1), "rejected_median": r_.median(), "rejected_p90": r_.quantile(.9),
                         "psi_accepted_vs_rejected": L.psi_numeric(a_.sample(min(len(a_), 300_000), random_state=0), r_.sample(min(len(r_), 300_000), random_state=0)) if len(a_) > 100 and len(r_) > 100 else np.nan})
    dist = pd.DataFrame(dist); dist["score_scale_caveat"] = np.where(dist.variable.str.startswith("Risk") & (dist.year >= 2014), "Vantage (rejects) vs FICO (accepted): not like-for-like", "")
    save_csv(dist, "reject_vs_accepted_distributions.csv")
    # (2) accept share by band
    def band(df):
        return pd.DataFrame({"year": df.year.values, "DTI": pd.cut(df.dti, [-0.01, 10, 20, 30, 40, 1e9], labels=["0-10", "10-20", "20-30", "30-40", "40+"]).astype(str),
                             "Loan amount": pd.cut(df.amt, [0, 5000, 10000, 15000, 20000, 30000, 1e9], labels=["<5k", "5-10k", "10-15k", "15-20k", "20-30k", "30k+"]).astype(str),
                             "Employment": pd.cut(df.emp, [-1, 0, 4, 9, 11], labels=["<1y", "1-4y", "5-9y", "10y+"]).astype(str),
                             "Risk_Score/FICO": pd.cut(df.risk, [0, 640, 660, 680, 700, 720, 1000], labels=["<640", "640-659", "660-679", "680-699", "700-719", "720+"]).astype(str)})
    ba, br = band(acc), band(rej); share_rows = []
    for var in ("DTI", "Loan amount", "Employment", "Risk_Score/FICO"):
        ca = ba[ba[var] != "nan"].groupby(["year", var]).size().rename("accepted"); cr = br[br[var] != "nan"].groupby(["year", var]).size().rename("rejected")
        t_ = pd.concat([ca, cr], axis=1).fillna(0).reset_index().rename(columns={var: "band"}); t_.insert(0, "variable", var); t_["accept_share"] = t_.accepted / (t_.accepted + t_.rejected); share_rows.append(t_)
    share = pd.concat(share_rows); save_csv(share, "accept_share_by_band.csv")
    overall = acc.groupby("year").size() / (acc.groupby("year").size() + rej.groupby("year").size())
    # (3) parcelling sensitivity on the development window 2013-2015, rejects WITH a Risk_Score
    Dd = D.iloc[np.concatenate([tr_idx, va_idx])]; fico_dev = Xall.iloc[np.concatenate([tr_idx, va_idx])]["fico"].values
    dev_df = pd.DataFrame({"risk": fico_dev, "dti": Xall.iloc[np.concatenate([tr_idx, va_idx])]["dti"].values, "amt": Dd.loan_amnt.values.astype(float), "emp": Xall.iloc[np.concatenate([tr_idx, va_idx])]["emp_length"].values,
                           "y": np.concatenate([ytr, yva]), "w": np.concatenate([np.full(len(tr_idx), w_tr), np.full(len(va_idx), w_va)])})
    r_edges = [-np.inf, 680, 700, 720, 740, np.inf]; d_edges = [-np.inf, 12, 20, 28, np.inf]            # the lowest accepted band is also used for scores below it (extrapolation)
    keyf = lambda df: pd.cut(df.risk, r_edges, labels=False).astype("float") * 10 + pd.cut(df.dti, d_edges, labels=False).astype("float")   # noqa: E731
    dev_df["band"] = keyf(dev_df); bad_rate_band = dev_df.dropna(subset=["band"]).groupby("band").apply(lambda g: np.average(g.y, weights=g.w))
    rej_dev = rej[rej.year.between(2013, 2015) & rej.risk.notna() & rej.dti.notna() & rej.dti.between(0, 100)].copy()
    n_rej_all = int(((rej.year.between(2013, 2015))).sum()); share_with_score = len(rej_dev) / n_rej_all
    rej_dev["band"] = keyf(rej_dev); rej_dev = rej_dev.dropna(subset=["band"])
    rej_dev["emp"] = rej_dev.emp.fillna(0.0); below_support = float((rej_dev.risk < 660).mean())
    rs_ = rej_dev.sample(min(len(rej_dev), 150_000 if not FAST else 30_000), random_state=SEED).copy(); scale_w = len(rej_dev) / len(rs_)
    def feat4(df): return np.column_stack([df.risk.fillna(690).clip(600, 850).values, df.dti.fillna(17.0).clip(0, 60).values, np.log(df.amt.clip(500, 40_000).values), df.emp.fillna(0).values])
    acc_tr = dev_df.sample(min(len(dev_df), 300_000), random_state=SEED); Xa = feat4(acc_tr); mu, sdv = Xa.mean(0), Xa.std(0)
    zf = lambda M: (M - mu) / sdv   # noqa: E731
    Xv_, Xo_ = zf(feat4(pd.DataFrame({"risk": Xva["fico"].values, "dti": Xva["dti"].fillna(Xtr["dti"].median()).values, "amt": D.loan_amnt.values[va_idx].astype(float), "emp": Xva["emp_length"].values}))), \
               zf(feat4(pd.DataFrame({"risk": Xoot["fico"].values, "dti": Xoot["dti"].fillna(Xtr["dti"].median()).values, "amt": D.loan_amnt.values[oot_idx].astype(float), "emp": Xoot["emp_length"].values})))
    fnames = ["Risk_Score/FICO", "DTI", "log loan amount", "employment length"]
    Xa = zf(Xa); Xa_fit, ya_fit, wa_fit = Xa, acc_tr.y.values, acc_tr.w.values
    sens_rows = []
    def fit_eval(Xf, yf, wf, label, k_):
        m_ = LogisticRegression(C=1e6, max_iter=400).fit(Xf, yf, sample_weight=wf)
        row = {"scenario": label, "k": k_, **{f"coef_{n_}": float(c_) for n_, c_ in zip(fnames, m_.coef_[0])}, "intercept": float(m_.intercept_[0]),
               "gini_accepted_valid": 2 * roc_auc_score(yva, m_.decision_function(Xv_)) - 1, "gini_accepted_oot": 2 * roc_auc_score(yoot, m_.decision_function(Xo_)) - 1}
        return row, m_
    base_row, base_m = fit_eval(Xa_fit, ya_fit, wa_fit, "accepted only", np.nan); sens_rows.append(base_row)
    Xr = zf(feat4(rs_)); acc_total_w = dev_df.w.sum(); rej_total_w = len(rej_dev)
    for k_ in (1.5, 2.0, 3.0):
        pbad = np.clip(rs_.band.map(bad_rate_band).fillna(bad_rate_band.mean()).values * k_, 0, 1)
        Xf = np.vstack([Xa_fit, Xr, Xr]); yf = np.r_[ya_fit, np.ones(len(Xr)), np.zeros(len(Xr))]
        wf = np.r_[wa_fit, scale_w * pbad, scale_w * (1 - pbad)]
        row, m_ = fit_eval(Xf, yf, wf, f"accepted + rejects parcelled at {k_}x band bad rate", k_)
        row["implied_through_the_door_bad_rate"] = float((np.average(ya_fit, weights=wa_fit) * acc_total_w + (scale_w * pbad).sum()) / (acc_total_w + rej_total_w))
        row["mean_reject_assigned_bad_rate"] = float(pbad.mean()); sens_rows.append(row)
    sens_rej = pd.DataFrame(sens_rows)
    for n_ in fnames: sens_rej[f"coef_{n_}_vs_accepted_only_pct"] = (sens_rej[f"coef_{n_}"] / sens_rej.loc[0, f"coef_{n_}"] - 1) * 100
    save_csv(sens_rej, "reject_parcelling_sensitivity.csv"); print(sens_rej[["scenario", "gini_accepted_valid", "gini_accepted_oot"] + [f"coef_{n_}" for n_ in fnames] + ["implied_through_the_door_bad_rate"]].round(4).to_string(index=False))
    reject_info = {"skipped": False, "window": "2013-2016", "rejected_applications_2013_2016": int(len(rej)), "accepted_applications_2013_2016_all_terms": int(len(acc)),
                   "accept_share_by_year": {int(k): float(v) for k, v in overall.items()}, "risk_score_coverage_rejects_by_year": {int(k): float(v) for k, v in rej.groupby("year").risk.apply(lambda s: s.notna().mean()).items()},
                   "risk_score_caveat": "Risk_Score is FICO until about Nov-2013 and VantageScore afterwards; accepted fico is always FICO. Blank for most rejects from 2015.",
                   "median_dti_accepted_vs_rejected_2016": [float(acc[acc.year == 2016].dti.median()), float(rej[rej.year == 2016].dti.median())],
                   "median_amount_accepted_vs_rejected_2016": [float(acc[acc.year == 2016].amt.median()), float(rej[rej.year == 2016].amt.median())],
                   "median_risk_accepted_vs_rejected_2013": [float(acc[acc.year == 2013].risk.median()), float(rej[rej.year == 2013].risk.median())],
                   "share_rejects_below_660_where_scored_2013_2014": float((rej[rej.year.between(2013, 2014)].risk.dropna() < 660).mean()),
                   "parcelling": {"label": "SENSITIVITY, not reject inference", "k_values": [1.5, 2.0, 3.0], "bands": "Risk_Score band (5) x DTI band (4), development window 2013-2015", "rejects_with_score_share_of_2013_2015_rejects": float(share_with_score),
                                  "share_of_scored_rejects_below_accepted_floor_660": below_support, "rejects_sampled": int(len(rs_)),
                                  "gini_oot_accepted_only": float(sens_rej.loc[0, "gini_accepted_oot"]), "gini_oot_by_k": {str(r_.k): float(r_.gini_accepted_oot) for r_ in sens_rej.iloc[1:].itertuples()},
                                  "implied_through_the_door_bad_rate_by_k": {str(r_.k): float(r_.implied_through_the_door_bad_rate) for r_ in sens_rej.iloc[1:].itertuples()},
                                  "max_abs_coef_change_pct": float(sens_rej.iloc[1:][[f"coef_{n_}_vs_accepted_only_pct" for n_ in fnames]].abs().max().max())}}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for j_, (var, bins_) in enumerate([("dti", np.linspace(0, 60, 41)), ("amt", np.linspace(0, 40_000, 41))]):
        ax[j_].hist(acc.loc[acc.year == 2015, var].dropna(), bins_, density=True, alpha=.6, color=NAVY, label="accepted"); ax[j_].hist(rej.loc[rej.year == 2015, var].dropna(), bins_, density=True, alpha=.5, color=RED, label="rejected")
        ax[j_].set_title({"dti": "DTI (%), 2015", "amt": "Loan amount ($), 2015"}[var]); ax[j_].legend(fontsize=8)
    sh_ = share[(share.variable == "DTI")].groupby("band").apply(lambda g: g.accepted.sum() / (g.accepted.sum() + g.rejected.sum())); ax[2].bar(sh_.index, sh_.values * 100, color=NAVY); ax[2].set_title("Accept share by DTI band, 2013-2016"); ax[2].set_ylabel("% of applications accepted")
    savefig(fig, "reject_vs_accepted.png", "Left and middle: accepted applications (blue) against rejected ones (red); rejects ask for similar amounts but carry higher DTI. Right: the share of applications accepted falls as DTI rises. Risk_Score is not plotted because it is blank for most rejects after 2014.")
    del rej, acc
EXTRA["reject_check"] = reject_info
reject_info if reject_info.get("skipped") else sens_rej.round(4)
""")

# ======================================================================================
md(r"""
## Stage 15 - Model roles and monotonicity evidence for the boosted model

**Roles (stated explicitly).**

| Role | Model | Why |
|---|---|---|
| **Decision model of record (explainable)** | **WoE points scorecard** | Every driver is a bin with points; reason codes and adverse-action statements come from it; anyone can recompute a score by hand. |
| **Challenger / benchmark** | **Monotone XGBoost** | Sets the accuracy ceiling for application-time data and wins the frozen accuracy rule on validation (`champion_rule.json`); in this repository it supplies the PD used for grading and the cut-off, with the scorecard supplying score and reasons. |
| Reference only | XGBoost + LendingClub grade | Not deployable; sizes what the excluded variables would add. |

The boosted model is only defensible next to a scorecard if its behaviour is **economically sensible, feature by feature**. Below, for the most important numeric features, we check (a) the monotone constraint that was applied, and (b) the *learned* response using
**partial dependence (PDP)** and **accumulated local effects (ALE)**, which stay reliable when features are correlated, against the expected sign.

**Banker's note.** Monotone constraints guarantee the direction only for the features that carry one; unconstrained features (where the business sign is unclear or the data disagreed) and interactions need evidence, not assertion. SR 11-7 style review asks exactly this
before a more complex model is allowed to influence a decision.

**Plain-English note.** For each important input, we nudge it up and down while holding everything else fixed and confirm that the boosted model moves the way common sense says (more debt, more risk; higher credit score, less risk).
""")

code(r"""
# ---- Stage 15: PDP / ALE monotonicity checks on the monotone XGBoost -------------------------------------------------------
bst_m = xgb_mono.get_booster()
margin_fn = lambda M: bst_m.predict(xgb.DMatrix(M), output_margin=True)   # noqa: E731
mono_sample = T_va[np.random.default_rng(SEED).choice(len(T_va), min(len(T_va), 4000 if not FAST else 1500), replace=False)]
num_imp = imp[imp.feature.isin(L.NUMERIC_FEATURES)].head(12 if not FAST else 8)
curves, mrows = [], []
for f_ in num_imp.feature:
    j_ = tree_names.index(f_); col = mono_sample[:, j_]
    if np.isnan(col).all(): continue
    grid = np.unique(np.nanquantile(col, np.linspace(0.02, 0.98, 15)))
    pdp = L.pdp_1d(margin_fn, mono_sample, j_, grid); ax_, ale = L.ale_1d(margin_fn, mono_sample, j_, 12)
    for g_, v_ in zip(grid, pdp): curves.append({"feature": f_, "kind": "PDP", "x": g_, "log_odds": v_})
    for g_, v_ in zip(ax_, ale): curves.append({"feature": f_, "kind": "ALE", "x": g_, "log_odds": v_})
    exp_ = L.EXPECTED_SIGN[f_]; sp, sa = L.monotone_summary(pdp, exp_), L.monotone_summary(ale, exp_)
    mrows.append({"feature": f_, "label": L.LABELS[f_], "shap_rank": int(np.where(imp.feature.values == f_)[0][0]) + 1, "expected_sign": exp_, "constraint_applied": int(constraint[f_]),
                  "pdp_direction": sp["direction"], "pdp_against_share": sp["against_share"], "ale_direction": sa["direction"], "ale_against_share": sa["against_share"],
                  "pdp_consistent_with_expected": sp["consistent"], "ale_consistent_with_expected": sa["consistent"],
                  "verdict": ("constrained and monotone" if constraint[f_] != 0 and sp["against_share"] < 1e-6 else
                              ("unconstrained: " + ("sign as expected" if sa["consistent"] else ("no expectation set" if exp_ == 0 else "REVIEW: ALE departs from expected sign"))))})
mono_ev = pd.DataFrame(mrows); save_csv(mono_ev, "pdp_ale_monotonicity.csv"); save_csv(pd.DataFrame(curves), "pdp_ale_curves.csv")
con_ = mono_ev[mono_ev.constraint_applied != 0]; unc_ = mono_ev[mono_ev.constraint_applied == 0]
roles = {"decision_model_of_record": "woe_scorecard", "decision_model_rationale": "points table, reason codes and adverse-action statements; fully explainable and recomputable by hand",
         "challenger_benchmark": "xgb_monotone", "challenger_rationale": "highest application-time accuracy; wins the frozen validation rule; supplies the PD for grading and the cut-off in this repository",
         "frozen_rule_champion": champion, "top_numeric_features_checked": int(len(mono_ev)), "constrained_among_checked": int(len(con_)),
         "constrained_pdp_exactly_monotone": int((con_.pdp_against_share < 1e-6).sum()), "constrained_ale_in_expected_direction": int(con_.ale_consistent_with_expected.fillna(False).sum()),
         "unconstrained_among_checked": int(len(unc_)), "unconstrained_ale_in_expected_direction": int(unc_.ale_consistent_with_expected.fillna(False).sum()),
         "unconstrained_with_no_expectation": int((unc_.expected_sign == 0).sum()), "features_flagged_for_review": mono_ev[mono_ev.verdict.str.startswith("unconstrained: REVIEW")].feature.tolist(),
         "promotion_conditions_for_xgb_as_decision_model": ["reason codes via SHAP or scorecard surrogate accepted by compliance", "monotonicity evidence above re-run on each refresh", "OOT gain over the scorecard stays above the frozen 0.010 AUC margin in the rolling-origin backtest"]}
save_json(roles, "model_roles.json"); EXTRA["model_roles"] = roles
print(mono_ev[["feature", "shap_rank", "expected_sign", "constraint_applied", "pdp_direction", "pdp_against_share", "ale_against_share", "verdict"]].round(3).to_string(index=False))
sel_ = mono_ev.feature.head(8).tolist(); fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax_, f_ in zip(axes.ravel(), sel_):
    cc_ = pd.DataFrame(curves); cc_ = cc_[cc_.feature == f_]
    for kd, colr in (("PDP", NAVY), ("ALE", ORANGE)):
        s_ = cc_[cc_.kind == kd]; ax_.plot(s_.x, s_.log_odds - s_.log_odds.mean(), "o-", ms=3, color=colr, label=kd)
    ax_.set_title(f"{f_}  (constraint {int(constraint[f_]):+d}, expected {L.EXPECTED_SIGN[f_]:+d})", fontsize=8); ax_.legend(fontsize=7)
fig.supylabel("effect on log-odds of default (centred)"); fig.tight_layout()
savefig(fig, "pdp_ale_top_features.png", "How the boosted model's predicted risk moves as one input changes, other inputs held as observed. Rising curves mean risk goes up with the feature. Constrained features are flat or one-directional by construction; the others are checked against business expectation.")
mono_ev.round(3)
""")

# ======================================================================================
md(r"""
## Finalise: save models, headline metrics, manifest

All serving logic lives in JSON (bins, coefficients, scaling, preprocessing parameters, XGBoost boosters) so the Streamlit app
can score an applicant **without retraining and without pickles from a different library version**. Joblib copies are also saved.
""")

code(r"""
import joblib
# ---- models -----------------------------------------------------------------------------------------
save_json(sc.to_dict(), "scorecard.json")
save_json(prep.to_dict(), "raw_prep.json"); save_json(prep_all.to_dict(), "tree_prep.json")
save_json({"names": tree_names, "monotone": list(mono_vec)}, "tree_features.json")
joblib.dump({"scorecard": sc, "l2": lr_models["l2"], "l1": lr_models["l1"], "elasticnet": lr_models["elasticnet"], "prep": prep, "woe_map": woe}, MOD / "logistic_ladder.joblib")
joblib.dump(xgb_tuned, MOD / "xgb_tuned.joblib"); joblib.dump(xgb_mono, MOD / "xgb_monotone.joblib")
xgb_tuned.get_booster().save_model(str(MOD / "xgb_tuned.json")); xgb_mono.get_booster().save_model(str(MOD / "xgb_monotone.json"))
save_json({"elasticnet_coef": dict(zip(names_lr, lr_models["elasticnet"].coef_[0])), "elasticnet_intercept": float(lr_models["elasticnet"].intercept_[0])}, "elasticnet_model.json")

# ---- app support files ---------------------------------------------------------------------------
sc_raw_cols = [v for v in L.ALL_FEATURES]
schema = {}
for c in L.NUMERIC_FEATURES:
    xs = Xtr[c]
    schema[c] = {"kind": "num", "label": L.LABELS[c], "group": L.GROUPS[c], "default": None if xs.dropna().empty else float(xs.median()),
                 "min": float(xs.quantile(.005)), "max": float(xs.quantile(.995)), "missing_share": float(xs.isna().mean()), "in_scorecard": c in sc_vars}
for c in L.CATEGORICAL_FEATURES:
    lv = prep_all.cat_params[c]["levels"]; schema[c] = {"kind": "cat", "label": L.LABELS[c], "group": "Applicant", "levels": lv, "default": lv[0], "in_scorecard": c in sc_vars}
save_json(schema, "feature_schema.json")
ex_idx = np.random.default_rng(3).choice(len(oot_idx), 12, replace=False)
exs = []
for i in ex_idx:
    r = Xoot.iloc[i].to_dict()
    exs.append({"features": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}, "actual_bad": int(yoot[i]), "pd_champion": float(ch[i]), "grade": L.grade_letter(int(g_oot[i]))})
save_json({"examples": exs}, "applicant_examples.json")
scored = pd.DataFrame({"issue_d": D.issue_d.values[oot_idx], "y": yoot, "loan_amnt": amt_oot, "score": S_oot, "grade": [L.grade_letter(int(g)) for g in g_oot]} | {f"pd_{k}": v for k, v in PC_oot.items()} | {"pd_champion": ch, "pd_champion_longrun_moc": ch_lr})
scored.to_csv(ART / "oot_scored.csv.gz", index=False, compression="gzip")
st = pd.DataFrame({"score": S_va, "y": yva}); st.to_csv(ART / "valid_scored.csv.gz", index=False, compression="gzip")
sbin = pd.DataFrame({"score": np.round(np.linspace(S_oot.min(), S_oot.max(), 40), 1)}); sbin["pd"] = L.score_to_pd(sbin.score)
save_csv(sbin, "score_to_pd_scorecardscale.csv")

# ---- headline metrics.json ---------------------------------------------------------------------------------
c_ = cmp_.set_index("model")
metrics = {
  "project": "LendingClub application PD scorecard (36-month loans)", "champion": champion, "champion_rule_file": "champion_rule.json",
  "data": {"source": "Kaggle wordsforthewise/lending-club accepted_2007_to_2018Q4", "rows_in_file": N_FILE, "snapshot": f"{SNAPSHOT:%b-%Y}", "train_rows_used": int(len(tr_idx)), "train_rows_available": n_train_full,
           "valid_rows": int(len(va_idx)), "oot_rows": int(len(oot_idx)), "sampling_note": SAMPLING_NOTE, "train_window": split.loc[0, "issue_window"], "valid_window": split.loc[1, "issue_window"], "oot_window": "Jan-2016 to Dec-2016",
           "bad_rate_train": float(ytr.mean()), "bad_rate_valid": float(yva.mean()), "bad_rate_oot": float(yoot.mean()), "n_features_candidate": len(L.ALL_FEATURES), "n_pass_iv": len(keep0), "n_after_corr": len(keep1), "n_after_vif": len(keep2), "n_scorecard_vars": len(sc_vars)},
  "oot": {k: {m: float(c_.loc[k, f"oot_{m}"]) for m in ("auc", "gini", "ks", "pr_auc", "brier", "ece")} | {"auc_ci": [float(c_.loc[k, "auc_ci_low"]), float(c_.loc[k, "auc_ci_high"])], "role": c_.loc[k, "role"]} for k in c_.index},
  "valid": {k: {"auc": float(vm.loc[k, "auc"]), "gini": float(vm.loc[k, "gini"]), "ks": float(vm.loc[k, "ks"]), "brier": float(vm.loc[k, "brier"]), "ece": float(vm.loc[k, "ece"])} for k in vm.index},
  "psi": psi_score, "label_sensitivity": cens, "cutoff": cutinfo, "bootstrap_reps": BOOT, "scorecard": {"pdo": L.PDO, "base_score": L.BASE_SCORE, "base_odds": L.BASE_ODDS},
  "monitoring_status": {t["metric"]: t["status"] for t in trig}, "n_grades_failing_binomial_oot": fails_bin, "n_grades_failing_vasicek_oot": fails_vas, "benchmark_fixes": EXTRA,
  "runtime_minutes": round((time.time() - T0) / 60, 1), "fast_mode": FAST}
save_json(metrics, "metrics.json"); save_json(CAPTIONS, "figure_captions.json")
save_json({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "optbinning": optbinning.__version__,
           "statsmodels": sm.__version__, "shap": shap.__version__, "platform": platform.platform(), "on_kaggle": ON_KAGGLE, "runtime_minutes": metrics["runtime_minutes"]}, "environment.json")
shutil.copy("lcpd_lib.py", ART / "lcpd_lib.py")
man = [{"file": str(p.relative_to(ART)).replace(os.sep, "/"), "kb": round(p.stat().st_size / 1024, 1)} for p in sorted(ART.rglob("*")) if p.is_file()]
save_json({"files": man, "n_files": len(man)}, "MANIFEST.json")
tick(f"DONE. {len(man)} artifact files; champion = {champion}")
pd.DataFrame(man).tail(8)
""")

# ======================================================================================
def build(lib_src):
    nb = nbf.v4.new_notebook()
    out = []
    for c in cells:
        c = nbf.v4.new_code_cell(c.source.replace("__LIB_SOURCE__", lib_src)) if (c.cell_type == "code" and "__LIB_SOURCE__" in c.source) else c
        out.append(c)
    nb["cells"] = out
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    lib_src = (ROOT / "lcpd_lib.py").read_text(encoding="utf-8")
    nb = build(lib_src)
    nbf.write(nb, ROOT / "LendingClubPDScorecard.ipynb")
    (ROOT / "kaggle").mkdir(exist_ok=True)
    print("cells:", len(nb.cells))
    # Kaggle's CLI does not return an executed .ipynb, so the pushed kernel is a thin wrapper that executes the real notebook
    # with nbconvert and leaves LendingClubPDScorecard.executed.ipynb next to the artifacts in /kaggle/working.
    import base64
    payload = base64.b64encode(nbf.writes(nb).encode("utf-8")).decode("ascii")
    NL = chr(10)
    runner = NL.join([
        "import base64, subprocess, sys",
        "NB = '" + payload + "'",
        "open('LendingClubPDScorecard.ipynb', 'wb').write(base64.b64decode(NB))",
        "r = subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'LendingClubPDScorecard.ipynb',",
        "                    '--output', 'LendingClubPDScorecard.executed.ipynb', '--ExecutePreprocessor.timeout=-1'], capture_output=True, text=True)",
        "print(r.stdout[-2000:]); print(r.stderr[-4000:])",
        "assert r.returncode == 0, 'inner notebook failed'"])
    wrapper = nbf.v4.new_notebook()
    wrapper["cells"] = [
        nbf.v4.new_markdown_cell("Kaggle runner for **LendingClubPDScorecard.ipynb** (embedded below as base64). It writes the notebook, executes it with nbconvert "
                                 "(outputs kept) and leaves `LendingClubPDScorecard.executed.ipynb` plus `artifacts/` in /kaggle/working."),
        nbf.v4.new_code_cell(runner)]
    wrapper["metadata"] = nb["metadata"]
    nbf.write(wrapper, ROOT / "kaggle" / "run_on_kaggle.ipynb")
