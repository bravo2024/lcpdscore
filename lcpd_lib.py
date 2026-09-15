"""lcpd_lib - shared code for the LendingClub application PD scorecard.

The notebook writes this exact file to disk, imports it, and copies it into artifacts/.
The Streamlit app and the tests import the same file, so training-time and serving-time
logic cannot drift apart.  Everything the app needs at serve time is stored as plain JSON
(bin tables, coefficients, prep parameters) - no sklearn pickles are needed to score.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1. Feature dictionary (application-time information only)
# --------------------------------------------------------------------------------------
# label, group, expected sign of the effect on PD (+1 risk up when value up, -1 risk down, 0 unclear)
FEATURE_META = {
    "loan_amnt": ("Loan amount requested ($)", "Loan", +1),
    "emp_length": ("Employment length (years)", "Applicant", -1),
    "annual_inc": ("Annual income ($)", "Applicant", -1),
    "dti": ("Debt-to-income ratio (%)", "Applicant", +1),
    "loan_to_income": ("Loan amount / annual income", "Loan", +1),
    "fico": ("FICO score at application", "Bureau", -1),
    "credit_history_months": ("Credit history length (months)", "Bureau", -1),
    "delinq_2yrs": ("Delinquencies in last 2 years", "Bureau", +1),
    "inq_last_6mths": ("Credit enquiries in last 6 months", "Bureau", +1),
    "mths_since_last_delinq": ("Months since last delinquency", "Bureau", -1),
    "mths_since_last_record": ("Months since last public record", "Bureau", -1),
    "mths_since_last_major_derog": ("Months since last major derogatory", "Bureau", -1),
    "open_acc": ("Open credit lines", "Bureau", 0),
    "pub_rec": ("Public records", "Bureau", +1),
    "revol_bal": ("Revolving balance ($)", "Bureau", 0),
    "revol_util": ("Revolving utilisation (%)", "Bureau", +1),
    "total_acc": ("Total credit lines ever", "Bureau", -1),
    "collections_12_mths_ex_med": ("Collections in last 12 months", "Bureau", +1),
    "acc_now_delinq": ("Accounts currently delinquent", "Bureau", +1),
    "tot_coll_amt": ("Total collection amount ever ($)", "Bureau", +1),
    "tot_cur_bal": ("Total current balance, all accounts ($)", "Bureau", 0),
    "total_rev_hi_lim": ("Total revolving credit limit ($)", "Bureau", -1),
    "acc_open_past_24mths": ("Accounts opened in past 24 months", "Bureau", +1),
    "avg_cur_bal": ("Average balance per account ($)", "Bureau", 0),
    "bc_open_to_buy": ("Open-to-buy on bankcards ($)", "Bureau", -1),
    "bc_util": ("Bankcard utilisation (%)", "Bureau", +1),
    "chargeoff_within_12_mths": ("Charge-offs in last 12 months", "Bureau", +1),
    "delinq_amnt": ("Past-due amount owed ($)", "Bureau", +1),
    "mo_sin_old_il_acct": ("Months since oldest instalment account", "Bureau", -1),
    "mo_sin_old_rev_tl_op": ("Months since oldest revolving account", "Bureau", -1),
    "mo_sin_rcnt_rev_tl_op": ("Months since newest revolving account", "Bureau", -1),
    "mo_sin_rcnt_tl": ("Months since newest account", "Bureau", -1),
    "mort_acc": ("Mortgage accounts", "Bureau", -1),
    "mths_since_recent_bc": ("Months since newest bankcard", "Bureau", -1),
    "mths_since_recent_bc_dlq": ("Months since bankcard delinquency", "Bureau", -1),
    "mths_since_recent_inq": ("Months since latest enquiry", "Bureau", -1),
    "mths_since_recent_revol_delinq": ("Months since revolving delinquency", "Bureau", -1),
    "num_accts_ever_120_pd": ("Accounts ever 120+ days past due", "Bureau", +1),
    "num_actv_bc_tl": ("Active bankcard accounts", "Bureau", 0),
    "num_actv_rev_tl": ("Active revolving accounts", "Bureau", 0),
    "num_bc_sats": ("Satisfactory bankcard accounts", "Bureau", 0),
    "num_bc_tl": ("Bankcard accounts", "Bureau", 0),
    "num_il_tl": ("Instalment accounts", "Bureau", 0),
    "num_op_rev_tl": ("Open revolving accounts", "Bureau", 0),
    "num_rev_accts": ("Revolving accounts", "Bureau", 0),
    "num_rev_tl_bal_gt_0": ("Revolving accounts with a balance", "Bureau", +1),
    "num_sats": ("Satisfactory accounts", "Bureau", 0),
    "num_tl_120dpd_2m": ("Accounts 120+ dpd in last 2 months", "Bureau", +1),
    "num_tl_30dpd": ("Accounts 30 dpd now", "Bureau", +1),
    "num_tl_90g_dpd_24m": ("Accounts 90+ dpd in last 24 months", "Bureau", +1),
    "num_tl_op_past_12m": ("Accounts opened in past 12 months", "Bureau", +1),
    "pct_tl_nvr_dlq": ("% of accounts never delinquent", "Bureau", -1),
    "percent_bc_gt_75": ("% of bankcards over 75% used", "Bureau", +1),
    "pub_rec_bankruptcies": ("Public-record bankruptcies", "Bureau", +1),
    "tax_liens": ("Tax liens", "Bureau", +1),
    "tot_hi_cred_lim": ("Total highest credit limit ($)", "Bureau", -1),
    "total_bal_ex_mort": ("Total balance excluding mortgage ($)", "Bureau", +1),
    "total_bc_limit": ("Total bankcard limit ($)", "Bureau", -1),
    "total_il_high_credit_limit": ("Total instalment high credit limit ($)", "Bureau", 0),
}
NUMERIC_FEATURES = list(FEATURE_META)
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
CAT_LABELS = {"home_ownership": "Home ownership", "verification_status": "Income verification",
              "purpose": "Loan purpose"}
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABELS = {k: v[0] for k, v in FEATURE_META.items()} | CAT_LABELS
EXPECTED_SIGN = {k: v[2] for k, v in FEATURE_META.items()}
GROUPS = {k: v[1] for k, v in FEATURE_META.items()} | {c: "Applicant" for c in CATEGORICAL_FEATURES}

# raw columns the loader must read from the LendingClub file
RAW_READ_COLS = sorted(set(
    ["id", "issue_d", "loan_status", "term", "application_type", "last_pymnt_d", "int_rate", "grade", "sub_grade",
     "earliest_cr_line", "fico_range_low", "fico_range_high", "emp_length", "loan_amnt"]
    + [c for c in NUMERIC_FEATURES if c not in ("emp_length", "loan_to_income", "fico", "credit_history_months")]
    + CATEGORICAL_FEATURES))

# Columns read ONLY for post-hoc evaluation (realised economics, default-definition checks). They are never features:
# ALL_FEATURES is built from FEATURE_META above and a test asserts the two sets are disjoint.
EVAL_ONLY_COLS = ["funded_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee"]
RAW_READ_COLS = sorted(set(RAW_READ_COLS) | set(EVAL_ONLY_COLS))

BAD_STATUS = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]
GOOD_STATUS = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]

# ---- default definition: 12-month performance window (no right-censoring) ---------------------------
# The file has only a snapshot status (no monthly delinquency history), so "90+ days past due within 12 months of issue"
# is proxied from two fields: final status in BAD_STATUS AND the month of the last payment made (months on book).
# A borrower whose last payment fell in month k has missed instalments k+1, k+2, k+3 by month k+3, so k <= 9 means 90+ DPD by month 12.
PERF_WINDOW_MONTHS = 12
BAD_MAX_MOB = 9


def months_on_book_last_payment(issue_d, last_pymnt_d):
    """Whole months from issue to the last payment received (0 when no payment was ever made)."""
    issue_d, last_pymnt_d = pd.to_datetime(issue_d), pd.to_datetime(last_pymnt_d)
    mob = (last_pymnt_d.dt.year - issue_d.dt.year) * 12 + (last_pymnt_d.dt.month - issue_d.dt.month)
    return mob.fillna(0).astype("float64")


def label_12m(loan_status, mob, max_mob=BAD_MAX_MOB):
    """1 = ended in Charged Off / Default AND stopped paying early enough to be 90+ DPD within the 12-month window; else 0."""
    return (pd.Series(loan_status).isin(BAD_STATUS).values & (np.asarray(mob) <= max_mob)).astype("float64")

# plain-English phrase for reason codes (used when the variable costs the applicant points)
REASON_PHRASES = {
    "fico": "Credit score (FICO) is lower than the best-scoring applicants",
    "dti": "Debts are high relative to income",
    "annual_inc": "Annual income is lower",
    "loan_amnt": "Loan amount requested is high",
    "loan_to_income": "Loan is large relative to income",
    "revol_util": "Credit cards are heavily used (high utilisation)",
    "bc_util": "Bankcards are heavily used (high utilisation)",
    "inq_last_6mths": "Several recent credit enquiries",
    "credit_history_months": "Short credit history",
    "delinq_2yrs": "Recent delinquencies on file",
    "pub_rec": "Public record on file",
    "acc_open_past_24mths": "Many new accounts opened recently",
    "tot_cur_bal": "Total balances are low relative to the best applicants",
    "mort_acc": "No or few mortgage accounts",
    "purpose": "Loan purpose carries higher historical risk",
    "home_ownership": "Home-ownership status carries higher historical risk",
    "verification_status": "Income verification status carries higher historical risk",
    "emp_length": "Shorter employment history",
    "total_rev_hi_lim": "Lower total revolving credit limit",
    "num_tl_op_past_12m": "Several accounts opened in the last 12 months",
    "mths_since_recent_inq": "A very recent credit enquiry",
    "num_rev_tl_bal_gt_0": "Many revolving accounts carry a balance",
    "total_bc_limit": "Lower total bankcard limit",
    "avg_cur_bal": "Average account balance differs from the best applicants",
    "pct_tl_nvr_dlq": "Lower share of never-delinquent accounts",
    "mo_sin_rcnt_tl": "Opened an account very recently",
    "mo_sin_old_rev_tl_op": "Newer oldest revolving account",
    "open_acc": "Number of open credit lines",
    "total_acc": "Number of total credit lines",
}


# Adverse-action wording (ECOA / Regulation B model-form style "principal reasons") for every model input.
# Reason codes from the scorecard are mapped to these statements; the plain-English phrase above stays for the app.
_AA_DELINQ = "Delinquent past or present credit obligations with others"
_AA_INQ = "Number of recent inquiries on credit bureau report"
_AA_OBLIG = "Excessive obligations in relation to income"
_AA_EXP = "Limited credit experience"
_AA_UTIL = "Too many accounts with balances / high utilisation of available credit"
_AA_AGE = "Length of credit history"
_AA_NEWACC = "Number of recently established accounts"
_AA_INC = "Income insufficient for amount of credit requested"
_AA_EMP = "Length of employment"
_AA_LIM = "Insufficient credit limits or available credit"
_AA_BUREAU = "Credit bureau information (score)"
ADVERSE_ACTION = {
    "fico": _AA_BUREAU, "dti": _AA_OBLIG, "annual_inc": _AA_INC, "loan_to_income": _AA_INC, "loan_amnt": _AA_INC,
    "emp_length": _AA_EMP, "credit_history_months": _AA_AGE, "delinq_2yrs": _AA_DELINQ, "inq_last_6mths": _AA_INQ,
    "mths_since_recent_inq": _AA_INQ, "revol_util": _AA_UTIL, "bc_util": _AA_UTIL, "percent_bc_gt_75": _AA_UTIL,
    "num_rev_tl_bal_gt_0": _AA_UTIL, "acc_open_past_24mths": _AA_NEWACC, "num_tl_op_past_12m": _AA_NEWACC,
    "mo_sin_rcnt_tl": _AA_NEWACC, "mo_sin_rcnt_rev_tl_op": _AA_NEWACC, "mths_since_recent_bc": _AA_NEWACC,
    "mo_sin_old_rev_tl_op": _AA_AGE, "mo_sin_old_il_acct": _AA_AGE, "total_acc": _AA_EXP, "open_acc": _AA_EXP,
    "mort_acc": _AA_EXP, "bc_open_to_buy": _AA_LIM, "tot_hi_cred_lim": _AA_LIM, "total_bc_limit": _AA_LIM,
    "total_rev_hi_lim": _AA_LIM, "total_il_high_credit_limit": _AA_LIM, "tot_cur_bal": _AA_OBLIG, "avg_cur_bal": _AA_OBLIG,
    "pub_rec": "Public record or collection filed", "pub_rec_bankruptcies": "Bankruptcy", "tax_liens": "Public record or collection filed",
    "pct_tl_nvr_dlq": _AA_DELINQ, "mths_since_last_delinq": _AA_DELINQ, "mths_since_last_major_derog": _AA_DELINQ,
    "num_accts_ever_120_pd": _AA_DELINQ, "num_tl_90g_dpd_24m": _AA_DELINQ, "num_tl_30dpd": _AA_DELINQ,
    "acc_now_delinq": _AA_DELINQ, "collections_12_mths_ex_med": "Public record or collection filed",
    "tot_coll_amt": "Public record or collection filed", "chargeoff_within_12_mths": _AA_DELINQ,
    "purpose": "Purpose of the requested credit", "home_ownership": "Housing status / length of residence",
    "verification_status": "Unable to verify income",
    "mths_since_last_record": "Public record or collection filed", "revol_bal": _AA_UTIL, "delinq_amnt": _AA_DELINQ,
    "mths_since_recent_bc_dlq": _AA_DELINQ, "mths_since_recent_revol_delinq": _AA_DELINQ, "num_tl_120dpd_2m": _AA_DELINQ,
    "num_actv_bc_tl": _AA_UTIL, "num_actv_rev_tl": _AA_UTIL, "num_op_rev_tl": _AA_UTIL, "num_bc_sats": _AA_EXP,
    "num_bc_tl": _AA_EXP, "num_il_tl": _AA_EXP, "num_rev_accts": _AA_EXP, "num_sats": _AA_EXP, "total_bal_ex_mort": _AA_OBLIG,
}
_missing_aa = [v for v in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if v not in ADVERSE_ACTION]
assert not _missing_aa, f"adverse-action wording missing for {_missing_aa}"


def adverse_action_text(var):
    return ADVERSE_ACTION.get(var, "Credit bureau information")


def reason_phrase(var):
    return REASON_PHRASES.get(var, f"{LABELS.get(var, var)} is less favourable than the best-scoring applicants")


# --------------------------------------------------------------------------------------
# 2. Raw -> engineered features (shared by training and the app)
# --------------------------------------------------------------------------------------
def parse_emp_length(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    out = s.str.extract(r"(\d+)")[0].astype("float64")
    out = out.where(~s.str.startswith("<", na=False), 0.0)
    return out.astype("float64")


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling frame (ALL_FEATURES) from raw LendingClub-style columns.

    Cleaning rules (all documented in the notebook): dti<0 or >=999 -> missing (sentinel codes),
    annual_inc<=0 -> missing.  Nothing here uses the target or any post-origination field.
    """
    r = raw.copy()
    out = pd.DataFrame(index=r.index)
    for c in NUMERIC_FEATURES:
        if c in ("emp_length", "loan_to_income", "fico", "credit_history_months"):
            continue
        out[c] = pd.to_numeric(r[c], errors="coerce") if c in r else np.nan
    # emp_length: text in the LC file, numeric when it comes from the app form
    if "emp_length" in r:
        out["emp_length"] = parse_emp_length(r["emp_length"]) if r["emp_length"].dtype == object or str(
            r["emp_length"].dtype).startswith("string") else pd.to_numeric(r["emp_length"], errors="coerce")
    else:
        out["emp_length"] = np.nan
    if "fico" in r:
        out["fico"] = pd.to_numeric(r["fico"], errors="coerce")
    else:
        out["fico"] = (pd.to_numeric(r["fico_range_low"], errors="coerce")
                       + pd.to_numeric(r["fico_range_high"], errors="coerce")) / 2.0
    if "credit_history_months" in r:
        out["credit_history_months"] = pd.to_numeric(r["credit_history_months"], errors="coerce")
    else:
        issue = pd.to_datetime(r["issue_d"], format="%b-%Y") if r["issue_d"].dtype == object else r["issue_d"]
        ecl = pd.to_datetime(r["earliest_cr_line"], format="%b-%Y")
        out["credit_history_months"] = ((issue.dt.year - ecl.dt.year) * 12 + (issue.dt.month - ecl.dt.month)).astype(
            "float64")
    out.loc[(out["dti"] < 0) | (out["dti"] >= 999), "dti"] = np.nan
    out.loc[out["annual_inc"] <= 0, "annual_inc"] = np.nan
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"]
    for c in CATEGORICAL_FEATURES:
        out[c] = r[c].astype("object").where(r[c].notna(), np.nan) if c in r else np.nan
    return out[ALL_FEATURES]


# --------------------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------------------
def auc_fast(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ks_stat(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(y)
    return float(sum(len(c) / n * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def reliability_table(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p)
    rows = []
    for i, c in enumerate(np.array_split(order, bins), 1):
        rows.append({"bin": i, "n": len(c), "mean_pred": p[c].mean(), "obs_rate": y[c].mean()})
    return pd.DataFrame(rows)


def all_metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_stat(y, p),
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)), "ece": ece(y, p), "mean_pred": float(p.mean()),
            "obs_rate": float(np.mean(y)), "n": int(len(y))}


def hosmer_lemeshow(y, p, g=10):
    from scipy.stats import chi2
    t = reliability_table(y, p, g)
    obs = t["obs_rate"] * t["n"]
    exp = t["mean_pred"] * t["n"]
    stat = float((((obs - exp) ** 2) / (exp * (1 - t["mean_pred"]))).sum())
    return {"chi2": stat, "df": g - 2, "p_value": float(1 - chi2.cdf(stat, g - 2))}


def psi_from_counts(e, a, eps=1e-4):
    e = np.asarray(e, float)
    a = np.asarray(a, float)
    pe = np.clip(e / e.sum(), eps, None)
    pa = np.clip(a / a.sum(), eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_numeric(expected, actual, bins=10):
    expected = pd.Series(expected).astype(float)
    actual = pd.Series(actual).astype(float)
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf

    def counts(s):
        c = np.histogram(s.dropna(), bins=edges)[0].tolist()
        c.append(int(s.isna().sum()))
        return c
    return psi_from_counts(counts(expected), counts(actual))


def psi_categorical(expected, actual):
    ce = pd.Series(expected).fillna("MISSING").value_counts()
    ca = pd.Series(actual).fillna("MISSING").value_counts()
    idx = ce.index.union(ca.index)
    return psi_from_counts(ce.reindex(idx, fill_value=0).values, ca.reindex(idx, fill_value=0).values)


def psi_label(v):
    return "stable (<0.10)" if v < 0.10 else ("moderate shift (0.10-0.25)" if v < 0.25 else "significant shift (>0.25)")


# --------------------------------------------------------------------------------------
# 4. WoE bin maps (own implementation; optbinning only proposes the cut points)
# --------------------------------------------------------------------------------------
def _fmt(v):
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:.2f}M"
    if av >= 1e4:
        return f"{v / 1e3:.0f}k"
    if av >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.4g}"


class WoEMap:
    """Stores bins + WoE per variable as plain Python, so scoring needs no optbinning.

    WoE = ln( share of GOODS in bin / share of BADS in bin ); higher WoE = safer bin.
    """

    def __init__(self, vars_=None):
        self.vars = vars_ or {}

    # ---- assignment of rows to bin indices (-1 = missing / unseen category)
    @staticmethod
    def _assign_num(x, splits):
        x = np.asarray(x, float)
        idx = np.searchsorted(np.asarray(splits, float), x, side="right") if len(splits) else np.zeros(len(x), int)
        return np.where(np.isnan(x), -1, idx)

    @staticmethod
    def _assign_cat(s, groups):
        mapping = {}
        for gi, g in enumerate(groups):
            for c in g:
                mapping[c] = gi
        return pd.Series(s).map(mapping).fillna(-1).astype(int).values

    def _assign(self, name, x):
        spec = self.vars[name]
        if spec["kind"] == "num":
            return self._assign_num(x, spec["splits"])
        return self._assign_cat(x, spec["groups"])

    def fit_var(self, name, x, y, kind, splits=None, groups=None, laplace=0.5):
        y = np.asarray(y)
        spec = {"kind": kind}
        if kind == "num":
            spec["splits"] = [float(s) for s in (splits if splits is not None else [])]
            nb = len(spec["splits"]) + 1
        else:
            spec["groups"] = [list(map(str, g)) for g in groups]
            nb = len(spec["groups"])
        idx = self._assign_num(x, spec["splits"]) if kind == "num" else self._assign_cat(
            pd.Series(x).astype("object"), spec["groups"])
        B, G = float(y.sum()), float(len(y) - y.sum())
        rows, woe = [], []
        for b in range(-1, nb):
            m = idx == b
            n = int(m.sum())
            if b == -1 and n == 0:
                spec["missing_woe"] = 0.0
                continue
            bad = int(y[m].sum())
            good = n - bad
            w = math.log(((good + laplace) / (G + laplace)) / ((bad + laplace) / (B + laplace)))
            iv = ((good / G) - (bad / B)) * w
            rows.append({"bin_index": b, "n": n, "share": n / len(y), "bad": bad, "bad_rate": bad / n if n else np.nan,
                         "woe": w, "iv_contrib": iv})
            if b == -1:
                spec["missing_woe"] = w
            else:
                woe.append(w)
        spec["woe"] = woe
        spec["labels"] = self._labels(spec)
        spec["table"] = rows
        spec["iv"] = float(sum(r["iv_contrib"] for r in rows))
        self.vars[name] = spec
        return spec

    @staticmethod
    def _labels(spec):
        if spec["kind"] == "num":
            s = spec["splits"]
            if not s:
                return ["all values"]
            lab = [f"< {_fmt(s[0])}"]
            lab += [f"{_fmt(s[i])} to < {_fmt(s[i + 1])}" for i in range(len(s) - 1)]
            lab.append(f">= {_fmt(s[-1])}")
            return lab
        out = []
        for g in spec["groups"]:
            t = ", ".join(g[:3]) + (f" (+{len(g) - 3} more)" if len(g) > 3 else "")
            out.append(t)
        return out

    def transform_var(self, name, x):
        spec = self.vars[name]
        idx = self._assign(name, x)
        woe = np.asarray(spec["woe"] + [spec.get("missing_woe", 0.0)], float)
        return woe[np.where(idx < 0, len(spec["woe"]), idx)]

    def transform(self, df, names=None):
        names = names or list(self.vars)
        return pd.DataFrame({n: self.transform_var(n, df[n].values if self.vars[n]["kind"] == "num"
                                                    else df[n].astype("object").values) for n in names},
                            index=df.index)

    def bin_index(self, name, x):
        return self._assign(name, x)

    def bin_label(self, name, idx):
        spec = self.vars[name]
        return "Missing / other" if idx < 0 else spec["labels"][idx]

    def bins_table(self):
        rows = []
        for n, s in self.vars.items():
            for r in s["table"]:
                rows.append({"variable": n, "bin": self.bin_label(n, r["bin_index"]), **r})
        return pd.DataFrame(rows)

    def to_json(self):
        return json.dumps(self.vars)

    @classmethod
    def from_json(cls, s):
        return cls(json.loads(s) if isinstance(s, str) else s)


def monotone_flag(table_rows):
    br = [r["bad_rate"] for r in table_rows if r["bin_index"] >= 0 and r["n"] > 0]
    if len(br) < 3:
        return True
    d = np.diff(br)
    return bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))


# --------------------------------------------------------------------------------------
# 5. Preprocessing for the raw logistic models and the trees
# --------------------------------------------------------------------------------------
class RawPrep:
    """Winsorise -> optional log1p -> median impute (+ missing flag) -> standardise; one-hot categoricals.

    All parameters are learned on TRAIN only and serialise to JSON.
    """

    def __init__(self, numeric, categorical, winsor=(0.005, 0.995), miss_flag_threshold=0.02, min_level_share=0.005):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.winsor, self.miss_thr, self.min_share = winsor, miss_flag_threshold, min_level_share
        self.num_params, self.cat_params = {}, {}

    def fit(self, df):
        for c in self.numeric:
            x = pd.to_numeric(df[c], errors="coerce")
            xn = x.dropna()
            lo, hi = (float(xn.quantile(self.winsor[0])), float(xn.quantile(self.winsor[1]))) if len(xn) else (0.0, 0.0)
            xc = x.clip(lo, hi)
            use_log = bool(len(xn) and xc.min() >= 0 and float(xc.dropna().skew()) > 2.0)
            xt = np.log1p(xc) if use_log else xc
            med = float(xt.median()) if xt.notna().any() else 0.0
            filled = xt.fillna(med)
            self.num_params[c] = {"lo": lo, "hi": hi, "log": use_log, "median": med, "mean": float(filled.mean()),
                                  "std": float(filled.std()) or 1.0, "miss_share": float(x.isna().mean()),
                                  "flag": bool(x.isna().mean() > self.miss_thr)}
        for c in self.categorical:
            vc = df[c].astype("object").value_counts(normalize=True)
            levels = [str(k) for k, v in vc.items() if v >= self.min_share]
            self.cat_params[c] = {"levels": levels, "baseline": levels[0]}
        return self

    def _num_block(self, df, standardise=True):
        cols, names = [], []
        for c in self.numeric:
            p = self.num_params[c]
            x = pd.to_numeric(df[c], errors="coerce")
            isna = x.isna().values
            xt = x.clip(p["lo"], p["hi"])
            if p["log"]:
                xt = np.log1p(xt)
            xt = xt.fillna(p["median"]).values
            if standardise:
                xt = (xt - p["mean"]) / p["std"]
            cols.append(xt)
            names.append(c)
            if p["flag"]:
                cols.append(isna.astype(float))
                names.append(c + "__missing")
        return np.column_stack(cols), names

    def _cat_block(self, df, drop_baseline=True):
        cols, names = [], []
        for c in self.categorical:
            p = self.cat_params[c]
            v = df[c].astype("object").astype(str).values
            v = np.where(np.isin(v, p["levels"]), v, p["baseline"])  # rare / unseen -> baseline
            for lev in p["levels"]:
                if drop_baseline and lev == p["baseline"]:
                    continue
                cols.append((v == lev).astype(float))
                names.append(f"{c}={lev}")
        return (np.column_stack(cols) if cols else np.zeros((len(df), 0))), names

    def transform(self, df):
        a, an = self._num_block(df)
        b, bn = self._cat_block(df)
        self.feature_names_ = an + bn
        return np.hstack([a, b]).astype(np.float32)

    def names(self):
        a = []
        for c in self.numeric:
            a.append(c)
            if self.num_params[c]["flag"]:
                a.append(c + "__missing")
        b = [f"{c}={lev}" for c in self.categorical for lev in self.cat_params[c]["levels"]
             if lev != self.cat_params[c]["baseline"]]
        return a + b

    def transform_trees(self, df):
        """Raw numerics (NaN kept, trees handle them) + full one-hot categoricals. Returns (matrix, names)."""
        a = np.column_stack([pd.to_numeric(df[c], errors="coerce").values for c in self.numeric]).astype(np.float32)
        b, bn = self._cat_block(df, drop_baseline=False)
        return np.hstack([a, b.astype(np.float32)]), list(self.numeric) + bn

    def to_dict(self):
        return {"numeric": self.numeric, "categorical": self.categorical, "winsor": list(self.winsor),
                "num_params": self.num_params, "cat_params": self.cat_params}

    @classmethod
    def from_dict(cls, d):
        o = cls(d["numeric"], d["categorical"], tuple(d["winsor"]))
        o.num_params, o.cat_params = d["num_params"], d["cat_params"]
        return o


# --------------------------------------------------------------------------------------
# 6. Scorecard scaling, calibration, master scale, reason codes
# --------------------------------------------------------------------------------------
PDO, BASE_SCORE, BASE_ODDS = 20.0, 600.0, 50.0  # 600 points at 50:1 good:bad, +20 points doubles the odds
FACTOR = PDO / math.log(2)
OFFSET = BASE_SCORE - FACTOR * math.log(BASE_ODDS)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def pd_to_score(pd_):
    return OFFSET - FACTOR * logit(pd_)


def score_to_pd(score):
    return sigmoid(-(np.asarray(score, float) - OFFSET) / FACTOR)


def platt_apply(raw_pd, cal):
    return sigmoid(cal["a"] + cal["b"] * logit(raw_pd))


def assign_grade(pd_, edges):
    """edges: ascending upper bounds of grades 1..K-1; returns 1..K (1 = best)."""
    return np.searchsorted(np.asarray(edges), np.asarray(pd_), side="right") + 1


GRADE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def grade_letter(g):
    return GRADE_NAMES[int(g) - 1]


def cutoff_curve(pd_, y, amt, margin, lgd, ead, n_points=200):
    """Approve if PD <= cut-off.  Profit per good = margin*amount; loss per bad = EAD*LGD*amount."""
    pd_, y, amt = np.asarray(pd_, float), np.asarray(y, float), np.asarray(amt, float)
    order = np.argsort(pd_)
    p, yy, aa = pd_[order], y[order], amt[order]
    gain = np.where(yy == 0, margin * aa, -ead * lgd * aa)
    cum_profit = np.cumsum(gain)
    cum_n = np.arange(1, len(p) + 1)
    cum_bad = np.cumsum(yy)
    pos = np.unique(np.linspace(0, len(p) - 1, n_points).astype(int))
    return pd.DataFrame({"cutoff_pd": p[pos], "approval_rate": cum_n[pos] / len(p),
                         "bad_rate_approved": cum_bad[pos] / cum_n[pos],
                         "profit_total": cum_profit[pos], "profit_per_application": cum_profit[pos] / len(p),
                         "profit_per_approved": cum_profit[pos] / cum_n[pos]})


# ---- scorecard objects ---------------------------------------------------------------
class Scorecard:
    """Points-based scorecard built from WoE bins and logistic coefficients (all JSON)."""

    def __init__(self, woe: WoEMap, variables, coefs, intercept, calibrator=None):
        self.woe, self.vars, self.coefs, self.intercept, self.cal = woe, list(variables), dict(coefs), float(
            intercept), calibrator

    def raw_logit(self, df):
        z = np.full(len(df), self.intercept)
        for v in self.vars:
            z = z + self.coefs[v] * self.woe.transform_var(v, df[v].values if self.woe.vars[v]["kind"] == "num"
                                                           else df[v].astype("object").values)
        return z

    def raw_pd(self, df):
        return sigmoid(self.raw_logit(df))

    def score(self, df):
        return OFFSET - FACTOR * self.raw_logit(df)

    def calibrated_pd(self, df):
        p = self.raw_pd(df)
        return platt_apply(p, self.cal) if self.cal else p

    def points_table(self):
        n = len(self.vars)
        rows = []
        for v in self.vars:
            spec = self.woe.vars[v]
            labels = list(spec["labels"]) + (["Missing / other"] if spec.get("missing_woe") is not None and any(
                r["bin_index"] == -1 for r in spec["table"]) else [])
            woes = list(spec["woe"]) + ([spec["missing_woe"]] if len(labels) > len(spec["woe"]) else [])
            tab = {r["bin_index"]: r for r in spec["table"]}
            for i, (lab, w) in enumerate(zip(labels, woes)):
                bi = i if i < len(spec["woe"]) else -1
                pts = -FACTOR * (self.coefs[v] * w + self.intercept / n) + OFFSET / n
                rows.append({"variable": v, "label": LABELS.get(v, v), "bin": lab, "woe": w, "coef": self.coefs[v],
                             "points": pts, "n_train": tab.get(bi, {}).get("n", np.nan),
                             "bad_rate_train": tab.get(bi, {}).get("bad_rate", np.nan)})
        return pd.DataFrame(rows)

    def reason_codes(self, df_row, top=3):
        """Top reasons = variables where the applicant sits furthest below the best bin's points."""
        pt = self.points_table()
        out = []
        for v in self.vars:
            sub = pt[pt.variable == v]
            val = df_row[v].values if self.woe.vars[v]["kind"] == "num" else df_row[v].astype("object").values
            bi = int(self.woe.bin_index(v, val)[0])
            lab = self.woe.bin_label(v, bi)
            mine = float(sub[sub.bin == lab].points.iloc[0]) if (sub.bin == lab).any() else float(sub.points.mean())
            out.append({"variable": v, "label": LABELS.get(v, v), "bin": lab, "points": mine,
                        "shortfall": float(sub.points.max()) - mine})
        out = sorted(out, key=lambda r: -r["shortfall"])
        res = []
        for r in out[:top]:
            if r["shortfall"] > 0.5:
                r["text"] = f"{reason_phrase(r['variable'])} (applicant bin: {r['bin']}, {r['shortfall']:.0f} points below the best bin)"
                r["adverse_action"] = adverse_action_text(r["variable"])
                res.append(r)
        return res

    def to_dict(self):
        return {"variables": self.vars, "coefs": self.coefs, "intercept": self.intercept, "woe": self.woe.vars,
                "calibrator": self.cal, "factor": FACTOR, "offset": OFFSET, "pdo": PDO, "base_score": BASE_SCORE,
                "base_odds": BASE_ODDS}

    @classmethod
    def from_dict(cls, d):
        return cls(WoEMap(d["woe"]), d["variables"], d["coefs"], d["intercept"], d.get("calibrator"))


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# Models the app may score. The two ref_* rows use LendingClub's own grade and rate and are not eligible models.
SCORABLE_MODELS = ("plain_lr", "l2_lr", "l1_lr", "elasticnet_lr", "woe_scorecard", "xgb_tuned", "xgb_monotone")
_LADDER_CACHE = {}


def _ladder(art):
    """The saved logistic ladder, loaded once per artifact folder."""
    key = str(art)
    if key not in _LADDER_CACHE:
        import joblib
        _LADDER_CACHE[key] = joblib.load(art / "models" / "logistic_ladder.joblib")
    return _LADDER_CACHE[key]


def model_pd(feat_df, art_dir, model):
    """Calibrated PD of any saved model for an engineered feature frame.

    Reproduces the notebook exactly: plain logistic uses the statsmodels coefficients on the shared
    standardised design matrix, the regularised logits use the saved sklearn fits, the scorecard uses
    its own WoE points, and the two boosters use the saved XGBoost JSON. Platt calibrators are applied
    last, exactly as on validation.
    """
    art = Path(art_dir)
    if model not in SCORABLE_MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {SCORABLE_MODELS}")
    cal = load_json(art / "calibrators.json")[model]
    if model == "woe_scorecard":
        sc = Scorecard.from_dict(load_json(art / "scorecard.json"))
        return platt_apply(sc.raw_pd(feat_df), cal)
    if model == "plain_lr":
        prep = RawPrep.from_dict(load_json(art / "raw_prep.json"))
        sm = pd.read_csv(art / "statsmodels_logit_raw.csv").set_index("feature")
        coef = sm.loc[prep.names(), "coef"].to_numpy(float)
        raw = sigmoid(np.asarray(prep.transform(feat_df)) @ coef + float(sm.loc["const", "coef"]))
        return platt_apply(raw, cal)
    if model in ("l2_lr", "l1_lr", "elasticnet_lr"):
        lad = _ladder(art)
        raw = lad[model.replace("_lr", "")].predict_proba(np.asarray(lad["prep"].transform(feat_df)))[:, 1]
        return platt_apply(raw, cal)
    import xgboost as xgb
    prep = RawPrep.from_dict(load_json(art / "tree_prep.json"))
    mat, _ = prep.transform_trees(feat_df)
    b = xgb.Booster()
    b.load_model(str(art / "models" / f"{model}.json"))
    return platt_apply(b.predict(xgb.DMatrix(mat)), cal)


def champion_pd(feat_df, art_dir, champion=None):
    """Calibrated PD of the frozen champion. Kept for callers that want the rule default."""
    art = Path(art_dir)
    champion = champion or load_json(art / "champion_rule.json")["champion"]
    return model_pd(feat_df, art, champion)


def score_applicant(raw: dict, art_dir, model=None):
    """Score one applicant from raw form values using only saved artifacts.

    Score and reason codes always come from the WoE scorecard, because it is the explainable one.
    PD and grade come from `model`, which defaults to the frozen champion.
    """
    art = Path(art_dir)
    sc = Scorecard.from_dict(load_json(art / "scorecard.json"))
    ms = load_json(art / "master_scale.json")
    model = model or load_json(art / "champion_rule.json")["champion"]
    df = engineer(pd.DataFrame([raw]))
    score = float(sc.score(df)[0])
    pd_sc = float(sc.calibrated_pd(df)[0])
    pd_m = float(model_pd(df, art, model)[0])
    grade = int(assign_grade([pd_m], ms["edges"])[0])
    return {"score": score, "pd": pd_m, "pd_scorecard": pd_sc, "model": model, "champion": model,
            "grade": grade_letter(grade), "grade_index": grade, "reasons": sc.reason_codes(df),
            "engineered": df.iloc[0].to_dict()}


# --------------------------------------------------------------------------------------
# 7. Calibration tests that respect default correlation (added after the benchmark review)
# --------------------------------------------------------------------------------------
RHO_BASE = 0.03                      # retail asset correlation used for the traffic light (Basel retail "other" is 0.03-0.16; 0.03 is the lenient end)
RHO_GRID = (0.01, 0.02, 0.03, 0.04)  # sensitivity range requested for the benchmark


def binomial_pvalues(k, n, pd_):
    """Plain binomial: one-sided (under / over prediction) and two-sided p-values. Assumes independent defaults."""
    from scipy.stats import binom
    k, n = int(k), int(n)
    p_under = float(binom.sf(k - 1, n, pd_))      # P(K >= k): small => more defaults than promised
    p_over = float(binom.cdf(k, n, pd_))          # P(K <= k): small => fewer defaults than promised
    return {"p_under": p_under, "p_over": p_over, "p_two": float(min(1.0, 2 * min(p_under, p_over)))}


def vasicek_test(k, n, pd_, rho=RHO_BASE, n_grid=1201):
    """Single-factor (Vasicek) calibration test with finite-n binomial noise.

    Default probability given the systematic factor z:  p(z) = Phi((Phi^-1(PD) - sqrt(rho) z) / sqrt(1 - rho)),  K | z ~ Binomial(n, p(z)).
    Integrating over z ~ N(0,1) gives the distribution of the default count under H0 'true PD = pd_' when defaults share one economy.
    rho -> 0 recovers the plain binomial test; larger rho widens the acceptance region (defaults cluster in time).
    """
    from scipy.stats import binom, norm
    pd_ = float(np.clip(pd_, 1e-9, 1 - 1e-9))
    k, n = int(k), int(n)
    z = np.linspace(-8, 8, n_grid)
    w = norm.pdf(z)
    w = w / w.sum()
    pz = norm.cdf((norm.ppf(pd_) - math.sqrt(rho) * z) / math.sqrt(1 - rho))
    p_under = float(np.sum(w * binom.sf(k - 1, n, pz)))
    p_over = float(np.sum(w * binom.cdf(k, n, pz)))
    return {"p_under": p_under, "p_over": p_over, "p_two": float(min(1.0, 2 * min(p_under, p_over)))}


def jeffreys_test(k, n, pd_):
    """Jeffreys (Bayesian, Beta(k+0.5, n-k+0.5) posterior) test of H0 'true PD = pd_'. Good for few defaults; still assumes independence."""
    from scipy.stats import beta
    k, n = int(k), int(n)
    post_le = float(beta.cdf(pd_, k + 0.5, n - k + 0.5))      # posterior prob that true PD <= promised PD
    p_under, p_over = post_le, 1 - post_le                     # small post_le => true PD very likely ABOVE the promise
    return {"p_under": p_under, "p_over": p_over, "p_two": float(min(1.0, 2 * min(p_under, p_over)))}


def calibration_intercept_slope(y, p):
    """Calibration-in-the-large (intercept with slope fixed at 1, offset logit(p)) and calibration slope (logit(y) ~ a + b logit(p)), with 95% CIs."""
    import statsmodels.api as sm
    y = np.asarray(y, float)
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    it = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit()
    sl = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
    ci_i, ci_s = np.asarray(it.conf_int())[0], np.asarray(sl.conf_int())[1]
    return {"intercept_citl": float(it.params[0]), "intercept_ci": [float(ci_i[0]), float(ci_i[1])],
            "slope": float(sl.params[1]), "slope_ci": [float(ci_s[0]), float(ci_s[1])], "n": int(len(y)),
            "mean_pred": float(np.mean(p)), "obs_rate": float(y.mean())}


def traffic_light(p_two, amber=0.05, red=0.01):
    return "RED" if p_two < red else ("AMBER" if p_two < amber else "GREEN")


def anchor_shift(p_raw, cal, target, weights=None):
    """Shift delta to add to the calibrator intercept so that the (weighted) mean calibrated PD equals `target`."""
    from scipy.optimize import brentq
    z = cal["b"] * logit(p_raw)
    w = np.ones(len(z)) if weights is None else np.asarray(weights, float)
    f = lambda d: float(np.average(sigmoid(cal["a"] + d + z), weights=w) - target)  # noqa: E731
    return float(brentq(f, -6.0, 6.0, xtol=1e-12))


# --------------------------------------------------------------------------------------
# 8. Discrimination: DeLong CI for AUC / Somers' D, CAP curve and accuracy ratio
# --------------------------------------------------------------------------------------
def delong_auc_ci(y, p, alpha=0.05):
    """AUC with the fast DeLong (Sun & Xu 2014) variance. Somers' D (= Gini for a binary outcome) = 2 AUC - 1, so its CI is the AUC CI doubled."""
    from scipy.stats import norm, rankdata
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tz = rankdata(np.concatenate([pos, neg]))
    tx, ty = rankdata(pos), rankdata(neg)
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    se = math.sqrt(var)
    zc = norm.ppf(1 - alpha / 2)
    return {"auc": float(auc), "auc_se": se, "auc_ci": [float(auc - zc * se), float(auc + zc * se)],
            "somers_d": float(2 * auc - 1), "somers_d_se": 2 * se,
            "somers_d_ci": [float(2 * (auc - zc * se) - 1), float(2 * (auc + zc * se) - 1)]}


def cap_curve(y, p, n_points=200):
    """Cumulative Accuracy Profile: share of bads captured vs share of population, riskiest first."""
    y = np.asarray(y, float)
    order = np.argsort(-np.asarray(p, float), kind="mergesort")
    cb = np.cumsum(y[order]) / y.sum()
    x = np.arange(1, len(y) + 1) / len(y)
    pos = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    return np.r_[0.0, x[pos]], np.r_[0.0, cb[pos]]


def accuracy_ratio(y, p):
    """AR = (area under CAP - 0.5) / (area under perfect CAP - 0.5). Equals the Gini coefficient for a binary outcome (checked in tests)."""
    y = np.asarray(y, float)
    order = np.argsort(-np.asarray(p, float), kind="mergesort")
    cb = np.r_[0.0, np.cumsum(y[order]) / y.sum()]
    area = float(np.sum((cb[1:] + cb[:-1]) / 2.0) / (len(cb) - 1))
    prev = y.mean()
    perfect = 1 - prev / 2.0
    return float((area - 0.5) / (perfect - 0.5)), area


# --------------------------------------------------------------------------------------
# 9. Monotonicity evidence for a boosted model: ALE and PDP on one feature
# --------------------------------------------------------------------------------------
def pdp_1d(predict_fn, X, j, grid):
    out = []
    for g in grid:
        Xg = X.copy()
        Xg[:, j] = g
        out.append(float(np.mean(predict_fn(Xg))))
    return np.asarray(out)


def ale_1d(predict_fn, X, j, n_bins=15):
    """First-order accumulated local effects of column j (on the scale predict_fn returns, e.g. log-odds). Rows with NaN in column j are ignored."""
    x = X[:, j]
    ok = ~np.isnan(x)
    Xo, xo = X[ok], x[ok]
    edges = np.unique(np.quantile(xo, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([]), np.array([])
    idx = np.clip(np.searchsorted(edges, xo, side="right") - 1, 0, len(edges) - 2)
    eff = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        lo, hi = Xo[m].copy(), Xo[m].copy()
        lo[:, j], hi[:, j] = edges[b], edges[b + 1]
        eff[b] = float(np.mean(predict_fn(hi) - predict_fn(lo)))
    ale = np.r_[0.0, np.cumsum(eff)]
    centers = (ale[:-1] + ale[1:]) / 2
    centers = centers - np.average(centers, weights=np.bincount(idx, minlength=len(edges) - 1) + 1e-9)
    return (edges[:-1] + edges[1:]) / 2, centers


def monotone_summary(curve, expected_sign, tol=1e-9):
    """Direction of a response curve (log-odds of default) and the share of its total movement that runs against `expected_sign`
    (+1 risk up with the feature, -1 risk down, 0 no expectation)."""
    c = np.asarray(curve, float)
    if len(c) < 3:
        return {"direction": "n/a", "against_share": float("nan"), "consistent": None}
    d = np.diff(c)
    up, down = float(d[d > tol].sum()), float(-d[d < -tol].sum())
    tot = up + down
    if tot < 1e-6:
        return {"direction": "flat", "against_share": 0.0, "consistent": None if expected_sign == 0 else True}
    sign = 1 if up >= down else -1
    against = abs(down if sign > 0 else up) / tot          # movement against its own dominant direction (0 = perfectly monotone)
    ok = None if expected_sign == 0 else bool(sign == expected_sign and against < 0.05)
    return {"direction": "risk rises with the feature" if sign > 0 else "risk falls with the feature",
            "against_share": float(against), "consistent": ok}


# --------------------------------------------------------------------------------------
# 10. Rejected applications (through-the-door population)
# --------------------------------------------------------------------------------------
def load_rejects(path, years=(2013, 2016), cache=None, chunksize=2_000_000):
    """Compact view of rejected_2007_to_2018Q4: year, month, amount requested, Risk_Score, DTI (%), employment length (years).

    Risk_Score is a FICO score up to roughly Nov-2013 and a VantageScore afterwards, and is blank for most rejects from 2015 on.
    """
    if cache and Path(cache).exists():
        return pd.read_parquet(cache)
    cols = ["Amount Requested", "Application Date", "Risk_Score", "Debt-To-Income Ratio", "Employment Length"]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        yr = ch["Application Date"].str[:4].astype(int)
        ch = ch[yr.between(*years)]
        if not len(ch):
            continue
        parts.append(pd.DataFrame({
            "year": ch["Application Date"].str[:4].astype(int).values.astype("int16"),
            "month": ch["Application Date"].str[5:7].astype(int).values.astype("int8"),
            "amt": ch["Amount Requested"].astype("float32").values,
            "risk": ch["Risk_Score"].astype("float32").values,
            "dti": pd.to_numeric(ch["Debt-To-Income Ratio"].astype(str).str.rstrip("%"), errors="coerce").astype("float32").values,
            "emp": parse_emp_length(ch["Employment Length"]).astype("float32").values}))
    out = pd.concat(parts, ignore_index=True)
    if cache:
        out.to_parquet(cache)
    return out
