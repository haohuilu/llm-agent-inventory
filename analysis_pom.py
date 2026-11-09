"""Additional analyses for the POM version (referee round 1). Writes utd24_runs/results/analysis1{1..6}_*.csv.

A11  AUROC with site-block-bootstrap intervals and positives per cell; TDC variants: own, other-model, agent-only,
     series-only, single feature q/q_SAA; leave-sites-out and leave-panel-out transfer.
A12  Reaction coefficient with site-clustered standard errors and a log specification.
A13  Rolling-origin outcome correction (factor refitted each test period on the six most recent realised periods).
A14  Review experiment on all panels with break-even review cost.
A15  EWMA + residual-quantile rule.
A16  Informed-treatment VDC AUROC and magnitude errors; VDC-coverage correlation after feedback; distinct VDC values.
usage: python3 analysis_pom.py
"""
import os, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import importlib.util
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.py"))
# re-use helpers from analysis.py without re-running it: copy the needed definitions
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.py")).read()
exec(_src.split("# ---------------------------------------------------------------- A2")[0])
_a5 = _src.split("# ---------------------------------------------------------------- A5")[1]
exec(_a5[_a5.index("def planner_order"):_a5.index("rows = []")])

def site_boot_auc(s, y, sites, B=500, seed=0):
    s, y, sites = np.asarray(s, float), np.asarray(y, int), np.asarray(sites)
    ok = ~np.isnan(s); s, y, sites = s[ok], y[ok], sites[ok]
    u = np.unique(sites); rng = np.random.default_rng(seed); idx = {k: np.where(sites == k)[0] for k in u}
    pt = auroc(s, y); vals = []
    for _ in range(B):
        pick = rng.choice(u, len(u), replace=True); ii = np.concatenate([idx[k] for k in pick])
        if y[ii].min() == y[ii].max(): continue
        vals.append(roc_auc_score(y[ii], s[ii]))
    return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def fmt(pt, lo, hi): return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"

# ---------------------------------------------------------------- A11
rows = []
PANEL_CAL = {}
for ds in DATASETS:
    D = {m: load(ds, "basic", m) for m, _ in MODELS}
    PANEL_CAL[ds] = pd.concat([d[d.split == "calib"].assign(ds=ds) for d in D.values()])
for ds in DATASETS:
    D = {m: load(ds, "basic", m) for m, _ in MODELS}
    other_panels = pd.concat([PANEL_CAL[o] for o in DATASETS if o != ds]).reset_index(drop=True)
    for m, name in MODELS:
        d = D[m]; cal, tst = d[d.split == "calib"], d[d.split == "test"]
        y = (~tst.covered).astype(int); sites = (tst.practice + "|" + tst.chemical).to_numpy()
        others = pd.concat([D[o][D[o].split == "calib"] for o in D if o != m])
        r = dict(dataset=ds, model=name, n_test=len(tst), n_short=int(y.sum()))
        r["vdc"] = fmt(*site_boot_auc(-tst.confidence, y, sites)) if tst.confidence.nunique() > 1 else "--"
        r["tdc_own"] = fmt(*site_boot_auc(tdc_risk(fit_tdc(cal, ds), tst, ds), y, sites))
        r["tdc_other"] = fmt(*site_boot_auc(tdc_risk(fit_tdc(others, ds), tst, ds), y, sites))
        r["agent_only"] = fmt(*site_boot_auc(tdc_risk(fit_tdc(others, ds, AGENT_F), tst, ds), y, sites))
        r["series_only"] = fmt(*site_boot_auc(tdc_risk(fit_tdc(others, ds, SERIES_F), tst, ds), y, sites))
        X = features(tst, ds); r["q_over_qsaa"] = fmt(*site_boot_auc(-X["q_qopt"].to_numpy(), y, sites))
        # leave-sites-out: fit on OTHER models' calibration months of the other half of sites, test on this model's test months of this half
        u = np.unique(sites); rng = np.random.default_rng(1); perm = rng.permutation(u); halves = [set(perm[: len(u) // 2]), set(perm[len(u) // 2:])]
        sc = np.full(len(tst), np.nan)
        for h in halves:
            tr = others[~(others.practice + "|" + others.chemical).isin(h)]; te = np.array([s in h for s in sites])
            sc[te] = tdc_risk(fit_tdc(tr, ds), tst[te], ds)
        r["leave_sites_out"] = fmt(*site_boot_auc(sc, y, sites))
        # leave-panel-out: fit on all models' calibration months of the other two panels
        # features are scale-free ratios, so the fit transfers across panels; the panel label is not used
        def feats_multi(df):
            return pd.concat([features(g.drop(columns="ds"), k) for k, g in df.groupby("ds")])
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        Xo = feats_multi(other_panels); yo = (~other_panels.covered).astype(int).loc[Xo.index]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xo, yo)
        r["leave_panel_out"] = fmt(*site_boot_auc(clf.predict_proba(features(tst, ds))[:, 1], y, sites))
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{RES}/analysis11_auroc_ci.csv", index=False); print("A11 done")

# ---------------------------------------------------------------- A12
def cluster_ols(x, y, g):
    x, y = np.asarray(x, float), np.asarray(y, float); X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for k in np.unique(g):
        ii = g == k; u = X[ii].T @ e[ii]; meat += np.outer(u, u)
    G = len(np.unique(g)); n = len(x); adj = G / (G - 1) * (n - 1) / (n - 2)
    V = adj * XtX_inv @ meat @ XtX_inv; return b[1], np.sqrt(V[1, 1])
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        a = load(ds, "agentic", m)
        if a is None or len(a) < 60: continue
        a = a.sort_values(["practice", "chemical", "month"]); g = a.groupby(["practice", "chemical"])
        a["prev_err"] = (g.demand.shift(1) - g.order.shift(1)) / g.order.shift(1); a["chg"] = (a.order - g.order.shift(1)) / g.order.shift(1)
        a["lerr"] = np.log(g.demand.shift(1) / g.order.shift(1)); a["lchg"] = np.log(a.order / g.order.shift(1))
        a["site"] = a.practice + "|" + a.chemical
        s = a.dropna(subset=["prev_err", "chg"]); s = s[(s.prev_err.abs() < 3) & (s.chg.abs() < 3)]
        b1, se1 = cluster_ols(s.prev_err, s.chg, s.site.to_numpy())
        s2 = a.replace([np.inf, -np.inf], np.nan).dropna(subset=["lerr", "lchg"]); s2 = s2[(s2.lerr.abs() < np.log(4)) & (s2.lchg.abs() < np.log(4))]
        b2, se2 = cluster_ols(s2.lerr, s2.lchg, s2.site.to_numpy())
        # anchoring specification of the human literature: order_t on last demand and own previous order (levels, scaled by history mean)
        hm = np.array([np.mean(HIST[ds].loc[i, "history"]) for i in a.decision_id]); a["o"] = a.order / hm; a["pd"] = g.demand.shift(1) / hm; a["po"] = g.order.shift(1) / hm
        s3 = a.dropna(subset=["pd", "po"]); s3 = s3[(s3.o < 4) & (s3.pd < 4) & (s3.po < 4)]
        X = np.c_[np.ones(len(s3)), s3.pd, s3.po]; bb = np.linalg.lstsq(X, s3.o, rcond=None)[0]
        rows.append(dict(dataset=ds, model=name, n=len(s), react=round(b1, 3), se_cluster=round(se1, 3), react_log=round(b2, 3), se_log_cluster=round(se2, 3),
                         n_sites=s.site.nunique(), w_last_demand=round(bb[1], 3), w_prev_order=round(bb[2], 3)))
pd.DataFrame(rows).to_csv(f"{RES}/analysis12_reaction_cluster.csv", index=False); print("A12 done")

# ---------------------------------------------------------------- A13 rolling-origin correction
rows = []
for ds in DATASETS:
    W = 6 if ds != "freshretail" else 7
    for cond in ["quantile", "agentic"]:
        for m, name in MODELS:
            d = load(ds, cond, m)
            if d is None or len(d) < 60: continue
            d = d.copy(); d["t"] = HIST[ds].loc[d.decision_id, "t"].to_numpy(); d["s"] = (d.demand - d.order) / d.order
            t = d[d.split == "test"].copy(); out = np.zeros(len(t)); fac = []
            for k, (idx, row) in enumerate(t.iterrows()):
                hist = d[(d.t < row.t) & (d.t >= row.t - W)].s
                f = 1 + conf_q(hist, TAU); out[k] = row.order * f; fac.append(f)
            cal = d[d.split == "calib"]; fixed = t.order * (1 + conf_q(cal.s, TAU))
            rows.append(dict(dataset=ds, cond=cond, model=name, raw_cov=round(t.covered.mean(), 3), raw_cost=round(cost_ratio(t.order, t), 3),
                             fixed_cov=round(float(np.mean(t.demand.values <= fixed)), 3), fixed_cost=round(cost_ratio(fixed, t), 3),
                             rolling_cov=round(float(np.mean(t.demand.values <= out)), 3), rolling_cost=round(cost_ratio(out, t), 3),
                             factor_first=round(fac[0], 3), factor_last=round(fac[-1], 3)))
pd.DataFrame(rows).to_csv(f"{RES}/analysis13_rolling_correction.csv", index=False); print("A13 done")

# ---------------------------------------------------------------- A14 review on all panels + break-even
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        d = load(ds, "agentic", m)
        if d is None or len(d) < 60: continue
        cal, t = d[d.split == "calib"], d[d.split == "test"].copy().reset_index(drop=True)
        t["order_human"] = [planner_order(HIST[ds].loc[i, "history"]) for i in t.decision_id]
        saa_cost_per_dec = nv_cost(t.q_opt, t.demand).mean()
        r = dict(dataset=ds, model=name, cost_none=cost_ratio(t.order, t))
        r["cost_random"], _ = review(t, None); r["cost_hand"], _ = review(t, hand_risk(t, cal)); r["cost_tdc"], _ = review(t, tdc_risk(fit_tdc(cal, ds), t, ds))
        y = (~t.covered).astype(int)
        r["auroc_hand"] = auroc(hand_risk(t, cal), y); r["auroc_tdc"] = auroc(tdc_risk(fit_tdc(cal, ds), t, ds), y)
        n_rev = sum(int(round(0.2 * len(mth))) for _, mth in t.groupby("month"))
        # break-even review cost per reviewed order, in units of SAA's mean cost per decision
        r["breakeven_c_tdc"] = (r["cost_none"] - r["cost_tdc"]) * len(t) / n_rev
        r["breakeven_c_hand"] = (r["cost_none"] - r["cost_hand"]) * len(t) / n_rev
        rows.append(r)
pd.DataFrame(rows).round(3).to_csv(f"{RES}/analysis14_review_all_panels.csv", index=False); print("A14 done")

# ---------------------------------------------------------------- A15 EWMA rule
rows = []
for ds in DATASETS:
    H_ = HIST[ds]; t = H_[H_.split == "test"]
    def ewma_order(h, alpha=0.3):
        h = np.asarray(h, float); lvl = h[0]; res = []
        for x in h[1:]:
            res.append(x - lvl); lvl = alpha * x + (1 - alpha) * lvl
        return lvl + np.quantile(res, TAU)
    o = np.array([ewma_order(h) for h in t.history]); dmd = t.demand.to_numpy(); q = t.q_opt.to_numpy()
    rows.append(dict(dataset=ds, ewma_cov=round(float(np.mean(dmd <= o)), 3), ewma_cost=round(nv_cost(o, dmd).sum() / nv_cost(q, dmd).sum(), 3),
                     saa_cov=round(float(np.mean(dmd <= q)), 3)))
pd.DataFrame(rows).to_csv(f"{RES}/analysis15_ewma.csv", index=False); print("A15 done")

# ---------------------------------------------------------------- A16 informed-treatment VDC, magnitude errors, feedback correlation, distinct values
rows = []
for ds in DATASETS + (["pbs"] if os.path.exists(f"{RES}/pbs_full_decisions_basic_gemma4.csv") else []):
    for m, name in MODELS:
        r = dict(dataset=ds, model=name)
        for cond in ["basic", "informed", "agentic"]:
            d = load(ds, cond, m)
            if d is None or len(d) < 60: continue
            t = d[d.split == "test"]; y = (~t.covered).astype(int)
            r[f"{cond}_auroc_vdc"] = round(auroc(-t.confidence, y), 2) if t.confidence.nunique() > 1 else np.nan
            r[f"{cond}_distinct"] = d.confidence.round(2).nunique()
            r[f"{cond}_mag_err"] = round(float(((t.order > 5 * t.demand) | (t.order < t.demand / 5)).mean()), 3)
            r[f"{cond}_rho_cov"] = round(spearmanr(d.confidence, d.covered.astype(int)).correlation, 2) if d.confidence.nunique() > 1 else np.nan
            r[f"{cond}_cov"] = round(t.covered.mean(), 3); r[f"{cond}_cost"] = round(cost_ratio(t.order, t), 2)
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{RES}/analysis16_vdc_by_treatment.csv", index=False); print("A16 done")
