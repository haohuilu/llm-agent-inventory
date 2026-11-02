"""Derived analyses reported in the paper (Sections 4-7 and the appendices), computed from the per-decision
result files written by run_mac.py. Writes utd24_runs/results/analysis{2..9}_*.csv.

usage: python3 analysis.py            (reads utd24_runs/results and utd24_runs/data; no model calls)

A2  VDC collapse: distinct values, modal share, normalised entropy, rank correlations          (Table 4)
A3  Feedback dynamics in the agent loop: reaction coefficient, direction, early/late coverage  (Table 6, D.x)
A4  Trace-based decision confidence (TDC) vs VDC: shortfall AUROC, own / other-model fits     (Table 5)
A5  E4 review at a 20% budget with the TDC-based index                                         (Table 10)
A6  Escalation to the quantile prompt instead of a planner                                    (Table E.x)
A7  TDC ablation: order-position features only, series features only, all (other-model fit)    (Tables 5, D.x)
A10 E4 with an SAA-quantile planner instead of the pull-to-centre planner                      (Table E.x)
A8  Paired block-bootstrap intervals for quantile minus basic coverage and cost               (Table D.x)
A9  Pipeline end to end: quantile order plus pooled / per-group correction                    (Table 11)
"""
import os, sys, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

DRIVE = os.path.abspath(os.environ.get("UTD24_DRIVE", "./utd24_runs"))
RES, DATA = f"{DRIVE}/results", f"{DRIVE}/data"
B, H = 9, 1; TAU = B / (B + H)
DATASETS = ["nhs", "pbs_k", "freshretail"]
MODELS = [("qwen2.5-7b-instruct", "Qwen"), ("llama3.1-8b", "Llama"), ("gemma4", "Gemma"),
          ("phi4-mini", "Phi"), ("gpt-6-luna", "GPT"), ("claude-haiku-5-5", "Claude")]
PKL = {"nhs": "nhs", "pbs_k": "pbs_k", "freshretail": "freshretail"}

def nv_cost(order, demand): return B * np.maximum(demand - order, 0) + H * np.maximum(order - demand, 0)
def cost_ratio(order, d): return nv_cost(order, d.demand).sum() / nv_cost(d.q_opt, d.demand).sum()
def conf_q(scores, tau):
    s = np.sort(np.asarray(scores, float)); s = s[~np.isnan(s)]; n = len(s)
    if n == 0: return 0.0
    k = min(max(int(np.ceil((n + 1) * tau)), 1), n); return s[k - 1]
def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, float); ok = ~np.isnan(s) & ~np.isnan(y)
    if ok.sum() < 10 or y[ok].min() == y[ok].max(): return np.nan
    return roc_auc_score(y[ok].astype(int), s[ok])

def load(ds, cond, m):
    p = f"{RES}/{ds}_full_decisions_{cond}_{m}.csv"
    if not os.path.exists(p) or os.path.getsize(p) < 10: return None
    d = pd.read_csv(p); d = d[d.order.notna()].copy()
    d["covered"] = d.covered.astype(bool); return d

HIST = {ds: pd.read_pickle(f"{DATA}/decisions_{PKL[ds]}_full.pkl").set_index("decision_id") for ds in DATASETS}

def features(d, ds):
    """TDC inputs observable when the order is placed."""
    h = HIST[ds].loc[d.decision_id, "history"]
    H_ = np.array([np.asarray(x, float) for x in h]); q = d.order.to_numpy(float)
    mean, last, mx = H_.mean(1), H_[:, -1], H_.max(1)
    cv = H_.std(1) / mean; trend = H_[:, -6:].mean(1) / H_[:, :6].mean(1)
    X = pd.DataFrame({"q_mean": q / mean, "q_qopt": q / d.q_opt.to_numpy(), "q_last": q / last, "q_max": q / mx,
                      "cv": cv, "trend": trend, "last_mean": last / mean}, index=d.index)
    return X.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 10)
AGENT_F, SERIES_F = ["q_mean", "q_qopt", "q_last", "q_max"], ["cv", "trend", "last_mean"]

def fit_tdc(train, ds, cols=None, gb=False):
    cols = cols or AGENT_F + SERIES_F
    X, y = features(train, ds)[cols], (~train.covered).astype(int)
    if y.nunique() < 2: return None
    clf = GradientBoostingClassifier(random_state=0) if gb else make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return clf.fit(X, y), cols
def tdc_risk(model, d, ds):
    if model is None: return np.full(len(d), np.nan)
    clf, cols = model; return clf.predict_proba(features(d, ds)[cols])[:, 1]

# ---------------------------------------------------------------- A2: VDC collapse
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        for cond in ["basic", "informed", "agentic"]:
            d = load(ds, cond, m)
            if d is None or len(d) < 60: continue
            c = d.confidence.dropna().round(2); vc = c.value_counts(normalize=True)
            ent = -(vc * np.log(vc)).sum(); norm = ent / np.log(len(vc)) if len(vc) > 1 else 0.0
            cv = features(d, ds)["cv"]
            r1 = spearmanr(c, cv.loc[c.index]).correlation if c.nunique() > 1 else np.nan
            r2 = spearmanr(c, d.covered.loc[c.index].astype(int)).correlation if c.nunique() > 1 else np.nan
            rows.append(dict(dataset=ds, model=name, cond=cond, n=len(d), distinct=c.nunique(), mode=vc.index[0],
                             mode_share=vc.iloc[0], norm_entropy=norm, rho_conf_volatility=r1, rho_conf_covered=r2))
pd.DataFrame(rows).round(2).to_csv(f"{RES}/analysis2_confidence_collapse.csv", index=False); print("A2 done")

# ---------------------------------------------------------------- A3: feedback dynamics
def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); X = np.c_[np.ones(len(x)), x]
    b, res, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    se = np.sqrt((r @ r) / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum()) if len(x) > 2 else np.nan
    return b[1], se
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        a, b_ = load(ds, "agentic", m), load(ds, "basic", m)
        if a is None or len(a) < 60: continue
        a = a.sort_values(["practice", "chemical", "month"]); a["t"] = HIST[ds].loc[a.decision_id, "t"].to_numpy()
        g = a.groupby(["practice", "chemical"])
        a["prev_err"] = (g.demand.shift(1) - g.order.shift(1)) / g.order.shift(1)
        a["chg"] = (a.order - g.order.shift(1)) / g.order.shift(1)
        s = a.dropna(subset=["prev_err", "chg"]); s = s[(s.prev_err.abs() < 3) & (s.chg.abs() < 3)]
        react, se = ols(s.prev_err, s.chg)
        sh, su = s[s.prev_err > 0], s[s.prev_err < 0]
        after_short = ols(sh.prev_err, sh.chg)[0] if len(sh) > 5 else np.nan
        after_surplus = ols(su.prev_err, su.chg)[0] if len(su) > 5 else np.nan
        chase_r = spearmanr(s.prev_err, s.chg).correlation
        dir_ok = (np.sign(s.chg) == np.sign(s.prev_err)).mean()
        half = (g.cumcount() < 6).to_numpy()                      # first six steps of each series vs the rest
        b_ = b_.set_index("decision_id").reindex(a.decision_id)
        rows.append(dict(dataset=ds, model=name, n=len(s), react=react, se=se, after_short=after_short, after_surplus=after_surplus,
                         chase_r=chase_r, dir_ok=dir_ok, agent_early=a.covered[half].mean(), agent_late=a.covered[~half].mean(),
                         basic_early=b_.covered[half].mean(), basic_late=b_.covered[~half].mean()))
pd.DataFrame(rows).round(2).to_csv(f"{RES}/analysis3_learning_dynamics.csv", index=False); print("A3 done")

# ---------------------------------------------------------------- A4 + A7: TDC
rows4, rows7 = [], []
for ds in DATASETS:
    for cond in ["basic", "agentic"]:
        D = {m: load(ds, cond, m) for m, _ in MODELS}; D = {m: d for m, d in D.items() if d is not None and len(d) >= 60}
        for m, name in MODELS:
            if m not in D: continue
            d = D[m]; cal, tst = d[d.split == "calib"], d[d.split == "test"]; y = (~tst.covered).astype(int)
            others = pd.concat([D[o][D[o].split == "calib"] for o in D if o != m])
            verb = auroc(-tst.confidence, y) if tst.confidence.nunique() > 1 else np.nan
            own = auroc(tdc_risk(fit_tdc(cal, ds), tst, ds), y)
            oth = auroc(tdc_risk(fit_tdc(others, ds), tst, ds), y)
            gb = auroc(tdc_risk(fit_tdc(others, ds, gb=True), tst, ds), y)
            rows4.append(dict(dataset=ds, cond=cond, model=name, verbalised=verb, trace_LR_own_calib=own,
                              trace_LR_other_models_calib=oth, trace_GB_other_models_calib=gb))
            if cond == "basic":   # ablation: fitted on the other models' calibration months, as in the paper's Tables 5 and D.x
                rows7.append(dict(dataset=ds, model=name, agent=auroc(tdc_risk(fit_tdc(others, ds, AGENT_F), tst, ds), y),
                                  series=auroc(tdc_risk(fit_tdc(others, ds, SERIES_F), tst, ds), y), all=oth,
                                  agent_own=auroc(tdc_risk(fit_tdc(cal, ds, AGENT_F), tst, ds), y),
                                  series_own=auroc(tdc_risk(fit_tdc(cal, ds, SERIES_F), tst, ds), y), all_own=own))
pd.DataFrame(rows4).round(2).to_csv(f"{RES}/analysis4_trace_confidence.csv", index=False)
# feature importance of a gradient-boosted TDC fitted on all six models' NHS calibration months (basic prompt)
D = pd.concat([load("nhs", "basic", m) for m, _ in MODELS]); D = D[D.split == "calib"]
gb, cols = fit_tdc(D, "nhs", gb=True)
pd.DataFrame({"feature": cols, "importance": gb.feature_importances_}).round(3).sort_values("importance", ascending=False).to_csv(f"{RES}/analysis4b_gb_importance.csv", index=False)
pd.DataFrame(rows7).round(2).to_csv(f"{RES}/analysis7_tdc_ablation.csv", index=False); print("A4, A7 done")

# ---------------------------------------------------------------- A5: E4 with the TDC index
def planner_order(hist, beta=0.6): h = np.asarray(hist, float); return beta * np.quantile(h, TAU) + (1 - beta) * h.mean()
def hand_risk(t, cal):
    site = (np.maximum(cal.demand - cal.order, 0) / cal.order).groupby([cal.practice, cal.chemical]).mean()
    hp = pd.Series(list(zip(t.practice, t.chemical))).map(site.to_dict()).fillna(site.mean()).to_numpy()
    return hp + np.maximum(t.q_opt.to_numpy() - t.order.to_numpy(), 0) / t.q_opt.to_numpy()
def review(t, risk, kf=0.2, seed=0):
    rng = np.random.default_rng(seed); pick = pd.Series(False, index=t.index)
    for _, mth in t.groupby("month"):
        k = int(round(kf * len(mth)))
        if k == 0: continue
        idx = rng.choice(mth.index, k, replace=False) if risk is None else pd.Series(risk, index=t.index).loc[mth.index].sort_values(ascending=False).index[:k]
        pick.loc[idx] = True
    final = np.where(pick, t.order_human, t.order)
    return cost_ratio(final, t), float(np.mean(t.demand.values <= final))
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        d = load(ds, "agentic", m)
        if d is None or len(d) < 60: d = load(ds, "basic", m)
        if d is None: continue
        cal, t = d[d.split == "calib"], d[d.split == "test"].copy().reset_index(drop=True)
        t["order_human"] = [planner_order(HIST[ds].loc[i, "history"]) for i in t.decision_id]
        r = dict(dataset=ds, model=name)
        r["cost_none"], r["cov_none"] = cost_ratio(t.order, t), t.covered.mean()
        r["cost_random"], r["cov_random"] = review(t, None)
        r["cost_stated"], r["cov_stated"] = review(t, -t.confidence.fillna(t.confidence.mean()).to_numpy())
        r["cost_hand"], r["cov_hand"] = review(t, hand_risk(t, cal))
        r["cost_trace"], r["cov_trace"] = review(t, tdc_risk(fit_tdc(cal, ds), t, ds))
        rows.append(r)
cols = ["dataset", "model"] + [f"cost_{k}" for k in ["none", "random", "stated", "hand", "trace"]] + [f"cov_{k}" for k in ["none", "random", "stated", "hand", "trace"]]
pd.DataFrame(rows)[cols].round(2).to_csv(f"{RES}/analysis5_e4_trace_review.csv", index=False); print("A5 done")

# ---------------------------------------------------------------- A6: escalation to the quantile prompt
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        b_, q = load(ds, "basic", m), load(ds, "quantile", m)
        if b_ is None or q is None: continue
        cal = b_[b_.split == "calib"]; t = b_[b_.split == "test"].copy()
        qo = q.set_index("decision_id").order.reindex(t.decision_id).to_numpy(); ok = ~np.isnan(qo); t = t[ok]; qo = qo[ok]
        risk = tdc_risk(fit_tdc(cal, ds), t, ds)
        r = dict(dataset=ds, model=name); outs = {"agent": t.order.to_numpy(), "quantile": qo}
        for kf in (0.2, 0.5):
            pick = np.zeros(len(t), bool)
            for _, mth in t.reset_index(drop=True).groupby("month"):
                k = int(round(kf * len(mth))); idx = pd.Series(risk[mth.index], index=mth.index).sort_values(ascending=False).index[:k]; pick[idx] = True
            outs[f"hybrid{int(kf*100)}"] = np.where(pick, qo, t.order)
        outs["max_of_both"] = np.maximum(t.order.to_numpy(), qo)
        for k, o in outs.items(): r[f"cost_{k}"] = cost_ratio(o, t)
        for k, o in outs.items(): r[f"cov_{k}"] = float(np.mean(t.demand.values <= o))
        rows.append(r)
pd.DataFrame(rows).round(2).to_csv(f"{RES}/analysis6_hybrid_escalation.csv", index=False); print("A6 done")

# ---------------------------------------------------------------- A8: paired block bootstrap, quantile minus basic
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        b_, q = load(ds, "basic", m), load(ds, "quantile", m)
        if b_ is None or q is None: continue
        t = b_[b_.split == "test"].set_index("decision_id"); qq = q[q.split == "test"].set_index("decision_id").reindex(t.index)
        ok = qq.order.notna(); t, qq = t[ok], qq[ok]
        series = (t.practice + "|" + t.chemical).to_numpy(); u = np.unique(series); rng = np.random.default_rng(0)
        d_cov = qq.covered.mean() - t.covered.mean(); d_cost = cost_ratio(qq.order, t) - cost_ratio(t.order, t)
        bc, bk = [], []
        for _ in range(1000):
            pick = rng.choice(u, len(u), replace=True); idx = np.concatenate([np.where(series == s)[0] for s in pick])
            tb, qb = t.iloc[idx], qq.iloc[idx]
            bc.append(qb.covered.mean() - tb.covered.mean()); bk.append(cost_ratio(qb.order, tb) - cost_ratio(tb.order, tb))
        rows.append(dict(dataset=ds, model=name, d_cov=round(d_cov, 3), lo=round(np.percentile(bc, 2.5), 3), hi=round(np.percentile(bc, 97.5), 3),
                         d_cost=round(d_cost, 2), cost_lo=round(np.percentile(bk, 2.5), 2), cost_hi=round(np.percentile(bk, 97.5), 2)))
pd.DataFrame(rows).to_csv(f"{RES}/analysis8_quantile_vs_basic_ci.csv", index=False); print("A8 done")

# ---------------------------------------------------------------- A9: pipeline (quantile order + correction)
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        q = load(ds, "quantile", m)
        if q is None: continue
        cal, t = q[q.split == "calib"], q[q.split == "test"].copy()
        s = (cal.demand - cal.order) / cal.order
        qp = conf_q(s, TAU); qg = {g: conf_q(s[cal.group == g], TAU) for g in cal.group.unique()}
        pooled = t.order * (1 + qp); grp = t.order * (1 + t.group.map(qg).fillna(qp))
        def mg(o): return pd.Series(t.demand.values <= o, index=t.index).groupby(t.group).mean().min()
        rows.append(dict(dataset=ds, model=name, q_cov=t.covered.mean(), q_cost=cost_ratio(t.order, t),
                         pipe_cov=float(np.mean(t.demand.values <= pooled)), pipe_cost=cost_ratio(pooled, t), pipe_min_grp=mg(pooled),
                         pipe_g_cov=float(np.mean(t.demand.values <= grp)), pipe_g_cost=cost_ratio(grp, t), pipe_g_min_grp=mg(grp), mult=round(1 + qp, 3)))
pd.DataFrame(rows).round({"q_cov": 2, "q_cost": 2, "pipe_cov": 2, "pipe_cost": 2, "pipe_min_grp": 2, "pipe_g_cov": 2, "pipe_g_cost": 2, "pipe_g_min_grp": 2}).to_csv(f"{RES}/analysis9_pipeline.csv", index=False); print("A9 done")

# ---------------------------------------------------------------- A10: E4 with an SAA-quantile planner
rows = []
for ds in DATASETS:
    for m, name in MODELS:
        d = load(ds, "agentic", m)
        if d is None or len(d) < 60: d = load(ds, "basic", m)
        if d is None: continue
        cal, t = d[d.split == "calib"], d[d.split == "test"].copy().reset_index(drop=True)
        t["order_human"] = t.q_opt
        r = dict(dataset=ds, model=name, cost_none=cost_ratio(t.order, t))
        r["cost_random"], _ = review(t, None); r["cost_hand"], _ = review(t, hand_risk(t, cal)); r["cost_trace"], _ = review(t, tdc_risk(fit_tdc(cal, ds), t, ds))
        rows.append(r)
pd.DataFrame(rows).round(2).to_csv(f"{RES}/analysis10_saa_planner.csv", index=False); print("A10 done")
