import matplotlib; matplotlib.use("Agg")      # figures go to files, no window
display = print                                # notebook display() -> print
import sys, os

# ===== cell 0 =====
#@title Configuration
MODE = "full"                    # "pilot" | "quick" | "full"
DATASET = "pbs"                   # "nhs" | "pbs" | "freshretail"
RUN = "all"                    # which models THIS run executes: "open" | "openai" | "anthropic" | "all" | a list of names
                                  # Results accumulate across runs; the analysis cells read every model found in results/.
# Every model (open-weight and API) sees the SAME decisions: a stratified set of sites with their full
# calibration+test histories, sized so that the total number of decisions is n_decisions.
# pilot : ~20-30 min on a T4 ->  90 decisions (15 sites x 6 periods), one medicine
# quick : ~1-2 h            -> 300 decisions (25 sites x 12 periods), two medicines
# full  : several hours     -> 600 decisions (50 sites x 12 periods), four medicines. Paper numbers.
SETTINGS = {"pilot": dict(n_items=1, per_group=6,  calib=3, test=3, n_samples=2, n_decisions=90),
            "quick": dict(n_items=2, per_group=10, calib=6, test=6, n_samples=3, n_decisions=300),
            "full":  dict(n_items=4, per_group=20, calib=6, test=6, n_samples=3, n_decisions=600)}[MODE]
N_PER_GROUP = SETTINGS["per_group"]           # candidate sites sampled per group before the decision cap is applied
HISTORY_MONTHS, CALIB_MONTHS, TEST_MONTHS = 24, SETTINGS["calib"], SETTINGS["test"]
N_SAMPLES, N_DECISIONS = SETTINGS["n_samples"], SETTINGS["n_decisions"]
CHEMICALS = {                     # NHS: BNF chemical code -> name, in priority order
    "metformin":    "0601022B0",  # steady, chronic
    "atorvastatin": "0212000B0",  # steady, trending up
    "amoxicillin":  "0501013B0",  # seasonal antibiotic
    "salbutamol":   "0301011R0",  # inhaler, seasonal
}
CHEMICALS = dict(list(CHEMICALS.items())[:SETTINGS["n_items"]])
B, H = 9, 1                        # under-stock / over-stock cost per unit  ->  target service level tau = 0.9
TAU = B / (B + H)

OPEN_MODELS = [                    # Ollama tags (4-bit GGUF); ~5 GB, ~5 GB, ~7 GB, ~2.5 GB on disk
    "qwen2.5:7b-instruct",
    "llama3.1:8b",
    "gemma4",
    "phi4-mini",
]
OLLAMA_URL = "http://localhost:11434"
# API models: (provider, run name). A run name may map to a base model plus settings in API_VARIANTS, so one model
# can appear under several configurations with separate caches and result files.
API_MODELS = [("openai", "gpt-6-luna"), ("anthropic", "claude-haiku-5-5")]
API_VARIANTS = {"gpt-6-luna":      dict(base="gpt-6-luna", reasoning_effort="none"),   # reasoning off (cheap, fast)
                "claude-haiku-5-5": dict(base="claude-haiku-5-5")}
PRICE_PER_M = {"gpt-6-luna": (0.10, 0.50), "gpt-6-luna-rlow": (0.10, 0.50), "claude-haiku-5-5": (0.10, 0.50)}   # USD per 1M (input, output) tokens
LLM_BUDGET_USD = 100.0              # hard cap, enforced before every API call

import os
DRIVE = os.path.abspath("./utd24_runs")     # data, caches and results live here, next to the notebook

if len(sys.argv) > 1: DATASET = sys.argv[1]            # nhs | pbs | freshretail
if len(sys.argv) > 2: RUN = sys.argv[2]                # open | openai | anthropic | all
STOP_AFTER_SECTION3 = len(sys.argv) > 3 and sys.argv[3] == "stop3"


# ===== cell 1 =====
# Install (first time only): pip install requests pandas numpy scikit-learn matplotlib tqdm openai anthropic python-dotenv datasets
for sub in ["data", "cache", "results"]: os.makedirs(f"{DRIVE}/{sub}", exist_ok=True)
import json, time, hashlib, re, math, warnings, requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
from io import StringIO
from tqdm import tqdm                      # plain text bars: no ipywidgets traffic between kernel and VS Code
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 140)
HEADERS = {"User-Agent": "utd24-research-notebook (academic use)"}
print("mode:", MODE, "| dataset:", DATASET, "| run:", RUN, "| tau =", TAU, "| folder:", DRIVE)

# ===== cell 2 =====
# API keys from a .env file next to the notebook (OPENAI_API_KEY=..., ANTHROPIC_API_KEY=...) or from the environment.
try:
    from dotenv import load_dotenv; load_dotenv(".env")
except Exception:
    pass
print({k: bool(os.environ.get(k)) for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]})

# ===== cell 3 =====
if DATASET == "nhs":   # cell is skipped (no error) for the other datasets, so Run All never halts here
    NHSBSA = "https://opendata.nhsbsa.net/api/3/action"

    def nhsbsa_sql(sql, resource_id):
        for attempt in range(5):
            r = requests.get(f"{NHSBSA}/datastore_search_sql", params={"resource_id": resource_id, "sql": sql}, headers=HEADERS, timeout=300)
            if r.status_code == 200:
                res = r.json().get("result", {})
                if "records" in res: return res["records"]
                if isinstance(res.get("result"), dict) and "records" in res["result"]: return res["result"]["records"]
                if "result_url" in res: return requests.get(res["result_url"], headers=HEADERS, timeout=300).json()
                raise RuntimeError(f"unexpected response: {str(res)[:300]}")
            time.sleep(2 ** attempt)
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")

    pkg = requests.get(f"{NHSBSA}/package_show", params={"id": "english-prescribing-data-epd"}, headers=HEADERS, timeout=120).json()
    res_names = sorted(r["name"] for r in pkg["result"]["resources"] if re.fullmatch(r"EPD_\d{6}", r["name"]))
    N_MONTHS = HISTORY_MONTHS + CALIB_MONTHS + TEST_MONTHS
    res_names = res_names[-N_MONTHS:]
    months = pd.to_datetime([n[-6:] for n in res_names], format="%Y%m")
    print("latest month:", res_names[-1][-6:], "| using", res_names[0][-6:], "→", res_names[-1][-6:], f"({len(months)} months)")

    frames = []
    for name, code_ in CHEMICALS.items():
        p = f"{DRIVE}/data/nhsbsa_{name}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
        else:
            parts = []
            for res, m in tqdm(list(zip(res_names, months)), desc=name):
                sql = (f"SELECT PRACTICE_CODE, PRACTICE_NAME, SUM(ITEMS) AS items FROM `{res}` "
                       f"WHERE BNF_CHEMICAL_SUBSTANCE = '{code_}' GROUP BY PRACTICE_CODE, PRACTICE_NAME")
                d = pd.DataFrame(nhsbsa_sql(sql, res)).rename(columns={"PRACTICE_CODE": "practice", "PRACTICE_NAME": "practice_name"})
                d["items"] = d["items"].astype(float); d["date"] = m; d["chemical"] = name
                parts.append(d[["practice", "practice_name", "date", "items", "chemical"]])
            df = pd.concat(parts, ignore_index=True); df.to_csv(p, index=False)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(raw.shape, "rows;", raw.practice.nunique(), "practices"); raw.head()

# ===== cell 4 =====
if DATASET == "nhs":   # cell is skipped (no error) for the other datasets, so Run All never halts here
    grp_path = f"{DRIVE}/data/practice_groups_imd.csv"
    if os.path.exists(grp_path):
        groups = pd.read_csv(grp_path)
    else:
        try:
            # 1. practice -> postcode (latest month, any of our medicines)
            pc = pd.DataFrame(nhsbsa_sql(f"SELECT DISTINCT PRACTICE_CODE, POSTCODE FROM `{res_names[-1]}` "
                                         f"WHERE BNF_CHEMICAL_SUBSTANCE = '{list(CHEMICALS.values())[0]}'", res_names[-1]))
            pc = pc.rename(columns={"PRACTICE_CODE": "practice", "POSTCODE": "postcode"}).dropna()
            pc["postcode"] = pc.postcode.str.replace(" ", "").str.upper()
            # 2. postcode -> 2011 LSOA via postcodes.io (bulk, 100 per request); IMD 2019 is on 2011 LSOAs
            lsoa = {}
            pcs = pc.postcode.unique().tolist()
            for i in tqdm(range(0, len(pcs), 100), desc="postcodes.io"):
                r = requests.post("https://api.postcodes.io/postcodes", json={"postcodes": pcs[i:i+100]}, headers=HEADERS, timeout=120)
                for item in r.json().get("result", []):
                    res = item.get("result")
                    code = (res or {}).get("codes", {}); c = code.get("lsoa11") or code.get("lsoa")
                    if c: lsoa[item["query"].replace(" ", "").upper()] = c
            pc["lsoa"] = pc.postcode.map(lsoa)
            # 3. LSOA -> IMD 2019 decile (official IoD2019 File 7; opendatacommunities as backup)
            imd = None
            for url in ["https://assets.publishing.service.gov.uk/media/5dc407b440f0b6379a7acc8d/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv",
                        "https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd2019%2Findices"]:
                try:
                    r = requests.get(url, headers=HEADERS, timeout=300); r.raise_for_status()
                    t = pd.read_csv(StringIO(r.text), low_memory=False)
                    code_col = [c for c in t.columns if "LSOA" in c and "code" in c.lower()][0]
                    dec_col  = [c for c in t.columns if "Index of Multiple Deprivation" in c and "Decile" in c]
                    if dec_col:
                        imd = t[[code_col, dec_col[0]]].rename(columns={code_col: "lsoa", dec_col[0]: "imd_decile"})
                    else:   # opendatacommunities long format
                        t = t[(t["Indices of Deprivation"].str.contains("Multiple", na=False)) & (t["Measurement"].str.contains("Decile", na=False))]
                        imd = t[["FeatureCode", "Value"]].rename(columns={"FeatureCode": "lsoa", "Value": "imd_decile"})
                    break
                except Exception as e:
                    print("IMD source failed:", url[:60], e)
            if imd is None: raise RuntimeError("no IMD source reachable")
            imd["imd_decile"] = pd.to_numeric(imd.imd_decile, errors="coerce")
            g = pc.merge(imd, on="lsoa", how="left").dropna(subset=["imd_decile"])
            # decile 1 = most deprived 10%. Quintile label: Q5 = most deprived (deciles 1-2) ... Q1 = least deprived (deciles 9-10)
            g["group"] = (6 - np.ceil(g.imd_decile / 2)).astype(int).map(lambda q: f"Q{q}")
            groups = g[["practice", "postcode", "lsoa", "imd_decile", "group"]].drop_duplicates("practice")
            groups["group_source"] = "IMD2019 quintile via postcode (postcodes.io + IoD2019)"
        except Exception as e:
            print("IMD route failed:", e, "\n→ falling back to prescribing-volume tertiles (NOT suitable for the equity analysis)")
            ls = raw.groupby("practice").items.mean().rename("volume").reset_index()
            groups = ls.copy(); groups["group"] = pd.qcut(groups.volume, 3, labels=["small", "medium", "large"]).astype(str)
            groups["group_source"] = "prescribing-volume tertile (fallback)"
        groups.to_csv(grp_path, index=False)
    print(groups.group_source.iloc[0]); print(groups.group.value_counts().sort_index())

# ===== cell 5 =====
if DATASET == "pbs":   # cell is skipped (no error) for the other datasets, so Run All never halts here
    PBS_FILES = ["dos-jul-2020-to-jun-2021-phrmcy-type.csv", "dos-jul-2021-to-jun-2022-phrmcy-type.csv", "dos-jul-2022-to-jun-2023-phrmcy-type.csv",
                 "dos-jul-2023-to-jun-2024-phrmcy-type.csv", "dos-jul-2024-to-jun-2025-phrmcy-type.csv", "dos-jul-2025-to-jun-2026-phrmcy-type.csv"]
    pbs_path = f"{DRIVE}/data/pbs_panel_v2.csv"
    if os.path.exists(pbs_path):
        raw = pd.read_csv(pbs_path, parse_dates=["date"])
    else:
        parts = []
        for fn in tqdm(PBS_FILES, desc="PBS"):
            fp = f"{DRIVE}/data/{fn}"
            if not os.path.exists(fp):
                r = requests.get(f"https://www.pbs.gov.au/statistics/dos-and-dop/files/{fn}", headers=HEADERS, timeout=600); r.raise_for_status()
                open(fp, "wb").write(r.content)
            t = pd.read_csv(fp, dtype={"MONTH_OF_SUPPLY": str, "ITEM_CODE": str, "PATIENT_CAT": str},
                            usecols=["MONTH_OF_SUPPLY", "ITEM_CODE", "PATIENT_CAT", "PRESCRIPTIONS"], encoding="latin-1")
            t["cat"] = t.PATIENT_CAT.str[0].map({"G": "general", "C": "concessional"}); t = t.dropna(subset=["cat"])
            parts.append(t.groupby(["MONTH_OF_SUPPLY", "ITEM_CODE", "cat"], as_index=False).PRESCRIPTIONS.sum())
        t = pd.concat(parts); t["date"] = pd.to_datetime(t.MONTH_OF_SUPPLY, format="%Y%m")
        try:
            drugmap = pd.read_csv("https://www.pbs.gov.au/statistics/dos-and-dop/files/pbs-item-drug-map.csv", dtype=str, encoding="latin-1")
            code_col = [c for c in drugmap.columns if "ITEM" in c.upper()][0]
            name_col = [c for c in drugmap.columns if "DRUG" in c.upper() and "NAME" in c.upper()][0]
            names = drugmap.drop_duplicates(code_col).set_index(code_col)[name_col].str.strip().str.lower().to_dict()
        except Exception as e:
            print("drug map not read:", e); names = {}
        label = t.ITEM_CODE.map(names).fillna("item").astype(str) + " (item " + t.ITEM_CODE + ")"
        raw = pd.DataFrame({"practice": t.ITEM_CODE + "_" + t.cat, "practice_name": label, "date": t.date,
                            "items": t.PRESCRIPTIONS.astype(float), "chemical": label, "cat": t.cat})
        raw.to_csv(pbs_path, index=False)

    # ---- units: national volumes are ~10^5 a month; small models drop digits at that scale. Present them in thousands. ----
    PBS_IN_THOUSANDS = os.environ.get("PBS_UNITS", "k") != "raw"
    if PBS_IN_THOUSANDS:
        raw["items"] = (raw["items"] / 1000).round(1)
        DATASET = "pbs_k"            # separate caches and result files from the raw-unit run

    months = pd.DatetimeIndex(sorted(raw.date.unique()))[-(HISTORY_MONTHS + CALIB_MONTHS + TEST_MONTHS):]
    raw = raw[raw.date.isin(months)]
    cnt = raw.groupby("practice").date.nunique(); raw = raw[raw.practice.isin(cnt[cnt == len(months)].index)]
    item_of = raw.practice.str.split("_").str[0]
    both = raw.assign(item=item_of).groupby("item").cat.nunique(); both = both[both == 2].index
    raw = raw[item_of.isin(both)]
    vol = raw.assign(item=raw.practice.str.split("_").str[0]).groupby("item").items.mean().nlargest(N_PER_GROUP * 4).index
    raw = raw[raw.practice.str.split("_").str[0].isin(vol)]
    groups = raw.drop_duplicates("practice")[["practice", "cat"]].rename(columns={"cat": "group"})
    groups["group_source"] = "PBS patient category (general vs concessional)"
    print(DATASET, "|", raw.shape, "rows;", raw.practice.nunique(), "series;", groups.group.value_counts().to_dict())
    print(raw.drop_duplicates("chemical").chemical.head(5).tolist())

# ===== cell 6 =====
if DATASET == "freshretail":   # cell is skipped (no error) for the other datasets, so Run All never halts here
    import subprocess, sys; subprocess.run([sys.executable, "-m", "pip", "-q", "install", "datasets"], check=False)
    from datasets import load_dataset
    fr_path = f"{DRIVE}/data/freshretail_panel.csv"
    if os.path.exists(fr_path):
        raw = pd.read_csv(fr_path, parse_dates=["date"])
    else:
        ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")
        fr = ds.to_pandas()
        fr["dt"] = pd.to_datetime(fr["dt"])
        # groups = cities; keep the 5 cities with most store-SKU series
        top_cities = fr.groupby("city_id").apply(lambda g: g[["store_id", "product_id"]].drop_duplicates().shape[0], include_groups=False).nlargest(5).index
        fr = fr[fr.city_id.isin(top_cities)]
        fr["series"] = fr.store_id.astype(str) + "_" + fr.product_id.astype(str)
        # complete series only, and not too sparse
        n_days = fr.dt.nunique()
        cnt = fr.groupby("series").dt.nunique(); fr = fr[fr.series.isin(cnt[cnt == n_days].index)]
        mean_sale = fr.groupby("series").sale_amount.mean(); fr = fr[fr.series.isin(mean_sale[mean_sale > mean_sale.quantile(0.5)].index)]
        rng = np.random.default_rng(0)
        keep = pd.concat([pd.Series(g.series.unique()).sample(min(N_PER_GROUP * 10, g.series.nunique()), random_state=0) for _, g in fr.groupby("city_id")])
        fr = fr[fr.series.isin(keep)]
        scale = 100.0 / fr.sale_amount.median()          # normalised sales -> integer-like units for the prompt
        raw = pd.DataFrame({"practice": fr.series.values, "practice_name": fr.store_id.astype(str).values, "date": fr.dt.values,
                            "items": np.round(fr.sale_amount.values * scale), "chemical": "sku_" + fr.product_id.astype(str).values,
                            "censored": (fr.stock_hour6_22_cnt.values > 0), "city": fr.city_id.astype(str).values})
        raw.to_csv(fr_path, index=False)
    groups = raw.drop_duplicates("practice")[["practice", "city"]].rename(columns={"city": "group"}); groups["group"] = "city_" + groups.group.astype(str)
    groups["group_source"] = "FreshRetailNet city"
    HISTORY_MONTHS, CALIB_MONTHS, TEST_MONTHS = 30, max(7, CALIB_MONTHS), max(7, TEST_MONTHS)   # days, not months
    months = pd.DatetimeIndex(sorted(raw.date.unique()))[-(HISTORY_MONTHS + CALIB_MONTHS + TEST_MONTHS):]
    raw = raw[raw.date.isin(months)]
    print(raw.shape, "rows;", raw.practice.nunique(), "series;", groups.group.value_counts().to_dict())

# ===== cell 7 =====
panel = raw.merge(groups[["practice", "group"]], on="practice", how="inner")
if "censored" not in panel.columns: panel["censored"] = False

# keep (site, item) series that cover every period in the window, and drop tiny series
n_per_series = panel.groupby(["practice", "chemical"]).date.nunique()
complete = n_per_series[n_per_series == len(months)].index
panel = panel.set_index(["practice", "chemical"]).loc[panel.set_index(["practice", "chemical"]).index.isin(complete)].reset_index()
mean_items = panel.groupby(["practice", "chemical"]).items.mean()
keep = mean_items[mean_items >= (0.02 if DATASET == "pbs_k" else 20)].index
panel = panel.set_index(["practice", "chemical"]).loc[panel.set_index(["practice", "chemical"]).index.isin(keep)].reset_index()
print("series after filters:", panel.groupby(["practice", "chemical"]).ngroups, "| sites:", panel.practice.nunique())

# sample sites evenly across groups
cands = panel.drop_duplicates("practice")[["practice", "group"]]
if len(cands) == 0: raise RuntimeError("no series survived the filters — check the dataset cell's printout")
chosen = pd.concat([g.sample(min(N_PER_GROUP, len(g)), random_state=0) for _, g in cands.groupby("group")])
panel = panel[panel.practice.isin(chosen.practice)].sort_values(["practice", "chemical", "date"])
print("sites:", panel.practice.nunique(), "| by group:", chosen.group.value_counts().sort_index().to_dict())

month_idx = {m: i for i, m in enumerate(months)}
panel["t"] = panel.date.map(month_idx)
wide = panel.pivot_table(index=["practice", "chemical", "group"], columns="t", values="items").reset_index()
cens = panel.pivot_table(index=["practice", "chemical", "group"], columns="t", values="censored", aggfunc="max").reset_index()
rows = []
for (_, r), (_, c) in zip(wide.iterrows(), cens.iterrows()):
    series = r[list(range(len(months)))].to_numpy(dtype=float)
    for t in range(HISTORY_MONTHS, len(months)):
        if bool(c[t]): continue                       # target period censored by a stock-out: demand unobserved
        rows.append(dict(practice=r.practice, chemical=r.chemical, group=r.group, t=t, month=months[t].strftime("%Y-%m-%d"),
                         split="test" if t >= len(months) - TEST_MONTHS else "calib",
                         history=series[t-HISTORY_MONTHS:t].tolist(), demand=series[t]))
dec = pd.DataFrame(rows)
dec["decision_id"] = dec.practice + "|" + dec.chemical + "|" + dec.month
dec["q_opt"]  = dec.history.apply(lambda h: float(np.quantile(h, TAU)))   # SAA: tau-quantile of history
dec["q_mean"] = dec.history.apply(np.mean)
dec.to_pickle(f"{DRIVE}/data/decisions_{DATASET}_{MODE}.pkl")
print(dec.split.value_counts().to_dict()); dec[["decision_id", "group", "split", "demand", "q_opt"]].head()

# ===== cell 8 =====
print(raw.groupby("date").practice.nunique().tail(6))

# ===== cell 9 =====
ROLE = {"nhs": "You are the stock manager for {chemical} at one GP practice in England.",
        "pbs": "You are the national supply planner for {chemical} under Australia's Pharmaceutical Benefits Scheme, for one patient category.",
        "pbs_k": "You are the national supply planner for {chemical} under Australia's Pharmaceutical Benefits Scheme, for one patient category.",
        "freshretail": "You are the stock manager for one fresh-food product ({chemical}) at one grocery store."}[DATASET]
PERIOD = {"nhs": "month", "pbs": "month", "pbs_k": "month", "freshretail": "day"}[DATASET]
UNIT_LINE = {"nhs": "Monthly items dispensed over the last {n} months, oldest to newest:",
             "pbs": "Monthly prescriptions supplied over the last {n} months, oldest to newest:",
             "pbs_k": "Monthly prescriptions supplied, in thousands, over the last {n} months, oldest to newest:",
             "freshretail": "Daily units sold over the last {n} days, oldest to newest:"}[DATASET]
UNIT_WORD = "thousand prescriptions" if DATASET == "pbs_k" else "unit"
# pbs_k is the only dataset whose history has decimals; its wording differs in the two places below.
# Every other dataset keeps the ORIGINAL wording byte-for-byte, so the finished runs are served from cache.
TASK_NOTE = " (same units as the history)" if DATASET == "pbs_k" else ""
NUM_WORD  = "<number>" if DATASET == "pbs_k" else "<integer>"
PROMPT_BASIC = ROLE + "\n" + UNIT_LINE + """
{hist}
{site}Costs: each UNIT short next PERIOD costs {b}; each UNIT left over at the end of the PERIOD costs {h}.
Task: choose the order quantity for next PERIODTASK_NOTE. Then state your probability that this quantity will be at least next PERIOD's demand.
Respond with one JSON object and nothing else: {{"order": NUM, "confidence": <probability between 0 and 1>}}""".replace("PERIOD", PERIOD).replace("UNIT", UNIT_WORD).replace("TASK_NOTE", TASK_NOTE).replace("NUM", NUM_WORD)

PROMPT_INFORMED = PROMPT_BASIC.replace(
    f"Task: choose the order quantity for next {PERIOD}",
    f"Because a shortfall costs {{b}} times more than a leftover, the cost-minimising policy orders enough to cover demand in about {{tau_pct}}% of {PERIOD}s.\nTask: choose the order quantity for next {PERIOD}")

def fmt_hist(xs):
    return ", ".join(f"{x:.1f}" if DATASET == "pbs_k" else f"{int(x)}" for x in xs)

def build_prompt(row, site_text="", informed=False):
    tpl = PROMPT_INFORMED if informed else PROMPT_BASIC
    noun = {"nhs": "practice", "pbs": "patient group", "pbs_k": "patient group", "freshretail": "store"}[DATASET]
    site = f"About this {noun}: {site_text}\n" if site_text else ""
    return tpl.format(chemical=row.chemical, n=len(row.history), hist=fmt_hist(row.history), site=site, b=B, h=H, tau_pct=int(round(TAU*100)))

def _num(x):
    if isinstance(x, (int, float)): return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x)); return float(m.group(0)) if m else None

def parse(text):
    if not text or not text.strip(): return None
    try:
        cands = re.findall(r"\{[^{}]*\}", text, re.S)
        for s in reversed(cands):
            try: j = json.loads(s)
            except Exception:
                try: j = json.loads(s.replace("'", '"'))
                except Exception: continue
            if "order" not in j or "confidence" not in j: continue
            o, c = _num(j["order"]), _num(j["confidence"])
            if o is None or c is None: continue
            if 1 < c <= 100: c /= 100
            if o < 0 or not (0 <= c <= 1): continue
            return {"order": o, "confidence": c}
        return None
    except Exception:
        return None

COST = f"{DRIVE}/results/llm_costs.csv"
def spent(): return float(pd.read_csv(COST).usd.sum()) if os.path.exists(COST) else 0.0
def log_cost(model, tin, tout, usd):
    new = not os.path.exists(COST)
    with open(COST, "a") as f:
        if new: f.write("ts,model,tokens_in,tokens_out,usd\n")
        f.write(f"{int(time.time())},{model},{tin},{tout},{usd:.6f}\n")
def cache_key(model, prompt, temperature, seed): return hashlib.sha256(json.dumps([model, prompt, temperature, seed]).encode()).hexdigest()
class Cache:
    def __init__(self, model):
        self.path = f"{DRIVE}/cache/{model.replace('/', '__').replace(':', '-')}.jsonl"; self.d = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                k, v = json.loads(line); self.d[k] = v
    def get(self, k): return self.d.get(k)
    def put(self, k, v):
        self.d[k] = v
        with open(self.path, "a") as f: f.write(json.dumps([k, v]) + "\n")

print(build_prompt(dec.iloc[0], informed=True))

# ===== cell 10 =====
# Open-weight models via Ollama (local, free). JSON mode guarantees a parsable object; think=False makes Gemma 4 answer directly.
def _ollama_ok():
    try: return requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).status_code == 200
    except Exception: return False

def ollama_chat(model, prompt, temperature=0.0, seed=0, num_predict=200):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "format": "json",
            "think": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict}}
    for attempt in range(4):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600); r.raise_for_status()
            msg = r.json().get("message", {})
            txt = msg.get("content") or ""
            if not txt.strip():
                m = re.search(r"\{[^{}]*\}", msg.get("thinking", "") or "", re.S)
                txt = m.group(0) if m else ""
            return txt
        except Exception as e:
            if attempt == 3: raise
            time.sleep(2 ** attempt)

def run_hf(model_name, prompts, temperature=0.0, seed=0):
    """'hf' specs go to Ollama, one call at a time."""
    if not _ollama_ok(): raise RuntimeError("Ollama is not running — open the Ollama app or run `ollama serve`")
    cache = Cache(model_name)
    keys = [cache_key(model_name + "#ollama-json-nothink", p, temperature, seed) for p in prompts]
    todo = [(k, p) for k, p in zip(keys, prompts) if cache.get(k) is None]
    for k, p in tqdm(todo, desc=f"{model_name} T={temperature}", leave=False):
        cache.put(k, ollama_chat(model_name, p, temperature, seed))
    return [cache.get(k) for k in keys]

def unload_hf():
    pass

# ===== cell 11 =====
def _est_cost(model, prompt, out_tokens=None):
    pin, pout = PRICE_PER_M.get(model, (2.0, 8.0))
    if out_tokens is None: out_tokens = 1500 if API_VARIANTS.get(model, {}).get("reasoning_effort", "none") != "none" else 200
    return (len(prompt) / 4) / 1e6 * pin + out_tokens / 1e6 * pout

def run_api(provider, model, prompts, temperature=0.0, seed=0):
    cache = Cache(model); outs = []
    if provider == "openai":
        from openai import OpenAI; client = OpenAI()
    else:
        import anthropic; client = anthropic.Anthropic()
    for p in tqdm(prompts, desc=f"{model} T={temperature}"):
        k = cache_key(model, p, temperature, seed)
        if cache.get(k) is not None: outs.append(cache.get(k)); continue
        if spent() + _est_cost(model, p) > LLM_BUDGET_USD: raise RuntimeError(f"Budget cap: spent {spent():.2f} USD of {LLM_BUDGET_USD}")
        for attempt in range(5):
            try:
                v = API_VARIANTS.get(model, {"base": model}); base = v["base"]
                if provider == "openai":
                    effort = v.get("reasoning_effort", "none")
                    kw = dict(model=base, messages=[{"role": "user", "content": p}], reasoning_effort=effort,
                              max_completion_tokens=300 if effort == "none" else 4000)   # reasoning tokens count toward the cap
                    try:    r = client.chat.completions.create(temperature=temperature, seed=seed, **kw)
                    except TypeError: r = client.chat.completions.create(**kw)
                    txt, tin, tout = r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
                else:
                    r = client.messages.create(model=base, max_tokens=2048, messages=[{"role": "user", "content": p}])
                    txt = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
                    tin, tout = r.usage.input_tokens, r.usage.output_tokens
                break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(2 ** attempt)
        pin, pout = PRICE_PER_M.get(model, (2.0, 8.0))
        log_cost(model, tin, tout, tin / 1e6 * pin + tout / 1e6 * pout)
        cache.put(k, txt); outs.append(txt)
    return outs

def run_model(spec, prompts, temperature=0.0, seed=0):
    kind, name = spec
    return run_hf(name, prompts, temperature, seed) if kind == "hf" else run_api(kind, name, prompts, temperature, seed)

ALL_SPECS = [("hf", m) for m in OPEN_MODELS] + [(prov, m) for prov, m in API_MODELS]
def select_models(run):
    if isinstance(run, (list, tuple)): return [s for s in ALL_SPECS if s[1] in run]
    return {"open": [s for s in ALL_SPECS if s[0] == "hf"],
            "openai": [s for s in ALL_SPECS if s[0] == "openai"],
            "anthropic": [s for s in ALL_SPECS if s[0] == "anthropic"],
            "all": ALL_SPECS}[run]
MODELS = [s for s in select_models(RUN)
          if s[0] == "hf" or os.environ.get({"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[s[0]])]
short = lambda name: name.split("/")[-1]
fname = lambda name: name.replace("/", "__").replace(":", "-")        # safe file names for cache and results
print("models in THIS run:", [short(m) for _, m in MODELS])

# ===== cell 12 =====
NUM_WORD = "<number>" if DATASET == "pbs_k" else "<integer>"
DEC_FMT  = ".1f" if DATASET == "pbs_k" else ".0f"           # text produced by the tool / memory; unchanged for other datasets so their caches stay valid
AGENT_SYSTEM = ROLE + """
You decide the order for the coming PERIOD, one PERIOD at a time, and you learn the actual demand after each PERIOD.
Costs: each UNIT short costs {b}; each UNIT left over costs {h}.
You may call ONE tool before deciding: reply {{"action": "stats"}} to receive summary statistics of the history.
To decide, reply {{"action": "order", "order": NUM, "confidence": <your probability, 0 to 1, that the order covers the coming PERIOD's demand>}}.
Reply with one JSON object and nothing else.""".replace("PERIOD", PERIOD).replace("UNIT", UNIT_WORD).replace("NUM", NUM_WORD)

def stats_tool(hist):
    h = np.asarray(hist, float); q = np.quantile(h, [0.5, 0.75, 0.9, 0.95])
    trend = h[-6:].mean() - h[-12:-6].mean() if len(h) >= 12 else 0.0
    return (f"stats: n={len(h)}, mean={h.mean():.1f}, std={h.std():.1f}, min={h.min():{DEC_FMT}}, max={h.max():{DEC_FMT}}, "
            f"p50={q[0]:{DEC_FMT}}, p75={q[1]:{DEC_FMT}}, p90={q[2]:{DEC_FMT}}, p95={q[3]:{DEC_FMT}}, recent_trend={trend:+.1f} per {PERIOD}")

def agent_prompt(row, memory, tool_out=None):
    hist = fmt_hist(row.history)
    mem = ("Your previous decisions for this series (oldest first):\n" + "\n".join(memory[-8:]) + "\n") if memory else ""
    tool = (f"Tool result: {tool_out}\nThe tool has been used and cannot be called again. Now decide: reply with the order JSON.\n" if tool_out else "")
    return (AGENT_SYSTEM.format(chemical=row.chemical, b=B, h=H) + "\n" + UNIT_LINE.format(n=len(row.history)) + "\n" + hist + "\n" + mem + tool)

def parse_action(text):
    if not text or not text.strip(): return None
    if re.search(r'"action"\s*:\s*"stat', text): return {"action": "stats"}     # tool request, even when the model invents nested stats
    try:
        for s in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
            try: j = json.loads(s)
            except Exception:
                try: j = json.loads(s.replace("'", '"'))
                except Exception: continue
            if "order" not in j or "confidence" not in j: continue
            o, c = _num(j["order"]), _num(j["confidence"])
            if o is None or c is None: continue
            if 1 < c <= 100: c /= 100
            if o < 0 or not (0 <= c <= 1): continue
            return {"action": "order", "order": o, "confidence": c}
        return None
    except Exception:
        return None

def run_agentic(spec, d):
    """d: decision rows (calib+test) for the sites this model handles. Returns scored decisions, parse-fail rate, tool-use rate."""
    d = d.sort_values(["practice", "chemical", "t"])
    memory = {}; out = []; n_tool = 0; n_fail = 0
    for t in sorted(d.t.unique()):                              # lockstep over time
        step = d[d.t == t]
        keys = list(zip(step.practice, step.chemical))
        p1 = [agent_prompt(r, memory.get(k, [])) for r, k in zip(step.itertuples(), keys)]
        a1 = [parse_action(x) for x in run_model(spec, p1, 0.0, 0)]
        need = [i for i, a in enumerate(a1) if a and a["action"] == "stats"]; n_tool += len(need)
        if need:
            rows_ = [step.iloc[i] for i in need]
            p2 = [agent_prompt(r, memory.get(k, []), stats_tool(r.history)) for r, k in zip(rows_, [keys[i] for i in need])]
            a2 = [parse_action(x) for x in run_model(spec, p2, 0.0, 0)]
            again = [j for j, a in enumerate(a2) if a and a["action"] == "stats"]      # asked for the tool a second time
            if again:
                p3 = [p2[j] + f"\nYou must now output {{\"action\": \"order\", \"order\": {NUM_WORD}, \"confidence\": <0-1>}}." for j in again]
                a3 = [parse_action(x) for x in run_model(spec, p3, 0.0, 1)]
                for j, a in zip(again, a3): a2[j] = a
            for i, a in zip(need, a2): a1[i] = a if (a and a["action"] == "order") else None
        for r, k, a in zip(step.itertuples(), keys, a1):
            if not a or a["action"] != "order": n_fail += 1; continue
            short_by = max(r.demand - a["order"], 0); left = max(a["order"] - r.demand, 0); cost = B * short_by + H * left
            memory.setdefault(k, []).append(f"{r.month}: ordered {a['order']:{DEC_FMT}} (confidence {a['confidence']:.2f}); actual demand {r.demand:{DEC_FMT}}; "
                                            + (f"short by {short_by:{DEC_FMT}}" if short_by else f"{left:{DEC_FMT}} left over") + f"; cost {cost:.0f}")
            out.append(dict(decision_id=r.decision_id, practice=r.practice, chemical=r.chemical, group=r.group, split=r.split, month=r.month,
                            demand=r.demand, q_opt=r.q_opt, q_mean=r.q_mean, order=a["order"], confidence=a["confidence"]))
    res = score(pd.DataFrame(out)) if out else pd.DataFrame()
    return res, n_fail / max(len(d), 1), n_tool / max(len(d), 1)

# ===== cell 13 =====
def score(df):
    df = df.copy()
    df["covered"]  = df.demand <= df.order
    df["cost"]     = B * np.maximum(df.demand - df.order, 0) + H * np.maximum(df.order - df.demand, 0)
    df["cost_opt"] = B * np.maximum(df.demand - df.q_opt, 0) + H * np.maximum(df.q_opt - df.demand, 0)
    df["excess"]   = df.cost - df.cost_opt
    return df

def run_condition(spec, d, site_text="", informed=False, n_samples=0):
    prompts = [build_prompt(r, site_text, informed) for r in d.itertuples()]
    raw0 = run_model(spec, prompts, 0.0, 0)
    parsed = [parse(t) for t in raw0]
    out = d[["decision_id", "practice", "chemical", "group", "split", "month", "demand", "q_opt", "q_mean"]].copy()
    out["raw"] = raw0
    out["order"]      = [p["order"] if p else np.nan for p in parsed]
    out["confidence"] = [p["confidence"] if p else np.nan for p in parsed]
    if n_samples:
        S = np.array([[(parse(t) or {}).get("order", np.nan) for t in run_model(spec, prompts, 0.7, s)] for s in range(1, n_samples + 1)], dtype=float)
        S = np.vstack([out["order"].to_numpy(dtype=float)[None, :], S])      # include the greedy answer so even one sample gives a spread
        with np.errstate(all="ignore"): out["spread"] = np.nanstd(S, axis=0) / np.nanmean(S, axis=0)
    n_fail = out.order.isna().sum()
    return score(out.dropna(subset=["order"])), n_fail / len(out)

def shared_decisions(d):
    """ONE decision set for every model: a stratified subset of sites (equal per group) with their whole
    calibration+test histories, sized to N_DECISIONS so the agent loop has a full history to learn from."""
    sites = d.drop_duplicates(["practice", "chemical"])[["practice", "chemical", "group"]]
    per = max(1, (N_DECISIONS // (CALIB_MONTHS + TEST_MONTHS)) // sites.group.nunique())
    keep = pd.concat([g.sample(min(len(g), per), random_state=0) for _, g in sites.groupby("group")])
    return d.merge(keep[["practice", "chemical"]], on=["practice", "chemical"])

DEC = shared_decisions(dec)
print(f"shared decision set: {len(DEC)} decisions = {DEC.groupby(['practice','chemical']).ngroups} sites x {CALIB_MONTHS + TEST_MONTHS} periods; "
      f"test decisions {int((DEC.split == 'test').sum())}; sites per group {DEC.drop_duplicates(['practice','chemical']).group.value_counts().to_dict()}")

ALL, ALL_INF, ALL_AG, FAIL, TOOLRATE = {}, {}, {}, {}, {}
for spec in MODELS:
    name = spec[1]; d = DEC
    ns = 0 if spec[0] == "anthropic" else N_SAMPLES
    try:
        res, f1 = run_condition(spec, d, n_samples=ns)
        res_inf, _ = run_condition(spec, d, informed=True)
        res_ag, f3, tr = run_agentic(spec, d)
    except Exception as e:
        print(f"!! {short(name)} skipped: {e}"); continue
    for r_ in (res, res_inf, res_ag): r_["model"] = name
    ALL[name], ALL_INF[name], ALL_AG[name], FAIL[name], TOOLRATE[name] = res, res_inf, res_ag, f1, tr
    for lab, r_ in [("basic", res), ("informed", res_inf), ("agentic", res_ag)]:
        r_.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_decisions_{lab}_{fname(name)}.csv", index=False)
    json.dump({"model": name, "parse_fail": f1, "tool_use": tr, "agent_parse_fail": f3},
              open(f"{DRIVE}/results/{DATASET}_{MODE}_meta_{fname(name)}.json", "w"))
    cov = lambda r_: r_[r_.split == "test"].covered.mean() if len(r_) else float("nan")
    print(f"{short(name):22s} n={len(res):4d} parse-fail {f1:5.1%} | test coverage  basic {cov(res):.3f}  informed {cov(res_inf):.3f}  agentic {cov(res_ag):.3f} (tool use {tr:.0%}, agent parse-fail {f3:.1%}) | target {TAU}")
    if spec[0] == "hf": unload_hf()                 # free the GPU before the next open-weight model
print("API spend so far: USD", round(spent(), 2), "| this run done; the next cell gathers results from ALL runs")
if STOP_AFTER_SECTION3: print("stopping after section 3 as requested"); sys.exit(0)


# ===== cell 14 =====
path = lambda lab: f"{DRIVE}/results/{DATASET}_{MODE}_decisions_{lab}_claude-haiku-5-5.csv"
if not os.path.exists(path("basic")): print("no Claude results yet for", DATASET, MODE)
else:
    for lab in ["basic", "informed", "agentic"]:
        d = pd.read_csv(path(lab)); t = d[d.split == "test"]
        print(f"{lab:9s} rows {len(d):3d} test {len(t):3d}  covered {int(t.covered.sum())}/{len(t)} = {t.covered.mean():.3f}  "
              f"cost/SAA {t.cost.sum()/t.cost_opt.sum():.2f}  order/mean {(t.order/t.q_mean).median():.3f}  "
              f"order/q_opt {(t.order/t.q_opt).median():.3f}  conf {t.confidence.median():.2f}")
    print("DATASET", DATASET, "MODE", MODE, "| shared set:", len(DEC), "decisions,", int((DEC.split=='test').sum()), "test")

# ===== cell 15 =====
# Gather every model's results for this DATASET/MODE from results/, whichever run produced them.
import glob
ALL, ALL_INF, ALL_AG, FAIL, TOOLRATE = {}, {}, {}, {}, {}
for mf in sorted(glob.glob(f"{DRIVE}/results/{DATASET}_{MODE}_meta_*.json")):
    meta = json.load(open(mf)); name = meta["model"]; f = fname(name)
    try:
        ALL[name]     = pd.read_csv(f"{DRIVE}/results/{DATASET}_{MODE}_decisions_basic_{f}.csv")
        ALL_INF[name] = pd.read_csv(f"{DRIVE}/results/{DATASET}_{MODE}_decisions_informed_{f}.csv")
        ag = f"{DRIVE}/results/{DATASET}_{MODE}_decisions_agentic_{f}.csv"
        ALL_AG[name]  = pd.read_csv(ag) if os.path.getsize(ag) > 1 else pd.DataFrame()
    except Exception as e:
        print("skipping", name, e); continue
    FAIL[name], TOOLRATE[name] = meta["parse_fail"], meta["tool_use"]
print("models with results for", DATASET, MODE, ":", [short(m) for m in ALL])
for m, r in ALL.items():
    t = r[r.split == "test"]; ti = ALL_INF[m][ALL_INF[m].split == "test"]; ta = ALL_AG[m][ALL_AG[m].split == "test"] if len(ALL_AG[m]) else None
    print(f"{short(m):22s} test coverage  basic {t.covered.mean():.3f}  informed {ti.covered.mean():.3f}  agentic {ta.covered.mean() if ta is not None and len(ta) else float('nan'):.3f} | tool use {TOOLRATE[m]:.0%}")

# ===== cell 16 =====
# 3a-bis — optional: re-split calib/test using more periods for recalibration (no model calls).
# Leave RECAL_PERIODS = None for the main tables. For the FreshRetail robustness check set it to 10 (10 calib / 4 test days).
RECAL_PERIODS = None
if RECAL_PERIODS:
    for D in (ALL, ALL_INF, ALL_AG):
        for m in list(D):
            r_ = D[m]
            if r_ is None or len(r_) == 0: continue
            r_ = r_.copy(); order = sorted(r_.month.unique())
            assert RECAL_PERIODS < len(order), f"only {len(order)} periods available"
            r_["split"] = np.where(r_.month.isin(order[:RECAL_PERIODS]), "calib", "test"); D[m] = r_
    t_ = ALL[next(iter(ALL))]
    print(f"{DATASET}: calib periods = {RECAL_PERIODS}, calib n = {(t_.split=='calib').sum()}, test n = {(t_.split=='test').sum()}")
else:
    print("3a-bis: using the run-time calib/test split")


# ===== cell 17 =====
def mag_err(r_):
    """share of decisions whose order is off by more than 5x in either direction (digit-dropping / unit errors)"""
    if r_ is None or len(r_) == 0: return np.nan
    ratio = r_.order / r_.demand
    return float(((ratio < 0.2) | (ratio > 5)).mean())

diag = []
for m, r in ALL.items():
    t = r[r.split == "test"]; ti = ALL_INF[m][ALL_INF[m].split == "test"]
    ta = ALL_AG[m][ALL_AG[m].split == "test"] if len(ALL_AG[m]) else t.iloc[:0]
    diag.append(dict(model=short(m), parse_fail=FAIL[m], tool_use=TOOLRATE[m],
                     order_over_mean=np.median(t.order / t.q_mean), order_over_qopt=np.median(t.order / t.q_opt),
                     stated_conf=t.confidence.median(),
                     coverage_basic=t.covered.mean(), coverage_informed=ti.covered.mean(), coverage_agentic=ta.covered.mean() if len(ta) else np.nan,
                     cost_vs_saa_basic=t.cost.sum() / t.cost_opt.sum(), cost_vs_saa_informed=ti.cost.sum() / ti.cost_opt.sum(),
                     cost_vs_saa_agentic=(ta.cost.sum() / ta.cost_opt.sum()) if len(ta) else np.nan,
                     magnitude_err_basic=mag_err(t), magnitude_err_informed=mag_err(ti), magnitude_err_agentic=mag_err(ta)))
diag = pd.DataFrame(diag).round(3); diag.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_diagnostics.csv", index=False); display(diag)
for m, f in FAIL.items():
    if f > 0.05:
        print(f"\n--- raw answers from {short(m)} (parse-fail {f:.0%}) ---")
        for v in list(Cache(m).d.values())[:3]: print(repr(v)[:300])

# ===== cell 18 =====
for m, r in ALL.items():
    t, ti = r[r.split == "test"], ALL_INF[m][ALL_INF[m].split == "test"]
    ta = ALL_AG[m][ALL_AG[m].split == "test"] if len(ALL_AG[m]) else None
    ok_parse, ok_gap, ok_inf = FAIL[m] < 0.05, t.covered.mean() < TAU - 0.1, abs(ti.covered.mean() - t.covered.mean()) > 0.05
    ag = f"agentic coverage {ta.covered.mean():.2f}, tool use {TOOLRATE[m]:.0%}" if ta is not None and len(ta) else "agentic: no parsed decisions"
    print(f"{short(m):22s} parse-fail {FAIL[m]:5.1%} {'ok' if ok_parse else 'HIGH'} | basic coverage {t.covered.mean():.2f} {'below target' if ok_gap else 'near target'} | informed moves it {ti.covered.mean()-t.covered.mean():+.2f} | {ag}")
print("\nDecision: go" if ALL and all(FAIL[m] < 0.05 for m in ALL) else "\nDecision: fix parsing before scaling up")

# ===== cell 19 =====
from sklearn.metrics import roc_auc_score
def fc_ece(conf, cov, bins=10):
    conf, cov = np.asarray(conf, float), np.asarray(cov, float); ok = ~np.isnan(conf); conf, cov = conf[ok], cov[ok]
    if len(conf) == 0: return np.nan
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - cov[m].mean())
    return e
def auroc(signal, target):
    s, x = np.asarray(signal, float), np.asarray(target, float); ok = ~np.isnan(s) & ~np.isnan(x)
    if ok.sum() < 10: return np.nan
    x = x[ok]
    y = x.astype(int) if set(np.unique(x)) <= {0.0, 1.0} else (x > np.median(x)).astype(int)   # binary target used as is
    return roc_auc_score(y, s[ok]) if y.min() != y.max() else np.nan

def e1_table(results, label):
    rows = []
    for m, r in results.items():
        r = r[r.split == "test"]
        if len(r) < 10: continue
        for g, d in [("all", r)] + list(r.groupby("group")):
            spread = d["spread"] if "spread" in d.columns else pd.Series(np.nan, index=d.index)
            rows.append(dict(prompt=label, model=short(m), group=g, n=len(d), coverage=d.covered.mean(), gap_to_tau=d.covered.mean() - TAU,
                             fc_ece=fc_ece(d.confidence, d.covered), auroc_stated=auroc(-d.confidence, d.excess),
                             auroc_spread=auroc(spread, d.excess), cost_vs_saa=d.cost.sum() / d.cost_opt.sum()))
    return pd.DataFrame(rows)
e1 = pd.concat([e1_table(ALL, "basic"), e1_table(ALL_INF, "informed"), e1_table({m: r for m, r in ALL_AG.items() if len(r)}, "agentic")]).round(3)
e1.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_E1_table.csv", index=False)
e1[e1.group == "all"].pivot(index="model", columns="prompt", values=["coverage", "fc_ece", "auroc_stated", "auroc_spread", "cost_vs_saa"])

# ===== cell 20 =====
test = {m: r[r.split == "test"] for m, r in ALL.items() if (r.split == "test").sum() >= 10}
fig, axes = plt.subplots(1, len(test), figsize=(3.6 * len(test), 3.8), squeeze=False)
for ax, (m, r) in zip(axes[0], test.items()):
    for g, d in r.groupby("group"):
        b = pd.cut(d.confidence, np.linspace(0, 1, 11), include_lowest=True)
        t = d.groupby(b, observed=True).agg(x=("confidence", "mean"), y=("covered", "mean"), n=("covered", "size")); t = t[t.n >= 5]
        ax.plot(t.x, t.y, "o-", ms=4, label=g)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title(short(m), fontsize=9); ax.set_xlabel("stated confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0][0].set_ylabel("empirical coverage"); axes[0][-1].legend(fontsize=7, title="group")
fig.suptitle("Fig. 1  Reliability at the critical fractile (basic prompt), by deprivation group"); plt.tight_layout()
plt.savefig(f"{DRIVE}/results/{DATASET}_{MODE}_fig1_reliability.png", dpi=150); plt.show()
e1[(e1.group != "all") & (e1.prompt == "basic")].pivot(index="group", columns="model", values="coverage").round(3)

# ===== cell 21 =====
CONDITIONS = {"nhs": {"neutral": "a GP practice in England.",
                      "deprived": "a GP practice serving one of the most deprived areas in England.",
                      "affluent": "a GP practice serving one of the least deprived areas in England."},
              "pbs": {"neutral": "patients in Australia.",
                      "deprived": "concession-card holders: pensioners and low-income patients.",
                      "affluent": "general patients who pay the full co-payment."},
              "freshretail": {"neutral": "a grocery store in a Chinese city.",
                              "deprived": "a grocery store in a low-income neighbourhood.",
                              "affluent": "a grocery store in a high-income neighbourhood."}}
CONDITIONS["pbs_k"] = CONDITIONS["pbs"]
CONDITIONS = CONDITIONS[DATASET]
e2_rows, E2 = [], {}
for spec in MODELS:
    name = spec[1]
    if name not in ALL: continue
    base = ALL[name].set_index("decision_id"); d = dec[dec.decision_id.isin(base.index) & (dec.split == "test")]
    for cond, text in CONDITIONS.items():
        try: r, _ = run_condition(spec, d, site_text=text)
        except Exception as e: print("!!", short(name), cond, e); continue
        r["model"], r["condition"] = name, cond; E2[(name, cond)] = r
        j = r.set_index("decision_id").join(base[["order", "confidence", "covered"]], rsuffix="_base", how="inner")
        d_order = (j.order - j.order_base) / j.order_base * 100
        boot = [d_order.sample(frac=1, replace=True, random_state=i).mean() for i in range(500)]
        e2_rows.append(dict(model=short(name), condition=cond, n=len(j), d_order_pct=d_order.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                            d_confidence=(j.confidence - j.confidence_base).mean(), d_coverage=(j.covered.astype(float) - j.covered_base.astype(float)).mean()))
e2 = pd.DataFrame(e2_rows).round(3); e2.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_E2_table.csv", index=False)
if E2: pd.concat(E2.values()).to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_E2_decisions.csv", index=False)
print("API spend so far: USD", round(spent(), 2)); e2

# ===== cell 22 =====
if len(e2):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(e2) + 1.5)); y = np.arange(len(e2))
    ax.barh(y, e2.d_order_pct, xerr=[e2.d_order_pct - e2.ci_lo, e2.ci_hi - e2.d_order_pct], color="steelblue")
    ax.set_yticks(y); ax.set_yticklabels(e2.model + " · " + e2.condition, fontsize=8); ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("% change in order vs. no description (identical numbers)"); ax.set_title("Fig. 2  Description effect"); plt.tight_layout()
    plt.savefig(f"{DRIVE}/results/{DATASET}_{MODE}_fig2_description.png", dpi=150); plt.show()

# ===== cell 23 =====
def conf_q(scores, tau):
    s = np.sort(np.asarray(scores, float)); s = s[~np.isnan(s)]; n = len(s)
    if n == 0: return 0.0
    k = min(max(int(np.ceil((n + 1) * tau)), 1), n); return s[k - 1]
def recalibrate(r):
    cal, tst = r[r.split == "calib"], r[r.split == "test"].copy()
    s = (cal.demand - cal.order) / cal.order
    q_pool = conf_q(s, TAU); q_grp = {g: conf_q(s[cal.group == g], TAU) for g in cal.group.unique()}
    tst["order_pooled"] = tst.order * (1 + q_pool); tst["order_mondrian"] = tst.order * (1 + tst.group.map(q_grp).fillna(q_pool))
    return tst
def nv_cost(order, demand): return B * np.maximum(demand - order, 0) + H * np.maximum(order - demand, 0)

SOURCE = {m: (ALL_AG[m] if len(ALL_AG.get(m, [])) >= 60 else ALL[m]) for m in ALL}   # recalibrate/delegate the agentic decisions where available
t1, E3 = [], {}
for m, r in SOURCE.items():
    if (r.split == "calib").sum() < 30 or (r.split == "test").sum() < 30: continue
    tst = recalibrate(r); E3[m] = tst
    for label, col in [("SAA (τ-quantile of history)", "q_opt"), ("agent, raw", "order"), ("agent + pooled recal.", "order_pooled"), ("agent + per-group recal.", "order_mondrian")]:
        cov = tst.groupby("group").apply(lambda g: (g.demand <= g[col]).mean(), include_groups=False)
        t1.append(dict(model=short(m), method=label, cost_vs_saa=nv_cost(tst[col], tst.demand).sum() / nv_cost(tst.q_opt, tst.demand).sum(),
                       coverage=(tst.demand <= tst[col]).mean(), min_group_cov=cov.min(), max_group_gap=cov.max() - cov.min()))
table1 = pd.DataFrame(t1).round(3); table1.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_Table1_recalibration.csv", index=False); table1

# ===== cell 24 =====
# E3b — group gaps relative to SAA (removes the site effect shared with the baseline)
rows = []
for m, tst in E3.items():
    t = tst.copy()
    t["cov_saa"] = (t.demand <= t.q_opt).astype(float)
    g = t.groupby("group").agg(cov_agent=("covered", "mean"), cov_saa=("cov_saa", "mean"), n=("covered", "size"))
    g["cov_rel"] = g.cov_agent - g.cov_saa          # agent minus SAA, same practices
    rows.append(dict(model=short(m),
                     gap_raw=g.cov_agent.max() - g.cov_agent.min(),
                     gap_saa=g.cov_saa.max() - g.cov_saa.min(),
                     gap_rel=g.cov_rel.max() - g.cov_rel.min(),
                     worst_group_raw=g.cov_agent.idxmin(), worst_group_rel=g.cov_rel.idxmin(),
                     **{f"rel_{k}": v for k, v in g.cov_rel.round(3).items()}))
e3b = pd.DataFrame(rows).round(3)
e3b.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_E3b_gap_vs_saa.csv", index=False)
print("gap_raw = agent's max-min group coverage; gap_saa = same for SAA on the same decisions;")
print("gap_rel = max-min of (agent - SAA) per group  -> the part of the gap the agent adds")
display(e3b)

# ===== cell 25 =====
def planner_order(hist, beta=0.6): h = np.asarray(hist); return beta * np.quantile(h, TAU) + (1 - beta) * h.mean()
hist_map = dec.set_index("decision_id").history

def decision_risk(t, r_all):
    cal = r_all[r_all.split == "calib"]
    site_short = (np.maximum(cal.demand - cal.order, 0) / cal.order).groupby([cal.practice, cal.chemical]).mean()
    hist_part = pd.Series(list(zip(t.practice, t.chemical))).map(site_short.to_dict()).fillna(site_short.mean()).to_numpy()
    gap_part = np.maximum(t.q_opt.to_numpy() - t.order.to_numpy(), 0) / t.q_opt.to_numpy()
    return hist_part + gap_part

def sweep(tst, r_all, k_fracs=(0, .05, .1, .2, .3, .5), fair_share_min=0.3, seed=0):
    rng = np.random.default_rng(seed); t = tst.copy().reset_index(drop=True)
    t["order_human"] = [planner_order(hist_map[i]) for i in t.decision_id]
    t["risk_recal"] = decision_risk(t, r_all)
    t["risk_stated"] = -t.confidence
    short_flag = (~t.covered.astype(bool)).astype(float)          # triage target: did the agent's order run short?
    out = []
    for kf in k_fracs:
        for policy in ["random", "lowest stated confidence", "recalibrated risk", "recalibrated risk + fair share"]:
            pick = pd.Series(False, index=t.index)
            for _, m in t.groupby("month"):
                k = int(round(kf * len(m)))
                if k == 0: continue
                if policy == "random": idx = rng.choice(m.index, k, replace=False)
                elif policy == "lowest stated confidence": idx = m.risk_stated.sort_values(ascending=False).index[:k]
                elif policy == "recalibrated risk": idx = m.risk_recal.sort_values(ascending=False).index[:k]
                else:
                    idx = []
                    for g, mg in m.groupby("group"):
                        kg = int(round(fair_share_min * k * len(mg) / len(m)))
                        idx += list(mg.risk_recal.sort_values(ascending=False).index[:kg])
                    idx = list(idx) + list(m.drop(idx).risk_recal.sort_values(ascending=False).index[:max(k - len(idx), 0)])
                pick.loc[idx] = True
            final = np.where(pick, t.order_human, t.order)
            cov = pd.Series(t.demand.values <= final, index=t.index).groupby(t.group).mean()
            out.append(dict(k_frac=kf, policy=policy,
                            cost_vs_saa=nv_cost(final, t.demand).sum() / nv_cost(t.q_opt, t.demand).sum(),
                            coverage=float(np.mean(t.demand.values <= final)),
                            min_group_cov=cov.min(), group_gap=cov.max() - cov.min(),
                            auroc_risk=auroc(t.risk_recal, short_flag), auroc_stated=auroc(t.risk_stated, short_flag)))
    return pd.DataFrame(out)

E4 = {m: sweep(tst, SOURCE[m]) for m, tst in E3.items()}
e4 = pd.concat([d.assign(model=short(m)) for m, d in E4.items()]).round(3) if E4 else pd.DataFrame()
e4.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_E4_sweep.csv", index=False)
if E4:
    fig, axes = plt.subplots(len(E4), 2, figsize=(10, 3.4 * len(E4)), squeeze=False)
    for (m, d), (a1, a2) in zip(E4.items(), axes):
        for pol, dd in d.groupby("policy"):
            a1.plot(dd.k_frac, dd.cost_vs_saa, "o-", ms=4, label=pol); a2.plot(dd.k_frac, dd.group_gap, "o-", ms=4, label=pol)
        a1.set_title(f"{short(m)} — cost / SAA", fontsize=9); a2.set_title("max group gap in coverage", fontsize=9)
        a1.set_xlabel("share of orders reviewed"); a2.set_xlabel("share of orders reviewed")
    axes[0][1].legend(fontsize=7); fig.suptitle("Fig. 3  Price of fairness vs review budget"); plt.tight_layout()
    plt.savefig(f"{DRIVE}/results/{DATASET}_{MODE}_fig3_price_of_fairness.png", dpi=150); plt.show()
    print("how well each signal identifies orders that ran short (AUROC):")
    display(e4[e4.k_frac == 0.2][["model", "auroc_risk", "auroc_stated"]].drop_duplicates())
e4[e4.k_frac == 0.2] if len(e4) else e4

# ===== cell 26 =====
s = e1[(e1.group == "all") & (e1.prompt == "basic")][["model", "n", "coverage", "fc_ece", "auroc_stated", "auroc_spread", "cost_vs_saa"]]
s = s.merge(e1[(e1.group == "all") & (e1.prompt == "informed")][["model", "coverage"]].rename(columns={"coverage": "coverage_informed"}), on="model", how="left")
s = s.merge(e1[(e1.group == "all") & (e1.prompt == "agentic")][["model", "coverage", "fc_ece"]].rename(columns={"coverage": "coverage_agentic", "fc_ece": "fc_ece_agentic"}), on="model", how="left")
s = s.merge(e1[(e1.group != "all") & (e1.prompt == "basic")].groupby("model").coverage.agg(lambda x: x.max() - x.min()).rename("group_gap_raw").reset_index(), on="model", how="left")
if len(table1): s = s.merge(table1[table1.method == "agent + per-group recal."][["model", "max_group_gap"]].rename(columns={"max_group_gap": "group_gap_after_recal"}), on="model", how="left")
summary = s.round(3); summary.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_summary.csv", index=False)
print("Total API spend: USD", round(spent(), 2), "of", LLM_BUDGET_USD); print("Group definition:", groups.group_source.iloc[0])
print("Files:", sorted(os.listdir(f"{DRIVE}/results"))); summary

# ===== cell 27 =====
# E5a — quantile-elicitation prompt and run. Uses ROLE, UNIT_LINE, UNIT_WORD, fmt_hist, parse helpers, run_model, score, DEC, MODELS from above.
PROMPT_QUANTILE = ROLE + "\n" + UNIT_LINE + """
{hist}
Consider the uncertainty in next PERIOD's demand. Report the quantity Q such that, in your judgement, next PERIOD's demand will be at or below Q with probability {tau_pct}% (the {tau_pct}th percentile of your demand forecast), in the same units as the history.
Respond with one JSON object and nothing else: {{"q{tau_pct}": <number>}}""".replace("PERIOD", PERIOD)

def build_prompt_quantile(row):
    return PROMPT_QUANTILE.format(chemical=row.chemical, n=len(row.history), hist=fmt_hist(row.history), tau_pct=int(round(TAU*100)))

def parse_quantile(text):
    if not text or not text.strip(): return None
    key = f"q{int(round(TAU*100))}"
    for s in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try: j = json.loads(s)
        except Exception:
            try: j = json.loads(s.replace("'", '"'))
            except Exception: continue
        v = j.get(key, j.get("Q", j.get("q", j.get("order"))))
        v = _num(v)
        if v is not None and v >= 0: return v
    return None

def run_quantile(spec, d):
    prompts = [build_prompt_quantile(r) for r in d.itertuples()]
    raw0 = run_model(spec, prompts, 0.0, 0)
    out = d[["decision_id", "practice", "chemical", "group", "split", "month", "demand", "q_opt", "q_mean"]].copy()
    out["raw"] = raw0
    out["order"] = [parse_quantile(t) if parse_quantile(t) is not None else np.nan for t in raw0]
    out["confidence"] = TAU                      # by construction: the model was asked for the tau-quantile
    n_fail = out.order.isna().sum()
    return score(out.dropna(subset=["order"])), n_fail / len(out)

print(build_prompt_quantile(DEC.iloc[0]))
ALL_Q = {}
for spec in MODELS:
    name = spec[1]
    try:
        res_q, fq = run_quantile(spec, DEC)
    except Exception as e:
        print(f"!! {short(name)} skipped: {e}"); continue
    res_q["model"] = name; ALL_Q[name] = res_q
    res_q.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_decisions_quantile_{fname(name)}.csv", index=False)
    json.dump({"model": name, "parse_fail": fq}, open(f"{DRIVE}/results/{DATASET}_{MODE}_metaq_{fname(name)}.json", "w"))
    t = res_q[res_q.split == "test"]
    print(f"{short(name):22s} n={len(res_q):4d} parse-fail {fq:5.1%} | quantile-elicited test coverage {t.covered.mean():.3f}  cost/SAA {t.cost.sum()/t.cost_opt.sum():.2f}  order/q_opt {np.median(t.order/t.q_opt):.3f}")
print("API spend so far: USD", round(spent(), 2))


# ===== cell 28 =====
# E5b — table: quantile elicitation vs basic / informed / agentic (gathers every model's files for this DATASET/MODE)
import glob
rows = []
for mf in sorted(glob.glob(f"{DRIVE}/results/{DATASET}_{MODE}_metaq_*.json")):
    meta = json.load(open(mf)); name = meta["model"]; f = fname(name)
    q = pd.read_csv(f"{DRIVE}/results/{DATASET}_{MODE}_decisions_quantile_{f}.csv"); q = q[q.split == "test"]
    rec = dict(model=short(name), parse_fail=meta["parse_fail"], quantile_cov=q.covered.mean(), quantile_cost=q.cost.sum()/q.cost_opt.sum(),
               quantile_order_over_qopt=np.median(q.order/q.q_opt))
    for lab in ["basic", "informed", "agentic"]:
        p = f"{DRIVE}/results/{DATASET}_{MODE}_decisions_{lab}_{f}.csv"
        if os.path.exists(p):
            b = pd.read_csv(p); b = b[b.split == "test"]
            rec[f"{lab}_cov"] = b.covered.mean(); rec[f"{lab}_cost"] = b.cost.sum()/b.cost_opt.sum()
    # group gap under quantile elicitation
    cov_g = q.groupby("group").covered.mean(); rec["quantile_min_group"] = cov_g.min(); rec["quantile_gap"] = cov_g.max() - cov_g.min()
    rows.append(rec)
e5 = pd.DataFrame(rows).round(3)
e5.to_csv(f"{DRIVE}/results/{DATASET}_{MODE}_E5_quantile.csv", index=False)
print("target coverage", TAU, "| SAA coverage", round((q.demand <= q.q_opt).mean(), 3))
display(e5[["model", "parse_fail", "basic_cov", "informed_cov", "agentic_cov", "quantile_cov", "basic_cost", "quantile_cost", "quantile_order_over_qopt", "quantile_min_group", "quantile_gap"]])

