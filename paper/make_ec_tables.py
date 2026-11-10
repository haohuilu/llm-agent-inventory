src=open('analysis.py').read()
exec(src.split("# ---------------------------------------------------------------- A2")[0])
import numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
def r2(x): return str(Decimal(repr(float(x))).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
NAME={"Qwen":"Qwen2.5-7B","Llama":"Llama-3.1-8B","Gemma":"Gemma-4-E4B","Phi":"Phi-4-mini","GPT":"GPT-6 Luna","Claude":"Claude Haiku 5.5"}
PAN=[("nhs","NHS"),("pbs_k","PBS, thousands"),("pbs","PBS, raw units"),("freshretail","FreshRetail")]
import os
out=[]
out.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Test coverage and cost relative to SAA under the basic, informed and agentic treatments, by panel (complements Tables 4, 8 and 9 of the main text; the quantile treatment is in Table 8). On PBS in raw units the cost of Llama and Gemma reflects the magnitude errors of Table~\ref{tab:vdc_treat}.}\label{tab:bytreat}
\setlength{\tabcolsep}{5pt}\begin{tabular}{@{}ll|ccc|ccc@{}}\toprule
 & & \multicolumn{3}{c|}{Coverage} & \multicolumn{3}{c}{Cost/SAA}\\
Panel & Model & basic & informed & agentic & basic & informed & agentic\\\midrule""")
for ds,lab in PAN:
    first=True
    for m,s in MODELS:
        cov=[];cost=[]
        for c in ['basic','informed','agentic']:
            p=f"{RES}/{ds}_full_decisions_{c}_{m}.csv"
            d=pd.read_csv(p); d=d[d.order.notna()]; d=d[d.split=='test']
            cov.append(r2(np.mean(d.demand<=d.order))); cost.append(r2(nv_cost(d.order.to_numpy(),d.demand.to_numpy()).sum()/nv_cost(d.q_opt.to_numpy(),d.demand.to_numpy()).sum()))
        out.append(f"{lab if first else ''} & {NAME[s]} & "+" & ".join(cov+cost)+r"\\"); first=False
    if ds!='freshretail': out.append(r"\midrule")
out[-1]=out[-1]+r"\bottomrule"
out.append(r"\end{tabular}"+"\n"+r"\end{table}"+"\n")
# vdc ci
v=pd.read_csv(RES+'/analysis19_vdc_auroc_ci_by_treatment.csv',dtype=str).fillna('')
out.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Shortfall AUROC of VDC on test decisions under the informed and agentic treatments, with 95\% site-bootstrap intervals (``--'': single value). Basic-treatment intervals are in Table 7 of the main text. No interval lies above 0.5.}\label{tab:vdcci}
\setlength{\tabcolsep}{5pt}\begin{tabular}{@{}l|cc|cc|cc@{}}\toprule
 & \multicolumn{2}{c|}{NHS} & \multicolumn{2}{c|}{PBS, thousands} & \multicolumn{2}{c}{FreshRetail}\\
Model & informed & agentic & informed & agentic & informed & agentic\\\midrule""")
for m,s in MODELS:
    cells=[]
    for ds in DATASETS:
        for c in ['informed','agentic']:
            r=v[(v.dataset==ds)&(v.treatment==c)&(v.model==s)].iloc[0]
            cells.append('--' if r.auroc=='const' else f"{r.auroc} [{r.lo}, {r.hi}]")
    out.append(f"{NAME[s]} & "+" & ".join(cells)+r"\\")
out[-1]+=r"\bottomrule"
out.append(r"\end{tabular}"+"\n"+r"\end{table}"+"\n")
open('/home/claude/paper_pom/ec_gen_a.tex','w').write("\n".join(out))
# SAA planner
a=pd.read_csv(RES+'/analysis10_saa_planner.csv')
o=[r"""\begin{table}[h]\centering\footnotesize
\caption{Review experiment with a planner who orders the sample $\tau$-quantile, 20\% review budget, agentic orders: cost relative to SAA without review and with the reviewed orders selected at random, by the heuristic index and by TDC.}\label{tab:saaplanner}
\begin{adjustbox}{max width=\textwidth}\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l|cccc|cccc|cccc@{}}\toprule
 & \multicolumn{4}{c|}{NHS} & \multicolumn{4}{c|}{PBS, thousands} & \multicolumn{4}{c}{FreshRetail}\\
Model & none & random & heur. & TDC & none & random & heur. & TDC & none & random & heur. & TDC\\\midrule"""]
for m,s in MODELS:
    cells=[]
    for ds in DATASETS:
        r=a[(a.dataset==ds)&(a.model==s)].iloc[0]; cells+= [f"{r.cost_none:.2f}",f"{r.cost_random:.2f}",f"{r.cost_hand:.2f}",f"{r.cost_trace:.2f}"]
    o.append(f"{NAME[s]} & "+" & ".join(cells)+r"\\")
o[-1]+=r"\bottomrule"
o.append(r"\end{tabular}\end{adjustbox}"+"\n"+r"\end{table}"+"\n")
open('/home/claude/paper_pom/ec_gen_b.tex','w').write("\n".join(o))
print("\n".join(out)); print("\n".join(o))
