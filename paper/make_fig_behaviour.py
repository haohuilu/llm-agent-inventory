"""Figure 3 of the POM version: (a) reaction coefficients with site-clustered 95% intervals; (b) composition of order
adjustments in the agent loop; (c) shortfall AUROC of VDC and TDC (other-model fit) with site-bootstrap intervals, NHS.
Reads utd24_runs/results; run analysis.py and analysis_pom.py first."""
import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","Times"],"mathtext.fontset":"stix",
    "font.size":8.5,"axes.titlesize":9,"axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6})
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); RES=f"{ROOT}/utd24_runs/results"
exec(open(f"{ROOT}/analysis.py").read().split("# ---------------------------------------------------------------- A2")[0].replace('DRIVE = os.path.abspath(os.environ.get("UTD24_DRIVE", "./utd24_runs"))',f'DRIVE = "{ROOT}/utd24_runs"'))
NAMES=["Qwen2.5-7B","Llama-3.1-8B","Gemma-4-E4B","Phi-4-mini","GPT-6 Luna","Claude Haiku 5.5"]; KEY=["Qwen","Llama","Gemma","Phi","GPT","Claude"]
C=["#2a78d6","#eb6834","#1baf7a"]; MK=["o","s","^"]; PAN=[("nhs","NHS"),("pbs_k","PBS"),("freshretail","FreshRetail")]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.75),gridspec_kw={"width_ratios":[1.05,1.15,1.0]})
# (a)
r=pd.read_csv(f"{RES}/analysis12_reaction_cluster.csv"); y=np.arange(6)[::-1]
for j,(ds,lab) in enumerate(PAN):
    d=r[r.dataset==ds].set_index("model").loc[KEY]
    ax[0].errorbar(d.react,y+(1-j)*0.22,xerr=1.96*d.se_cluster,fmt=MK[j],ms=4,lw=0.9,color=C[j],label=lab,capsize=0)
ax[0].axvline(0,color="0.6",lw=0.6); ax[0].axvline(1,color="0.3",lw=0.7,ls=":"); ax[0].text(0.97,2.5,"full correction",ha="right",va="center",fontsize=7,color="0.3",rotation=90)
ax[0].set_yticks(y); ax[0].set_yticklabels(NAMES); ax[0].set_xlim(-0.05,1.05); ax[0].set_xlabel("reaction coefficient"); ax[0].set_title("(a) Reaction to the previous error")
ax[0].legend(loc="upper center",bbox_to_anchor=(0.5,-0.27),ncol=3,frameon=False,handletextpad=0.1,columnspacing=0.7)
# (b) composition of steps, pooled over panels
comp=[]
for m,k in MODELS:
    S=[]
    for ds,_ in PAN:
        a=load(ds,"agentic",m).sort_values(["practice","chemical","month"]); g=a.groupby(["practice","chemical"])
        a["pe"]=(g.demand.shift(1)-g.order.shift(1))/g.order.shift(1); a["ch"]=(a.order-g.order.shift(1))/g.order.shift(1)
        s=a.dropna(subset=["pe","ch"]); S.append(s[(s.pe.abs()<3)&(s.ch.abs()<3)])
    s=pd.concat(S); same=(np.sign(s.ch)==np.sign(s.pe))&(s.ch!=0); comp.append([(s.ch==0).mean(),same.mean(),((s.ch!=0)&~same).mean()])
comp=np.array(comp); left=np.zeros(6)
for j,(lab,col,h) in enumerate([("unchanged","0.80",""),("toward error","#2a78d6",""),("away from error","#eb6834","////")]):
    ax[1].barh(y,comp[:,j],left=left,height=0.62,color=col,edgecolor="white",linewidth=0.8,hatch=h,label=lab); left+=comp[:,j]
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlim(0,1); ax[1].set_xlabel("share of steps in the agent loop"); ax[1].set_title("(b) Order adjustment after feedback")
ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,-0.27),ncol=3,frameon=False,handlelength=1.1,columnspacing=0.7,handletextpad=0.3)
# (c)
a=pd.read_csv(f"{RES}/analysis11_auroc_ci.csv"); d=a[a.dataset=="nhs"].set_index("model").loc[KEY]
def parse(s):
    if s=="--" or pd.isna(s): return None
    v=[float(x) for x in re.findall(r"[0-9.]+",s)]; return v
for i,k in enumerate(KEY):
    v=parse(d.loc[k,"vdc"]); t=parse(d.loc[k,"tdc_other"])
    if v: ax[2].errorbar(v[0],y[i]+0.14,xerr=[[v[0]-v[1]],[v[2]-v[0]]],fmt="o",mfc="white",ms=4,lw=0.9,color="0.35",label="VDC" if i==0 else None)
    else: ax[2].text(0.5,y[i]+0.14,"single value",fontsize=6.5,color="0.4",ha="center",va="center")
    ax[2].errorbar(t[0],y[i]-0.14,xerr=[[t[0]-t[1]],[t[2]-t[0]]],fmt="s",ms=4,lw=0.9,color="#2a78d6",label="TDC" if i==0 else None)
ax[2].axvline(0.5,color="0.3",lw=0.7,ls=":"); ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_xlim(0.25,1.0); ax[2].set_xlabel("shortfall AUROC, NHS"); ax[2].set_title("(c) Statement versus behaviour")
ax[2].legend(loc="upper center",bbox_to_anchor=(0.5,-0.27),ncol=2,frameon=False,columnspacing=0.9,handletextpad=0.2)
for x in ax: x.grid(axis="x",lw=0.3,alpha=0.5); x.tick_params(length=2.5,width=0.5)
fig.tight_layout(w_pad=1.2)
for ext in ("pdf","png"): fig.savefig(f"{HERE}/fig4_agent_behaviour.{ext}",dpi=300,bbox_inches="tight")
print(np.round(comp,2))
