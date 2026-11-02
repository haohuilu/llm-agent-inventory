"""Elicitation ladder figure (paper Fig. 3): coverage under basic / informed / agentic / quantile."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","Times"],
                     "mathtext.fontset":"stix","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                     "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5})
DS=[("nhs","NHS (monthly)"),("pbs_k","PBS (monthly, thousands)"),("freshretail","FreshRetail (daily)")]
MODELS=[("qwen2.5-7b-instruct","Qwen2.5-7B"),("llama3.1-8b","Llama-3.1-8B"),("gemma4","Gemma-4-E4B"),
        ("phi4-mini","Phi-4-mini"),("gpt-6-luna","GPT-6 Luna"),("claude-haiku-5-5","Claude Haiku 5.5")]
COLS=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860"]
MARK=["o","s","^","D","v","P"]
COND=[("basic_cov","Basic"),("informed_cov","Informed"),("agentic_cov","Agentic"),("quantile_cov","Quantile")]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
for ax,(ds,title) in zip(axes,DS):
    t=pd.read_csv(f"{ds}_full_E5_quantile.csv"); t["model"]=t["model"].str.replace(":","-"); t=t.set_index("model")
    d=pd.read_csv(f"{ds}_full_decisions_basic_gemma4.csv"); d=d[d.split=="test"]
    saa=float((d.demand<=d.q_opt).mean())
    x=np.arange(4)
    for (m,lab),c,mk in zip(MODELS,COLS,MARK):
        if m not in t.index: continue
        y=[t.loc[m,k] for k,_ in COND]
        ax.plot(x,y,marker=mk,ms=4.5,lw=1.3,color=c,label=lab,zorder=3)
    ax.axhline(0.9,ls=":",lw=1,color="black",zorder=1)
    ax.axhline(saa,ls="--",lw=1,color="0.45",zorder=1)
    # reference labels outside the axes, to the right, stacked if close
    ytau,ysaa=0.9,saa
    if abs(ytau-ysaa)<0.035:
        hi,lo=max(ytau,ysaa),min(ytau,ysaa); ytau_l=hi+0.012 if ytau>=ysaa else lo-0.012; ysaa_l=lo-0.012 if ytau>=ysaa else hi+0.012
    else: ytau_l,ysaa_l=ytau,ysaa
    ax.text(3.25,ytau_l,r"$\tau$",va="center",ha="left",fontsize=8,clip_on=False)
    ax.text(3.25,ysaa_l,f"SAA\n{saa:.2f}",va="center",ha="left",fontsize=7,color="0.35",clip_on=False,linespacing=0.9)
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in COND]); ax.set_xlim(-0.3,3.2)
    ax.set_title(title); ax.grid(axis="y",lw=0.4,alpha=0.5); ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Coverage"); axes[0].set_ylim(0.5,1.02)
h,l=axes[0].get_legend_handles_labels()
h+= [Line2D([0],[0],ls=":",color="black",lw=1),Line2D([0],[0],ls="--",color="0.45",lw=1)]
l+= [r"target $\tau=0.9$","SAA coverage"]
fig.legend(h,l,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.12),handlelength=2.2,columnspacing=1.1)
fig.tight_layout(rect=(0,0.04,1,1),w_pad=1.6)
for ext in ("pdf","png"): fig.savefig(f"../../paper/fig5_elicitation_ladder.{ext}",dpi=300,bbox_inches="tight")
print("ok")
