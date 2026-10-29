import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats

alpha=0.05
RES=[]
LETTER_NAMES={"A":"flank/irregular","B":"semi-open","C":"open games/French",
"D":"closed/semi-closed","E":"Indian defences"}

def format_p(p):
    if p==0 or p<1e-300:
        return "<1e-300"
    return f"{p:.3e}" if p<1e-3 else f"{p:.4f}"

def verdict_diff(l,h,delta):
    if l>=delta or h<=-delta:
        return "matters"
    if l>-delta and h<delta:
        return "negligible"
    return "inconclusive"

def onehot_to_label(df,prefix,dropped):
    cols=[c for c in df.columns if c.startswith(prefix)]
    if not cols:
        sys.exit(f"No {prefix}*columns found in the input.")
    labels=np.array([c[len(prefix):] for c in cols])
    m=df[cols].to_numpy()
    out=pd.Series(labels[m.argmax(axis=1)],index=df.index)
    out[m.sum(axis=1)==0]=dropped
    return out

def load(path,a):
    df=pd.read_csv(path)
    need=["white_score","white_elo","black_elo","elo_gap","total_moves","ends_in_checkmate"]
    miss=[c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Missing columns in {path}:{miss}")
    df["speed"]=onehot_to_label(df,"speed_",a.dropped_speed)
    df["eco_letter"]=onehot_to_label(df,"eco_letter_",a.dropped_letter)
    df["strength"]=(df["white_elo"]+df["black_elo"])/2
    q25,q75=df["strength"].quantile([0.25,0.75])
    df["is_top"]=df["strength"]>=q75
    df["is_bottom"]=df["strength"]<=q25
    print(f"Loaded {len(df):,} games. Top quartile strength>={q75:.0f}, bottom<={q25:.0f}.")
    print(f"Speeds: {df['speed'].value_counts().to_dict()}")
    return df

def add(name,test,stat_label,stat,dof,p,est,ci,verdict):
    RES.append(dict(test=name,method=test,stat_label=stat_label,statistic=stat,df=dof,p=p,
    estimate=est,ci=ci, reject=bool(p<alpha),practical=verdict))

def test1(df,rng):
    print("\n"+"-"*78)
    print("TEST 1 - One-sample t-test: does White keep a first-move edge in near-equal games?")
    print("="*78)
    print("Subset:|elo_gap|<=25")
    print("H0: mu=0.5  |  H1: mu>0.5 (mean White score)   one-sided, alpha=0.05, smallest meaningful diff=0.01")
    sub=df[df["elo_gap"].abs()<=25]
    x=sub["white_score"].to_numpy(float)
    n=len(x)
    mean,sd= x.mean(),x.std(ddof=1)
    se=sd/np.sqrt(n)
    t=(mean-0.5)/se
    dof=n-1
    p=stats.t.sf(t,dof)
    tc=stats.t.ppf(1-alpha/2,dof)
    l,h=mean-tc*se,mean+tc*se
    low1=mean-stats.t.ppf(1-alpha,dof)*se
    v=verdict_diff(l-0.5,h-0.5,0.01)
    print(f"\nn = {n:,} of {len(df):,} games   mean White score = {mean:.5f}   (all games: {df['white_score'].mean():.5f})")
    print(f"t = {t:.3f}  df = {dof:,}  p (one-sided) = {format_p(p)}  -> {'REJECT H0' if p < alpha else 'FAIL TO REJECT H0'}")
    print(f"Estimate {mean:.5f}   95% CI (two-sided) [{l:.5f}, {h:.5f}]   one-sided 95% lower bound {low1:.5f}")
    print(f"Edge over 0.5={mean-0.5:.4f}, CI [{l-0.5:.4f}, {h-0.5:.4f}] vs threshold 0.01  ->  {v}")
    print("\nContext: White score by rating-gap band")
    bins= [-np.inf,-200,-100,-25,25,100,200,np.inf]
    lab= ["<-200","-200..-100","-100..-25","-25..25","25..100","100..200",">200"]
    band=pd.cut(df["elo_gap"],bins=bins,labels=lab)
    print(df.groupby(band,observed=True)["white_score"].agg(["mean", "size"]).round(4).to_string())
    add("Test 1","One-sample t","t",t,f"{dof:,}",p,f"{mean:.4f}",f"[{l:.4f}, {h:.4f}]",v)

def two_sample_t(a,b):
    va,vb=a.var(ddof=1),b.var(ddof=1)
    se=np.sqrt(va/len(a)+vb/len(b))
    diff=a.mean()-b.mean()
    t=diff/se
    dof=(va/len(a) +vb/len(b))**2/((va/len(a))**2 /(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    p=2*stats.t.sf(abs(t),dof)
    tc=stats.t.ppf(1-alpha/2,dof)
    return diff,t,dof,p,diff-tc*se,diff+tc*se

def test2(df, rng):
    print("\n"+"-"*78)
    print("TEST 2- two_sample_t two-sample t-test: do stronger players play longer games?")
    print("-"*78)
    print("H0: mu_top-mu_bottom=0  |  H1:!=0   two-sided, alpha=0.05, smallest meaningful diff=3 moves")
    top = df.loc[df["is_top"], "total_moves"].to_numpy(float)
    bot = df.loc[df["is_bottom"], "total_moves"].to_numpy(float)
    print(f"\ntop: n={len(top):,} mean={top.mean():.2f} (sd {top.std(ddof=1):.2f})   "
          f"bottom: n={len(bot):,} mean={bot.mean():.2f} (sd {bot.std(ddof=1):.2f})")
    diff,t,dof,p,lo,hi=two_sample_t(top,bot)
    v=verdict_diff(lo,hi,3.0)
    print(f"t = {t:.3f}  df (two_sample_t-Satterthwaite)={dof:,.1f}  p={format_p(p)}->{'REJECT H0' if p<alpha else 'FAIL TO REJECT H0'}")
    print(f"Estimate (top-bottom)={diff:.3f} moves 95% CI [{lo:.3f}, {hi:.3f}]  vs threshold 3->{v}")
    sp=np.sqrt((top.var(ddof=1)+bot.var(ddof=1))/2)
    print("\nconfounder check-same comparison within each speed:")
    for name,g in df.groupby("speed"):
        if len(g)<400:
            continue
        s=g["strength"]
        q25,q75=s.quantile([0.25, 0.75])
        ta,ba=g.loc[s>=q75, "total_moves"].to_numpy(float), g.loc[s<=q25,"total_moves"].to_numpy(float)
        if len(ta)<30 or len(ba)<30:
            continue
        d_, _, _, p_, l_, h_=two_sample_t(ta, ba)
        print(f"{name:<12} diff={d_:7.2f} moves 95% CI [{l_:6.2f},{h_:6.2f}]  p={format_p(p_)}")
    add("Test 2","two_sample_t t","t", t, f"{dof:,.0f}",p,f"{diff:.2f} moves",f"[{lo:.2f},{hi:.2f}]",v)

def test3(df,rng):
    print("\n"+"-"*78)
    print("TEST 3 - Two-proportion z-test: do strong players' games end in checkmate less often?")
    print("-"*78)
    print("H0: p_top-p_bottom=0  |  H1:!=0   two-sided, alpha=0.05, smallest meaningful diff=3 percentage points")
    a,b=df[df["is_top"]], df[df["is_bottom"]]
    x1,n1,x2,n2=int(a["ends_in_checkmate"].sum()),len(a),int(b["ends_in_checkmate"].sum()),len(b)
    p1,p2=x1/n1,x2/n2
    pp=(x1+x2)/(n1+n2)
    se0=np.sqrt(pp*(1-pp)*(1/n1+1/n2))
    z=(p1-p2)/se0
    p=2*stats.norm.sf(abs(z))
    se1=np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    zc=stats.norm.ppf(1-alpha/2)
    diff=p1-p2
    lo,hi=diff-zc*se1,diff+zc*se1
    v=verdict_diff(lo,hi,0.03)
    print(f"\ntop: {x1:,}/{n1:,}={p1:.3%}, bottom: {x2:,}/{n2:,}={p2:.3%}")
    print(f"z={z:.3f}  p={format_p(p)} -> {'REJECT H0' if p<alpha else 'FAIL TO REJECT H0'}")
    print(f"Estimate (top-bottom)={diff*100:.3f} 95% CI [{lo*100:.3f},{hi*100:.3f}]"
          f"vs threshold 3 pp -> {v}")
    print(f"Risk ratio={p1/p2:.3f}")
    add("Test 3","Two-proportion z","z",z,"n/a",p,f"{diff*100:.2f} pp",f"[{lo*100:.2f},{hi*100:.2f}]",v)

def cramers_v(tbl):
    chi2=stats.chi2_contingency(tbl,correction=False)[0]
    return np.sqrt(chi2/(tbl.sum()*(min(tbl.shape)-1)))

def test4(df,rng,n_boot=1000,min_count=200):
    print("\n"+"-"*78)
    print("TEST 4- Chi-square test of independence: opening family vs game speed")
    print("-"*78)
    print("H0: opening family and speed are independent  |  H1: not independent")
    print("upper-tail chi2, alpha=0.05, smallest meaningful difference: Cramer's V=0.10")
    rows=df["speed"]
    cols=df["eco_letter"].map(lambda c: LETTER_NAMES.get(c,c))
    tab=pd.crosstab(rows, cols)
    tab=tab.loc[tab.sum(axis=1)>=min_count, tab.sum(axis=0)>=min_count]
    print("\nObserved counts:\n"+tab.to_string())
    print("\nRow percentages (opening mix within each speed):\n" +(tab.div(tab.sum(axis=1),axis=0)*100).round(2).to_string())
    t=tab.to_numpy()
    chi2,p,dof,exp=stats.chi2_contingency(t,correction=False)
    n=t.sum()
    V=cramers_v(t)
    probs=(t/n).ravel()
    boots=[]
    for i in range(n_boot):
        b=rng.multinomial(n, probs).reshape(t.shape)
        if(b.sum(axis=0)==0).any() or (b.sum(axis=1)==0).any():
            continue
        boots.append(cramers_v(b))
    lo,hi=np.percentile(boots,[2.5,97.5])
    v="Matters" if lo>=0.10 else "Negligible" if hi<0.10 else "Inconclusive"
    print(f"\nchi2={chi2:.3f}  df={dof}  critical (0.05)={stats.chi2.ppf(0.95, dof):.3f}  p={format_p(p)} "
          f"-> {'REJECT H0' if p < alpha else 'FAIL TO REJECT H0'}   (min expected count={exp.min():.1f})")
    print(f"Cramer's V={V:.4f}   95% bootstrap CI [{lo:.4f}, {hi:.4f}]  vs threshold 0.10  ->  {v}")
    resid=(t-exp)/np.sqrt(exp)
    print("\nStandardised residuals (|r|>3=cells driving the result):\n" +
          pd.DataFrame(resid,index=tab.index,columns=tab.columns).round(2).to_string())
    add("Test 4","Chi-square","chi2",chi2, f"{dof}",p,f"V = {V:.3f}", f"[{lo:.3f}, {hi:.3f}]",v)
    return sorted(
        [
            (tab.index[i],tab.columns[j],float(resid[i,j]))
            for i in range(len(tab.index))
            for j in range(len(tab.columns))
        ],
        key=lambda cell: abs(cell[2]),
        reverse=True,
    )[:5]

def test5(df, rng):
    print("\n"+"-"*78)
    print("TEST 5-F-test for two variances: are strong players more consistent in game length?")
    print("-"*78)
    print("H0: var_top=var_bottom | H1:!=two-sided, alpha=0.05, smallest meaningful diff: ratio 1.25(or 0.80)")
    a=df.loc[df["is_top"],"total_moves"].to_numpy(float)
    b=df.loc[df["is_bottom"],"total_moves"].to_numpy(float)
    d1,d2=len(a)-1,len(b)-1
    F=a.var(ddof=1)/b.var(ddof=1)
    p=2*min(stats.f.cdf(F, d1, d2), stats.f.sf(F, d1, d2))
    lo=F/stats.f.ppf(1-alpha/2,d1,d2)
    hi=F/stats.f.ppf(alpha/2,d1,d2)
    v="Matters" if lo>=1.25 or hi<=0.80 else "Negligible" if lo>0.80 and hi<1.25 else "Inconclusive"
    print(f"\ntop: n={len(a):,} var={a.var(ddof=1):.2f} (sd {a.std(ddof=1):.2f})"
          f"bottom: n={len(b):,} var={b.var(ddof=1):.2f} (sd {b.std(ddof=1):.2f})")
    print(f"F={F:.4f}  df=({d1:,}, {d2:,})  p={format_p(p)} -> {'REJECT H0' if p<alpha else 'FAIL TO REJECT H0'}")
    print(f"Estimate (var_top/var_bottom)={F:.4f} 95% CI [{lo:.4f},{hi:.4f}]  vs band [0.80, 1.25] ->{v}")
    print(f"Skewness of total_moves: top={stats.skew(a):.2f}, bottom={stats.skew(b):.2f}  "
          f"(the F-test assumes normality, so judge by the ratio CI)")
    add("Test 5","F-test","F",F,f"{d1:,},{d2:,}",p,f"{F:.3f}",f"[{lo:.3f},{hi:.3f}]",v)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--dropped-speed",default="blitz")
    ap.add_argument("--dropped-letter",default="A")
    a=ap.parse_args()

    rng=np.random.default_rng(a.seed)
    df=load(a.src,a)
    largest_residual_cells=[]
    for fn in (test1,test2,test3,test4,test5):
        result=fn(df,rng)
        if fn is test4:
            largest_residual_cells=result

    res=pd.DataFrame(RES)
    print("\n"+"-"*78)
    print("SUMMARY")
    print("-"*78)
    show=res[["test","method","statistic","df","p","estimate","ci","reject","practical"]].copy()
    show["statistic"]=show["statistic"].map(lambda v: f"{v:.3f}")
    show["p"]=show["p"].map(format_p)
    print(show.to_string(index=False))
    print(f"\n(Bonferroni alpha for 5 tests={alpha / 5:.3f}; if you also report the other tests you ran, "
          f"divide 0.05 by the total number of tests run.)")
    with open("final_results.json", "w") as f:
        json.dump(RES, f, indent=2, default=float)
    print("Saved final_results.json")
    
if __name__=="__main__":
    main()
