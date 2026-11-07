"""decayonly_fresh: fresh unseen seeds 30-39, 60 tasks, all arms paired per seed vs the parent run here.
A = _pr50_decayonly, B = _wd00084 (matched constant decay, primary), B2 = _wd0005 (descriptive).
95% t CI df=9 (t=2.262); blocks of 15 tasks.
Rule (preregistered):
  A bars vs parent: mean >= +0.002 AND >= 7/10 improved AND CI_lo > 0 -> A PASS (screen; nonpromoting until confirmed) else A FAIL.
  Timing read on D = A - B (paired), G = A gain vs parent:
    TIMING-SPECIFIC: CI_lo(D) > 0 and D >= 0.5*G (requires G > 0)
    AMOUNT-ONLY:     CI_lo(D) <= 0 or D < 0.25*G or G <= 0
    otherwise MIXED. B2 is descriptive only."""
import json, statistics as st
D='outputs/ipmnist_screening/decayonly_fresh_screen/fresh/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
S=range(30,40); arms={'parent':P,'A_decayonly':P+'_pr50_decayonly','B_wd00084':P+'_wd00084','B2_wd0005':P+'_wd0005'}
def acc(c,s): a=json.load(open(f'{D}{c}_seed{s}.json'))['per_task_accuracy']; assert len(a)==60; return a
X={k:{s:acc(c,s) for s in S} for k,c in arms.items()}
def summ(name,diffs):
    g=[sum(v)/60 for v in diffs]; m=st.mean(g); se=st.stdev(g)/10**.5; lo,hi=m-2.262*se,m+2.262*se; n=sum(x>0 for x in g)
    bl=[st.mean(sum(v[k:k+15])/15 for v in diffs) for k in range(0,60,15)]
    print(f'{name}: '+' '.join(f's{s} {x:+.6f}' for s,x in zip(S,g)))
    print(f'  mean {m:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {n}/10 blocks '+' / '.join(f'{b:+.6f}' for b in bl)); return m,lo,n
for k in X: print(f'{k} mean acc {st.mean(sum(X[k][s])/60 for s in S):.6f}')
d=lambda a,b:[[i-j for i,j in zip(X[a][s],X[b][s])] for s in S]
G,Glo,Gn=summ('A vs parent',d('A_decayonly','parent'))
summ('B vs parent',d('B_wd00084','parent')); summ('B2 vs parent',d('B2_wd0005','parent'))
Dm,Dlo,_=summ('A - B',d('A_decayonly','B_wd00084')); summ('A - B2',d('A_decayonly','B2_wd0005'))
bars=G>=0.002 and Gn>=7 and Glo>0
print(f'A BARS: mean>=+0.002 {G>=0.002}, >=7/10 {Gn>=7}, CI_lo>0 {Glo>0} -> {"PASS" if bars else "FAIL"}')
t='TIMING-SPECIFIC' if G>0 and Dlo>0 and Dm>=0.5*G else ('AMOUNT-ONLY' if G<=0 or Dlo<=0 or Dm<0.25*G else 'MIXED')
print(f'TIMING READ {t} (D {Dm:+.6f}, G {G:+.6f}, CI_lo(D) {Dlo:+.6f})')
