import json,math,sys
W='wave_identmap50_r_idet003_s005625_anneal_polyak/shards/'; R='wave_identmap50_r_idet003_s005625_anneal/shards/'
P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'; A=P+'_polyak_w100h0'
seeds=[int(s) for s in sys.argv[1:]] or [0,1,2]
tc={2:4.302652729911275,9:2.2621571627409915,19:2.093024054408263}
L=lambda f:json.load(open(f))['per_task_accuracy']
m=lambda a,i=0,j=60:sum(a[i:j])/(j-i)
rows=[]
print('seed parent arm delta')
for s in seeds:
    p=L(R+f'{P}_seed{s}.json'); a=L(W+f'{A}_seed{s}.json'); rows.append((p,a))
    print(s,'%.6f %.6f %+.6f'%(m(p),m(a),m(a)-m(p)))
d=[m(a)-m(p) for p,a in rows]; n=len(d); mu=sum(d)/n; sd=math.sqrt(sum((x-mu)**2 for x in d)/(n-1)); h=tc[n-1]*sd/math.sqrt(n)
print('paired mean %+.6f  95%% CI [%+.6f, %+.6f]  improved %d/%d'%(mu,mu-h,mu+h,sum(x>0 for x in d),n))
for i,j in ((0,15),(15,30),(30,45),(45,60)):
    print('tasks %d-%d %+.6f'%(i+1,j,sum(m(a,i,j)-m(p,i,j) for p,a in rows)/n))
