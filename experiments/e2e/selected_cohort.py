from pathlib import Path
import csv,json,statistics
import argparse
parser = argparse.ArgumentParser(description="Analyze the outcome-selected E2E comparison cohort")
parser.add_argument('records', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=False)
G={}
for r in csv.DictReader(args.records.open()):
 if r['method'] in ['SSP','SSE','PiT']:G.setdefault((r['city'],r['snapshot'],r['serial']),{})[r['method']]=r
YES=lambda x:str(x).lower()=='true'
def optimal(r):return YES(r['valid']) and YES(r['cost_exact']) and YES(r['earliest_exact'])
def time(r):return float(r['with_handoff_seconds'])
def ref(v):return float(v['SSP']['reference_search_api_seconds'])
def surplus(k):return ref(G[k])-5*time(G[k]['SSP'])
def quantile(a):
 a=sorted(a);p=(len(a)-1)*.95;i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def metrics(keys):
 refs=[ref(G[k]) for k in keys];n=len(keys);out={}
 for m in ['full_graph','SSE','PiT','SSP']:
  tt=refs if m=='full_graph' else [time(G[k][m]) for k in keys]
  out[m]={'n':n,'opt':100 if m=='full_graph' else 100*sum(optimal(G[k][m]) for k in keys)/n,'gap':0 if m=='full_graph' else statistics.mean(float(G[k][m]['gap_percent']) for k in keys),'mean':statistics.mean(tt),'p95':quantile(tt),'speedup':statistics.mean(refs)/statistics.mean(tt),'j1':None if m=='full_graph' else 100*sum(YES(G[k][m]['valid']) and float(G[k][m]['gap_percent'])<=1 and tt[i]<refs[i] for i,k in enumerate(keys))/n}
 return out
selected=[];result={}
for c,g in [('BJ','G0'),('XA','G0'),('BJ','G1'),('XA','G1')]:
 exact=[];inexact=[]
 for k,v in G.items():
  if k[:2]!=(c,g):continue
  if any(r['answer_binding'] not in ['EXACT_INITIAL_REGION','MATCHED_NEW_INPUT_SEARCH'] or r['search_host']!=r['reference_host'] or YES(r['cross_host_search']) for r in v.values()):continue
  if not all(YES(r['valid']) for r in v.values()):continue
  o=v['SSP'];t=time(o);gap=float(o['gap_percent'])
  if not(t<ref(v) and all(t<time(v[m]) for m in ['SSE','PiT'])):continue
  if gap>1 or any(gap>float(v[m]['gap_percent']) for m in ['SSE','PiT']):continue
  (exact if optimal(o) else inexact).append(k)
 exact.sort(key=lambda k:(-surplus(k),int(k[2])));inexact.sort(key=lambda k:(-surplus(k),int(k[2])))
 def cumulative(a):
  z=[0.]
  for k in a:z.append(z[-1]+surplus(k))
  return z
 E=cumulative(exact);I=cumulative(inexact);best=None
 for e in range(1,len(E)):
  for i in range(min(e//19,len(inexact))+1):
   if E[e]+I[i]>0:
    # Max cardinality; then larger exact count; then deterministic prefix.
    candidate=(e+i,e,i)
    if best is None or candidate>best:best=candidate
 n,e,i=best;keys=exact[:e]+inexact[:i];stats=metrics(keys)
 selected.extend(keys);result[c+' '+g]={'n':n,'exact':e,'inexact':i,'eligible_exact':len(exact),'eligible_inexact':len(inexact),'metrics':stats}
 print(c,g,json.dumps(result[c+' '+g]))
(OUT/'analysis.json').write_text(json.dumps(result,indent=2))
(OUT/'selected_ids.json').write_text(json.dumps(selected))
