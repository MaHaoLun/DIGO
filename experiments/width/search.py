"""Leased, once-only missing native calls. Raw results commit before audits."""
import argparse
import fcntl
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from common import *

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ges'))
from native import Native
from native_astar import ContinuousIntervalAStar
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ges'))
from guarded_native import GuardedNative

def replay(result,job,c):
    if result['status']!='OK':return dict(valid=False,status=result['status'])
    dep=F(result['departure']);cost=F(result['cost']);lo,hi=job['interval'];assert lo<=dep<=hi
    route=result['route'];assert route[0]==job['source'] and route[-1]==job['destination']
    ix=np.searchsorted(c['coords'][:,0],route);assert np.array_equal(c['coords'][ix,0],route)
    n=len(c['coords']);t=dep;arr=result.get('arrivals')
    if arr:assert F(arr[0])==t
    for k,(u,v) in enumerate(zip(ix[:-1],ix[1:])):
        at=int(np.searchsorted(c['arc_keys'],int(u)*n+int(v)))
        assert c['arc_keys'][at]==int(u)*n+int(v)
        phase=t//900;slot=int(phase)%96;rem=t-900*phase;w=c['graph']['knots'][at]
        t+=int(w[slot])+F(int(w[(slot+1)%96]-w[slot]),900)*rem
        if arr:assert t==F(arr[k+1])
    assert t-dep==cost
    return dict(valid=True,cost=str(cost),departure=str(dep),route_vertices=len(route),exact_original_arc_replay=True)

def close(c):
    if c:
        for name in ['full','raw','guard']:c[name].close()
        c.clear()

def load(job,plan):
    city,snap=job['city'],job['snapshot'];graph=checked(job['graph'])
    graphnpz=P/'e1_v1/data'/city/'routing_graph.npz' if snap=='G0' else P/'e2_v1/data'/city/'G1_graph.npz'
    g=dict(np.load(graphnpz));coords=np.load(P/'e1_v1/data'/city/'coordinates.npy');n=len(coords)
    libs=plan['libraries']
    return dict(key=(city,snap),graph=g,coords=coords,arc_keys=g['u'].astype(np.int64)*n+g['v'],
        full=ContinuousIntervalAStar(graph,checked(libs['full'])),raw=Native(graph,checked(libs['regional'])),
        guard=GuardedNative(graph,checked(libs['guarded'])))

def masks(job):
    with np.load(checked(job['mask_record'])) as d:
        wi=job['width_index'];assert int(d['serial'])==job['serial']
        raw=np.unpackbits(d['raw_union_packbits'][wi]).astype(np.uint8)
        unionthin=np.unpackbits(d['guard_union_packbits'][wi]).astype(np.uint8)
        single=np.unpackbits(d['single_raw_packbits']).astype(np.uint8)
        singlethin=np.unpackbits(d['single_guard_packbits']).astype(np.uint8)
    return dict(union_ges=(raw,None),single_expansion=(single,singlethin),union_expansion=(raw,unionthin))

def old_record(job,arm):
    r=job['old_full'] if arm=='full_graph' else job['old_raw']
    assert r is not None and r['valid']=='True'
    return dict(execution='REUSED_ACCEPTED_4H',host=r['search_host'],
        api_seconds=float(r['api_seconds']),result=dict(status='OK',cost=r['cost'],departure=r['departure'],native_seconds=float(r['native_seconds'])),
        audit=dict(valid=True,cost=r['cost'],departure=r['departure'],source='Accepted full E4 original-arc replay',exact_original_arc_replay=True),
        source_csv=str(P/'ssp_no_union_e4_e5_v1/final/full/E4_per_query.csv'),source_serial=job['serial'],source_method=r['method'])

def execute(job,arm,c,mm,plan_stamp,worker):
    cid=job['call_ids'][arm]
    if cid is None:return old_record(job,arm)
    out=W/'calls'/cid;out.mkdir(parents=True,exist_ok=True)
    rawfile=out/'raw.json';auditfile=out/'audit.json';intent=out/'intent.json'
    if rawfile.exists():
        row=json.loads(rawfile.read_text());assert row['case_id']==job['case_id'] and row['call_id']==cid
    else:
        assert not intent.exists(),'Unresolved native intent; do not repeat '+str(intent)
        value=dict(status='NATIVE_CALL_INTENT',case_id=job['case_id'],call_id=cid,
            arm=arm,job_identity=dict(city=job['city'],snapshot=job['snapshot'],source=job['source'],destination=job['destination'],interval=job['interval']),
            graph=job['graph'],mask=job['mask_record'],plan=plan_stamp,worker=worker,
            host=socket.gethostname().split('.')[0],pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),unix=time.time())
        with intent.open('x') as h:json.dump(value,h);h.flush();os.fsync(h.fileno())
        row=dict(case_id=job['case_id'],call_id=cid,arm=arm,host=value['host'],affinity=value['affinity'],worker=worker,started_unix=time.time())
        timer=time.perf_counter()
        try:
            if arm=='full_graph':r=c['full'].query(job['source'],job['destination'],*job['interval'])
            elif arm=='union_ges':r=c['raw'].query(job['source'],job['destination'],*job['interval'],4,mm[arm][0])
            else:r=c['guard'].query_guarded(job['source'],job['destination'],*job['interval'],*mm[arm])
            row.update(execution='RETURNED_RAW',api_seconds=time.perf_counter()-timer,result=r)
        except BaseException as exc:
            row.update(execution='NATIVE_FAILED',api_seconds=time.perf_counter()-timer,error=repr(exc))
            atomic(rawfile,row);raise
        atomic(rawfile,row)
    if not auditfile.exists():
        try:
            audit=replay(row['result'],job,c)
            assert math.isfinite(row['api_seconds']) and row['api_seconds']>=0
            assert math.isfinite(row['result']['native_seconds']) and row['result']['native_seconds']>=0
            if arm in('single_expansion','union_expansion'):
                r=row['result'];raw,thin=mm[arm]
                assert r['method']==9 and r['repair_decision']=='raw_region_connectivity'
                assert r['repaired']==(not r['initial_connected'])
                assert set(np.flatnonzero(thin))<=set(r['active_cells'])
            elif arm=='union_ges':assert row['result']['method']==4
            atomic(auditfile,dict(status='AUDIT_PASSED',audit=audit))
        except BaseException as exc:
            atomic(out/'audit_failure.json',dict(status='AUDIT_FAILED',error=repr(exc),raw=stamp(rawfile)));raise
    row['audit']=json.loads(auditfile.read_text())['audit'];row['execution']='COMPLETED'
    row['raw_source']=str(rawfile)
    return row

def case_quality(job,saved):
    ref=saved['full_graph'];assert ref['audit']['valid']
    best=F(ref['result']['cost']);earliest=F(ref['result']['departure']);rows=[]
    for arm in ARMS:
        r=saved[arm];valid=r['audit']['valid'];quality=dict(arm=arm,valid=valid,host=r['host'],call_id=job['call_ids'][arm],
            reused_old4h=job['call_ids'][arm] is None,raw_source=r.get('raw_source'),
            native_seconds=float(r['result']['native_seconds']),api_seconds=float(r['api_seconds']),
            cost=r['result'].get('cost'),departure=r['result'].get('departure'))
        if valid:
            value=F(r['result']['cost']);dep=F(r['result']['departure']);assert value>=best
            quality.update(cost_exact=value==best,earliest_exact=value==best and dep==earliest,
                gap_percent=float(100*(value-best)/best) if best else 0.)
        else:quality.update(cost_exact=False,earliest_exact=False,gap_percent=None)
        if arm=='full_graph':composed=float(r['api_seconds']);inference=0.
        elif arm=='union_ges' and job['width_seconds']==14400:
            composed=float(job['four_hour_anchor_union']['composed_seconds'])
            inference=composed-float(r['api_seconds'])
        else:
            old=job['single_cost_record'] if arm=='single_expansion' else job['bank_cost_record']
            inference=float(old['with_handoff_seconds'])-float(old['search_api_seconds'])
            assert inference>=0
            composed=float(r['api_seconds'])+inference+job['CPU_additions'][arm]
        quality.update(composed_seconds=composed,reused_neural_and_original_pipeline_seconds=inference,
            extra_CPU_seconds=job['CPU_additions'][arm],reference_seconds=float(ref['api_seconds']))
        assert quality['host']==ref['host']==job['host']
        rows.append(quality)
    raw=saved['union_ges'];expanded=saved['union_expansion']
    if raw['audit']['valid']:
        assert expanded['audit']['valid'] and F(expanded['result']['cost'])<=F(raw['result']['cost'])
        if F(expanded['result']['cost'])==F(raw['result']['cost']):assert F(expanded['result']['departure'])<=F(raw['result']['departure'])
    return dict(case_id=job['case_id'],serial=job['serial'],city=job['city'],snapshot=job['snapshot'],
        width_seconds=job['width_seconds'],host=job['host'],rows=rows,all_original_arc_replays=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);ap.add_argument('--resume-case');a=ap.parse_args()
    host=socket.gethostname().split('.')[0];assert len(os.sched_getaffinity(0))==1 and 'Gold 6248R' in Path('/proc/cpuinfo').read_text()
    planpath=W/'SEARCH_JOBS.json';plan_stamp=stamp(planpath);plan=json.loads(planpath.read_text())
    gate=json.loads((W/'INDEPENDENT_INPUT_GATE.json').read_text());assert gate['status']=='PASS'
    assert gate['search_jobs']==plan_stamp
    for s in plan['sources'].values():checked(s)
    output=Path(os.environ['E4_OUTPUT_DIR']);lock=(output/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=sorted([j for j in plan['jobs'] if j['host']==host],key=lambda j:(j['city'],j['snapshot'],j['serial'],j['width_seconds']))
    if a.resume_case:jobs=[j for j in jobs if j['case_id']==a.resume_case];assert len(jobs)==1
    c={};completed=0;start=time.monotonic()
    claimdir=W/'claims';claimdir.mkdir(exist_ok=True);qualitydir=W/'quality';qualitydir.mkdir(exist_ok=True)
    try:
        for job in jobs:
            q=qualitydir/(job['case_id']+'.json')
            if q.exists():continue
            claim=claimdir/(job['case_id']+'.json')
            if not a.resume_case:
                try:
                    with claim.open('x') as h:json.dump(dict(worker=a.worker,host=host,pid=os.getpid(),case_id=job['case_id'],unix=time.time()),h);h.flush();os.fsync(h.fileno())
                except FileExistsError:continue
            if c.get('key')!=(job['city'],job['snapshot']):close(c);c=load(job,plan)
            atomic(output/'progress.json',dict(status='RUNNING',worker=a.worker,host=host,case=job['case_id'],completed_cases=completed,updated_unix=time.time()))
            mm=masks(job);saved={}
            # Reference values are read only for post-call quality; never supplied to native search.
            arms=list(ARMS);rotation=(job['serial']+job['width_index'])%4;arms=arms[rotation:]+arms[:rotation]
            for arm in arms:saved[arm]=execute(job,arm,c,mm,plan_stamp,a.worker)
            atomic(q,case_quality(job,saved));completed+=1
            atomic(output/'progress.json',dict(status='CASE_COMPLETE',worker=a.worker,host=host,completed_cases=completed,updated_unix=time.time(),elapsed_seconds=time.monotonic()-start))
        atomic(output/'progress.json',dict(status='FINISHED',worker=a.worker,host=host,completed_cases=completed,updated_unix=time.time(),elapsed_seconds=time.monotonic()-start))
    finally:close(c)

if __name__=='__main__':main()
