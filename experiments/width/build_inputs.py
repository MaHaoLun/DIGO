"""Construct all frozen masks and once-measured extra CPU interface work.

No predictor/solver calls. Existing original16-bank and actual-single outputs
are retained, along with their complete paid component costs.
"""
import argparse
import csv
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from common import *

def expand(raw,scores,support,cutoff):
    raw=np.asarray(raw,bool)
    point=raw.reshape(-1,64,64)
    full=binary_dilation(point,structure=np.ones((1,3,3),bool)).reshape(-1,4096)&support
    return raw.reshape(-1,4096)|(full&(scores.reshape(-1,4096).astype(np.float64)>=cutoff))

def delivered(raw,guard=None):
    if guard is None:
        payload=np.packbits(raw).tobytes()
        out=np.unpackbits(np.frombuffer(payload,dtype=np.uint8)).copy()
        assert len(payload)==512 and np.array_equal(out,raw)
    else:
        payload=np.packbits(np.stack((raw,guard)),axis=1).tobytes()
        out=np.unpackbits(np.frombuffer(payload,dtype=np.uint8).reshape(2,512),axis=1).copy()
        assert len(payload)==1024 and np.array_equal(out[0],raw) and np.array_equal(out[1],guard)
    return payload

def generate(city,snap,cohort):
    entries=sorted([r for r in cohort['records'] if(r['city'],r['snapshot'])==(city,snap)],key=lambda r:r['parent_index'])
    cal=json.loads((W/'calibration'/f'{city}_{snap}.json').read_text())
    cutoff=cal['guard_threshold'];threshold=cal['base_threshold']
    assert all(r['model']==cal['model'] and r['base_threshold']==threshold for r in entries)
    cells=np.load(P/'e1_v1/data'/city/'cell64.npy');support=np.bincount(cells,minlength=4096)>0
    parent=dict(np.load(P/'e4_full_v2/inputs'/city/snap/'queries.npz'))
    if snap=='G0':
        d=modeldir(city,snap)/'test_predictions'
        ids=np.load(d/'query_ids.npy',mmap_mode='r');bank=np.load(d/'scores.npy',mmap_mode='r')
    else:bank=None;ids=None
    lastchunk=None;chunkdata=None;outputs=[]
    output=W/'mask_records';output.mkdir(exist_ok=True)
    for position,e in enumerate(entries):
        serial=e['serial'];target=output/f'{serial:05d}.npz'
        if target.exists():
            with np.load(target) as previous:
                assert int(previous['serial'])==serial and float(previous['cutoff'])==cutoff
            outputs.append(stamp(target));continue
        i=e['parent_index'];gid=e['group_id'];assert int(parent['group_ids'][i])==gid
        if snap=='G0':
            assert np.array_equal(ids[i*16:(i+1)*16],gid*16+np.arange(16))
            scores=np.array(bank[i*16:(i+1)*16]);score_source=modeldir(city,snap)/'test_predictions/scores.npy'
        else:
            begin=i//64*64;end=min(begin+64,len(parent['group_ids']))
            chunk=P/'e4_full_v2/predictions'/f'{city}_G1'/f'chunk_{begin:05d}_{end:05d}.npz'
            if chunk!=lastchunk:
                with np.load(chunk) as d:chunkdata={k:np.array(d[k]) for k in ['group_ids','scores']}
                lastchunk=chunk
            assert int(chunkdata['group_ids'][i-begin])==gid
            scores=np.array(chunkdata['scores'][i-begin]);score_source=chunk
        assert scores.shape==(16,4096) and scores.dtype==np.float32 and np.isfinite(scores).all()
        ep=cells[np.asarray(e['node_indices'])]
        rawpoints=(scores.astype(np.float64)>=threshold)&support;rawpoints[:,ep]=True
        assert np.array_equal(np.packbits(rawpoints.any(0)),parent['initial_unions_packbits'][i])
        single_source=checked(e['single_scores'])
        with np.load(single_source) as d:
            singlescore=np.array(d['scores']).reshape(1,4096)
            single_expected=np.array(d['measured_packbits'])
            assert int(d['group_id'])==gid
        # One actual single extension is reusable over every window, since no
        # departure/interval participates in this CPU mask operation.
        t=time.perf_counter()
        singleraw=(singlescore.astype(np.float64)>=threshold)&support;singleraw[:,ep]=True
        singlethin=expand(singleraw,singlescore,support,cutoff)[0]
        singleraw=singleraw[0]
        delivered(singleraw,singlethin)
        single_seconds=time.perf_counter()-t
        assert np.array_equal(np.packbits(singleraw),single_expected)
        raws=[];guards=[];raw_seconds=[];guard_seconds=[]
        for w in WIDTHS:
            k=w//900
            t=time.perf_counter()
            rp=(scores[:k].astype(np.float64)>=threshold)&support;rp[:,ep]=True
            raw=rp.any(0);delivered(raw)
            rawtime=time.perf_counter()-t
            t=time.perf_counter()
            rp=(scores[:k].astype(np.float64)>=threshold)&support;rp[:,ep]=True
            thin=expand(rp,scores[:k],support,cutoff).any(0)
            raw_again=rp.any(0);delivered(raw_again,thin)
            guardtime=time.perf_counter()-t
            assert np.array_equal(raw_again,raw) and not(raw&~thin).any()
            raws.append(np.packbits(raw));guards.append(np.packbits(thin))
            raw_seconds.append(0. if w==14400 else rawtime);guard_seconds.append(guardtime)
        # Preserve this extra CPU observation once; later restarts load it.
        tmp=target.with_suffix('.tmp.npz')
        with tmp.open('wb') as f:
            np.savez_compressed(f,serial=serial,group_id=gid,threshold=threshold,cutoff=cutoff,
                raw_union_packbits=np.stack(raws),guard_union_packbits=np.stack(guards),
                single_raw_packbits=np.packbits(singleraw),single_guard_packbits=np.packbits(singlethin),
                raw_extra_seconds=np.asarray(raw_seconds),guard_extra_seconds=np.asarray(guard_seconds),
                single_extra_seconds=single_seconds,CPU_host=socket.gethostname().split('.')[0],
                CPU_affinity=np.asarray(sorted(os.sched_getaffinity(0))),
                score_source=str(score_source),actual_single_source=str(single_source),
                source_bank_conditions=16,source_batch=16 if snap=='G0' else 2,
                paid_single_conditions=1,extra_work='CPU prefix/expansion plus in-process512/1024byte mask delivery')
            f.flush();os.fsync(f.fileno())
        tmp.replace(target);outputs.append(stamp(target))
        if position%64==0:atomic(W/'inputs_progress'/f'{city}_{snap}.json',dict(saved=position+1,expected=len(entries),native_calls=0,forward_calls=0,unix=time.time()))
    immutable(W/f'INPUTS_{city}_{snap}.json',dict(status='COMPLETE_FIXED_BANK_AND_ACTUAL_SINGLE_MASKS',
        city=city,snapshot=snap,ODs=len(entries),calibration=stamp(W/'calibration'/f'{city}_{snap}.json'),
        cohort=stamp(W/'cohort.json'),files=outputs,query_forward_calls=0,native_calls=0,
        neural_cost='reuse complete original16bank/actualbatch1 pipeline; all new CPU work added conservatively'))
    print(json.dumps(dict(city=city,snapshot=snap,ODs=len(entries),status='MASKS_READY')),flush=True)

def plan():
    source=W/'cohort.json';cohort=json.loads(source.read_text());assert len(cohort['records'])==5275
    for city,snap,n in SETTINGS:
        x=json.loads((W/f'INPUTS_{city}_{snap}.json').read_text());assert x['ODs']==n
        for f in x['files']:checked(f)
    selected={r['serial']:r for r in cohort['records']}
    oldrows={}
    with(P/'ssp_no_union_e4_e5_v1/final/full/E4_per_query.csv').open() as f:
        for r in csv.DictReader(f):
            s=int(r['serial'])
            if s in selected and r['method'] in ['global_continuous_Astar','GES_deferred_shared_endpoint']:
                oldrows[s,r['method']]=r
    assert len(oldrows)==10550
    jobs=[];callindex=[];aliases=0;hosts=os.environ['DIGO_WORKER_HOSTS'].split(',')
    cost_fields=['with_handoff_seconds','search_api_seconds','input_prepare_seconds',
        'h2d_and_gpu_rng_setup_seconds','model_sampling_seconds','d2h_seconds',
        'region_decode_union_seconds','handoff_encode_seconds','handoff_receive_decode_seconds',
        'cost_host','source_cost_record','cost_record_path']
    for e in sorted(cohort['records'],key=lambda r:r['serial']):
        s=e['serial'];path=W/'mask_records'/f'{s:05d}.npz'
        with np.load(path) as d:m={k:np.array(d[k]) for k in ['raw_union_packbits','guard_union_packbits','single_raw_packbits','single_guard_packbits','raw_extra_seconds','guard_extra_seconds','single_extra_seconds']}
        for wi,w in enumerate(WIDTHS):
            case=f's{s:05d}_w{w}';host=e['host'] if w==14400 else hosts[(s+wi)%4]
            same=np.array_equal(m['raw_union_packbits'][wi],m['single_raw_packbits']) and np.array_equal(m['guard_union_packbits'][wi],m['single_guard_packbits'])
            callids={arm:f'{case}_{arm}' for arm in ARMS}
            if w==14400:callids['full_graph']=callids['union_ges']=None
            if same:callids['union_expansion']=callids['single_expansion'];aliases+=1
            job=dict(case_id=case,serial=s,city=e['city'],snapshot=e['snapshot'],host=host,
                source=e['source'],destination=e['destination'],node_indices=e['node_indices'],
                interval=[e['lo'],e['lo']+w],width_seconds=w,width_index=wi,
                mask_record=stamp(path),graph=e['graph'],model=e['model'],call_ids=callids,
                old_full=oldrows[s,'global_continuous_Astar'] if w==14400 else None,
                old_raw=oldrows[s,'GES_deferred_shared_endpoint'] if w==14400 else None,
                anchor=e['old_reference'] if w==14400 else None,
                bank_cost_record={k:e['bank_cost_record'].get(k,'') for k in cost_fields},
                single_cost_record={k:e['single_cost_record'].get(k,'') for k in cost_fields},
                CPU_additions=dict(union_ges=float(m['raw_extra_seconds'][wi]),
                    single_expansion=float(m['single_extra_seconds']),union_expansion=float(m['guard_extra_seconds'][wi]),full_graph=0.),
                four_hour_anchor_union=e['old_union'] if w==14400 else None)
            jobs.append(job)
            for arm in ARMS:
                if callids[arm] is None:continue
                if arm=='union_expansion' and same:continue
                callindex.append(dict(call_id=callids[arm],case_id=case,host=host,city=e['city'],snapshot=e['snapshot'],
                    source=e['source'],destination=e['destination'],interval=job['interval'],arm=arm,
                    method='full' if arm=='full_graph' else 4 if arm=='union_ges' else 9,mask_record=job['mask_record']))
    assert len(jobs)==21100 and len(callindex)==73850-aliases
    amendment=dict(version='width_scheduling_v1',protocol=stamp(W/'WIDTH_PROTOCOL.json'),
        four_hour='original reference host retained',short_width='paired full+methods host=(serial+width_index)%4, same model Gold6248R; no historical reference time substituted',
        host_order=hosts,capacity='C18 untilD complete then24; shared physical leases; original graph/model/selection immutable')
    immutable(W/'SCHEDULING_AMENDMENT.json',amendment)
    base=P/'e4_full_v2/builds/formal_v2_r2'
    libraries=dict(full=stamp(base/'libcontinuous_interval_astar_v1.so'),
        regional=stamp(base/'libges_astar_full_v2.so'),
        guarded=stamp(P/'rolling_interval_distance_20261028/builds/guarded_v1/libguarded_regional_v1.so'))
    immutable(W/'CALL_INDEX.json',dict(status='ALL_MISSING_CALL_IDENTITIES_FROZEN',logical_rows=84400,
        old4h_reuse_rows=10550,unique_new_calls=len(callindex),exact_method9_alias_rows=aliases,
        no_new_model_query_calls=True,libraries=libraries,calls=callindex))
    immutable(W/'SEARCH_JOBS.json',dict(status='FROZEN_READY_FOR_INDEPENDENT_INPUT_REVIEW',jobs=jobs,
        logical_rows=84400,old4h_rows=10550,unique_calls=len(callindex),aliases=aliases,
        protocol=stamp(W/'WIDTH_PROTOCOL.json'),scheduling=stamp(W/'SCHEDULING_AMENDMENT.json'),
        calibrations=stamp(W/'CALIBRATION_ALL_FROZEN.json'),libraries=libraries,
        sources={n:stamp(Path(__file__).parent/n) for n in ['common.py','build_inputs.py','calibration.py','search.py']}))
    atomic(ROOT/'status/width.json',dict(status='ALL_INPUTS_AND_CALL_INDEX_FROZEN_PENDING_INDEPENDENT_REVIEW',
        logical_rows=84400,old4h_reuses=10550,new_unique_calls=len(callindex),aliases=aliases,
        native_calls=0,query_forward_calls=0,updated_unix=time.time()))
    print(json.dumps(dict(status='ALL_INPUTS_FROZEN',cases=len(jobs),new_unique_calls=len(callindex),aliases=aliases)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',choices=['G0','G1']);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    require_registration()
    if a.plan_only:plan();return
    assert a.snapshot
    cohort=json.loads((W/'cohort.json').read_text())
    for city in ['BJ','XA']:generate(city,a.snapshot,cohort)

if __name__=='__main__':main()
