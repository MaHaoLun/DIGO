"""Complete native64 component inference with atomic, resumable OD-group shards."""
import argparse
import fcntl
import json
import os
import time
import numpy as np
import torch
from common import RUN,SparseData,atomic,parents,ev
import models


def generate(args):
    parents();city=args.city;snapshot=args.snapshot;method=args.method
    out=RUN/'training'/f'{city}_{snapshot}_{method}_seed42'
    result=json.loads((out/'result.json').read_text());assert result['status']=='TRAINED_FULL'
    assert json.loads((out/'training_audit.json').read_text())['status']=='PASSED'
    role=args.role;assert role in (['Vcal','base_test'] if snapshot=='G0' else ['Acal','mixed_test'])
    d=SparseData(city,snapshot,role);folder=out/(role+'_generation');folder.mkdir(exist_ok=True)
    lock=(folder/f'worker_{args.shard:02d}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    # Deterministic cuDNN avoids source self-repeat noise during binding audits.
    torch.backends.cudnn.deterministic=True
    device='cuda:0';model=(models.sse(city) if method=='SSE' else models.pit()).to(device)
    selected=torch.load(out/'selected.pt',map_location=device,weights_only=False)
    model.load_state_dict(selected['model']);model.eval();dp=models.diffusion(device) if method=='PiT' else None
    ppg=d.ppg;groups=len(d)//ppg;begun=time.monotonic();saved=0;generated=0
    sampler=model;proof_group=None;proof_image=None;executor=None
    execution='eager released denoiser' if method=='PiT' else 'released OD-only classifier'
    graph_gate=RUN/'audits/PiT_graph_equivalence.json'
    parallel_gate=RUN/'audits/PiT_parallel_equivalence.json'
    integration_gate=RUN/'audits/PiT_executor_integration.json'
    parallel_ready=(parallel_gate.exists() and integration_gate.exists()
        and json.loads(parallel_gate.read_text())['status']=='PASSED'
        and json.loads(integration_gate.read_text())['status']=='ENGINEERING_CHECK_PASSED')
    if method=='PiT' and parallel_ready:
        from pit_executor import Executor
        width=4
        decision_file=RUN/'audits/PiT_concurrency_decision.json'
        if decision_file.exists():
            decision=json.loads(decision_file.read_text())
            if decision['status']=='ACCEPTED_FOR_FORMAL_INFERENCE':
                probe=json.loads((RUN/'audits/PiT_concurrency_probe.json').read_text())
                integration=json.loads((RUN/'audits/PiT_concurrency_integration.json').read_text())
                assert probe['status']=='PASSED' and integration['status']=='ENGINEERING_CHECK_PASSED'
                assert decision['width_by_point_batch']==probe['proposed_width_by_point_batch']=={'16':4,'2':16}
                assert integration['concurrent_groups']==16 and integration['proof_cache_consumed_once']
                assert integration['source_proof_full_channels_and_final_RNG_exact']
                assert integration['subsequent_short_wave_groups']==8
                assert all(error==0. for error in integration['short_wave_full_three_channel_errors'])
                width=decision['width_by_point_batch'][str(ppg)]
        executor=Executor(model,dp,d,args.shard,folder,result['selected_epoch'],width=width)
        execution=executor.execution
    elif method=='PiT' and graph_gate.exists() and json.loads(graph_gate.read_text())['status']=='PASSED':
        from pit_graph import GraphModel
        candidate=GraphModel(model,ppg);proof_group=args.shard*32
        assert proof_group<groups
        lo,hi=proof_group*ppg,(proof_group+1)*ppg
        cond=torch.as_tensor(np.array(d.odt[lo:hi]),device=device)
        seed=ev.identity_seed(20260912,int(d.query_ids[lo]//ppg),city,city+'_'+snapshot+'_canonical_fifo_continuous',role)
        generator=torch.Generator(device=device).manual_seed(seed)
        reference=models.pit_sample(model,dp,cond,generator);state=generator.get_state()
        generator.manual_seed(seed);captured=models.pit_sample(candidate,dp,cond,generator)
        assert torch.isfinite(reference).all() and torch.isfinite(captured).all()
        same=bool(torch.equal(reference,captured) and torch.equal(state,generator.get_state()))
        error=float((reference-captured).abs().max())
        if same:
            sampler=candidate;proof_image=captured;execution='captured unchanged denoiser;all1000 eager DDPM transitions'
        else:
            proof_image=reference;execution='eager released denoiser;capture rejected by exact source check'
        already_saved=(folder/f'chunk_{args.shard:06d}.npz').exists()
        atomic(folder/f'engine_audit_{args.shard:02d}_{time.time_ns()}.json',dict(status='PASSED_EXACT_CAPTURE' if same else 'USING_EAGER_SOURCE',
            city=city,snapshot=snapshot,role=role,shard=args.shard,selected_epoch=result['selected_epoch'],
            complete_reverse_steps=1000,point_batch=ppg,registered_first_group=proof_group,
            full3_channel_max_abs_error=error,Gaussian_state_exact=bool(torch.equal(state,generator.get_state())),
            eager_reference_denoiser_images=ppg*1000,captured_comparison_denoiser_images=ppg*1000,
            additional_check_denoiser_images=ppg*1000*(2 if already_saved else 1),
            designated_first_sample_reused=not already_saved,existing_first_chunk=already_saved,
            graph_warmup_forward_images=ppg*3,graph_capture_forward_invocations=1,
            note='first proof image reused when its designated chunk is not already saved; no GT used; no changed DDPM transitions'))
    def sequential_samples(begin,end):
        for group in range(begin,end):
            lo,hi=group*ppg,(group+1)*ppg
            cond=torch.as_tensor(np.array(d.odt[lo:hi]),device=device)
            gid=int(d.query_ids[lo]//ppg)
            seed=ev.identity_seed(20260912,gid,city,city+'_'+snapshot+'_canonical_fifo_continuous',role)
            generator=torch.Generator(device=device).manual_seed(seed)
            yield group,(proof_image if group==proof_group else models.pit_sample(sampler,dp,cond,generator))
    # Storage shards contain 32 complete groups. Each PiT OD still uses its
    # prescribed separate 16/2 point batch and its own generator, never GT.
    try:
        for chunk in range(args.shard, (groups+31)//32, args.shards):
            begin=chunk*32;end=min(begin+32,groups);file=folder/f'chunk_{chunk:06d}.npz'
            if file.exists():
                with np.load(file) as old:
                    assert int(old['selected_epoch'])==result['selected_epoch']
                    assert np.array_equal(old['query_ids'],d.query_ids[begin*ppg:end*ppg])
                    assert old['scores'].shape==((end-begin)*ppg,4096)
                saved+=(end-begin)*ppg;continue
            t0=time.monotonic();scores=np.empty(((end-begin)*ppg,4096),np.float32)
            signed_min=None;signed_max=None;below=0;above=0
            with torch.no_grad():
                if method=='SSE':
                    lo,hi=begin*ppg,end*ppg
                    pairs=torch.as_tensor(np.array(d.pairs[lo:hi]),device=device)
                    endpoints=torch.as_tensor(np.array(d.endpoints[lo:hi]),device=device)
                    scores[:]=models.sse_scores(model,pairs,endpoints).cpu().numpy()
                else:
                    samples=executor.sample_range(begin,end) if executor is not None else sequential_samples(begin,end)
                    for group,image in samples:
                        assert torch.isfinite(image).all(),'nonfinite DDPM output before clipping'
                        raw=image[:,0].cpu().numpy();low=float(raw.min());high=float(raw.max())
                        signed_min=low if signed_min is None else min(signed_min,low)
                        signed_max=high if signed_max is None else max(signed_max,high)
                        below+=int((raw < -1).sum());above+=int((raw > 1).sum())
                        scores[(group-begin)*ppg:(group-begin+1)*ppg]=np.clip((image[:,0].cpu().numpy()+np.float32(1))/np.float32(2),0,1).reshape(ppg,4096)
            assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
            temporary=file.with_suffix('.tmp.npz')
            np.savez(temporary,scores=scores,query_ids=d.query_ids[begin*ppg:end*ppg],
                selected_epoch=np.int64(result['selected_epoch']),nfe=np.int64(1000 if method=='PiT' else 1),
                signed_min=np.float64(signed_min if signed_min is not None else 0),
                signed_max=np.float64(signed_max if signed_max is not None else 0),
                signed_below_minus_one=np.int64(below),signed_above_one=np.int64(above))
            temporary.replace(file);saved+=len(scores);generated+=len(scores)
            atomic(folder/f'progress_{args.shard:02d}.json',dict(status='GENERATING',city=city,snapshot=snapshot,
                method=method,role=role,shard=args.shard,shards=args.shards,saved_points=saved,
                newly_generated_points=generated,last_chunk=chunk,last_chunk_seconds=time.monotonic()-t0,
                elapsed_seconds=time.monotonic()-begun))
            print(json.dumps(dict(chunk=chunk,points=len(scores),saved=saved,seconds=time.monotonic()-t0)),flush=True)
        atomic(folder/f'complete_{args.shard:02d}.json',dict(status='SHARD_COMPLETE',shard=args.shard,
            shards=args.shards,saved_points=saved,newly_generated_points=generated,execution=execution,
            elapsed_seconds=time.monotonic()-begun))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def assemble(args):
    parents();out=RUN/'training'/f'{args.city}_{args.snapshot}_{args.method}_seed42'
    d=SparseData(args.city,args.snapshot,args.role);folder=out/(args.role+'_generation')
    for shard in range(args.shards):
        value=json.loads((folder/f'complete_{shard:02d}.json').read_text())
        assert value['status']=='SHARD_COMPLETE' and value['shards']==args.shards
    target=out/('calibration_predictions' if args.role in ('Vcal','Acal') else 'test_predictions')
    target.mkdir(exist_ok=True)
    scores=np.lib.format.open_memmap(target/'scores.tmp.npy',mode='w+',dtype=np.float32,shape=(len(d),4096))
    seen=np.zeros(len(d),bool);nfe=1000 if args.method=='PiT' else 1
    for chunk,lo in enumerate(range(0,len(d),32*d.ppg)):
        hi=min(len(d),lo+32*d.ppg)
        with np.load(folder/f'chunk_{chunk:06d}.npz') as part:
            assert np.array_equal(part['query_ids'],d.query_ids[lo:hi]) and int(part['nfe'])==nfe
            scores[lo:hi]=part['scores'];seen[lo:hi]=True
    assert seen.all();scores.flush();del scores
    (target/'scores.tmp.npy').replace(target/'scores.npy');np.save(target/'query_ids.npy',d.query_ids)
    atomic(target/'acceptance.json',dict(status='GENERATED_FULL',rows=len(d),groups=len(d)//d.ppg,
        points_per_OD=d.ppg,nfe=nfe,role=args.role,method=args.method,shards=args.shards,seed=42))
    if args.role in ('Vcal','Acal'):
        scores=np.load(target/'scores.npy',mmap_mode='r')
        summary=ev.exact_threshold_sweep(scores,d.truth(),d.endpoints,d.support);summary.pop('coarse_curve')
        # The shared SSP sweeper defaults its diagnostic point metadata to1.
        # Label every diagnostic with the actual method sampling budget.
        for point in ('best','max_q_point','empty_core_boundary'):
            summary[point]['nfe']=nfe
        atomic(out/'calibration.json',dict(status='THRESHOLD_FROZEN',threshold=summary['best']['threshold'],
            role=args.role,rows=len(d),summary=summary,nfe=nfe))
    else:
        from metrics import compute
        compute(args.city,args.snapshot,args.method)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--city',choices=['BJ','XA'],required=True)
    p.add_argument('--snapshot',choices=['G0','G1'],required=True);p.add_argument('--method',choices=['SSE','PiT'],required=True)
    p.add_argument('--role',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--assemble',action='store_true');a=p.parse_args();assert 0<=a.shard<a.shards
    assemble(a) if a.assemble else generate(a)
