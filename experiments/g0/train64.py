"""Full-budget paired-seed training, Vselect selection, Vcal calibration, test."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
import model64 as mt
import evaluator64 as ev
from point_data import Dataset
from p0_inventory import atomic

def save_pt(path,payload):
    tmp=path.with_suffix(".tmp.pt");torch.save(payload,tmp);tmp.replace(path)

def preload(run,city,role,device):
    data=Dataset(run,city,role);n=len(data)
    x=torch.empty((n,2,64,64),dtype=torch.float32,device=device)
    cond=torch.empty((n,5),dtype=torch.float32,device=device)
    qids=(data.group_ids[:,None]*16+np.arange(16)[None,:]).reshape(-1)
    endpoints=np.empty((n,2),np.int64);seen=np.zeros(n,bool)
    for batch in data.iter_batches(128):
        idx=np.searchsorted(qids,batch["query_id"])
        assert np.array_equal(qids[idx],batch["query_id"]) and not seen[idx].any()
        target=torch.from_numpy(idx).to(device)
        x[target]=torch.from_numpy(batch["image"]).to(device)
        cond[target]=torch.from_numpy(batch["odt"]).to(device)
        endpoints[idx]=batch["endpoints"];seen[idx]=True
    assert seen.all()
    torch.cuda.synchronize()
    return dict(image=x,odt=cond,query_ids=qids,endpoints=endpoints,support=data.road_support,
                metadata=data.metadata)

@torch.no_grad()
def evaluate(model,data,device,threshold=None,save_dir=None):
    n=len(data["image"]);scores=np.empty((n,4096),np.float32)
    gt=np.empty((n,4096),bool)
    old_training=model.training;model.eval()
    try:
        for begin in range(0,n,16):
            end=min(n,begin+16)
            assert data["query_ids"][begin]//16==data["query_ids"][end-1]//16
            noise=ev.fixed_query_noise(data["query_ids"][begin:end],eval_seed=20260912,
                city=data["metadata"]["city"],snapshot_id=data["metadata"]["snapshot_id"],
                role=data["metadata"]["role"],shape=(2,64,64))
            generated=mt.sample(model,data["odt"][begin:end],torch.from_numpy(noise).to(device),nfe=1)
            if not bool(torch.isfinite(generated).all()):raise FloatingPointError("nonfinite prediction")
            signed=generated[:,0].cpu().numpy()
            scores[begin:end]=np.clip((signed+np.float32(1))/np.float32(2),0,1).reshape(-1,4096)
            gt[begin:end]=(data["image"][begin:end,0]>0).cpu().numpy().reshape(-1,4096)
        if threshold is None:
            sweep=ev.exact_threshold_sweep(scores,gt,data["endpoints"],data["support"])
            summary={k:v for k,v in sweep.items() if k!="coarse_curve"}
        else:
            pred=(scores>=threshold)&data["support"][None,:]
            pred[np.arange(n)[:,None],data["endpoints"]]=True
            tp=int((pred & gt).sum());npred=int(pred.sum());ngt=int(gt.sum())
            summary=ev._point(tp,npred,ngt,n,threshold)
            per_group=ev._count_rows(pred,gt).reshape(-1,16,3).sum(1)
            if save_dir:
                save_dir.mkdir(parents=True,exist_ok=True)
                np.save(save_dir/"group_counts.npy",per_group)
                np.save(save_dir/"query_ids.npy",data["query_ids"])
                # SSP G0/E3 can reuse these genuine point predictions.
                np.save(save_dir/"scores.npy",scores)
        summary.update(metadata=data["metadata"],nfe=1,prediction_batch=16)
        return summary
    finally:model.train(old_training)

def select(records):
    maximum=max(r["max_q"] for r in records)
    return min((r for r in records if r["max_q"]>=maximum-1e-8),
        key=lambda r:(r["best"]["mean_predicted_cells"],-r["best"]["precision_micro"],r["epoch"]))

def run_train(args):
    run=args.run;city=args.city;method=args.method;seed=args.seed
    name=f"{city}_{method}_seed{seed}";out=run/"training"/name;out.mkdir(parents=True,exist_ok=True)
    lock=(out/"worker.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((run/"code/protocol.json").read_text())
    generation=json.loads((run/"data"/city/"generation_report.json").read_text())
    assert generation["status"]=="GENERATED_FULL"
    assert json.loads((run/"data"/city/"independent_acceptance.json").read_text())["status"]=="PASSED"
    assert json.loads((run/"p1/ssp64_checks.json").read_text())["status"]=="PASSED"
    assert seed in (42,43,44) and method in protocol["samplers"]
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True;device="cuda:0"
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=mt.DiT().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01)
    ng=torch.Generator(device=device).manual_seed(seed+200)
    tg=torch.Generator(device=device).manual_seed(seed+300)
    started=time.monotonic();carried=0.
    config=dict(city=city,method=method,seed=seed,protocol=protocol,
                gpu_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),run_dir=str(out))
    if (out/"config.json").exists():
        saved_cfg=json.loads((out/"config.json").read_text())
        assert {k:v for k,v in saved_cfg.items() if k!="gpu_visible"}=={k:v for k,v in config.items() if k!="gpu_visible"}
    else:atomic(out/"config.json",config)
    initial_rng=mt._rng_state()
    atomic(out/"progress.json",dict(status="PRELOADING",city=city,method=method,seed=seed))
    train=preload(run,city,method+"_train",device)
    valid=preload(run,city,"Vselect",device)
    mt._restore_rng(initial_rng)
    n=len(train["image"]);assert n==protocol["training_points_per_method"][city]
    assert len(valid["image"])==protocol["Vselect_points"][city]
    batches=math.ceil(n/64);total_updates=batches*50
    state=dict(epoch=0,cursor=0,global_update=0,actual_rows=0,epoch_loss_sum=0.,selection=[],elapsed_seconds=0.)
    permutation=None
    def checkpoint():
        state["elapsed_seconds"]=carried+time.monotonic()-started
        payload=dict(config=config,state=state.copy(),model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_permutation=permutation,ng=ng.get_state(),tg=tg.get_state(),rng=mt._rng_state(),
            total_updates=total_updates,train_rows=n,batch_size=64)
        save_pt(out/"last.pt",payload)
        atomic(out/"progress.json",dict(status="RUNNING",city=city,method=method,seed=seed,
            total_updates=total_updates,train_rows=n,**state,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    if args.resume:
        saved=torch.load(out/"last.pt",map_location=device,weights_only=False)
        assert saved["total_updates"]==total_updates and saved["train_rows"]==n
        assert saved["config"]["seed"]==seed and saved["config"]["method"]==method
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        state=saved["state"];permutation=saved["sampler_permutation"]
        ng.set_state(saved["ng"].cpu());tg.set_state(saved["tg"].cpu());mt._restore_rng(saved["rng"])
        carried=state["elapsed_seconds"];started=time.monotonic()
    else:
        assert not (out/"last.pt").exists(),"Explicit own-run resume required"
        checkpoint();save_pt(out/"initial.pt",dict(model=model.state_dict(),seed=seed,actual_rows=0,
            optimizer_state_empty=not optimizer.state,random_initialization=True,rng=initial_rng))
    def terminate(signum,frame):raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        model.train()
        while state["epoch"]<50:
            epoch=state["epoch"]
            if permutation is None:
                permutation=np.random.default_rng(np.random.SeedSequence([seed+100,epoch])).permutation(n)
                assert state["cursor"]==0
            while state["cursor"]<n:
                end=min(n,state["cursor"]+64)
                idx=torch.from_numpy(permutation[state["cursor"]:end]).to(device)
                image=train["image"][idx];cond=train["odt"][idx]
                noise,times=mt.draw_noise_times(image,ng,tg)
                optimizer.zero_grad(set_to_none=True)
                loss=mt.cfm_loss(model,image,cond,noise,times,10)
                if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite loss")
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError("nonfinite gradient")
                update=state["global_update"]+1
                lr=mt.lr_at_update(update,3e-4,total_updates,"cosine",100)
                for g in optimizer.param_groups:g["lr"]=lr
                optimizer.step()
                state["actual_rows"]+=end-state["cursor"]
                state["epoch_loss_sum"]+=float(loss.detach())*(end-state["cursor"])
                state.update(cursor=end,global_update=update)
                if update%500==0:checkpoint()
            assert state["cursor"]==n and state["actual_rows"]==(epoch+1)*n
            ledger=dict(epoch=epoch+1,rows=n,global_update=state["global_update"],
                actual_rows=state["actual_rows"],mean_loss=state["epoch_loss_sum"]/n,
                permutation_rows=n,all_rows_once=True,lr=lr)
            if (epoch+1)%5==0:
                with mt._evaluation_context(model):summary=evaluate(model,valid,device)
                record=dict(epoch=epoch+1,**summary);state["selection"].append(record)
                atomic(out/"selection"/f"epoch_{epoch+1:03d}.json",record)
                save_pt(out/f"epoch_{epoch+1:03d}.pt",dict(model=model.state_dict(),epoch=epoch+1,
                    config=config,summary=record))
            atomic(out/"epochs"/f"epoch_{epoch+1:03d}.json",ledger)
            state.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.)
            permutation=None;checkpoint()
        assert state["global_update"]==total_updates and state["actual_rows"]==50*n
        chosen=select(state["selection"])
        selected=torch.load(out/f"epoch_{chosen['epoch']:03d}.pt",map_location=device,weights_only=False)
        model.load_state_dict(selected["model"]);save_pt(out/"selected.pt",selected)
        atomic(out/"selection_result.json",dict(status="CHECKPOINT_SELECTED",selected_epoch=chosen["epoch"],
            selected_on="Vselect",criterion=protocol["primary_selection_metric"],record=chosen))
        del train,valid
        torch.cuda.empty_cache()
        calibration=preload(run,city,"Vcal",device)
        cal=evaluate(model,calibration,device);threshold=cal["best"]["threshold"]
        atomic(out/"calibration.json",dict(status="THRESHOLD_FROZEN",threshold=threshold,
            calibrated_on="Vcal",selected_epoch=chosen["epoch"],summary=cal))
        del calibration;torch.cuda.empty_cache()
        test=preload(run,city,"base_test",device)
        metrics=evaluate(model,test,device,threshold=threshold,save_dir=out/"test_predictions")
        atomic(out/"test_metrics.json",metrics)
        result=dict(status="COMPLETE",city=city,method=method,seed=seed,epochs=50,
            global_update=state["global_update"],actual_rows=state["actual_rows"],train_rows=n,
            selected_epoch=chosen["epoch"],threshold=threshold,test=metrics,
            elapsed_seconds=carried+time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic(out/"result.json",result);atomic(out/"progress.json",result)
        return result
    except BaseException as e:
        atomic(out/"failure.json",dict(status="FAILED",error=repr(e),traceback=traceback.format_exc(),state=state))
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--city",choices=["BJ","XA"],required=True)
    p.add_argument("--method",choices=["cluster","uniform_grid","hierarchical_grid"],required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--resume",action="store_true")
    a=p.parse_args();print(json.dumps(run_train(a)),flush=True)

if __name__=="__main__":main()
