"""Audit real saved recovery states and paired starts without changing runs."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import model64 as mt
from p0_inventory import atomic

def audit(run):
    torch.set_num_threads(2)
    expected_model=mt.DiT().state_dict()
    reports=[];initials={};vselect_truth={}
    for path in sorted((run/"training").glob("*/last.pt")):
        saved=torch.load(path,map_location="cpu",weights_only=False)
        cfg=saved["config"];state=saved["state"];city=cfg["city"]
        n=800000 if city=="BJ" else 400000
        assert saved["train_rows"]==n and saved["batch_size"]==64
        assert saved["total_updates"]==50*(n//64)
        epoch,cursor,update=(state[k] for k in ("epoch","cursor","global_update"))
        assert 0<=epoch<=50 and 0<=cursor<=n and cursor%64==0
        assert state["actual_rows"]==epoch*n+cursor
        assert update==epoch*(n//64)+cursor//64
        perm=saved["sampler_permutation"]
        if perm is None:assert cursor==0
        else:assert np.array_equal(np.sort(perm),np.arange(n))
        assert set(saved["model"])==set(expected_model)
        for key,value in saved["model"].items():
            assert value.shape==expected_model[key].shape and value.dtype==torch.float32
            assert bool(torch.isfinite(value).all())
        assert tuple(saved["model"]["pos"].shape)==(1,256,128)
        optimizer=saved["optimizer"]
        for group in optimizer["param_groups"]:
            assert tuple(group["betas"])==(.9,.95) and group["weight_decay"]==.01
            if update:
                rate=mt.lr_at_update(update,3e-4,saved["total_updates"],"cosine",100)
                assert abs(group["lr"]-rate)<1e-15
        for value in optimizer["state"].values():
            assert int(value["step"].item())==update
            assert torch.isfinite(value["exp_avg"]).all() and torch.isfinite(value["exp_avg_sq"]).all()
        assert update==0 or len(optimizer["state"])>0
        assert set(saved["rng"])=={"python","numpy","torch","cuda"}
        assert len(saved["rng"]["cuda"])==1
        assert saved["ng"].dtype==saved["tg"].dtype==torch.uint8
        for e in range(1,epoch+1):
            ledger=json.loads((path.parent/"epochs"/f"epoch_{e:03d}.json").read_text())
            assert ledger["rows"]==ledger["permutation_rows"]==n
            assert ledger["actual_rows"]==e*n and ledger["global_update"]==e*(n//64)
            assert math.isfinite(ledger["mean_loss"])
        initial=torch.load(path.parent/"initial.pt",map_location="cpu",weights_only=False)
        assert initial["seed"]==cfg["seed"] and initial["actual_rows"]==0
        assert initial["optimizer_state_empty"] and initial["random_initialization"]
        pair=(city,cfg["seed"])
        if pair in initials:
            assert all(torch.equal(initials[pair][k],v) for k,v in initial["model"].items())
        else:initials[pair]=initial["model"]
        selections=[]
        for selection in state["selection"]:
            expected=50000 if city=="BJ" else 24992
            assert selection["metadata"]["role"]=="Vselect"
            assert selection["metadata"]["rows"]==selection["best"]["rows"]==expected
            assert selection["nfe"]==1 and selection["prediction_batch"]==16
            truth=selection["best"]["gt_cells"]
            if city in vselect_truth:assert vselect_truth[city]==truth
            else:vselect_truth[city]=truth
            selections.append(selection["epoch"])
        reports.append(dict(name=path.parent.name,status="PASSED",epoch=epoch,cursor=cursor,
            global_update=update,total_updates=saved["total_updates"],actual_rows=state["actual_rows"],
            train_rows=n,full_recovery_fields=True,full_current_permutation=True,
            optimizer_steps_match=True,completed_epoch_ledgers=epoch,full_Vselect_evaluations=selections))
    different_seed_checks=[]
    for city in ("BJ","XA"):
        seeds=sorted(seed for c,seed in initials if c==city)
        for i,seed in enumerate(seeds):
            for other in seeds[i+1:]:
                assert not torch.equal(initials[(city,seed)]["pos"],initials[(city,other)]["pos"])
                different_seed_checks.append(dict(city=city,seeds=[seed,other],distinct=True))
    result=dict(status="PASSED",recorded_unix=time.time(),runs=reports,
        paired_initialization_groups=[dict(city=city,seed=seed) for city,seed in initials],
        paired_initial_weights_identical=True,common_Vselect_gt_cell_counts=vselect_truth,
        different_seed_initial_weights=different_seed_checks,
        no_training_state_changed=True,no_content_hashes=True)
    atomic(run/"p1"/f"training_state_audit_{time.time_ns()}.json",result)
    print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();audit(a.run)

if __name__=="__main__":main()
