"""Verify all formal runs and independently recount every saved test prediction."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import model64 as mt
from point_data import Dataset
from p0_inventory import atomic

METHODS=("cluster","uniform_grid","hierarchical_grid")
SEEDS=(42,)

def truth(run,city):
    dataset=Dataset(run,city,"base_test")
    ids=(dataset.group_ids[:,None]*16+np.arange(16)).reshape(-1)
    gt=np.empty((len(ids),4096),bool);endpoints=np.empty((len(ids),2),np.int64)
    seen=np.zeros(len(ids),bool)
    for batch in dataset.iter_batches(128):
        positions=np.searchsorted(ids,batch["query_id"])
        assert np.array_equal(ids[positions],batch["query_id"]) and not seen[positions].any()
        gt[positions]=batch["image"][:,0].reshape(-1,4096)>0
        endpoints[positions]=batch["endpoints"];seen[positions]=True
    assert seen.all()
    return ids,gt,endpoints,dataset.road_support

def audit(run):
    expected=[(city,method,seed) for city in ("BJ","XA") for method in METHODS for seed in SEEDS]
    missing=[dict(city=c,method=m,seed=s) for c,m,s in expected
             if not (run/"training"/f"{c}_{m}_seed{s}"/"result.json").exists()]
    if missing:
        result=dict(status="INCOMPLETE",required_runs=6,missing_runs=missing,
                    test_truth_not_opened=True,recorded_unix=time.time())
        atomic(run/"audits/completion_latest.json",result);return result
    torch.set_num_threads(2)
    torch.manual_seed(42)
    fresh_seed42=mt.DiT().state_dict()
    protocol=json.loads((run/"code/protocol.json").read_text())
    contract=json.loads((run/"code/reporting_protocol.json").read_text())
    assert contract["training_seeds"]==[42] and contract["required_runs"]==6
    cost_profile=json.loads((run/"cost_profiles/full_v1/summary.json").read_text())
    assert cost_profile["status"]=="PASSED" and cost_profile["additional_point_paths_computed"]==3752064
    rows=json.loads((run/"artifacts/e1_results.json").read_text())
    assert len(rows)==6
    reports=[];initials={}
    for city in ("BJ","XA"):
        folder=run/"data"/city
        assert json.loads((folder/"independent_acceptance.json").read_text())["status"]=="PASSED"
        measured=json.loads((run/"cost_profiles/full_v1"/(city+"_profile.json")).read_text())
        assert measured["status"]=="PASSED" and measured["all_frozen_routes_exactly_match_original"]
        assert measured["points"]==(2573136 if city=="BJ" else 1178928)
        plan=json.loads((folder/"plan.json").read_text())
        groups=np.load(folder/"groups.npy");cells=np.load(folder/"cell64.npy")
        eligible=np.load(folder/"train_eligible_64_cell_pairs.npy")
        assert np.array_equal(eligible,np.unique(eligible))
        assert len(eligible)==plan["common_train_eligible_ordered_64_cell_pairs"]
        qids,gt,endpoints,support=truth(run,city)
        assert len(qids)==(100000 if city=="BJ" else 50000)
        for method in METHODS:
            group_ids=np.load(folder/(method+"_train_group_ids.npy"));pairs=groups[group_ids]
            codes=np.unique(cells[pairs[:,0]]*4096+cells[pairs[:,1]])
            assert np.isin(codes,eligible).all()
            coverage=100*len(codes)/len(eligible)
            metrics=[]
            for seed in SEEDS:
                out=run/"training"/f"{city}_{method}_seed{seed}"
                result=json.loads((out/"result.json").read_text())
                assert result["status"]=="COMPLETE" and result["epochs"]==50
                assert (result["city"],result["method"],result["seed"])==(city,method,seed)
                config=json.loads((out/"config.json").read_text())
                assert (config["city"],config["method"],config["seed"])==(city,method,seed)
                assert config["protocol"]==protocol
                assert result["test"]["rows"]==result["test"]["metadata"]["rows"]==len(qids)
                assert result["test"]["metadata"]["role"]=="base_test"
                train_rows=800000 if city=="BJ" else 400000
                assert result["train_rows"]==train_rows==len(group_ids)*16
                assert result["actual_rows"]==50*train_rows
                assert result["global_update"]==50*(train_rows//64)
                selected=json.loads((out/"selection_result.json").read_text())
                cal=json.loads((out/"calibration.json").read_text())
                assert selected["selected_on"]=="Vselect" and cal["calibrated_on"]=="Vcal"
                assert selected["selected_epoch"]==cal["selected_epoch"]==result["selected_epoch"]
                assert cal["threshold"]==result["threshold"]
                assert cal["summary"]["metadata"]["role"]=="Vcal"
                assert cal["summary"]["metadata"]["rows"]==(50000 if city=="BJ" else 25008)
                records=[json.loads((out/"selection"/f"epoch_{e:03d}.json").read_text()) for e in range(5,51,5)]
                assert all(r["metadata"]["role"]=="Vselect" and
                           r["metadata"]["rows"]==(50000 if city=="BJ" else 24992) for r in records)
                maximum=max(r["max_q"] for r in records)
                candidates=[r for r in records if r["max_q"]>=maximum-1e-8]
                winner=sorted(candidates,key=lambda r:(r["best"]["mean_predicted_cells"],
                    -r["best"]["precision_micro"],r["epoch"]))[0]
                assert winner["epoch"]==selected["selected_epoch"]
                weights=torch.load(out/"selected.pt",map_location="cpu",weights_only=False)
                epoch_weights=torch.load(out/f"epoch_{winner['epoch']:03d}.pt",map_location="cpu",weights_only=False)
                assert weights["config"]["protocol"]==protocol
                assert (weights["config"]["city"],weights["config"]["method"],weights["config"]["seed"])==(city,method,seed)
                assert set(weights["model"])==set(epoch_weights["model"])==set(fresh_seed42)
                assert all(v.shape==fresh_seed42[k].shape and torch.isfinite(v).all() for k,v in weights["model"].items())
                assert all(torch.equal(weights["model"][k],v) for k,v in epoch_weights["model"].items())
                initial=torch.load(out/"initial.pt",map_location="cpu",weights_only=False)
                assert initial["seed"]==42 and initial["actual_rows"]==0
                assert initial["random_initialization"] and initial["optimizer_state_empty"]
                assert set(initial["model"])==set(fresh_seed42)
                assert all(torch.equal(initial["model"][k],v) for k,v in fresh_seed42.items())
                key=(city,seed)
                if key in initials:assert all(torch.equal(initials[key][k],v) for k,v in initial["model"].items())
                else:initials[key]=initial["model"]
                stored_ids=np.load(out/"test_predictions/query_ids.npy")
                assert np.array_equal(stored_ids,qids)
                scores=np.load(out/"test_predictions/scores.npy",mmap_mode="r")
                assert scores.shape==(len(qids),4096) and scores.dtype==np.float32
                counts=np.empty((len(qids),3),np.int64)
                for begin in range(0,len(qids),1024):
                    end=min(len(qids),begin+1024);block=scores[begin:end]
                    assert np.isfinite(block).all() and np.all((block>=0)&(block<=1))
                    predicted=(block>=cal["threshold"])&support[None,:]
                    predicted[np.arange(end-begin)[:,None],endpoints[begin:end]]=True
                    tp=(predicted & gt[begin:end]).sum(1,dtype=np.int64)
                    counts[begin:end,0]=tp
                    counts[begin:end,1]=predicted.sum(1,dtype=np.int64)-tp
                    counts[begin:end,2]=gt[begin:end].sum(1,dtype=np.int64)-tp
                assert np.array_equal(counts.reshape(-1,16,3).sum(1),np.load(out/"test_predictions/group_counts.npy"))
                tp,fp,fn=map(int,counts.sum(0))
                assert [tp,fp,fn]==[result["test"][k] for k in ("tp","fp","fn")]
                values=np.array([tp/(tp+fp),tp/(tp+fn),2*tp/(2*tp+fp+fn)])*100
                metrics.append(values)
                reports.append(dict(city=city,method=method,seed=seed,status="PASSED",test_points=len(qids),
                    test_groups=len(qids)//16,tp=tp,fp=fp,fn=fn,selected_epoch=winner["epoch"],
                    full_saved_test_predictions_recounted=True,paired_initialization_verified=True,
                    fresh_seed42_initialization_independently_recreated=True,configuration_identity_verified=True))
                print(json.dumps(reports[-1]),flush=True)
            row=next(r for r in rows if r["city"]==city and r["method"]==method)
            assert row["od_coverage_numerator"]==len(codes) and row["od_coverage_denominator"]==len(eligible)
            assert abs(row["od_coverage_percent"]-coverage)<1e-12
            assert len(metrics)==1 and row["training_seed"]==42 and row["training_seeds"]==1
            metric=metrics[0]
            for j,name in enumerate(("precision","recall","f1")):
                assert abs(row[name+"_percent"]-metric[j])<1e-10
    assert len(reports)==6 and sum(r["test_points"] for r in reports)==450000
    result=dict(status="PASSED",verified_runs=len(reports),required_runs=6,verified_rows=6,
        full_test_point_predictions_recounted=sum(r["test_points"] for r in reports),
        common_test_counts={"BJ":100000,"XA":50000},reports=reports,no_content_hashes=True,
        recorded_unix=time.time(),latex_compilation_not_covered_by_this_audit=True)
    atomic(run/"audits/completion_latest.json",result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--require-complete",action="store_true");a=p.parse_args();result=audit(a.run)
    print(json.dumps(result),flush=True)
    if a.require_complete:assert result["status"]=="PASSED"

if __name__=="__main__":main()
