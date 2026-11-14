"""Accepted E1 mathematics plus complete two-departure G1 inference."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from g1_data import Dataset


def bindings(run):
    config=json.loads((Path(run)/'protocol.json').read_text());parent=Path(config['parent_e1_run'])
    sys.path.insert(0,str(parent/'code'))
    import model64 as mt
    import evaluator64 as ev
    assert Path(mt.__file__).resolve()==(parent/'code/model64.py').resolve()
    assert Path(ev.__file__).resolve()==(parent/'code/evaluator64.py').resolve()
    return config,parent,mt,ev


def preload(run,city,role,device):
    dataset=Dataset(run,city,role);n=len(dataset);qids=dataset.query_ids
    images=torch.empty((n,2,64,64),dtype=torch.float32,device=device)
    odt=torch.empty((n,5),dtype=torch.float32,device=device)
    endpoints=np.empty((n,2),np.int64);seen=np.zeros(n,bool)
    for batch in dataset.iter_batches(128):
        positions=np.searchsorted(qids,batch['query_id'])
        assert np.array_equal(qids[positions],batch['query_id']) and not seen[positions].any()
        target=torch.from_numpy(positions).to(device)
        images[target]=torch.from_numpy(batch['image']).to(device)
        odt[target]=torch.from_numpy(batch['odt']).to(device)
        endpoints[positions]=batch['endpoints'];seen[positions]=True
    assert seen.all();torch.cuda.synchronize()
    return dict(image=images,odt=odt,query_ids=qids,endpoints=endpoints,support=dataset.road_support,
        metadata=dataset.metadata,affected_groups=dataset.affected_groups,
        affected_points=dataset.point_flags[:,:,-1].reshape(-1).copy())


def select(records):
    maximum=max(r['max_q'] for r in records)
    return min((r for r in records if r['max_q']>=maximum-1e-8),
        key=lambda r:(r['best']['mean_predicted_cells'],-r['best']['precision_micro'],r['epoch']))


def point_counts(scores,gt,endpoints,support,threshold):
    count=np.empty((len(scores),3),np.int64)
    for start in range(0,len(scores),512):
        end=min(len(scores),start+512)
        pred=(scores[start:end]>=threshold)&support[None,:]
        pred[np.arange(end-start)[:,None],endpoints[start:end]]=True
        tp=(pred & gt[start:end]).sum(1,dtype=np.int64)
        count[start:end,0]=tp
        count[start:end,1]=pred.sum(1,dtype=np.int64)-tp
        count[start:end,2]=gt[start:end].sum(1,dtype=np.int64)-tp
    return count


def summarize_counts(counts,threshold,ev):
    tp,fp,fn=map(int,counts.sum(0));n=len(counts)
    result=ev._point(tp,tp+fp,tp+fn,n,threshold)
    result.update(tp=tp,fp=fp,fn=fn,rows=n)
    return result


@torch.no_grad()
def evaluate(model,data,run,device,threshold=None,save_dir=None,frozen_threshold=None):
    _,_,mt,ev=bindings(run);n=len(data['query_ids'])
    scores=np.empty((n,4096),np.float32);gt=np.empty((n,4096),bool)
    training=model.training;model.eval()
    try:
        for start in range(0,n,2):
            end=start+2;assert data['query_ids'][start]//2==data['query_ids'][end-1]//2
            noise=ev.fixed_query_noise(data['query_ids'][start:end],eval_seed=20260912,
                city=data['metadata']['city'],snapshot_id=data['metadata']['snapshot_id'],
                role=data['metadata']['role'],shape=(2,64,64))
            result=mt.sample(model,data['odt'][start:end],torch.from_numpy(noise).to(device),nfe=1)
            if not bool(torch.isfinite(result).all()):raise FloatingPointError('nonfinite G1 prediction')
            scores[start:end]=np.clip((result[:,0].cpu().numpy()+np.float32(1))/np.float32(2),0,1).reshape(-1,4096)
            gt[start:end]=(data['image'][start:end,0]>0).cpu().numpy().reshape(-1,4096)
        if save_dir:
            save_dir=Path(save_dir);save_dir.mkdir(parents=True,exist_ok=True)
            np.save(save_dir/'scores.npy',scores);np.save(save_dir/'query_ids.npy',data['query_ids'])
        if threshold is None:
            summary=ev.exact_threshold_sweep(scores,gt,data['endpoints'],data['support'])
            summary={k:v for k,v in summary.items() if k!='coarse_curve'}
        else:
            counts=point_counts(scores,gt,data['endpoints'],data['support'],threshold)
            full=summarize_counts(counts,threshold,ev)
            affected=np.repeat(data['affected_groups'],2)
            primary=summarize_counts(counts[affected],threshold,ev)
            only=summarize_counts(counts[data['affected_points']],threshold,ev)
            summary=dict(full_mixed_test=full,affected_group_test=primary,affected_points_only=only,
                affected_groups=int(data['affected_groups'].sum()),affected_point_rows=int(data['affected_points'].sum()),
                main_role='common affected complete groups retaining both departures')
            if save_dir:
                np.save(save_dir/'point_counts.npy',counts);np.save(save_dir/'group_counts.npy',counts.reshape(-1,2,3).sum(1))
                np.save(save_dir/'affected_group_mask.npy',data['affected_groups'])
                np.save(save_dir/'affected_point_mask.npy',data['affected_points'])
            if frozen_threshold is not None:
                frozen=point_counts(scores,gt,data['endpoints'],data['support'],frozen_threshold)
                summary['fully_frozen_NoFT_diagnostic']=dict(
                    full_mixed_test=summarize_counts(frozen,frozen_threshold,ev),
                    affected_group_test=summarize_counts(frozen[affected],frozen_threshold,ev),
                    affected_points_only=summarize_counts(frozen[data['affected_points']],frozen_threshold,ev),
                    G1_calibration_labels=0,reused_same_raw_predictions=True)
                if save_dir:np.save(save_dir/'frozen_threshold_point_counts.npy',frozen)
        summary.update(metadata=data['metadata'],nfe=1,prediction_batch=2,
            road_support_and_endpoint_policy='common support filtering and known endpoint insertion')
        return summary
    finally:model.train(training)
