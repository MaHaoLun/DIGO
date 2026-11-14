"""Full E2 prediction, calibration, selection, optimizer and parent-lineage audit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from g1_data import Dataset
from g1_runtime import bindings
from gt_cache import atomic_json

METHODS=('no_ft','random_ft','sdc_mixed_ft')


def truth(run,city,role):
    d=Dataset(run,city,role);qids=d.query_ids
    gt=np.empty((len(qids),4096),bool);endpoints=np.empty((len(qids),2),np.int64);seen=np.zeros(len(qids),bool)
    for batch in d.iter_batches(128):
        positions=np.searchsorted(qids,batch['query_id'])
        assert np.array_equal(qids[positions],batch['query_id']) and not seen[positions].any()
        gt[positions]=batch['image'][:,0].reshape(-1,4096)>0
        endpoints[positions]=batch['endpoints'];seen[positions]=True
    assert seen.all()
    return dict(dataset=d,query_ids=qids,gt=gt,endpoints=endpoints,support=d.road_support,
        group_mask=d.affected_groups,point_mask=d.point_flags[:,:,-1].reshape(-1))


def predictions(path,data):
    assert np.array_equal(np.load(path/'query_ids.npy'),data['query_ids'])
    scores=np.load(path/'scores.npy',mmap_mode='r')
    assert scores.shape==(len(data['query_ids']),4096) and scores.dtype==np.float32
    for start in range(0,len(scores),512):
        x=scores[start:start+512];assert np.isfinite(x).all() and np.all((x>=0)&(x<=1))
    return scores


def recount(scores,data,threshold):
    count=np.empty((len(scores),3),np.int64)
    for start in range(0,len(scores),512):
        end=min(len(scores),start+512);mask=scores[start:end]>=threshold
        mask[:,~data['support']]=False
        for j in range(end-start):mask[j,data['endpoints'][start+j]]=True
        actual=data['gt'][start:end]
        tp=np.logical_and(mask,actual).sum(1,dtype=np.int64)
        count[start:end,0]=tp;count[start:end,1]=mask.sum(1,dtype=np.int64)-tp
        count[start:end,2]=actual.sum(1,dtype=np.int64)-tp
    return count


def check_summary(counts,summary):
    tp,fp,fn=map(int,counts.sum(0));assert summary['rows']==len(counts)
    assert [summary[k] for k in ('tp','fp','fn')]==[tp,fp,fn]
    expected=[tp/(tp+fp),tp/(tp+fn),2*tp/(2*tp+fp+fn)]
    assert np.allclose(expected,[summary[k] for k in ('precision_micro','recall_micro','f1_micro')],rtol=0,atol=1e-12)


def check_sweep(scores,data,stored,ev):
    rebuilt=ev.exact_threshold_sweep(scores,data['gt'],data['endpoints'],data['support'])
    assert abs(rebuilt['max_q']-stored['max_q'])<1e-12
    assert rebuilt['best']['threshold']==stored['best']['threshold']
    for key in ('rows','tp','fp','fn','gt_cells'):
        assert rebuilt['best'][key]==stored['best'][key]
    return rebuilt


def sampled_replay(mt,ev,weights,scores,data,device):
    d=data['dataset'];n=len(data['query_ids'])
    gids=np.repeat(d.gids,2);pairs=np.column_stack((gids//d.n,gids % d.n))
    from p0_inventory import BBOX
    box=BBOX[d.city];lo=np.asarray([box[0],box[2]]);span=np.asarray([box[1]-box[0],box[3]-box[2]])
    odt=np.empty((n,5),np.float32)
    odt[:,:2]=2*(d.coords[pairs[:,0],1:]-lo)/span-1;odt[:,2:4]=2*(d.coords[pairs[:,1],1:]-lo)/span-1
    odt[:,4]=2*np.tile([43200,44100],len(d.gids))/86400-1
    full=torch.from_numpy(odt).to(device)
    model=mt.DiT().to(device);model.load_state_dict(weights,strict=True);model.eval();errors=[]
    with torch.no_grad():
        for group in (0,len(d.gids)//2,len(d.gids)-1):
            start=2*group;qids=data['query_ids'][start:start+2]
            noise=ev.fixed_query_noise(qids,eval_seed=20260912,city=d.city,
                snapshot_id=d.metadata['snapshot_id'],role='mixed_test',shape=(2,64,64))
            actual=mt.sample(model,full[start:start+2],torch.from_numpy(noise).to(device),nfe=1)
            actual=np.clip((actual[:,0].cpu().numpy()+np.float32(1))/np.float32(2),0,1).reshape(2,4096)
            error=float(np.max(np.abs(actual-scores[start:start+2])));assert error<=1e-5
            errors.append(error)
    del model,full;torch.cuda.empty_cache()
    return dict(point_predictions=6,max_absolute_error=max(errors),tolerance=1e-5,
        scope='first,middle,last complete two-departure OD groups; numerical model/output binding sample')


def audit(run,device):
    config,parent,mt,ev=bindings(run);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    gate=json.loads((run/'data_gate.json').read_text());assert gate['status']=='BOTH_CITIES_DATA_ACCEPTED' and gate['protocol']==config
    expected=[run/'training'/f'{c}_{m}_seed42'/'result.json' for c in ('BJ','XA') for m in METHODS]
    assert all(p.exists() for p in expected), 'All six full strategy results are required'
    reports=[];test_total=cal_total=select_total=0;initials={};started=time.monotonic();cpu=time.process_time()
    for city in ('BJ','XA'):
        budget=config['budgets'][city]
        truths={role:truth(run,city,role) for role in ('Aselect','Acal','mixed_test')}
        base_path=parent/'training'/f'{city}_hierarchical_grid_seed42'/'selected.pt'
        base=torch.load(base_path,map_location='cpu',weights_only=False)
        old_threshold=json.loads((base_path.parent/'calibration.json').read_text())['threshold']
        assert base['config']['seed']==42 and base['epoch']==50
        for method in METHODS:
            out=run/'training'/f'{city}_{method}_seed42'
            r=json.loads((out/'result.json').read_text());cfg=json.loads((out/'config.json').read_text())
            assert r['status']=='COMPLETE' and (r['city'],r['method'],r['seed'])==(city,method,42)
            assert cfg['protocol']==config and cfg['parent_checkpoint']==str(base_path)
            assert (cfg['city'],cfg['method'],cfg['seed'])==(city,method,42)
            selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
            assert selected['config']['protocol']==config
            if method=='no_ft':
                assert all(r[k]==0 for k in ('epochs','train_rows','actual_rows','global_update','selected_epoch'))
                assert set(selected['model'])==set(base['model'])
                assert all(torch.equal(selected['model'][k],v) for k,v in base['model'].items())
                frozen=json.loads((out/'frozen_NoFT_reference.json').read_text())
                assert frozen['original_G0_threshold']==old_threshold and frozen['formal_optimizer_updates']==0
            else:
                n=budget['Afit_points_per_FT'];total=budget['updates_per_FT']
                assert r['epochs']==5 and r['train_rows']==n and r['global_update']==total and r['actual_rows']==5*n
                last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);state=last['state']
                assert state['epoch']==5 and state['cursor']==0 and state['actual_rows']==5*n and state['global_update']==total
                assert last['total_updates']==total and last['train_rows']==n and last['batch_size']==64 and last['sampler_permutation'] is None
                assert set(last['rng'])=={'python','numpy','torch','cuda'} and len(last['rng']['cuda'])==1
                assert last['ng'].dtype==last['tg'].dtype==torch.uint8
                for group in last['optimizer']['param_groups']:
                    assert tuple(group['betas'])==(.9,.95) and group['weight_decay']==.01 and group['eps']==1e-8
                    assert abs(group['lr']-mt.lr_at_update(total,1e-5,total,'cosine',100))<1e-15
                assert last['optimizer']['state']
                for moment in last['optimizer']['state'].values():
                    assert int(moment['step'])==total and torch.isfinite(moment['exp_avg']).all() and torch.isfinite(moment['exp_avg_sq']).all()
                for epoch in range(1,6):
                    ledger=json.loads((out/'epochs'/f'epoch_{epoch:03d}.json').read_text())
                    assert ledger['rows']==ledger['permutation_rows']==n and ledger['all_rows_once']
                    assert ledger['actual_rows']==epoch*n and ledger['global_update']==epoch*(n//64)
                    assert np.isfinite(ledger['mean_loss'])
                initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
                assert initial['copied_exact_parent_weights'] and initial['optimizer_state_empty'] and initial['optimizer_reset']
                assert initial['global_update']==initial['actual_rows']==0
                assert all(torch.equal(initial['model'][k],v) for k,v in base['model'].items())
                if city in initials:
                    assert torch.equal(initial['ng'],initials[city]['ng']) and torch.equal(initial['tg'],initials[city]['tg'])
                else:initials[city]=initial
                records=[]
                for epoch in range(1,6):
                    record=json.loads((out/'selection'/f'epoch_{epoch:03d}.json').read_text())
                    assert record['epoch']==epoch and record['metadata']['role']=='Aselect'
                    assert record['metadata']['rows']==budget['Aselect_points'] and record['nfe']==1 and record['prediction_batch']==2
                    scores=predictions(out/'selection_predictions'/f'epoch_{epoch:03d}',truths['Aselect'])
                    check_sweep(scores,truths['Aselect'],record,ev);records.append(record);select_total+=len(scores)
                maximum=max(q['max_q'] for q in records)
                winner=min((q for q in records if q['max_q']>=maximum-1e-8),
                    key=lambda q:(q['best']['mean_predicted_cells'],-q['best']['precision_micro'],q['epoch']))
                assert winner['epoch']==r['selected_epoch']==selected['epoch']
                selection=json.loads((out/'selection_result.json').read_text());assert selection['selected_on']=='Aselect' and selection['selected_epoch']==winner['epoch']
                epoch_model=torch.load(out/f"epoch_{winner['epoch']:03d}.pt",map_location='cpu',weights_only=False)
                assert all(torch.equal(selected['model'][k],v) for k,v in epoch_model['model'].items())
            cal=json.loads((out/'calibration.json').read_text());assert cal['calibrated_on']=='Acal' and cal['threshold']==r['threshold']
            assert cal['summary']['metadata']['role']=='Acal' and cal['summary']['metadata']['rows']==budget['Acal_points']
            assert cal['summary']['nfe']==1 and cal['summary']['prediction_batch']==2
            scores=predictions(out/'calibration_predictions',truths['Acal']);check_sweep(scores,truths['Acal'],cal['summary'],ev);cal_total+=len(scores)
            test=truths['mixed_test'];scores=predictions(out/'test_predictions',test);counts=recount(scores,test,r['threshold'])
            assert np.array_equal(counts,np.load(out/'test_predictions/point_counts.npy'))
            assert np.array_equal(counts.reshape(-1,2,3).sum(1),np.load(out/'test_predictions/group_counts.npy'))
            assert np.array_equal(test['group_mask'],np.load(out/'test_predictions/affected_group_mask.npy'))
            assert np.array_equal(test['point_mask'],np.load(out/'test_predictions/affected_point_mask.npy'))
            check_summary(counts,r['test']['full_mixed_test'])
            check_summary(counts[np.repeat(test['group_mask'],2)],r['test']['affected_group_test'])
            check_summary(counts[test['point_mask']],r['test']['affected_points_only'])
            assert r['test']['metadata']['role']=='mixed_test' and r['test']['metadata']['rows']==budget['mixed_test_points']
            assert r['test']['nfe']==1 and r['test']['prediction_batch']==2
            assert r['G1_calibration_rows']==budget['Acal_points']
            if method=='no_ft':
                frozen=recount(scores,test,old_threshold);diag=r['test']['fully_frozen_NoFT_diagnostic']
                assert np.array_equal(frozen,np.load(out/'test_predictions/frozen_threshold_point_counts.npy'))
                assert diag['G1_calibration_labels']==0 and diag['reused_same_raw_predictions']
                check_summary(frozen,diag['full_mixed_test']);check_summary(frozen[np.repeat(test['group_mask'],2)],diag['affected_group_test'])
                check_summary(frozen[test['point_mask']],diag['affected_points_only'])
            binding=sampled_replay(mt,ev,selected['model'],scores,test,device)
            report=dict(city=city,method=method,seed=42,status='PASSED',full_test_predictions=len(scores),
                primary_test_points=int(np.repeat(test['group_mask'],2).sum()),parent_weights_and_reset_verified=True,
                full_calibration_rebuilt=True,all_selection_candidates_rebuilt=method!='no_ft',
                all_full_test_counts_recomputed=True,sampled_model_binding=binding,
                epochs=r['epochs'],updates=r['global_update'],actual_rows=r['actual_rows'])
            reports.append(report);test_total+=len(scores);print(json.dumps(report),flush=True)
    assert len(reports)==6 and test_total==90000 and cal_total==36000 and select_total==120000
    result=dict(status='PASSED',cases=reports,strategy_cases=6,full_FT_runs=4,NoFT_cases=2,
        full_test_predictions_recounted=test_total,full_calibration_predictions_rebuilt=cal_total,
        all_epoch_selection_predictions_rebuilt=select_total,sampled_model_output_binding_points=36,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        no_hashes=True,recorded_unix=time.time(),latex_compilation_not_covered=True)
    atomic_json(run/'audits/model_completion.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();audit(args.run,args.device)


if __name__=='__main__':main()
