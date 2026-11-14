"""Four full five-epoch FT jobs and two calibrated weights-frozen references."""
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
from gt_cache import atomic_json
from g1_runtime import bindings,preload,evaluate,select


def save_pt(path,value):
    temporary=path.with_suffix('.tmp.pt');torch.save(value,temporary);temporary.replace(path)


def train(args):
    run=args.run;config,parent,mt,ev=bindings(run)
    gate=json.loads((run/'data_gate.json').read_text())
    assert gate['status']=='BOTH_CITIES_DATA_ACCEPTED' and gate['G1_rational_reference_status']=='PASSED'
    assert gate['protocol']==config
    assert json.loads((parent/'E1_ACCEPTANCE.json').read_text())['status']=='ACCEPTED'
    assert config['training_seeds']==[42] and config['training']['training_seed']==42
    city,method=args.city,args.method;out=run/'training'/f'{city}_{method}_seed42';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text());assert result['status']=='COMPLETE'
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        return result
    assert torch.cuda.device_count()==1;device='cuda:0';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    started=time.monotonic();carried=0.
    parent_path=parent/'training'/f'{city}_hierarchical_grid_seed42'/'selected.pt'
    parent_weights=torch.load(parent_path,map_location='cpu',weights_only=False)
    pc=parent_weights['config']
    assert (pc['city'],pc['method'],pc['seed'])==(city,'hierarchical_grid',42)
    assert parent_weights['epoch']==50 and tuple(parent_weights['model']['pos'].shape)==(1,256,128)
    parent_cal=json.loads((parent_path.parent/'calibration.json').read_text())
    frozen_threshold=parent_cal['threshold']
    model=mt.DiT().to(device);model.load_state_dict(parent_weights['model'],strict=True)
    budget=config['budgets'][city]
    assert budget['FT_epochs']==5 and budget['batch_size']==64
    cfg=dict(city=city,method=method,seed=42,protocol=config,parent_checkpoint=str(parent_path),
        parent_selected_epoch=50,parent_checkpoint_bytes=parent_path.stat().st_size,
        parent_checkpoint_mtime_ns=parent_path.stat().st_mtime_ns,
        gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),run_dir=str(out),
        accepted_math_module=mt.__file__,accepted_evaluator_module=ev.__file__)
    if (out/'config.json').exists():
        original=json.loads((out/'config.json').read_text())
        assert {k:v for k,v in original.items() if k!='gpu_visible'}=={k:v for k,v in cfg.items() if k!='gpu_visible'}
    else:atomic_json(out/'config.json',cfg)
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    identity=dict(pid=os.getpid(),start_time=stat[19],arguments=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x])
    atomic_json(out/'worker_status.json',dict(status='RUNNING',**identity,city=city,method=method))
    def interrupt(signum,frame):raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        if method!='no_ft':
            role='random_Afit' if method=='random_ft' else 'sdc_Afit'
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01,eps=1e-8)
            ng=torch.Generator(device=device).manual_seed(242);tg=torch.Generator(device=device).manual_seed(342)
            initial_rng=mt._rng_state()
            atomic_json(out/'progress.json',dict(status='PRELOADING',city=city,method=method,global_update=0))
            data=preload(run,city,role,device);valid=preload(run,city,'Aselect',device)
            mt._restore_rng(initial_rng)
            n=len(data['query_ids']);assert n==budget['Afit_points_per_FT']
            assert n%64==0 and len(valid['query_ids'])==budget['Aselect_points']
            total=n//64*5;assert total==budget['updates_per_FT']
            state=dict(epoch=0,cursor=0,global_update=0,actual_rows=0,epoch_loss_sum=0.,selection=[],elapsed_seconds=0.)
            permutation=None
            def checkpoint():
                state['elapsed_seconds']=carried+time.monotonic()-started
                save_pt(out/'last.pt',dict(config=cfg,state=state.copy(),model=model.state_dict(),
                    optimizer=optimizer.state_dict(),sampler_permutation=permutation,
                    ng=ng.get_state(),tg=tg.get_state(),rng=mt._rng_state(),
                    total_updates=total,train_rows=n,batch_size=64))
                atomic_json(out/'progress.json',dict(status='RUNNING',city=city,method=method,
                    total_updates=total,train_rows=n,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),**state))
            if args.resume:
                saved=torch.load(out/'last.pt',map_location=device,weights_only=False)
                assert saved['total_updates']==total and saved['train_rows']==n
                assert saved['config']['city']==city and saved['config']['method']==method
                assert {k:v for k,v in saved['config'].items() if k!='gpu_visible'}=={k:v for k,v in cfg.items() if k!='gpu_visible'}
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                state=saved['state'];permutation=saved['sampler_permutation'];carried=state['elapsed_seconds']
                ng.set_state(saved['ng'].cpu());tg.set_state(saved['tg'].cpu());mt._restore_rng(saved['rng'])
            else:
                assert not (out/'last.pt').exists(), 'Explicit own-run resume is required'
                assert not optimizer.state and all(torch.equal(v.cpu(),parent_weights['model'][k]) for k,v in model.state_dict().items())
                save_pt(out/'initial.pt',dict(model=model.state_dict(),config=cfg,global_update=0,actual_rows=0,
                    copied_exact_parent_weights=True,optimizer_state_empty=True,optimizer_reset=True,rng=initial_rng,
                    ng=ng.get_state(),tg=tg.get_state()))
                checkpoint()
            if args.initialize_only:
                assert state['global_update']==0
                nstate,tstate=ng.get_state(),tg.get_state();global_state=mt._rng_state()
                image=data['image'][:64];odt=data['odt'][:64]
                optimizer.zero_grad(set_to_none=True)
                noise,times=mt.draw_noise_times(image,ng,tg);loss=mt.cfm_loss(model,image,odt,noise,times,10)
                assert bool(torch.isfinite(loss));loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert bool(torch.isfinite(norm))
                optimizer.zero_grad(set_to_none=True);ng.set_state(nstate);tg.set_state(tstate);mt._restore_rng(global_state)
                assert not optimizer.state and all(torch.equal(v.cpu(),parent_weights['model'][k]) for k,v in model.state_dict().items())
                check=dict(status='FULL_PRELOAD_STARTUP_VERIFIED',fit_rows=n,Aselect_rows=len(valid['query_ids']),
                    actual_optimizer_updates=0,engineering_forward_backward_points=64,
                    initial_parent_weights_unchanged=True,explicit_generators_restored=True,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),loss=float(loss),gradient_norm=float(norm))
                checkpoint();atomic_json(out/'startup_check.json',check)
                atomic_json(out/'worker_status.json',dict(status='STARTUP_ONLY_COMPLETE',**identity))
                return check
            model.train()
            while state['epoch']<5:
                epoch=state['epoch']
                if permutation is None:
                    permutation=np.random.default_rng(np.random.SeedSequence([142,epoch])).permutation(n)
                    assert state['cursor']==0
                while state['cursor']<n:
                    end=state['cursor']+64;indices=torch.from_numpy(permutation[state['cursor']:end]).to(device)
                    image=data['image'][indices];odt=data['odt'][indices]
                    noise,times=mt.draw_noise_times(image,ng,tg)
                    optimizer.zero_grad(set_to_none=True);loss=mt.cfm_loss(model,image,odt,noise,times,10)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite G1 loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite G1 gradient')
                    update=state['global_update']+1;rate=mt.lr_at_update(update,1e-5,total,'cosine',100)
                    for group in optimizer.param_groups:group['lr']=rate
                    optimizer.step();state['actual_rows']+=64;state['epoch_loss_sum']+=float(loss.detach())*64
                    state.update(cursor=end,global_update=update)
                    if update%100==0:checkpoint()
                assert state['actual_rows']==(epoch+1)*n and state['cursor']==n
                with mt._evaluation_context(model):
                    summary=evaluate(model,valid,run,device,save_dir=out/'selection_predictions'/f'epoch_{epoch+1:03d}')
                record=dict(epoch=epoch+1,**summary);state['selection'].append(record)
                atomic_json(out/'selection'/f'epoch_{epoch+1:03d}.json',record)
                save_pt(out/f'epoch_{epoch+1:03d}.pt',dict(model=model.state_dict(),epoch=epoch+1,config=cfg,summary=record))
                atomic_json(out/'epochs'/f'epoch_{epoch+1:03d}.json',dict(epoch=epoch+1,rows=n,
                    permutation_rows=n,all_rows_once=True,global_update=state['global_update'],
                    actual_rows=state['actual_rows'],mean_loss=state['epoch_loss_sum']/n,lr=rate))
                state.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.);permutation=None;checkpoint()
            assert state['global_update']==total and state['actual_rows']==5*n
            chosen=select(state['selection']);selected=torch.load(out/f"epoch_{chosen['epoch']:03d}.pt",map_location=device,weights_only=False)
            model.load_state_dict(selected['model']);save_pt(out/'selected.pt',selected)
            atomic_json(out/'selection_result.json',dict(status='CHECKPOINT_SELECTED',selected_epoch=chosen['epoch'],
                selected_on='Aselect',record=chosen))
            training_result=dict(epochs=5,train_rows=n,actual_rows=state['actual_rows'],
                global_update=state['global_update'],selected_epoch=chosen['epoch'])
            del data,valid;torch.cuda.empty_cache()
        else:
            assert not args.resume and not args.initialize_only
            atomic_json(out/'frozen_NoFT_reference.json',dict(parent_checkpoint=str(parent_path),
                original_G0_threshold=frozen_threshold,formal_optimizer_updates=0,weights_frozen=True))
            save_pt(out/'selected.pt',dict(model=model.state_dict(),config=cfg,epoch=0,parent_selected_epoch=50))
            training_result=dict(epochs=0,train_rows=0,actual_rows=0,global_update=0,selected_epoch=0)
        calibration=preload(run,city,'Acal',device)
        cal=evaluate(model,calibration,run,device,save_dir=out/'calibration_predictions')
        threshold=cal['best']['threshold']
        atomic_json(out/'calibration.json',dict(status='THRESHOLD_FROZEN',calibrated_on='Acal',
            selected_epoch=training_result['selected_epoch'],threshold=threshold,summary=cal))
        del calibration;torch.cuda.empty_cache()
        test=preload(run,city,'mixed_test',device)
        result_test=evaluate(model,test,run,device,threshold=threshold,save_dir=out/'test_predictions',
            frozen_threshold=frozen_threshold if method=='no_ft' else None)
        if method=='no_ft':
            assert all(torch.equal(v.cpu(),parent_weights['model'][k]) for k,v in model.state_dict().items())
        result=dict(status='COMPLETE',city=city,method=method,seed=42,**training_result,
            threshold=threshold,test=result_test,parent_selected_epoch=50,
            elapsed_seconds=carried+time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            G1_calibration_rows=config['budgets'][city]['Acal_points'],NoFT_weights_unchanged=method=='no_ft',
            no_content_hashes=True)
        atomic_json(out/'result.json',result);atomic_json(out/'progress.json',result)
        atomic_json(out/'worker_status.json',dict(status='COMPLETE',**identity,completed_unix=time.time()))
        return result
    except BaseException as exc:
        failure=dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),recorded_unix=time.time())
        atomic_json(out/'failures'/('failure_'+str(time.time_ns())+'.json'),failure)
        atomic_json(out/'worker_status.json',dict(status='FAILED',**identity,error=repr(exc)))
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--city',choices=('BJ','XA'),required=True)
    p.add_argument('--method',choices=('no_ft','random_ft','sdc_mixed_ft'),required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--initialize-only',action='store_true')
    print(json.dumps(train(p.parse_args())),flush=True)


if __name__=='__main__':main()
