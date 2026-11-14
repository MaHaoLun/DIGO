"""Full matched-budget training of authentic E3 region components, seed42 only."""
import argparse
import fcntl
import json
import math
import os
import random
import signal
import time
import traceback
import numpy as np
import torch
from common import RUN,SparseData,atomic,parents,save_pt,select,ev
import models


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def preload(data,method,device):
    cond=torch.as_tensor(np.array(data.odt),device=device)
    if method=='PiT':
        x=torch.empty((len(data),3,64,64),device=device,dtype=torch.float32)
        for lo in range(0,len(data),1024):
            x[lo:lo+1024]=torch.from_numpy(data.images(np.arange(lo,min(lo+1024,len(data))))).to(device)
        result=dict(image=x,odt=cond)
    else:
        result=dict(truth=torch.from_numpy(data.truth()).to(device),
            pairs=torch.from_numpy(np.array(data.pairs)).to(device),endpoints=torch.from_numpy(np.array(data.endpoints)).to(device))
    torch.cuda.synchronize();return result


@torch.no_grad()
def validate(model,valid,tensors,method,dp,out,epoch):
    model.eval();device=next(model.parameters()).device
    if method=='PiT':
        ng=torch.Generator(device=device).manual_seed(20260912)
        tg=torch.Generator(device=device).manual_seed(20260913);loss=0.
        for lo in range(0,len(valid),64):
            image=tensors['image'][lo:lo+64];cond=tensors['odt'][lo:lo+64]
            t=torch.randint(0,1000,(len(image),),device=device,generator=tg)
            noise=torch.randn(image.shape,device=device,generator=ng)
            loss+=float(dp.p_losses(model,image,t,y=cond,noise=noise,loss_type='huber'))*len(image)
        return dict(epoch=epoch,rows=len(valid),mean_fixed_noise_huber=loss/len(valid),
            checkpoint_rule='fixed final epoch; diagnostic loss does not select checkpoint',generated_regions=0)
    folder=out/'selection'/f'epoch_{epoch:03d}';folder.mkdir(parents=True,exist_ok=True)
    scores=np.lib.format.open_memmap(folder/'scores.npy',mode='w+',dtype=np.float32,shape=(len(valid),4096))
    for lo in range(0,len(valid),512):
        block=models.sse_scores(model,tensors['pairs'][lo:lo+512],tensors['endpoints'][lo:lo+512])
        assert torch.isfinite(block).all();scores[lo:lo+512]=block.cpu().numpy()
    scores.flush();np.save(folder/'query_ids.npy',valid.query_ids)
    summary=ev.exact_threshold_sweep(scores,valid.truth(),valid.endpoints,valid.support)
    summary.pop('coarse_curve');summary.update(epoch=epoch,role=valid.role,rows=len(valid))
    atomic(folder/'metrics.json',summary);return summary


def train(args):
    protocol=parents();city=args.city;snapshot=args.snapshot;method=args.method
    out=RUN/'training'/f'{city}_{snapshot}_{method}_seed42';out.mkdir(parents=True,exist_ok=True)
    handle=(out/'worker.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=json.loads((RUN/'audits'/f'{city}_{method}_core.json').read_text());assert gate['status']=='PASSED'
    gate=json.loads((RUN/'audits'/f'{city}_prepared.json').read_text());assert gate['status']=='PASSED'
    assert not (out/'result.json').exists(),'completed run cannot be relaunched'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    device='cuda:0';model=(models.sse(city) if method=='SSE' else models.pit()).to(device)
    dp=models.diffusion(device) if method=='PiT' else None
    parent_path=None
    if snapshot=='G1':
        parent=RUN/'training'/f'{city}_G0_{method}_seed42'
        accepted=json.loads((parent/'result.json').read_text());assert accepted['status']=='TRAINED_FULL'
        assert json.loads((parent/'training_audit.json').read_text())['status']=='PASSED'
        parent_path=parent/'selected.pt';p=torch.load(parent_path,map_location=device,weights_only=False)
        model.load_state_dict(p['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=2,factor=.8,threshold=.001) if method=='SSE' else None
    ng=torch.Generator(device=device).manual_seed(242);tg=torch.Generator(device=device).manual_seed(342)
    trainrole='hierarchical_grid_train' if snapshot=='G0' else 'sdc_Afit'
    selectrole='Vselect' if snapshot=='G0' else 'Aselect'
    data=SparseData(city,snapshot,trainrole);valid=SparseData(city,snapshot,selectrole)
    n=len(data);epochs=50 if snapshot=='G0' else 5
    assert n==protocol['training_data']['points'][city][snapshot] and n%64==0
    total_updates=n//64*epochs;started=time.monotonic();carried=0.
    config=dict(city=city,snapshot=snapshot,method=method,seed=42,protocol=protocol,
        training_rows=n,epochs=epochs,batch_size=64,optimizer='Adam',lr=.001,
        parent_selected=str(parent_path) if parent_path else None,
        precision='FP32 with TF32',gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        devices=torch.cuda.device_count(),data_parallel=method=='PiT' and torch.cuda.device_count()>1)
    # Transport environment is logged by persistent job specs. It changes no
    # model, sampling, optimizer or row-budget configuration.
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        assert {k:v for k,v in old.items() if k!='gpu_visible'}=={k:v for k,v in config.items() if k!='gpu_visible'}
    else:atomic(out/'config.json',config)
    initial=rng_state();atomic(out/'progress.json',dict(status='PRELOADING',**{k:config[k] for k in ('city','snapshot','method','seed')}))
    tensors=preload(data,method,device);vt=preload(valid,method,device);restore_rng(initial)
    state=dict(epoch=0,cursor=0,global_update=0,actual_rows=0,epoch_loss_sum=0.,selection=[],elapsed_seconds=0.)
    permutation=None
    def checkpoint(status='RUNNING'):
        state['elapsed_seconds']=carried+time.monotonic()-started
        payload=dict(config=config,state=state.copy(),model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None,permutation=permutation,rng=rng_state(),
            ng=ng.get_state(),tg=tg.get_state(),total_updates=total_updates)
        save_pt(out/'last.pt',payload)
        atomic(out/'progress.json',dict(status=status,city=city,snapshot=snapshot,method=method,seed=42,
            train_rows=n,total_updates=total_updates,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),**state))
    if args.resume:
        saved=torch.load(out/'last.pt',map_location=device,weights_only=False)
        assert saved['total_updates']==total_updates
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        state=saved['state'];permutation=saved['permutation'];restore_rng(saved['rng'])
        ng.set_state(saved['ng'].cpu());tg.set_state(saved['tg'].cpu());carried=state['elapsed_seconds'];started=time.monotonic()
    else:
        assert not (out/'last.pt').exists(),'own-run resume must be explicit'
        save_pt(out/'initial.pt',dict(model=model.state_dict(),config=config,optimizer_state_empty=not optimizer.state,
            rng=initial,actual_rows=0,parent_selected=config['parent_selected']))
        checkpoint()
    parallel=torch.nn.DataParallel(model) if config['data_parallel'] else model
    if args.admission_only:
        assert state['global_update']==0 and not optimizer.state
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};saved_rng=rng_state()
        saved_ng=ng.get_state();saved_tg=tg.get_state();t0=time.monotonic();model.train()
        optimizer.zero_grad(set_to_none=True)
        if method=='PiT':
            image=tensors['image'][:64];cond=tensors['odt'][:64]
            t=torch.randint(0,1000,(64,),device=device,generator=tg)
            noise=torch.randn(image.shape,device=device,generator=ng)
            loss=dp.p_losses(parallel,image,t,y=cond,noise=noise,loss_type='huber')
        else:
            scores=models.sse_scores(model,tensors['pairs'][:64],tensors['endpoints'][:64])
            loss=torch.nn.functional.binary_cross_entropy(scores,tensors['truth'][:64].float())
        loss.backward();torch.cuda.synchronize();assert torch.isfinite(loss)
        elapsed=time.monotonic()-t0;model.load_state_dict(before);optimizer.zero_grad(set_to_none=True)
        restore_rng(saved_rng);ng.set_state(saved_ng);tg.set_state(saved_tg)
        assert all(torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items()) and not optimizer.state
        atomic(out/'admission.json',dict(status='PASSED',full_training_rows=n,full_selection_rows=len(valid),
            engineering_forward_backward_rows=64,optimizer_updates=0,weights_and_RNG_restored=True,
            batch_forward_backward_seconds=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            config=config))
        checkpoint('ADMITTED_ZERO_UPDATES');print((out/'admission.json').read_text(),flush=True)
        fcntl.flock(handle,fcntl.LOCK_UN);handle.close();return
    assert (out/'admission.json').exists(),'full preload admission must precede formal updates'
    stopped=[]
    def request_stop(signum,frame):stopped.append(signum)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        while state['epoch']<epochs:
            ep=state['epoch'];model.train()
            if permutation is None:
                assert state['cursor']==0
                permutation=np.random.default_rng(np.random.SeedSequence([142,ep])).permutation(n)
            while state['cursor']<n:
                end=min(n,state['cursor']+64);idx=torch.as_tensor(permutation[state['cursor']:end],device=device)
                optimizer.zero_grad(set_to_none=True)
                if method=='PiT':
                    image=tensors['image'][idx];cond=tensors['odt'][idx]
                    t=torch.randint(0,1000,(len(idx),),device=device,generator=tg)
                    noise=torch.randn(image.shape,device=device,generator=ng)
                    loss=dp.p_losses(parallel,image,t,y=cond,noise=noise,loss_type='huber')
                else:
                    output=models.sse_scores(model,tensors['pairs'][idx],tensors['endpoints'][idx])
                    loss=torch.nn.functional.binary_cross_entropy(output,tensors['truth'][idx].float())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise FloatingPointError('nonfinite training gradient')
                optimizer.step();batch=end-state['cursor']
                state['actual_rows']+=batch;state['epoch_loss_sum']+=float(loss.detach())*batch
                state.update(cursor=end,global_update=state['global_update']+1)
                if state['global_update']%500==0:
                    checkpoint();print(json.dumps(dict(epoch=ep+1,cursor=end,update=state['global_update'],loss=float(loss))),flush=True)
                if stopped:
                    checkpoint('STOPPED_WITH_CHECKPOINT');raise InterruptedError(f'signal {stopped[-1]}')
            assert state['actual_rows']==(ep+1)*n and state['cursor']==n
            evaluate=(ep+1)%5==0 if snapshot=='G0' else True
            if evaluate:
                summary=validate(model,valid,vt,method,dp,out,ep+1);state['selection'].append(summary)
                atomic(out/'selection'/f'epoch_{ep+1:03d}.json',summary)
                if scheduler:scheduler.step(summary['best']['recall_micro'])
                save_pt(out/f'epoch_{ep+1:03d}.pt',dict(model=model.state_dict(),epoch=ep+1,config=config,summary=summary))
            atomic(out/'epochs'/f'epoch_{ep+1:03d}.json',dict(epoch=ep+1,rows=n,actual_rows=state['actual_rows'],
                global_update=state['global_update'],mean_loss=state['epoch_loss_sum']/n,all_rows_once=True,
                permutation_rows=len(permutation),lr=optimizer.param_groups[0]['lr']))
            state.update(epoch=ep+1,cursor=0,epoch_loss_sum=0.);permutation=None;checkpoint()
        assert state['global_update']==total_updates and state['actual_rows']==n*epochs
        chosen=select(state['selection']) if method=='SSE' else state['selection'][-1]
        selected=torch.load(out/f"epoch_{chosen['epoch']:03d}.pt",map_location='cpu',weights_only=False)
        save_pt(out/'selected.pt',selected)
        atomic(out/'selection_result.json',dict(status='CHECKPOINT_SELECTED',selected_epoch=chosen['epoch'],
            role=selectrole,criterion='common min(R,F1)' if method=='SSE' else 'preregistered final epoch',record=chosen))
        result=dict(status='TRAINED_FULL',city=city,snapshot=snapshot,method=method,seed=42,
            epochs=epochs,train_rows=n,actual_rows=state['actual_rows'],global_update=state['global_update'],
            selected_epoch=chosen['epoch'],elapsed_seconds=carried+time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),inference_complete=False)
        atomic(out/'result.json',result);atomic(out/'progress.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        atomic(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),state=state,
            last_checkpoint_preserved=True));raise
    finally:fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--city',choices=['BJ','XA'],required=True)
    p.add_argument('--snapshot',choices=['G0','G1'],required=True);p.add_argument('--method',choices=['SSE','PiT'],required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--admission-only',action='store_true');train(p.parse_args())
