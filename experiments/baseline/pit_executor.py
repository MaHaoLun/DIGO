"""Gate actual selected weights before concurrent independent OD execution."""
import time
import numpy as np
import torch
from common import atomic,ev
from pit_graph import GraphModel
from pit_parallel import sample_wave
import models


class Executor:
    def __init__(self,model,process,data,shard,folder,selected_epoch,width=4):
        self.model=model;self.process=process;self.data=data;self.device=next(model.parameters()).device
        self.width=width;self.wrappers=[GraphModel(model,data.ppg) for _ in range(width)]
        self.streams=[torch.cuda.Stream(device=self.device) for _ in range(width)]
        self.proofs={};self.parallel=False;started=time.monotonic()
        first=shard*32;groups=list(range(first,min(first+width,len(data)//data.ppg)))
        assert len(groups)==width
        conditions,generators=self.inputs(groups);references=[];states=[]
        for cond,generator in zip(conditions,generators):
            references.append(models.pit_sample(model,process,cond,generator));states.append(generator.get_state())
        conditions,generators=self.inputs(groups)
        actual=sample_wave(self.wrappers,process,conditions,generators,self.streams)
        errors=[];rng_equal=[]
        for ref,got,state,generator in zip(references,actual,states,generators):
            assert torch.isfinite(ref).all() and torch.isfinite(got).all()
            errors.append(float((ref-got).abs().max()));rng_equal.append(bool(torch.equal(state,generator.get_state())))
        self.parallel=all(e==0. for e in errors) and all(rng_equal)
        self.execution=(f'{width} independent OD streams;unchanged per-OD batch and all1000 DDPM transitions'
            if self.parallel else 'eager released denoiser;parallel execution rejected by exact source check')
        chosen=actual if self.parallel else references
        self.proofs=dict(zip(groups,chosen))
        saved=(folder/f'chunk_{shard:06d}.npz').exists();ppg=data.ppg
        atomic(folder/f'engine_audit_{shard:02d}_{time.time_ns()}.json',dict(
            status='PASSED_EXACT_PARALLEL' if self.parallel else 'USING_EAGER_SOURCE',
            city=data.city,snapshot=data.snapshot,role=data.role,shard=shard,selected_epoch=selected_epoch,
            complete_reverse_steps=1000,point_batch_per_OD=ppg,concurrent_OD_groups=width,
            registered_first_groups=groups,full3_channel_max_abs_errors=errors,
            all_Gaussian_states_exact=all(rng_equal),
            eager_reference_denoiser_images=width*ppg*1000,
            parallel_comparison_denoiser_images=width*ppg*1000,
            additional_check_denoiser_images=width*ppg*1000*(2 if saved else 1),
            graph_warmup_forward_images=width*ppg*3,graph_capture_forward_invocations=width,
            designated_first_samples_reused=not saved,existing_first_chunk=saved,
            elapsed_seconds=time.monotonic()-started,
            note='full actual selected-model source comparison; only independent execution overlaps; no GT or modified query/noise/transition budget'))

    def inputs(self,groups):
        conditions=[];generators=[];d=self.data
        for group in groups:
            lo,hi=group*d.ppg,(group+1)*d.ppg
            conditions.append(torch.as_tensor(np.array(d.odt[lo:hi]),device=self.device))
            seed=ev.identity_seed(20260912,int(d.query_ids[lo]//d.ppg),d.city,
                d.city+'_'+d.snapshot+'_canonical_fifo_continuous',d.role)
            generators.append(torch.Generator(device=self.device).manual_seed(seed))
        return conditions,generators

    def sample_range(self,begin,end):
        for start in range(begin,end,self.width):
            groups=list(range(start,min(start+self.width,end)))
            needed=[group for group in groups if group not in self.proofs];computed={}
            if needed:
                conditions,generators=self.inputs(needed)
                if self.parallel:
                    images=sample_wave(self.wrappers,self.process,conditions,generators,self.streams)
                else:
                    images=[models.pit_sample(self.model,self.process,c,g) for c,g in zip(conditions,generators)]
                computed=dict(zip(needed,images))
            for group in groups:
                image=self.proofs.pop(group) if group in self.proofs else computed[group]
                yield group,image
