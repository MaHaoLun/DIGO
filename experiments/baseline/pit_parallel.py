"""Independent per-OD CUDA streams; original point batch and all steps preserved."""
import torch
import models


@torch.no_grad()
def sample_wave(wrappers,process,conditions,generators,streams):
    n=len(conditions);assert 0<n<=len(wrappers) and len(generators)==n and len(streams)>=n
    device=conditions[0].device;current=torch.cuda.current_stream(device);images=[]
    for j,cond in enumerate(conditions):
        streams[j].wait_stream(current)
        with torch.cuda.stream(streams[j]):
            images.append(torch.randn((len(cond),3,64,64),device=device,generator=generators[j]))
    for step in reversed(range(1000)):
        for j,cond in enumerate(conditions):
            with torch.cuda.stream(streams[j]):
                t=torch.full((len(cond),),step,device=device,dtype=torch.long)
                images[j]=models.pit_step(wrappers[j],process,images[j],t,step,cond,generators[j])
    for stream in streams[:n]:current.wait_stream(stream)
    return images
