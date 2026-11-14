"""Resource admission measurement, separate from formal quality experiments."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import model64 as mt
from train64 import preload
from p0_inventory import atomic

def worker(run,city,slot,steps,out):
    torch.set_num_threads(2);torch.manual_seed(4242+slot)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    model=mt.DiT().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01)
    data=preload(run,city,"hierarchical_grid_train","cuda:0")
    valid=preload(run,city,"Vselect","cuda:0")
    ng=torch.Generator(device="cuda").manual_seed(4442+slot)
    tg=torch.Generator(device="cuda").manual_seed(4542+slot)
    atomic(out/f"ready_{slot}.json",dict(status="READY"))
    while not (out/"go.json").exists():time.sleep(.2)
    torch.cuda.synchronize();start=time.monotonic()
    for j in range(steps):
        x=data["image"][j*64:(j+1)*64];c=data["odt"][j*64:(j+1)*64]
        noise,t=mt.draw_noise_times(x,ng,tg);opt.zero_grad(set_to_none=True)
        loss=mt.cfm_loss(model,x,c,noise,t,10);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval()
    with torch.no_grad():
        noise=torch.zeros_like(valid["image"][:16]);mt.sample(model,valid["odt"][:16],noise)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    r=dict(status="COMPLETE",city=city,slot=slot,steps=steps,formal_updates=0,
        elapsed_seconds=elapsed,point_updates_per_second=steps*64/elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic(out/f"worker_{slot}.json",r)

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--city",choices=["BJ","XA"],required=True)
    p.add_argument("--slot",type=int,default=0);p.add_argument("--steps",type=int,default=128)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    worker(a.run,a.city,a.slot,a.steps,a.out)

if __name__=="__main__":main()
