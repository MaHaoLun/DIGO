"""Capture the unchanged released denoiser; every DDPM transition remains eager."""
import torch


class GraphModel:
    def __init__(self,model,batch):
        assert not model.training
        self.module=model;device=next(model.parameters()).device
        assert device.type=='cuda'
        self.x=torch.zeros((batch,3,64,64),device=device)
        self.t=torch.full((batch,),500,device=device,dtype=torch.long)
        self.cond=torch.zeros((batch,5),device=device)
        stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):model(self.x,self.t,self.cond)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph),torch.no_grad():
            self.output=model(self.x,self.t,self.cond)

    @torch.no_grad()
    def __call__(self,x,t,cond):
        assert x.shape==self.x.shape and t.shape==self.t.shape and cond.shape==self.cond.shape
        self.x.copy_(x);self.t.copy_(t);self.cond.copy_(cond)
        self.graph.replay()
        return self.output
