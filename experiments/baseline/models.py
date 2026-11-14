"""Load released model classes; explicit algebraic/storage adapters only."""
import importlib.util
from types import SimpleNamespace
import torch
from common import RUN


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m


def sse(city):
    src=module('released_sse',RUN/'external_src/JIT-TP/SSE/Searchsp/ss_grids.py')
    h=SimpleNamespace(use_node2vec=True,pretrained_node_emb=str(RUN/'data'/city/'SSE_features/node_embedding.npy'),
        pretrained_region_emb=str(RUN/'data'/city/'SSE_features/region_embedding.npy'),
        region_num=4096,d_node=64,d_m1=256,d_m2=512,d_m3=256,d_m4=320,dropout=.2,batch_size=64)
    return src.SSRegionPred(h)


def sse_scores(model,pairs,endpoints):
    o,d=model.od_layer(pairs[:,0],pairs[:,1])
    query=model.mlp(torch.cat((o,model.region_emb(endpoints[:,0]),d,model.region_emb(endpoints[:,1])),dim=1))
    # The released concatenation, linear projection and arithmetic mean are
    # algebraically separable. Keep every parameter, feature and gradient path.
    # A preflight compares scores/loss/all active gradients with source forward.
    weight=model.fc_layer2.weight.mean(dim=0)
    query_part=torch.nn.functional.linear(query,weight[:256].unsqueeze(0))
    region_part=torch.nn.functional.linear(model.region_emb.weight,weight[256:].unsqueeze(0)).T
    return torch.sigmoid(query_part+region_part)


def pit():
    src=module('released_dot_unet',RUN/'external_src/DOT/model/denoiser.py')
    return src.Unet(dim=64,channels=3,init_dim=4,dim_mults=(1,2,4),condition='odt')


def diffusion(device):
    src=module('released_dot_diffusion',RUN/'external_src/DOT/model/diffusion.py')
    class DeviceDiffusion(src.DiffusionProcess):
        def extract(self,a,t,x_shape):
            return a.gather(-1,t).reshape(t.shape[0],*((1,)*(len(x_shape)-1)))
    process=DeviceDiffusion(1000,'linear')
    for name in ('betas','sqrt_recip_alphas','sqrt_alphas_cumprod','sqrt_one_minus_alphas_cumprod','posterior_variance'):
        setattr(process,name,getattr(process,name).to(device))
    return process


@torch.no_grad()
def pit_step(model,process,x,t,step,cond,generator=None):
    beta=process.extract(process.betas,t,x.shape)
    denom=process.extract(process.sqrt_one_minus_alphas_cumprod,t,x.shape)
    recip=process.extract(process.sqrt_recip_alphas,t,x.shape)
    mean=recip*(x-beta*model(x,t,cond)/denom)
    if step==0:return mean
    variance=process.extract(process.posterior_variance,t,x.shape)
    noise=torch.randn_like(x) if generator is None else torch.randn(x.shape,device=x.device,generator=generator)
    return mean+torch.sqrt(variance)*noise


@torch.no_grad()
def pit_sample(model,process,cond,generator):
    x=torch.randn((len(cond),3,64,64),device=cond.device,generator=generator)
    for i in reversed(range(1000)):
        t=torch.full((len(cond),),i,device=cond.device,dtype=torch.long)
        x=pit_step(model,process,x,t,i,cond,generator)
    return x
