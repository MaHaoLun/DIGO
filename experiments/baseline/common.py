"""Shared E3 contracts. All numerical work runs on the migrated execution host."""
import json
import os
from pathlib import Path
import sys
import numpy as np


ROOT=Path(os.environ['DIGO_WORKSPACE'])
RUN=ROOT/'runs/sdc_ssp64_od49_56/e3_v1'
E1=RUN.parent/'e1_v1'
E2=RUN.parent/'e2_v1'
sys.path.insert(0,str(E1/'code'))
from p0_inventory import atomic, BBOX
import evaluator64 as ev


def parents():
    for p in (E1/'E1_ACCEPTANCE.json',E2/'E2_ACCEPTANCE.json'):
        value=json.loads(p.read_text())
        assert 'ACCEPT' in str(value.get('status','')), (p,value.get('status'))
    protocol=json.loads((RUN/'protocol.json').read_text())
    assert protocol['training_seeds']==[42] and not protocol['hash_verification']
    return protocol


def roles(snapshot):
    return ['hierarchical_grid_train','Vselect','Vcal','base_test'] if snapshot=='G0' else ['sdc_Afit','Aselect','Acal','mixed_test']


class SparseData:
    def __init__(self,city,snapshot,role):
        self.folder=RUN/'data'/city/snapshot/role
        self.meta=json.loads((self.folder/'acceptance.json').read_text())
        assert self.meta['status']=='PREPARED_FULL'
        for name in ('query_ids','pairs','odt','endpoints','offsets','cells','daytime','progress','support'):
            setattr(self,name,np.load(self.folder/(name+'.npy'),mmap_mode='r'))
        self.ppg=16 if snapshot=='G0' else 2
        self.city=city;self.snapshot=snapshot;self.role=role
        assert len(self.query_ids)==self.meta['points'] and len(self.offsets)==len(self.query_ids)+1
    def __len__(self):return len(self.query_ids)
    def images(self,indices):
        indices=np.asarray(indices);x=np.full((len(indices),3,4096),-1.,np.float32)
        for j,i in enumerate(indices):
            sl=slice(self.offsets[i],self.offsets[i+1]);c=self.cells[sl]
            x[j,0,c]=1.;x[j,1,c]=self.daytime[sl];x[j,2,c]=self.progress[sl]
        return x.reshape(-1,3,64,64)
    def truth(self):
        out=np.zeros((len(self),4096),bool)
        for i in range(len(self)):
            out[i,self.cells[self.offsets[i]:self.offsets[i+1]]]=True
        return out


def save_pt(path,value):
    import torch
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)


def select(records):
    maximum=max(r['max_q'] for r in records)
    return min((r for r in records if r['max_q']>=maximum-1e-8),
        key=lambda r:(r['best']['mean_predicted_cells'],-r['best']['precision_micro'],r['epoch']))
