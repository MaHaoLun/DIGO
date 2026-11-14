"""G1 role-only sparse-label reader, with two query points per global OD ID."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np


class Dataset:
    def __init__(self, run, city, role):
        self.run=Path(run);self.folder=self.run/'data'/city;self.city=city;self.role=role
        config=json.loads((self.run/'protocol.json').read_text())
        accepted=json.loads((self.folder/'data_acceptance.json').read_text())
        assert accepted['status']=='DATA_ACCEPTED'
        expected=dict(random_Afit=config['budgets'][city]['Afit_points_per_FT'],
            sdc_Afit=config['budgets'][city]['Afit_points_per_FT'],
            Aselect=config['budgets'][city]['Aselect_points'],Acal=config['budgets'][city]['Acal_points'],
            mixed_test=config['budgets'][city]['mixed_test_points'])
        assert role in expected
        self.gids=np.sort(np.load(self.folder/(role+'_gids.npy')))
        assert len(np.unique(self.gids))==len(self.gids) and len(self.gids)*2==expected[role]
        self.coords=np.load(self.folder/'coordinates.npy',mmap_mode='r');self.n=len(self.coords)
        self.cells=np.load(self.folder/'cell64.npy',mmap_mode='r')
        self.road_support=np.bincount(self.cells,minlength=4096)>0
        self.query_ids=(self.gids[:,None]*2+np.arange(2)).reshape(-1)
        self.point_flags=np.load(self.folder/(role+'_point_flags.npy'),mmap_mode='r')
        self.affected_groups=np.load(self.folder/(role+'_affected_group_mask.npy'))
        assert self.point_flags.shape==(len(self.gids),2,7) and self.affected_groups.shape==(len(self.gids),)
        assert np.array_equal(self.affected_groups,self.point_flags[:,:,-1].any(1))
        with np.load(self.folder/'cache_index.npz') as index:
            positions=np.searchsorted(index['group_ids'],self.gids)
            assert np.array_equal(index['group_ids'][positions],self.gids)
            self.paths=index['paths'][positions];self.source_rows=index['rows'][positions]
        self.metadata=dict(city=city,role=role,snapshot_id=city+'_G1_canonical_fifo_continuous',
            rows=len(self.gids)*2,groups=len(self.gids),departures=[43200,44100],grid_dim=64,
            group_id_semantics='source_index*n_nodes+destination_index',points_per_group=2,
            affected_groups=int(self.affected_groups.sum()),affected_points=int(self.point_flags[:,:,-1].sum()))
        self.parent=Path(config['parent_e1_run'])

    def __len__(self):return len(self.gids)*2

    def iter_batches(self,batch_size=128):
        sys.path.insert(0,str(self.parent/'code'))
        from p0_inventory import BBOX
        box=BBOX[self.city];lo=np.asarray([box[0],box[2]]);span=np.asarray([box[1]-box[0],box[3]-box[2]])
        work={}
        for i,path in enumerate(self.paths):work.setdefault(str(path),[]).append(i)
        cache_root=(self.run/'cache'/self.city).resolve()
        for relative,indices in sorted(work.items()):
            file=(self.run/relative).resolve();assert cache_root in file.parents
            ix=np.asarray(indices,np.int64);ix=ix[np.argsort(self.source_rows[ix])]
            gids=self.gids[ix];rows=self.source_rows[ix]
            with h5py.File(file,'r') as f:
                assert f.attrs['status']=='COMPLETE' and f.attrs['city']==self.city
                assert np.array_equal(f['group_ids'][rows],gids)
                count=len(f['group_ids'])
                for dep_index,departure in enumerate((43200,44100)):
                    for start in range(0,len(ix),batch_size):
                        end=min(len(ix),start+batch_size);these=gids[start:end];row=rows[start:end]+dep_index*count
                        begins=f['cell_offsets'][row];ends=f['cell_offsets'][row+1]
                        image=np.full((len(row),2,4096),-1.,np.float32)
                        for j,(begin,stop) in enumerate(zip(begins,ends)):
                            cells=f['cells'][begin:stop];progress=f['progress'][begin:stop]
                            image[j,0,cells]=1.;image[j,1,cells]=progress
                        pairs=np.column_stack((these//self.n,these % self.n))
                        odt=np.empty((len(row),5),np.float32)
                        odt[:,:2]=2*(self.coords[pairs[:,0],1:]-lo)/span-1
                        odt[:,2:4]=2*(self.coords[pairs[:,1],1:]-lo)/span-1
                        odt[:,4]=2*departure/86400-1
                        yield dict(image=image.reshape(-1,2,64,64),odt=odt,
                            query_id=these*2+dep_index,group_id=these,endpoints=self.cells[pairs],departure=departure)
