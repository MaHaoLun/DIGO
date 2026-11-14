"""Independent complete emitted-route, label, evidence and selection acceptance."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import resource
import time
import h5py
import numpy as np
from gt_cache import FLAG_NAMES, accounting, atomic_json


def weight_at(knots, edge, entry):
    phase = float(entry) % 86400.
    slot = int(phase//900); fraction = (phase-900*slot)/900.
    return int(knots[edge, slot])+(int(knots[edge, (slot+1) % 96])-int(knots[edge, slot]))*fraction


def replay(gids, n, offsets, nodes, arrival, costs, keys, knots):
    count = len(gids); departures = np.repeat([43200,44100], count)
    assert np.all((nodes>=0)&(nodes<n))
    assert len(offsets)==2*count+1 and offsets[0]==0 and offsets[-1]==len(nodes)==len(arrival)
    assert np.all(np.diff(offsets)>=2) and np.isfinite(arrival).all() and np.isfinite(costs).all()
    assert np.array_equal(nodes[offsets[:-1]], np.tile(gids//n, 2))
    assert np.array_equal(nodes[offsets[1:]-1], np.tile(gids % n, 2))
    assert np.array_equal(arrival[offsets[:-1]], departures)
    boundary = np.ones(len(nodes)-1, bool); boundary[offsets[1:-1]-1]=False
    before, after = nodes[:-1][boundary], nodes[1:][boundary]
    edge_keys = before.astype(np.int64)*n+after
    edge = np.searchsorted(keys, edge_keys); assert np.array_equal(keys[edge], edge_keys)
    entry, actual = arrival[:-1][boundary], arrival[1:][boundary]
    phase=np.remainder(entry,86400.);slot=(phase//900).astype(np.int64);r=phase-slot*900
    expected=entry+knots[edge,slot]+(knots[edge,(slot+1)%96]-knots[edge,slot])*r/900
    error=float(np.max(np.abs(expected-actual),initial=0))
    assert error<1e-7 and np.all(actual>entry) and np.all(costs>0)
    assert np.max(np.abs(arrival[offsets[1:]-1]-departures-costs),initial=0)<1e-7
    return error


def check(run, city, qualification_only=False):
    start,cpu=time.monotonic(),time.process_time();folder=run/'data'/city
    config=json.loads((run/'protocol.json').read_text());coords=np.load(folder/'coordinates.npy');n=len(coords)
    cells=np.load(folder/'cell64.npy');old=np.load(folder/'G0_graph.npz');new=np.load(folder/'G1_graph.npz')
    keys=new['u'].astype(np.int64)*n+new['v'];knots0,knots1=old['knots'],new['knots']
    assert np.array_equal(old['u'],new['u']) and np.array_equal(old['v'],new['v'])
    assert np.all(keys[1:]>keys[:-1]) and knots0.shape==knots1.shape
    with np.load(folder/'detection.npz') as detection:
        significant=set(map(int,detection['shifted_pair_keys']));hot=set(map(int,detection['hotspots']))
    all_flags={};all_hot={};files=[];point_paths=0;max_edge=max_progress=0.;witness_count=0
    for file in sorted((run/'cache'/city).glob('*/s*.h5')):
        if '.tmp.' in file.name:continue
        with h5py.File(file,'r') as f:
            assert f.attrs['status']=='COMPLETE' and f.attrs['city']==city
            assert f.attrs['points_per_group']==2 and f.attrs['grid_dim']==64
            assert json.loads(f.attrs['flag_names'])==list(FLAG_NAMES)
            gids=f['group_ids'][:];count=len(gids)
            assert len(np.unique(gids))==count and np.all(gids[1:]>gids[:-1])
            assert np.all(gids//n==int(f.attrs['source']))
            data={stage:{key:f[stage+'/'+key][:] for key in ('path_offsets','node_indices','arrival_times','cost_seconds')}
                  for stage in ('G0','G1')}
            for stage,weights in (('G0',knots0),('G1',knots1)):
                z=data[stage];max_edge=max(max_edge,replay(gids,n,z['path_offsets'],z['node_indices'],
                    z['arrival_times'],z['cost_seconds'],keys,weights))
            flags=f['point_flags'][:];celloff=f['cell_offsets'][:];sparse=f['cells'][:];progress=f['progress'][:]
            hotoff=f['internal_hot_offsets'][:];hotcells=f['internal_hot_cells'][:]
            assert flags.shape==(2*count,len(FLAG_NAMES))
            assert len(celloff)==len(hotoff)==2*count+1
            assert celloff[0]==hotoff[0]==0 and celloff[-1]==len(sparse)==len(progress) and hotoff[-1]==len(hotcells)
            assert progress.dtype==np.float32 and np.isfinite(progress).all()
            for point,departure in enumerate((43200,44100)):
                for i,gid in enumerate(gids):
                    row=point*count+i;g0,g1=data['G0'],data['G1']
                    sl=slice(g1['path_offsets'][row],g1['path_offsets'][row+1])
                    oldsl=slice(g0['path_offsets'][row],g0['path_offsets'][row+1])
                    path=g1['node_indices'][sl];arrival=g1['arrival_times'][sl]
                    previous=g0['node_indices'][oldsl];cost=g1['cost_seconds'][row]
                    last={int(cells[node]):float(t) for node,t in zip(path,arrival)}
                    expected_cells=np.asarray(sorted(last),np.uint16)
                    actual=slice(celloff[row],celloff[row+1])
                    assert np.array_equal(sparse[actual],expected_cells)
                    expected_progress=np.asarray([2*(last[int(cell)]-departure)/cost-1 for cell in expected_cells],np.float32)
                    error=float(np.max(np.abs(progress[actual]-expected_progress),initial=0))
                    max_progress=max(max_progress,error);assert error<1e-7
                    internal=set(int(cells[node]) for node in path[1:-1]) & hot
                    endpoint=bool({int(cells[path[0]]),int(cells[path[-1]])} & hot)
                    assert np.array_equal(hotcells[hotoff[row]:hotoff[row+1]],np.asarray(sorted(internal),np.uint16))
                    pair_hit=changed=False
                    for j,(u,v) in enumerate(zip(path[:-1],path[1:])):
                        if min(int(u),int(v))*n+max(int(u),int(v)) not in significant:continue
                        pair_hit=True;edge=int(np.searchsorted(keys,int(u)*n+int(v)))
                        changed |= weight_at(knots0,edge,arrival[j])!=weight_at(knots1,edge,arrival[j])
                    path_change=not np.array_equal(previous,path)
                    expected=[pair_hit,changed,bool(internal),endpoint and not internal,path_change,
                              bool(g0['cost_seconds'][row]!=cost),changed or bool(internal) or path_change]
                    assert np.array_equal(flags[row],expected), (str(file),int(gid),departure,'evidence')
                    all_flags.setdefault(int(gid),np.empty((2,len(FLAG_NAMES)),bool))[point]=expected
                    all_hot.setdefault(int(gid),[None,None])[point]=internal
            witnesses=json.loads(f.attrs['rational_witnesses']);assert len(witnesses)==4
            for w in witnesses:
                point=(w['departure']-43200)//900;stage=w['stage'];z=data[stage];row=point*count
                path=z['node_indices'][z['path_offsets'][row]:z['path_offsets'][row+1]]
                assert int(gids[0] % n)==w['target']
                t=Fraction(w['departure']);weights=knots0 if stage=='G0' else knots1
                for u,v in zip(path[:-1],path[1:]):
                    edge=int(np.searchsorted(keys,int(u)*n+int(v)));slot=int(t//900)%96;r=t-(t//900)*900
                    t+=int(weights[edge,slot])+Fraction(int(weights[edge,(slot+1)%96]-weights[edge,slot]),900)*r
                assert t==Fraction(int(w['numerator']),int(w['denominator']))
                assert abs(float(t)-z['arrival_times'][z['path_offsets'][row+1]-1])<1e-7
            point_paths+=4*count;witness_count+=4;files.append(str(file.relative_to(run)))
        if len(files)%64==0:print(json.dumps(dict(city=city,validation_files=len(files),point_paths=point_paths)),flush=True)
    assert len(all_flags)*4==point_paths
    qualifications=[]
    for role in ('mixed_test','Aselect','Acal','sdc_Afit'):
        path=folder/(role+'_qualification.json')
        if not path.exists():
            assert qualification_only;continue
        q=json.loads(path.read_text());assert q['status']=='QUALIFIED_FULL_QUOTA'
        order=np.load(folder/(role+'_target_order_gids.npy'));screened=q['screened_groups']
        eligible=np.asarray([all_flags[int(gid)][:,-1].any() for gid in order[:screened]])
        assert int(eligible.sum())==q['eligible_groups_in_all_screened_chunks']
        chosen=np.load(folder/(role+'_targeted_gids.npy'));required=q['required_target_groups']
        assert len(chosen)==required and np.array_equal(chosen,order[:screened][eligible][:required])
        nth=np.flatnonzero(eligible)[required-1]
        expected_screened=min(len(order),(int(nth)//config['sampler']['qualification_chunk_groups']+1)*config['sampler']['qualification_chunk_groups'])
        assert screened==expected_screened
        qualifications.append(dict(role=role,screened_groups=screened,qualified_groups=int(eligible.sum()),selected=required))
    role_reports={}
    if not qualification_only:
        selection=json.loads((folder/'selection_manifest.json').read_text());assert selection['final_shared_holdout_frozen']
        b=config['budgets'][city]
        for role,points in (('mixed_test',b['mixed_test_points']),('Aselect',b['Aselect_points']),
                            ('Acal',b['Acal_points']),('sdc_Afit',b['Afit_points_per_FT']),('random_Afit',b['Afit_points_per_FT'])):
            gids=np.sort(np.load(folder/(role+'_gids.npy')))
            assert len(gids)*2==points and len(np.unique(gids))==len(gids)
            expected=np.asarray([all_flags[int(gid)] for gid in gids])
            assert np.array_equal(expected,np.load(folder/(role+'_point_flags.npy')))
            group_h=expected[:,:,-1].any(1)
            assert np.array_equal(group_h,np.load(folder/(role+'_affected_group_mask.npy')))
            if role!='random_Afit':
                targets=np.load(folder/(role+'_targeted_gids.npy'))
                ordinary=np.load(folder/(role+'_ordinary_gids.npy'))
                assert np.array_equal(gids,np.sort(np.concatenate((ordinary,targets))))
                target=np.load(folder/(role+'_targeted.npy'));assert int(target.sum())==len(gids)//5
                assert np.array_equal(target,np.isin(gids,targets))
            inside=[all_hot[int(gid)][point] for gid in gids for point in range(2)]
            visited=set().union(*inside)
            role_reports[role]=dict(points=points,groups=len(gids),affected_groups=int(group_h.sum()),
                affected_points=int(expected[:,:,-1].sum()),main_affected_group_points=int(group_h.sum())*2,
                hotspot_hit_points=sum(bool(s) for s in inside),hotspot_hit_percent=100*sum(bool(s) for s in inside)/points,
                visited_hotspots=len(visited),hotspot_denominator=len(hot),
                hotspot_coverage_percent=100*len(visited)/len(hot) if hot else None)
        generated=json.loads((folder/'generation_report.json').read_text())
        assert generated['status']=='GT_QUALIFICATION_AND_SELECTED_LABELS_COMPLETE'
        for role,report in role_reports.items():
            for key in ('points','groups','affected_groups','affected_points'):
                assert generated['roles'][role][key]==report[key]
    paid=accounting(run,city)
    assert paid['known_completed_point_path_restorations']>=point_paths
    assert paid['completed_sssp_calls']>=4*len(files)
    result=dict(status='QUALIFICATION_CACHE_VALIDATED' if qualification_only else 'DATA_ACCEPTED',city=city,
        scope='all complete cache files, including rejected and unneeded qualification candidates',
        complete_cache_files=len(files),unique_cached_groups=len(all_flags),replayed_point_paths=point_paths,
        independently_rebuilt_G1_point_labels=2*len(all_flags),independent_rational_replays=witness_count,
        max_edge_replay_error=max_edge,max_progress_error=max_progress,qualifications=qualifications,
        role_reports=role_reports,accounting=paid,formal_training_updates=0,
        rebuilt_component_point_counts={name:int(sum(value[:,i].sum() for value in all_flags.values())) for i,name in enumerate(FLAG_NAMES)},
        elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_hashes=True,recorded_unix=time.time())
    name='qualification_cache_validation.json' if qualification_only else 'data_acceptance.json'
    atomic_json(folder/name,result);print(json.dumps(result),flush=True)
    old.close();new.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--city',choices=('XA','BJ'),required=True);p.add_argument('--qualification-only',action='store_true')
    a=p.parse_args();check(a.run,a.city,a.qualification_only)


if __name__=='__main__':main()
