"""Frozen width-study locations and crash-safe output helpers; no hashes."""
from pathlib import Path
import json
import os

ROOT = Path(os.environ['DIGO_ROLLING_RUN']).resolve()
P = ROOT.parent
W = ROOT/'width'
SETTINGS = [('BJ','G0',800),('XA','G0',344),('BJ','G1',2971),('XA','G1',1160)]
WIDTHS = (1800,3600,7200,14400)
ARMS = ('full_graph','union_ges','single_expansion','union_expansion')

def stamp(path):
    path=Path(path);s=path.stat()
    return dict(path=str(path),size=s.st_size,mtime_ns=s.st_mtime_ns)

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as h:
        json.dump(value,h,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
    tmp.replace(path)

def immutable(path,value):
    path=Path(path)
    if path.exists():assert json.loads(path.read_text())==value,path
    else:atomic(path,value)

def checked(metadata):
    path=Path(metadata['path']);assert stamp(path)==metadata,path
    return path

def modeldir(city,snapshot):
    return P/('e1_v1/training' if snapshot=='G0' else 'e2_v1/training')/(f'{city}_hierarchical_grid_seed42' if snapshot=='G0' else f'{city}_sdc_mixed_ft_seed42')

def require_registration():
    data=json.loads((W/'WIDTH_PROTOCOL.json').read_text())
    assert data['version']=='width_v3.1_banked16_all5275'
    assert data['counts']==dict(BJ_G0=800,XA_G0=344,BJ_G1=2971,XA_G1=1160)
    return data
