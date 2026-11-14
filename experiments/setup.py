"""Stage experiment code and configure input/output paths in a fresh workspace."""
import argparse
import json
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    root = parser.parse_args().workspace.resolve()
    here = Path(__file__).resolve().parent
    runs = root / 'runs/sdc_ssp64_od49_56'
    runs.mkdir(parents=True, exist_ok=False)
    e1, e2, e3 = [runs / f'e{i}_v1' for i in range(1, 4)]
    for index, (name, target) in enumerate(zip(('g0', 'g1', 'baseline'), (e1, e2, e3)), 1):
        cfg = json.loads((here / f'e{index}.json').read_text())
        cfg.update(remote_root=str(root), run=str(target))
        if index == 2:
            cfg.update(parent_e1_run=str(e1), required_parent_acceptance=str(e1/'E1_ACCEPTANCE.json'))
        if index == 3:
            cfg.update(parent_e1=str(e1), parent_e2=str(e2))
        shutil.copytree(here/name, target/'code', ignore=shutil.ignore_patterns('__pycache__'))
        (target/'protocol.json').write_text(json.dumps(cfg, indent=2)+'\n')
    print(runs)

if __name__ == '__main__':
    main()
