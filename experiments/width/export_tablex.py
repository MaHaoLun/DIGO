"""Export all48 learned rows only after independent full84400 acceptance.

No experimental calls or outcome filtering. Root/D own final TeX formatting.
"""
import csv
import json
from pathlib import Path
from common import *

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def main():
    require_registration()
    gatepath=ROOT/'reviews/width_final_acceptance.json'
    gate=json.loads(gatepath.read_text());assert gate['status']=='PASS_ALL84400_WIDTH_ROWS'
    result=W/'results';aggregation=json.loads((result/'AGGREGATION.json').read_text())
    if(result/'TABLEX_EXPORT.json').exists():
        previous=json.loads((result/'TABLEX_EXPORT.json').read_text())
        for key in ['source_summary','producer_aggregation','independent_acceptance','export','reference_export']:checked(previous[key])
        print(json.dumps(dict(status='EXISTING_ACCEPTED48ROW_EXPORT_REUSED',learned_rows=48,new_model_calls=0,new_solver_calls=0)),flush=True)
        return
    assert aggregation['logical_rows']==84400 and aggregation['cases']==21100
    assert aggregation['unique_new_native_calls']==72801 and aggregation['old4h_rows']==10550
    source=result/'summary.csv';rows=read(source);assert len(rows)==64
    mapping={'union_ges':'Union + GES','single_expansion':'Single + expansion','union_expansion':'Union + expansion'}
    by={(r['city'],r['snapshot'],int(r['width_seconds']),r['arm']):r for r in rows}
    assert len(by)==64
    learned=[];references=[]
    for city,snap,n in SETTINGS:
        for w in WIDTHS:
            ref=by[city,snap,w,'full_graph'];assert int(ref['n'])==n
            references.append(dict(city=city,snapshot=snap,interval_hours=str(w/3600).removesuffix('.0'),
                n=n,reference_mean_seconds=ref['composed_mean_seconds'],reference_p95_seconds=ref['composed_p95_seconds']))
            for arm in ['union_ges','single_expansion','union_expansion']:
                r=by[city,snap,w,arm];assert int(r['n'])==n
                learned.append(dict(city=city,snapshot=snap,interval_hours=str(w/3600).removesuffix('.0'),
                    method=mapping[arm],n=n,valid_percent=r['valid_percent'],optimal_percent=r['optimal_percent'],
                    gap_mean_percent=r['gap_mean_percent'],composed_mean_seconds=r['composed_mean_seconds'],
                    composed_p95_seconds=r['composed_p95_seconds'],reference_mean_seconds=r['reference_mean_seconds'],
                    speedup=r['ratio_of_means'],bank_paid_conditions=r['bank_paid_conditions'],
                    region_prefix_conditions=r['region_prefix_conditions'],source_arm=arm))
    assert len(learned)==48 and len(references)==16
    anchor=json.loads((W/'full_4h_anchor.json').read_text())['settings']
    for city,snap,n in SETTINGS:
        r=by[city,snap,14400,'union_ges'];a=anchor[city+'_'+snap]
        for k in ['optimal_percent','composed_mean_seconds','ratio_of_means']:assert abs(float(r[k])-a[k])<1e-10
    write_csv(result/'TableX_learned48.csv',learned)
    write_csv(result/'TableX_reference16.csv',references)
    report=dict(status='ALL48_LEARNED_ROWS_EXPORTED_WITH_INDEPENDENT_FULL_ACCEPTANCE',
        learned_rows=48,reference_rows=16,logical_rows=84400,ODs=5275,
        counts=dict(BJ_G0=800,XA_G0=344,BJ_G1=2971,XA_G1=1160),
        full_4h_anchor_reproduced=True,old4h_rows=10550,unique_new_native_calls=72801,
        source_summary=stamp(source),producer_aggregation=stamp(result/'AGGREGATION.json'),
        independent_acceptance=stamp(gatepath),export=stamp(result/'TableX_learned48.csv'),
        reference_export=stamp(result/'TableX_reference16.csv'),
        evidence_scope='Retrospective TableIII selected cohort, same5275 ODs for each width; no new outcome selection or claim of independent generalization.',
        neural_budget='All Union windows pay original16-condition bank; only region prefix changes2/4/8/16. Actual Single pays one originalbatch1 condition. Additional CPU prefix/expansion/handoff is counted conservatively.',
        timing_scope='Composed separately observed prediction/interface and actual searchAPI costs, same-host paired references; not same-call onlineE2E.',
        point_and_rolling_scopes='Separate studies; this export does not certify PointV2 or900second adaptation.',
        raw_data_rounding='Preserved source CSV numeric strings; final paper formatting is downstream.',
        new_model_calls=0,new_solver_calls=0,hash_verification=False)
    immutable(result/'TABLEX_EXPORT.json',report)
    atomic(ROOT/'status/width.json',dict(status='INDEPENDENTLY_ACCEPTED_FULL_WIDTH_DATA_AND48ROW_EXPORT',
        logical_rows=84400,cases=21100,ODs=5275,learned_export_rows=48,
        original_worker_dispatches=24,all_original_workers_exit0=True,
        old4h_reuse_rows=10550,new_native_calls=72801,method9_alias_rows=1049,
        independent_acceptance=report['independent_acceptance'],export_receipt=stamp(result/'TABLEX_EXPORT.json'),
        evidence_scope=report['evidence_scope'],performance_target='Four-hour raw anchor exactly reproduced; other widths/expansions retain actual measured results.',
        query_model_calls=0,required_G0_calibration_points=75008,hash_verification=False))
    print(json.dumps({k:report[k] for k in ['status','learned_rows','reference_rows','logical_rows','full_4h_anchor_reproduced']}),flush=True)

if __name__=='__main__':main()
