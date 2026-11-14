"""Qualify frozen target order, then emit complete selected-role manifests."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback
import numpy as np
from gt_cache import Cache, FLAG_NAMES, accounting, atomic_array, atomic_json


def frozen_array(path, value):
    if path.exists(): assert np.array_equal(np.load(path), value), str(path)
    else: atomic_array(path, value)


def run(args):
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        cache = Cache(args.run, args.city, args.workers)
    except BaseException as exc:
        atomic_json(args.run/'receipts'/args.city/('setup_failure_'+str(time.time_ns())+'.json'),
            dict(status='SETUP_FAILED_BEFORE_GT_CALLS', error=repr(exc), traceback=traceback.format_exc(),
                 elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu_started,
                 formal_point_paths_computed=0))
        raise
    config, folder = cache.config, cache.folder
    process_stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    worker = dict(pid=os.getpid(), start_time=process_stat[19],
        arguments=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        city=args.city, only_role=args.only_role, attempt=cache.attempt)
    atomic_json(folder/'GT_worker_status.json', dict(status='RUNNING', **worker, formal_training_updates=0))
    def interrupt(signum, frame):
        cache.stop.set(); raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    session_status = 'FAILED'
    roles = ('mixed_test', 'Aselect', 'Acal', 'sdc_Afit')
    wanted_roles = (args.only_role,) if args.only_role else roles
    try:
        reports = []
        for role in wanted_roles:
            target_file = folder/(role+'_targeted_gids.npy')
            summary_file = folder/(role+'_qualification.json')
            if summary_file.exists():
                prior = json.loads(summary_file.read_text()); assert prior['status'] == 'QUALIFIED_FULL_QUOTA'
                assert target_file.exists(); reports.append(prior); continue
            static = json.loads((folder/'static_preflight.json').read_text())['roles'][role]
            candidate_order = np.load(folder/(role+'_target_order_gids.npy'))
            required = static['required_targeted_groups']; selected = []; screened = eligible_total = 0
            chunk_size = config['sampler']['qualification_chunk_groups']
            for first in range(0, len(candidate_order), chunk_size):
                part = candidate_order[first:first+chunk_size]
                cache.make_block(f'qual_{role}_{first:07d}', part)
                flags = cache.flags(part); eligible = flags[:, :, FLAG_NAMES.index('affected')].any(1)
                screened += len(part); eligible_total += int(eligible.sum())
                remaining = required-len(selected)
                selected.extend(map(int, part[eligible][:remaining]))
                atomic_json(folder/(role+'_qualification_progress.json'), dict(status='QUALIFYING',
                    role=role, screened_groups=screened, eligible_groups_in_all_screened_chunks=eligible_total,
                    selected_target_groups=len(selected), required_target_groups=required,
                    all_screened_points_charged=True, cached_groups=len(cache.index)))
                if len(selected) == required: break
            if len(selected) != required:
                failure = dict(status='INSUFFICIENT_QUALIFIED_TARGET_CAPACITY', city=args.city, role=role,
                    candidate_groups=len(candidate_order), required_target_groups=required,
                    eligible_target_groups=eligible_total, selected_target_groups=len(selected), screened_groups=screened,
                    no_threshold_or_budget_relaxation=True, accounting=accounting(args.run, args.city))
                atomic_json(folder/(role+'_capacity_failure.json'), failure)
                raise RuntimeError(json.dumps(failure))
            selected = np.asarray(selected, np.int64); assert len(np.unique(selected)) == required
            frozen_array(target_file, selected)
            result = dict(status='QUALIFIED_FULL_QUOTA', city=args.city, role=role,
                required_target_groups=required, selected_target_groups=len(selected), screened_groups=screened,
                eligible_groups_in_all_screened_chunks=eligible_total,
                qualification_computes_complete_chunks=True, selected_by_frozen_order=True,
                no_model_predictions_used=True, accounting=accounting(args.run, args.city))
            atomic_json(summary_file, result); reports.append(result)
            print(json.dumps(result), flush=True)
            cache.save_index()
        if args.only_role:
            session_status = 'COMPLETE'
            return dict(status='ONE_ROLE_QUALIFIED_FULL_QUOTA', city=args.city, role=args.only_role,
                        reports=reports, formal_training_updates=0)
        selected_roles = {}
        for role in roles:
            ordinary = np.load(folder/(role+'_ordinary_gids.npy'))
            targeted = np.load(folder/(role+'_targeted_gids.npy'))
            merged = np.concatenate((ordinary, targeted)); order = np.argsort(merged)
            assert len(np.unique(merged)) == len(merged)
            selected_roles[role] = merged[order]
            frozen_array(folder/(role+'_gids.npy'), merged[order])
            frozen_array(folder/(role+'_targeted.npy'), np.concatenate((np.zeros(len(ordinary), bool),
                                                                      np.ones(len(targeted), bool)))[order])
        selected_roles['random_Afit'] = np.sort(np.load(folder/'random_Afit_gids.npy'))
        selection = dict(status='FINAL_SHARED_ROLE_MANIFESTS_FROZEN', city=args.city,
            protocol_version=config['version'], selected_groups={name:len(gids) for name,gids in selected_roles.items()},
            model_blind=True, formal_training_updates=0, final_shared_holdout_frozen=True)
        manifest = folder/'selection_manifest.json'
        if manifest.exists(): assert json.loads(manifest.read_text()) == selection
        else: atomic_json(manifest, selection)
        # Ordinary and Random groups are never filtered by affected status.
        all_selected = np.unique(np.concatenate(list(selected_roles.values())))
        cache.make_block('selected_missing', all_selected)
        role_summaries = {}
        for role, gids in selected_roles.items():
            flags = cache.flags(gids)
            point_affected = flags[:, :, FLAG_NAMES.index('affected')]
            group_affected = point_affected.any(1)
            atomic_array(folder/(role+'_point_flags.npy'), flags)
            atomic_array(folder/(role+'_affected_group_mask.npy'), group_affected)
            role_summaries[role] = dict(groups=len(gids), points=2*len(gids),
                affected_groups=int(group_affected.sum()), affected_points=int(point_affected.sum()),
                affected_group_retained_points=int(group_affected.sum())*2,
                source_quota_is_not_affected_fraction=True)
        cache.save_index(); cache.check_inputs()
        actual = accounting(args.run, args.city)
        result = dict(status='GT_QUALIFICATION_AND_SELECTED_LABELS_COMPLETE', city=args.city,
            roles=role_summaries, qualifications=reports, unique_cached_groups=len(cache.index),
            unique_successful_point_paths=4*len(cache.index), all_selected_groups=len(all_selected),
            final_shared_holdout_frozen=True, no_model_predictions_used=True,
            independent_acceptance='PENDING_FULL_REPLAY', formal_training_updates=0,
            accounting=actual, elapsed_seconds=time.monotonic()-started,
            process_cpu_seconds=time.process_time()-cpu_started,
            elapsed_scope='this invocation including setup; cumulative invocation receipts are separate',
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(folder/'generation_report.json', result)
        session_status = 'COMPLETE'
        print(json.dumps(result), flush=True); return result
    except BaseException as exc:
        cache.stop.set()
        atomic_json(folder/('GT_failure_'+str(time.time_ns())+'.json'), dict(status='FAILED',
            error=repr(exc), traceback=traceback.format_exc(), accounting=accounting(args.run, args.city),
            elapsed_seconds=time.monotonic()-started, completed_cache_groups=len(cache.index)))
        raise
    finally:
        receipt = dict(status=session_status, attempt=cache.attempt, city=args.city,
            only_role=args.only_role, elapsed_seconds=time.monotonic()-started,
            process_cpu_seconds=time.process_time()-cpu_started,
            scope='this invocation including graph setup, cached-data reading, qualification and emission',
            completed_cache_groups=len(cache.index), formal_training_updates=0)
        receipts = args.run/'receipts'/args.city
        atomic_json(receipts/('gt_session_'+cache.attempt+'.json'), receipt)
        sessions = [json.loads(p.read_text()) for p in receipts.glob('gt_session_*.json')]
        cost = accounting(args.run, args.city)
        cost.update(total_GT_session_process_cpu_seconds=sum(s['process_cpu_seconds'] for s in sessions),
            sum_GT_session_elapsed_seconds=sum(s['elapsed_seconds'] for s in sessions),
            elapsed_sum_scope='sum of process invocation wall times; not cross-city end-to-end latency',
            sessions=sessions)
        atomic_json(folder/'GT_cost_accounting.json', cost)
        atomic_json(folder/'GT_worker_status.json', dict(status=session_status, **worker,
            completed_unix=time.time(), completed_cache_groups=len(cache.index), formal_training_updates=0))
        cache.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--city', choices=('XA', 'BJ'), required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--only-role', choices=('mixed_test', 'Aselect', 'Acal', 'sdc_Afit'))
    args = p.parse_args(); run(args)


if __name__ == '__main__': main()
