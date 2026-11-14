"""Actual96-stage SSP FT; dev selection and final confirmation remain separate."""
import argparse
import csv
import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from common import ROOT, PARENT, OLD, atomic, frozen_json, stamp
from model_io import mt, save_pt, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['BJ', 'XA'], required=True)
    parser.add_argument('--warm-epochs', choices=[16, 32, 64, 128], type=int, required=True)
    parser.add_argument('--epochs', choices=[8, 16, 32, 64], type=int, required=True)
    parser.add_argument('--wait-data-seconds', type=int, default=86400)
    args = parser.parse_args()
    city = args.city
    assert torch.cuda.is_available() and len(os.sched_getaffinity(0)) == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    runname = f'w{args.warm_epochs:03d}_e{args.epochs:02d}'
    out = ROOT / f'training/rolling/{city}/{runname}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = OLD / f'base_warmup/{city}/selected.pt' if args.warm_epochs == 16 else ROOT / f'training/warmup/{city}/e{args.warm_epochs:03d}/selected.pt'
    parent_stamp = stamp(parent)
    weights = torch.load(parent, map_location='cpu', weights_only=False)
    assert int(weights['epoch']) == args.warm_epochs
    if args.warm_epochs == 16:
        assert json.loads((parent.parent / 'ACCEPTANCE.json').read_text())['status'] == 'COMMON_G0_WHOLE_DAY_WARMUP_ACCEPTED'
    model = mt.DiT().cuda()
    model.load_state_dict(weights['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95), weight_decay=.01, eps=1e-8)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ng = torch.Generator(device='cuda').manual_seed(242)
    tg = torch.Generator(device='cuda').manual_seed(342)
    steps = args.epochs * 4
    total_steps = steps * 96
    rolepath = ROOT / f'inputs/rolling_roles/{city}.npz'
    config = dict(city=city, candidate=runname, warm_epochs=args.warm_epochs,
        FT_epochs_per_slot=args.epochs, steps_per_epoch=4, steps_per_slot=steps,
        total_steps=total_steps, slots=96, fit_points_per_slot=256, calibration_points_per_slot=64,
        development_ODs=128, parent=parent_stamp, cohort=stamp(rolepath),
        seed=42, batch=64, optimizer_reset_per_slot=False, final_test_read=False,
        peak_lr=1e-5, schedule='global cosine over96slots, warmup100', hash_verification=False)
    frozen_json(out / 'config.json', config)
    state = dict(completed_slots=0, global_update=0, train_rows_seen=0,
        prior_threshold=float(weights['threshold']))
    del weights
    if (out / 'completion.json').exists():
        print((out / 'completion.json').read_text(), flush=True)
        return
    if (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', map_location='cuda', weights_only=False)
        assert saved['config'] == config
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        ng.set_state(saved['ng'].cpu())
        tg.set_state(saved['tg'].cpu())
        mt._restore_rng(saved['rng'])
    support = np.bincount(np.load(PARENT / f'e1_v1/data/{city}/cell64.npy'), minlength=4096) > 0
    scratch = Path('/tmp') / f'{city}_{runname}_last_update.pt'
    actual_scratch = Path(os.environ['E4_LOCAL_SCRATCH']) / scratch.name

    def checkpoint(path, sync=False):
        save_pt(path, dict(config=config, state=dict(state), model=model.state_dict(),
            optimizer=optimizer.state_dict(), ng=ng.get_state(), tg=tg.get_state(), rng=mt._rng_state()), sync=sync)

    def progress(status, **extra):
        record = dict(status=status, city=city, candidate=runname, **state,
            pid=os.getpid(), host=os.environ.get('E4_EXECUTION_HOST'),
            gpu_uuid=os.environ.get('E4_SELECTED_GPU_UUID'), updated_unix=time.time(), **extra)
        atomic(out / 'progress.json', record)
        atomic(ROOT / f'status/rolling_{city.lower()}.json', record)

    if not (out / 'last.pt').exists():
        checkpoint(out / 'last.pt')
    for slot in range(state['completed_slots'] + 1, 97):
        folder = ROOT / f'data/rolling/{city}/v{slot:03d}'
        wait_start = time.monotonic()
        while not (folder / 'prepared.json').exists():
            progress('WAITING_FOR_SHARED_SLOT_LABELS', slot=slot, worker_started=True)
            assert time.monotonic() - wait_start < args.wait_data_seconds, 'Data wait expired; do not repeat any prior update'
            time.sleep(15)
        prepared = json.loads((folder / 'prepared.json').read_text())
        assert prepared['status'] == 'ALL_SHARED_SLOT_LABELS_READY_FOR_INDEPENDENT_REVIEW'
        assert prepared['cohort'] == config['cohort']
        assert stamp(parent) == parent_stamp
        phase = out / f'v{slot:03d}'
        phase.mkdir(exist_ok=True)
        assert state['global_update'] == (slot - 1) * steps
        # A partially executed phase has an external scratch checkpoint and must be
        # recovered explicitly, never replayed from the prior phase's model.
        journalpath = phase / 'updates.jsonl'
        assert not journalpath.exists(), 'Partial slot found; recover its saved scratch checkpoint before resuming'
        before = parent if slot == 1 else out / f'v{slot - 1:03d}/model.pt'
        frozen_json(phase / 'before_state.json', dict(previous_slot=slot - 1,
            global_update=state['global_update'], checkpoint=stamp(before),
            optimizer_continues=True, RNG_continues=True, graph=prepared['graph']))
        train = dict(np.load(folder / 'train.npz'))
        cal = dict(np.load(folder / 'cal.npz'))
        dev = dict(np.load(folder / 'dev.npz'))
        assert len(train['image']) == 256 and len(cal['image']) == 64 and len(dev['image']) == 128
        image = torch.from_numpy(train['image']).cuda()
        conditions = torch.from_numpy(train['odt']).cuda()
        model.train()
        update_started = time.time()
        compute_total = checkpoint_total = 0.
        losses = []
        with journalpath.open('x') as journal:
            for local in range(steps):
                number = (slot - 1) * steps + local + 1
                epoch, block = divmod(local, 4)
                indexes = np.random.default_rng(142 + (slot - 1) * args.epochs + epoch).permutation(256)[block * 64:(block + 1) * 64]
                intent = dict(status='UPDATE_INTENT', slot=slot, global_update=number,
                    local_update=local + 1, rows=indexes.tolist(), unix=time.time())
                journal.write(json.dumps(intent) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                lr = mt.lr_at_update(number, 1e-5, total_steps, schedule='cosine', warmup=100)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                index = torch.from_numpy(indexes).cuda()
                batch, odt = image[index], conditions[index]
                torch.cuda.synchronize()
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                noise, times = mt.draw_noise_times(batch, ng, tg)
                loss = mt.cfm_loss(model, batch, odt, noise, times, fg_weight=10)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                torch.cuda.synchronize()
                compute_seconds = time.monotonic() - started
                state['global_update'] = number
                state['train_rows_seen'] += 64
                started = time.monotonic()
                checkpoint(scratch, sync=True)
                checkpoint_seconds = time.monotonic() - started
                done = dict(status='COMMITTED', slot=slot, global_update=number, rows=64,
                    loss=float(loss.detach().cpu()), lr=lr, optimizer_compute_seconds=compute_seconds,
                    local_checkpoint_seconds=checkpoint_seconds, checkpoint_host_path=str(actual_scratch),
                    checkpoint_worker_path=str(scratch),
                    checkpoint_mount_binding=dict(worker='/tmp', host=os.environ['E4_LOCAL_SCRATCH']),
                    optimizer_state_continued=True, RNG_continued=True)
                journal.write(json.dumps(done) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                compute_total += compute_seconds
                checkpoint_total += checkpoint_seconds
                losses.append(done['loss'])
                progress('TRAINING', slot=slot)
        update_finished = time.time()
        del image, conditions
        # Commit one portable full state per slot and retain pre/post lineage.
        model_identity = dict(candidate=runname, slot=slot, global_update=state['global_update'],
            initial_G0=parent_stamp, graph=prepared['graph'])
        progress('CALIBRATING', slot=slot)
        calibration = evaluate(model, cal, phase / 'calibration', city, slot, 'cal', support, model_identity)
        progress('EVALUATING_DEVELOPMENT', slot=slot)
        development = evaluate(model, dev, phase / 'development', city, slot, 'dev', support,
            model_identity, threshold=calibration['threshold'])
        state.update(completed_slots=slot, prior_threshold=calibration['threshold'])
        checkpoint(phase / 'model.pt')
        checkpoint(out / 'last.pt')
        atomic(phase / 'result.json', dict(status='SLOT_COMPLETE_PENDING_INDEPENDENT_AUDIT',
            city=city, candidate=runname, slot=slot, release_available_clock_seconds=(slot - 1) * 900,
            evaluate_departure_seconds=(slot - 1) * 900, update_start_unix=update_started,
            update_end_unix=update_finished, updates=steps, epochs=args.epochs,
            cumulative_updates=state['global_update'], optimizer_reset=False,
            optimizer_compute_seconds=compute_total, checkpoint_seconds=checkpoint_total,
            mean_train_loss=float(np.mean(losses)), calibration=calibration, development=development,
            graph=prepared['graph'], before_state=stamp(phase / 'before_state.json'),
            after_state=stamp(phase / 'model.pt'), final_test_read=False))
        progress('SLOT_COMPLETE', slot=slot)
    rows = []
    for slot in range(1, 97):
        result = json.loads((out / f'v{slot:03d}/result.json').read_text())
        metric = result['development']
        rows.append(dict(city=city, method='SSP', slot=slot, clock_seconds=(slot - 1) * 900,
            points=metric['points'], tp=metric['tp'], fp=metric['fp'], fn=metric['fn'],
            precision_percent=100 * metric['precision'], recall_percent=100 * metric['recall'],
            f1_percent=100 * metric['f1'], macro_point_f1_percent=100 * metric['macro_point_f1'],
            optimizer_compute_seconds=result['optimizer_compute_seconds']))
    with (out / 'development.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    f1 = np.asarray([row['f1_percent'] for row in rows])
    tp, fp, fn = [sum(row[key] for row in rows) for key in ['tp', 'fp', 'fn']]
    summary = dict(status='COMPLETE_DEVELOPMENT_DAY_PENDING_INDEPENDENT_AUDIT', city=city,
        candidate=runname, slots=96, points=12288, median_slot_micro_f1_percent=float(np.median(f1)),
        pooled_micro_f1_percent=200 * tp / (2 * tp + fp + fn),
        mean_slot_micro_f1_percent=float(f1.mean()), p10_slot_micro_f1_percent=float(np.quantile(f1, .1)),
        minimum_slot_micro_f1_percent=float(f1.min()), slots_below90=int(np.count_nonzero(f1 < 90)),
        full_development_target_met=bool(np.median(f1) >= 90), final_test_read=False,
        global_updates=state['global_update'], train_rows=state['train_rows_seen'],
        final_checkpoint=stamp(out / 'last.pt'), config=config)
    atomic(out / 'completion.json', summary)
    progress('FINISHED_DEVELOPMENT', development_summary=summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
