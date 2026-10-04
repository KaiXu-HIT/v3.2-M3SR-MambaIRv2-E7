"""E7 image-paired validation and 10,000-resample image-cluster bootstrap.

E6 trained checkpoints and matched seeds are reused. Separate test workers
mirror E6's sorted-data BasicSR validation, but retain per-image PSNR/SSIM.
Bootstrap resamples image IDs while retaining all seeds for each sampled image.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr.evaluate_e6_multiseed import (DATASETS, checkpoint_provenance,
                                               read_plan, validate_configs)

BOOTSTRAP_REPLICATES = 10000


def validate_e6_report(plan, report, provenance):
    if report['seeds'] != plan['seeds'] or tuple(report['per_dataset']) != DATASETS:
        raise ValueError('E7 needs the complete matched E6 seed/dataset report.')
    for label in ('rgb', 'udrv2'):
        for seed in plan['seeds']:
            expected = provenance[label][seed]['sha256']
            recorded = report['checkpoints'][label][str(seed)]['sha256']
            if expected != recorded:
                raise ValueError(f'E6 reported {label} seed {seed} weights changed.')


def pair_model_rows(baseline, final, seeds):
    """Join only exact dataset/image/seed matches; reject missing or duplicates."""
    def index(rows):
        result = {}
        for row in rows:
            key = row['dataset'], row['image_id'], int(row['seed'])
            if key in result:
                raise ValueError(f'Duplicate E7 per-image result: {key}')
            if not all(np.isfinite(float(row[field])) for field in ('psnr', 'ssim')):
                raise ValueError(f'Nonfinite E7 metric: {key}')
            result[key] = row
        return result
    rgb, e4 = index(baseline), index(final)
    if set(rgb) != set(e4):
        raise ValueError('RGB and UDR-v2 did not test the exact same image/seed keys.')
    if not rgb or set(key[2] for key in rgb) != set(seeds):
        raise ValueError('E7 is missing a planned seed.')
    order = {name: i for i, name in enumerate(DATASETS)}
    rows = []
    for dataset, image, seed in sorted(rgb, key=lambda key:
                                       (order[key[0]], key[1], key[2])):
        b, v = rgb[(dataset, image, seed)], e4[(dataset, image, seed)]
        rows.append(dict(dataset=dataset, image_id=image, seed=seed,
                         baseline_psnr=float(b['psnr']),
                         udrv2_psnr=float(v['psnr']),
                         delta_psnr=float(v['psnr'])-float(b['psnr']),
                         baseline_ssim=float(b['ssim']),
                         udrv2_ssim=float(v['ssim']),
                         delta_ssim=float(v['ssim'])-float(b['ssim'])))
    for dataset in DATASETS:
        images = {row['image_id'] for row in rows if row['dataset'] == dataset}
        if not images:
            raise ValueError(f'E7 has no images for {dataset}.')
        for image in images:
            observed = [row['seed'] for row in rows if row['dataset'] == dataset
                        and row['image_id'] == image]
            if sorted(observed) != sorted(seeds):
                raise ValueError(f'E7 image {dataset}/{image} lacks all paired seeds.')
    return rows


def reconcile_e6(rows, e6_report, seeds, tolerance=1e-6):
    """Image means must reproduce E6's original aggregate test worker."""
    for dataset in DATASETS:
        for seed_index, seed in enumerate(seeds):
            selected = [row for row in rows if row['dataset'] == dataset
                        and row['seed'] == seed]
            if not selected:
                raise ValueError(f'E7 has no {dataset} rows for seed {seed}.')
            for label, prefix in (('rgb', 'baseline'), ('udrv2', 'udrv2')):
                for metric in ('psnr', 'ssim'):
                    actual = statistics.mean(row[f'{prefix}_{metric}'] for row in selected)
                    expected = e6_report['per_dataset'][dataset][label]
                    expected = expected['metrics'][metric]['by_seed'][seed_index]
                    if abs(actual - expected) > tolerance:
                        raise ValueError(
                            f'E7 image means differ from E6 at {dataset}/{label}/'
                            f'seed {seed}/{metric}: {actual} versus {expected}.')


def image_cluster_bootstrap(rows, seeds, rng, replicates=BOOTSTRAP_REPLICATES):
    """Resample paired images, carrying their complete seed vector together."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['image_id'], {})[row['seed']] = row
    image_ids = sorted(grouped)
    if not image_ids:
        raise ValueError('Bootstrap needs at least one image.')
    for image in image_ids:
        if set(grouped[image]) != set(seeds):
            raise ValueError(f'Bootstrap image {image} lacks matched seeds.')
    image_values = np.asarray([
        statistics.mean(grouped[image][seed]['delta_psnr'] for seed in seeds)
        for image in image_ids], dtype=np.float64)
    if not np.isfinite(image_values).all():
        raise ValueError('Bootstrap deltas must be finite.')
    indices = rng.integers(0, len(image_ids), size=(replicates, len(image_ids)))
    sampled_means = image_values[indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [.025, .975])
    point = float(image_values.mean())
    extremes = [dict(image_id=image_ids[int(i)],
                     mean_delta_psnr=float(image_values[i]),
                     seed_deltas=[float(grouped[image_ids[int(i)]][seed]['delta_psnr'])
                                  for seed in seeds])
                for i in range(len(image_ids))]
    ranked = sorted(extremes, key=lambda item:
                    (item['mean_delta_psnr'], item['image_id']))
    per_seed = {str(seed): dict(
        mean_delta_psnr=statistics.mean(grouped[image][seed]['delta_psnr']
                                        for image in image_ids),
        positive_image_ratio=statistics.mean(
            grouped[image][seed]['delta_psnr'] > 0 for image in image_ids))
        for seed in seeds}
    if low > 0:
        verdict = 'positive_ci_excludes_zero'
    elif point > 0 and low <= 0 <= high:
        verdict = 'positive_trend_ci_crosses_zero'
    elif high < 0:
        verdict = 'negative_ci_excludes_zero'
    else:
        verdict = 'inconclusive'
    return dict(images=len(image_ids), seeds=list(seeds),
                mean_delta_psnr=point,
                ci95_psnr=[float(low), float(high)],
                bootstrap_replicates=replicates,
                positive_image_ratio=float(np.mean(image_values > 0)),
                median_delta_psnr=float(np.median(image_values)),
                p25_delta_psnr=float(np.percentile(image_values, 25)),
                p75_delta_psnr=float(np.percentile(image_values, 75)),
                worst10=ranked[:10],
                best10=list(reversed(ranked[-10:])),
                per_seed=per_seed, verdict=verdict)


def analyze(rows, seeds, bootstrap_seed, replicates=BOOTSTRAP_REPLICATES):
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError('The formal E7 experiment requires exactly 10,000 resamples.')
    streams = np.random.SeedSequence(bootstrap_seed).spawn(len(DATASETS))
    results = {}
    for dataset, stream in zip(DATASETS, streams):
        subset = [row for row in rows if row['dataset'] == dataset]
        results[dataset] = image_cluster_bootstrap(
            subset, seeds, np.random.default_rng(stream), replicates)
    return dict(seeds=list(seeds), bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=replicates,
                unit='paired image ID; all matched seeds stay in its cluster',
                datasets=results,
                note='Image bootstrap interval is conditional on the observed trained seeds; E6 reports seed-to-seed variation separately.')


def write_outputs(output, rows, report):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'E7_per_image.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'dataset', 'image_id', 'seed', 'baseline_psnr', 'udrv2_psnr',
            'delta_psnr', 'baseline_ssim', 'udrv2_ssim', 'delta_ssim'))
        writer.writeheader()
        writer.writerows(rows)
    (output/'E7_statistics.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8')
    lines = ['# E7 image-level paired bootstrap', '',
             f"{report['bootstrap_replicates']:,} paired image resamples; seeds {report['seeds']}; "
             f"bootstrap RNG seed {report['bootstrap_seed']}.", '',
             '| Dataset | Images | Mean ΔPSNR | 95% CI | Positive image ratio | Median | P25 / P75 | Verdict |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for dataset in DATASETS:
        item = report['datasets'][dataset]
        low, high = item['ci95_psnr']
        lines.append(f"| {dataset} | {item['images']} | {item['mean_delta_psnr']:+.4f} | "
                     f"[{low:+.4f}, {high:+.4f}] | "
                     f"{item['positive_image_ratio']:.1%} | "
                     f"{item['median_delta_psnr']:+.4f} | "
                     f"{item['p25_delta_psnr']:+.4f} / {item['p75_delta_psnr']:+.4f} | "
                     f"{item['verdict']} |")
    lines += ['', 'A wholly positive CI supports a stronger statistical claim. '
              'A CI crossing zero supports only a positive average trend when the mean is positive.',
              'JSON includes worst-10/best-10 images and per-seed image ratios. '
              'The CSV preserves every exact image/seed RGB–UDR-v2 pair.', '']
    (output/'E7_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def worker(config_path, output_path):
    """Mirror E6's sorted BasicSR validation, retaining each image metric."""
    import torch
    from basicsr.data import build_dataset, build_dataloader
    from basicsr.metrics import calculate_metric
    from basicsr.models import build_model
    from basicsr.utils import make_exp_dirs, set_random_seed
    from basicsr.utils.img_util import tensor2img
    from basicsr.utils.options import parse_options

    sys.argv = [sys.argv[0], '-opt', str(config_path)]
    opt, _ = parse_options(str(ROOT), is_train=False)
    make_exp_dirs(opt)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(opt)
    rows = []
    for _, dsopt in sorted(opt['datasets'].items()):
        dataset = build_dataset(dsopt)
        dataset.paths.sort(key=lambda item: item['lq_path'])
        loader = build_dataloader(dataset, dsopt, num_gpu=1, dist=False,
                                  sampler=None, seed=opt['manual_seed'])
        set_random_seed(opt['manual_seed'])
        for item in loader:
            model.feed_data(item)
            model.test()
            visuals = model.get_current_visuals()
            metric_data = dict(img=tensor2img([visuals['result']]),
                               img2=tensor2img([visuals['gt']]))
            metrics = {name: float(calculate_metric(metric_data, metric_opt))
                       for name, metric_opt in opt['val']['metrics'].items()}
            rows.append(dict(dataset=dsopt['name'],
                             image_id=Path(item['gt_path'][0]).stem,
                             seed=opt['manual_seed'],
                             psnr=metrics['psnr'], ssim=metrics['ssim']))
            # Match the original E6 validation's lifetime of image tensors.
            del model.gt, model.lq, model.output
            if hasattr(model, 'depth'):
                del model.depth
            torch.cuda.empty_cache()
    Path(output_path).write_text(json.dumps(rows, indent=2, allow_nan=False),
                                 encoding='utf-8')


def run(args):
    plan = read_plan(args.plan)
    e6_report = json.loads(Path(args.e6_report).read_text(encoding='utf-8'))
    provenance = checkpoint_provenance(plan, ['rgb', 'udrv1', 'udrv2'])
    validate_e6_report(plan, e6_report, provenance)
    config_paths = dict(rgb=args.rgb_config, udrv2=plan['e4_test_config'])
    configs = {label: yaml.safe_load(Path(path).read_text(encoding='utf-8'))
               for label, path in config_paths.items()}
    # validate_configs expects v1 too; only RGB/v2 are re-evaluated for E7.
    v1 = yaml.safe_load(Path(args.udrv1_config).read_text(encoding='utf-8'))
    validate_configs(dict(rgb=configs['rgb'], udrv1=v1, udrv2=configs['udrv2']))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = {}
    for seed in plan['seeds']:
        for label in ('rgb', 'udrv2'):
            config = deepcopy(configs[label])
            config['name'] = f'E7_{label}_s{seed}_image_test'
            config['manual_seed'] = seed
            config['path']['pretrain_network_g'] = provenance[label][seed]['path']
            config['path']['strict_load_g'] = True
            cfg_path = output/f'{label}_s{seed}.yml'
            json_path = output/f'{label}_s{seed}_images.json'
            cfg_path.write_text(yaml.safe_dump(config, sort_keys=False,
                                               allow_unicode=True), encoding='utf-8')
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--worker', str(cfg_path), str(json_path)],
                           cwd=ROOT, check=True)
            raw.setdefault(label, []).extend(json.loads(json_path.read_text(encoding='utf-8')))
    rows = pair_model_rows(raw['rgb'], raw['udrv2'], plan['seeds'])
    reconcile_e6(rows, e6_report, plan['seeds'])
    report = analyze(rows, plan['seeds'], args.bootstrap_seed)
    report.update(e6_plan=str(Path(args.plan).resolve()),
                  e6_report=str(Path(args.e6_report).resolve()),
                  checkpoint_sha256={label: {str(seed): provenance[label][seed]['sha256']
                                             for seed in plan['seeds']}
                                     for label in ('rgb', 'udrv2')},
                  metric='original uint8 Y-channel x4 crop border 4')
    write_outputs(output, rows, report)
    print('Saved E7 per-image results and 10,000-resample bootstrap to', output)


def self_test():
    seeds = [10, 11, 12]
    baseline, final = [], []
    for dataset in DATASETS:
        for index in range(20):
            for seed in seeds:
                base = 30 + index / 100 + seed / 1000
                delta = .01 + index / 1000
                baseline.append(dict(dataset=dataset, image_id=f'image{index:03d}',
                                     seed=seed, psnr=base, ssim=.9))
                final.append(dict(dataset=dataset, image_id=f'image{index:03d}',
                                  seed=seed, psnr=base + delta, ssim=.901))
    rows = pair_model_rows(baseline, final, seeds)
    assert len(rows) == 20 * 3 * 5
    first = analyze(rows, seeds, 2026)
    second = analyze(rows, seeds, 2026)
    assert first == second
    assert all(item['ci95_psnr'][0] > 0 for item in first['datasets'].values())
    try:
        pair_model_rows(baseline, final[:-1], seeds)
    except ValueError:
        pass
    else:
        raise AssertionError('E7 accepted an unpaired image/seed result.')
    print('PASS: exact image/seed pairing, 10,000 deterministic cluster bootstraps and positive CI')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(ROOT/'experiments/E6_plan/E6_plan.json'))
    parser.add_argument('--e6-report', default=str(ROOT/'results/E6_multiseed/E6_multiseed.json'))
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--udrv1-config', default=str(ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml'))
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    parser.add_argument('--output', default=str(ROOT/'results/E7_image_statistics'))
    parser.add_argument('--worker', nargs=2, metavar=('CONFIG', 'IMAGE_JSON'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    elif args.self_test:
        self_test()
    else:
        run(args)


if __name__ == '__main__':
    main()
