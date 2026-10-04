"""Offline E7 contracts for image pairing, E6 reconciliation and reports."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr.e7_image_statistics import (DATASETS, analyze,
                                             image_cluster_bootstrap,
                                             pair_model_rows, reconcile_e6,
                                             write_outputs)
from scripts.udr.evaluate_e6_multiseed import summarize


def main():
    seeds = [10, 11, 12]
    baseline, final = [], []
    for dataset in DATASETS:
        for image in range(5):
            for seed in seeds:
                base = 30 + image / 10 + seed / 1000
                baseline.append(dict(dataset=dataset, image_id=f'image{image:02d}',
                                     seed=seed, psnr=base, ssim=.9))
                final.append(dict(dataset=dataset, image_id=f'image{image:02d}',
                                  seed=seed, psnr=base + .02, ssim=.901))
    rows = pair_model_rows(baseline, final, seeds)
    runs = {label: {} for label in ('rgb', 'udrv1', 'udrv2')}
    for seed in seeds:
        for label in runs:
            runs[label][seed] = {}
            for dataset in DATASETS:
                source = (final if label == 'udrv2' else baseline)
                values = [item for item in source if item['dataset'] == dataset
                          and item['seed'] == seed]
                runs[label][seed][dataset] = dict(
                    psnr=sum(item['psnr'] for item in values) / len(values),
                    ssim=sum(item['ssim'] for item in values) / len(values))
    e6 = summarize(runs, seeds, list(runs))
    reconcile_e6(rows, e6, seeds)
    wrong = deepcopy(e6)
    wrong['per_dataset']['Set5']['udrv2']['metrics']['psnr']['by_seed'][0] += .01
    try:
        reconcile_e6(rows, wrong, seeds)
    except ValueError:
        pass
    else:
        raise AssertionError('E7 accepted metrics that disagree with E6.')
    report = analyze(rows, seeds, 2026)
    assert all(abs(item['mean_delta_psnr'] - .02) < 1e-10
               for item in report['datasets'].values())
    assert all(item['verdict'] == 'positive_ci_excludes_zero'
               for item in report['datasets'].values())
    assert all(len(item['worst10']) == 5 and len(item['best10']) == 5
               for item in report['datasets'].values())
    mixed = [dict(image_id='negative', seed=seed, delta_psnr=-.05)
             for seed in seeds] + [dict(image_id='positive', seed=seed,
                                        delta_psnr=.07) for seed in seeds]
    uncertain = image_cluster_bootstrap(mixed, seeds, np.random.default_rng(7))
    assert uncertain['mean_delta_psnr'] > 0
    assert uncertain['ci95_psnr'][0] < 0 < uncertain['ci95_psnr'][1]
    assert uncertain['verdict'] == 'positive_trend_ci_crosses_zero'
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        write_outputs(output, rows, report)
        assert (output/'E7_per_image.csv').is_file()
        assert (output/'E7_statistics.json').is_file()
        assert (output/'E7_summary.md').is_file()
    print('PASS: E7 exact E6 reconciliation, paired CSV, CI verdicts and worst/best images')


if __name__ == '__main__':
    main()
