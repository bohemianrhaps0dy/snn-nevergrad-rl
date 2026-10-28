"""
make_grid_settings.py -- turn tune_hyperparams.py's results into the per-cell settings that
experiment_runner.py reads (SETTINGS_FILE in its config, or --settings FILE).

Two rules decide which settings a tuned cell gets:

  default         compare the tuned settings with the current ones on the held-out seeds.
                  tune_hyperparams.py scores both on the same K inner runs, so the K
                  differences are paired. A cell uses its tuned settings when their mean is
                  higher and a paired t-test over the K runs gives p < P_MAX; otherwise it
                  keeps the current settings.
  --always-tuned  every tuned cell uses its tuned settings, whatever the held-out
                  comparison says (no test). The comparison is still written next to each
                  cell (tuned_minus_current, p, ...), so it stays on record.

Both cases are written out with their values, so the grid doesn't depend on module
defaults that may change later. The spike generator the cell was tuned with is written
too (poprate cells only), and experiment_runner.py runs the cell with it.

Caveat: the tuner scores a setting by RandomSearch's best network, not by what the grid's
optimizers reach.

    python make_grid_settings.py                     # tuning/result-*.json -> grid_settings.json
    python make_grid_settings.py --p-max 0.01 --out grid_settings_strict.json
    python make_grid_settings.py --always-tuned --results 'tuning_grid/result-*.json' \\
        --out grid_settings_tuned.json
"""
import argparse
import glob
import json

import numpy as np
from scipy import stats

KEYS = ('w_scale', 'r_max', 'r_min', 'lat_min', 'lat_window')
P_MAX = 0.05


def grid_key(environment, hidden_sizes, encoding, decoding, n_per_dim):
    """experiment_runner.cell_key() without the trial."""
    topo = 'x'.join(str(h) for h in hidden_sizes) or 'none'
    return "%s|%s|%s|%s|n%d" % (environment, topo, encoding, decoding, int(n_per_dim))


def choose(tuned, current, holdout_vals, baseline_vals, p_max=P_MAX, extra=None,
           always_tuned=False):
    """The settings one cell should use, with the evidence behind the choice."""
    h = np.asarray(holdout_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    diff = float((h - b).mean())
    p = float(stats.ttest_rel(h, b).pvalue) if len(h) > 1 else 1.0
    if p != p:                       # identical runs: the t-test is undefined
        p = 1.0
    use = True if always_tuned else (diff > 0 and p < p_max)
    cfg = tuned if use else current
    out = {k: float(cfg[k]) for k in KEYS}
    out.update(source='tuned' if use else 'current',
               tuned_minus_current=round(diff, 2), p=round(p, 4),
               runs_tuned_better=int((h > b).sum()), runs=int(len(h)))
    out.update(extra or {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='tuning/result-*.json',
                    help="tune_hyperparams.py's per-cell result files (glob)")
    ap.add_argument('--p-max', type=float, default=P_MAX)
    ap.add_argument('--always-tuned', action='store_true',
                    help='use the tuned settings in every tuned cell, without the test')
    ap.add_argument('--out', default='grid_settings.json')
    args = ap.parse_args()

    cells, sources = {}, {}
    for f in sorted(glob.glob(args.results)):
        r = json.load(open(f))
        c = r['cell']
        key = grid_key(c['environment'], c['hidden_sizes'], c['encoding'], c['decoding'],
                       c['n_per_dim'])
        if key in cells:
            raise SystemExit("%s and %s are both results for %s (two spike generators?). "
                             "Keep one, or point --results at one of them."
                             % (sources[key], f, key))
        generator = None                     # only poprate's encoder uses it
        if c['encoding'] == 'poprate':
            generator = r.get('settings', {}).get('generator') or c.get('generator')
        cells[key] = choose(r['best_config'], r['baseline_config'], r['holdout_vals'],
                            r['baseline_vals'], args.p_max,
                            extra=dict(tuned_raw=round(r['holdout_raw'], 2),
                                       current_raw=round(r['baseline_raw'], 2),
                                       tuned_random_actions=round(r['holdout_random_actions'], 4),
                                       current_random_actions=round(r['baseline_random_actions'], 4),
                                       generator=generator,
                                       result_file=f),
                            always_tuned=args.always_tuned)
        sources[key] = f
    if not cells:
        raise SystemExit("no files match %s" % args.results)
    if args.always_tuned:
        rule = ("tuned in every tuned cell (no test); the held-out comparison with the "
                "current settings is kept for the record")
    else:
        rule = ("tuned if mean(tuned - current) > 0 and paired t-test p < %g over the K "
                "held-out inner runs; current otherwise" % args.p_max)
    with open(args.out, 'w') as fh:
        json.dump(dict(rule=rule, cells=cells), fh, indent=1)
    n_tuned = sum(v['source'] == 'tuned' for v in cells.values())
    print("%d cells: %d tuned, %d current -> %s" % (len(cells), n_tuned, len(cells) - n_tuned,
                                                    args.out))


if __name__ == '__main__':
    main()
