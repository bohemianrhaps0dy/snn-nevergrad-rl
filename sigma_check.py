"""
sigma_check.py -- does PARAM_SIGMA = 'range/6' help the optimizers that step from a point?

Runs DiagonalCMA, OnePlusOne, PSO and NoisyRL3 on Acrobot (6,)/n4, both encodings, 5 seeds,
up to 40 x d, with the mutation step set to (W_HIGH - W_LOW) / 6 instead of Nevergrad's
default of 1 weight unit. The default side is already in the budget sweep (budget_sweep/runs):
the same cells, optimizers, seeds, starting weights, training episodes and report episodes,
so every range/6 run pairs with the default run of the same seed. TwoPointsDE and
RandomSearch are left out: they ask the same candidates under any sigma (checked to rounding
error), so their sweep runs already stand for both.

Everything else is the sweep's (budget_sweep.py does the work): the search, the scoring at
each level, the one-run-per-level treatment of NoisyRL3 (its levels: 20 and 40 x d, which the
sweep has too), the settings, resume after a stop. Only PARAM_SIGMA differs, and it is passed
to every run, so optimize_snn.py is not edited.

    python sigma_check.py --dry-run                 # runs and their cost; runs nothing
    mkdir -p logs && nohup python sigma_check.py --cores 40 > logs/sigma_check.log 2>&1 &
    python sigma_check.py --status
    python sigma_check.py --summary                 # the paired comparison (also mid-run)
    python sigma_check.py --stop

It can run while the budget sweep runs: its results go to sigma_check/, its own folder.
Needs budget_sweep.py of Oct 30 or later (the one that passes PARAM_SIGMA to each run).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget_sweep as B                                     # noqa: E402

# ============================ what to run ============================
SIGMA = 'range/6'              # the step to test (the sweep's runs give the other side)
BASE_DIR = 'budget_sweep'      # the budget sweep's folder: the default-sigma runs
OUT_DIR = 'sigma_check'
CELLS = dict(ENVIRONMENTS=['acrobot'], ENCODINGS=['poprate', 'poplatency'],
             SIZES=[([6], 4)],
             OPTIMIZERS=['DiagonalCMA', 'OnePlusOne', 'PSO', 'NoisyRL3'], SEEDS=5,
             LEVELS={'acrobot': (10, 20, 30, 40)},
             PER_RUN_LEVELS={'acrobot': (20, 40)})
REFERENCE = 'TwoPointsDE'      # shown from the sweep for scale (sigma doesn't change it)
# --smoke (testing only): tiny budgets, both sides run here
SMOKE = dict(SEEDS=2, LEVELS={'acrobot': (0.25, 0.5)},
             PER_RUN_LEVELS={'acrobot': (0.25, 0.5)}, EVAL_EPISODES=2, REPORT_EPISODES=5)
# ======================================================================


def configure(sigma, out_dir, extra=None):
    for k, v in CELLS.items():
        setattr(B, k, v)
    for k, v in (extra or {}).items():
        setattr(B, k, v)
    B.OUT_DIR = out_dir
    snn, O = B.modules()
    O.PARAM_SIGMA = None if sigma in (None, 'default') else sigma


def runs_by_seed(folder, only_optimizers=None):
    """{(encoding, optimizer, seed): {level: row}} of the scored levels of the (6,)/n4
    Acrobot runs in folder/runs, and the parametrizations they recorded."""
    out, sigmas = {}, {}
    saved = B.OUT_DIR
    B.OUT_DIR = folder
    try:
        results = B.load_results()
    finally:
        B.OUT_DIR = saved
    for r in results:
        t = r['task']
        c = t['cell']
        if (c['environment'], c['hidden_sizes'], c['n_per_dim']) != ('acrobot', [6], 4):
            continue
        if only_optimizers and t['optimizer'] not in only_optimizers:
            continue
        key = (c['encoding'], t['optimizer'], int(t['seed']))
        for row in r['rows']:
            if 'raw' not in row:
                continue
            if t['kind'] == 'per-level' and float(row['per_weight']) != float(t['run_k']):
                continue
            out.setdefault(key, {})[float(row['per_weight'])] = row
        sigmas.setdefault(t['optimizer'], set()).add(str(r['key'].get('param_sigma')))
    return out, sigmas


def sign_p(wins, losses):
    """Two-sided sign test p (ties dropped)."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, p)


def fmt_p(p):
    return '-' if p is None else ('%.3f' % p)


def compare(base_dir, out_dir):
    new, new_s = runs_by_seed(out_dir)
    old, old_s = runs_by_seed(base_dir, set(CELLS['OPTIMIZERS']) | {REFERENCE})
    if not new:
        print("no %s runs with a score yet in %s/runs" % (SIGMA, out_dir))
        return
    print("sigma check, Acrobot (6,)/n4: default step (the sweep, %s/) vs %s (%s/)"
          % (base_dir, SIGMA, out_dir))
    print("recorded PARAM_SIGMA: sweep %s | this check %s"
          % (", ".join("%s %s" % (k, "/".join(sorted(v))) for k, v in sorted(old_s.items())
                       if k != REFERENCE),
             ", ".join("%s %s" % (k, "/".join(sorted(v))) for k, v in sorted(new_s.items()))))
    bad = [k for k, v in old_s.items() if k != REFERENCE and v != {'None'}]
    if bad:
        print("WARNING: the sweep's runs of %s were not made with the default step, so they "
              "are not the other side of this comparison" % ", ".join(bad))
    print("\nper cell: median raw reward, default -> %s (seeds where %s is better / pairs; "
          "pass counts default -> %s; pass: raw >= -100)" % (SIGMA, SIGMA, SIGMA))
    pooled = {}
    levels_all = sorted({L for v in new.values() for L in v})
    for enc in CELLS['ENCODINGS']:
        print("\n%s" % enc)
        print("  %-12s" % 'optimizer' + "".join("%28s" % ("%gxd" % L) for L in levels_all))
        ref = [old.get((enc, REFERENCE, s), {}) for s in range(CELLS['SEEDS'])]
        line = "  %-12s" % (REFERENCE + '*')
        for L in levels_all:
            vals = [r[L]['raw'] for r in ref if L in r]
            line += "%28s" % (("%.0f (%d/%d pass)" % (np.median(vals), sum(v >= -100 for v in vals),
                                                     len(vals))) if vals else '-')
        print(line)
        for name in CELLS['OPTIMIZERS']:
            line = "  %-12s" % name
            for L in levels_all:
                pairs = []
                for s in range(CELLS['SEEDS']):
                    a = old.get((enc, name, s), {}).get(L)
                    b = new.get((enc, name, s), {}).get(L)
                    if a is not None and b is not None:
                        pairs.append((a['raw'], b['raw']))
                if not pairs:
                    line += "%28s" % '-'
                    continue
                a = [x for x, _ in pairs]
                b = [y for _, y in pairs]
                w = sum(y > x for x, y in pairs)
                line += "%28s" % ("%.0f -> %.0f  %d/%d  %d->%d"
                                  % (np.median(a), np.median(b), w, len(pairs),
                                     sum(x >= -100 for x in a), sum(y >= -100 for y in b)))
                pooled.setdefault((name, L), []).extend(pairs)
            print(line)
    print("  * the sweep's runs; its candidates are the same under any sigma")

    print("\nboth encodings together: median of (%s - default) per seed, seeds better / worse, "
          "sign-test p" % SIGMA)
    print("  %-12s" % 'optimizer' + "".join("%24s" % ("%gxd" % L) for L in levels_all))
    for name in CELLS['OPTIMIZERS']:
        line = "  %-12s" % name
        for L in levels_all:
            pairs = pooled.get((name, L))
            if not pairs:
                line += "%24s" % '-'
                continue
            d = [y - x for x, y in pairs]
            w, l = sum(v > 0 for v in d), sum(v < 0 for v in d)
            line += "%24s" % ("%+.0f  %d/%d  p %s" % (np.median(d), w, l, fmt_p(sign_p(w, l))))
        print(line)
    n_new = sum(len(v) for v in new.values())
    print("\n%d scored levels so far. With 10 pairs (5 seeds x 2 encodings), 9 or 10 seeds "
          "the same way gives p <= 0.02; 8 of 10 gives p = 0.11." % n_new)


def main():
    args = sys.argv[1:]
    smoke = '--smoke' in args
    sigma, out_dir, base_dir = SIGMA, OUT_DIR, BASE_DIR
    if '--sigma' in args:                       # testing: run the default side here too
        i = args.index('--sigma')
        sigma = args[i + 1]
        del args[i:i + 2]
    if '--out-dir' in args:
        i = args.index('--out-dir')
        out_dir = args[i + 1]
        del args[i:i + 2]
    if '--base-dir' in args:
        i = args.index('--base-dir')
        base_dir = args[i + 1]
        del args[i:i + 2]
    if smoke:
        args.remove('--smoke')
        out_dir = out_dir + '_smoke' if out_dir == OUT_DIR else out_dir
    configure(sigma, out_dir, SMOKE if smoke else None)
    B.cmd_summary = lambda *a, **k: compare(base_dir, out_dir)   # after a run, and --summary
    if '--summary' in args and not B.load_results():
        print("no runs saved yet in %s" % B.out_path('runs'))
        return
    sys.argv = [sys.argv[0]] + args
    B.main()


if __name__ == '__main__':
    main()
