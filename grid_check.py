"""
grid_check.py -- why do CMA, TBPSA and PSO fail on the larger Acrobot cells?

Run it in the pipeline folder (next to experiment_results_grid.json); it reads the results
and the saved networks, runs no simulation, and can run while the grid runs.

    python grid_check.py
"""
import collections as C
import json
import os
import pickle
import statistics as st

import numpy as np

RESULTS = 'experiment_results_grid.json'
WEIGHTS_DIR = 'experiment_results_grid_weights'
OPTIMIZERS = ['TwoPointsDE', 'NgIohTuned', 'RandomSearch', 'PSO', 'CMA', 'TBPSA']
EDGE = 0.99          # a weight counts as "at the bound" when |w| >= EDGE * w_scale

cells = json.load(open(RESULTS))['cells']


def w_scale(cell, opt):
    s = cell.get('settings') or {}
    if s.get('w_scale') is not None:
        return float(s['w_scale'])
    p = opt.get('param') or {}
    if p.get('sigma_rule') == 'range/6' and p.get('sigma'):
        return 3.0 * float(p['sigma'])          # sigma = 2w / 6
    return None


def at_bound(key, name, w):
    path = os.path.join(WEIGHTS_DIR, key.replace('|', '__') + '.pkl')
    if w is None or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            W = pickle.load(f).get(name)
    except Exception:
        return None
    if W is None:
        return None
    flat = np.concatenate([np.ravel(np.asarray(m, dtype=float)) for m in W])
    return float(np.mean(np.abs(flat) >= EDGE * w))


# 1. what each optimizer actually ran with
print("=== 1. parametrization per optimizer: sigma rule | noise flag | what Nevergrad ran: tasks")
seen = C.defaultdict(C.Counter)
for c in cells.values():
    for name, o in (c.get('optimizers') or {}).items():
        p = o.get('param') or {}
        seen[name][(str(p.get('sigma_rule')), str(p.get('noisy')), str(o.get('sub_optim')))] += 1
for name in sorted(seen):
    for (rule, noisy, sub), k in sorted(seen[name].items()):
        print("  %-36s %-18s noisy=%-6s %-28s %d" % (name, rule, noisy, sub, k))

# 2. by environment, optimizer and n_per_dim
rows = C.defaultdict(list)
for key, c in cells.items():
    for name, o in (c.get('optimizers') or {}).items():
        if name not in OPTIMIZERS:
            continue
        rows[(c['env'], name, int(c['n_per_dim']))].append(dict(
            raw=o.get('reward_mean'), goal=o.get('goal_share'),
            rnd=o.get('random_actions'), silent=o.get('silent'),
            edge=at_bound(key, name, w_scale(c, o)), dim=c.get('genome_dim')))


def med(xs, scale=1.0, fmt='%.0f'):
    xs = [x for x in xs if x is not None]
    return (fmt % (scale * st.median(xs))) if xs else '-'


print("\n=== 2. median per environment x optimizer x n_per_dim")
print("  raw: reward | goal: share of report episodes that reach the goal | random: actions "
      "drawn at random | silent: steps with no output spike | bound: weights at +-w")
for env in ('acrobot', 'mountaincar'):
    print("\n%s" % env)
    print("  %-14s %4s %6s %8s %7s %8s %8s %8s"
          % ('optimizer', 'n', 'tasks', 'raw', 'goal', 'random', 'silent', 'bound'))
    for name in OPTIMIZERS:
        for n in sorted({k[2] for k in rows if k[0] == env and k[1] == name}):
            r = rows[(env, name, n)]
            print("  %-14s %4d %6d %8s %6s%% %7s%% %7s%% %7s%%"
                  % (name, n, len(r), med([x['raw'] for x in r]),
                     med([x['goal'] for x in r], 100), med([x['rnd'] for x in r], 100, '%.1f'),
                     med([x['silent'] for x in r], 100, '%.1f'),
                     med([x['edge'] for x in r], 100)))
print("\nA uniform random start has about %.0f%% of its weights at the bound."
      % (100 * (1 - EDGE)))
