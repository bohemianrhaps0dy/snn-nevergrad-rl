"""
budget_sweep.py -- what each evaluations-per-weight budget buys, what it costs, and the
largest budget rule the grid can afford in a time window.

It runs the grid's own search on a few cells at several budgets and records, at every
budget, the Gym score of each optimizer's recommendation and the CPU time the run took to
get there. From those times it predicts how long the whole grid (run_pipeline.py's GRID)
takes on N cores under a rule budget = k x d, and finds the largest k that fits the
window: one k for both environments, and one k per environment.

    python budget_sweep.py --dry-run          # runs, and the sweep's time on 16 ... 288 cores
    python budget_sweep.py --smoke            # a few minutes on tiny budgets, 4 at a time
    mkdir -p logs && nohup python budget_sweep.py --cores 64 > logs/budget_sweep.log 2>&1 &
    python budget_sweep.py --status           # progress and time left, from any shell
    python budget_sweep.py --stop             # stop it and every run it started
    python budget_sweep.py --summary          # score tables, and the largest k for 16 ... 288 cores
    python budget_sweep.py --summary --cores 64 --hours 24 --grid-optimizers 8

A SHARED SERVER. The sweep starts only with --cores N (or CORES set below): it keeps at
most N runs going, one core each, at nice 10. With --leave-free M it also waits before
starting a run while the machine's 1-minute load average leaves fewer than M cores idle,
so it uses spare cores and gives them back as others start work. Costs are measured in
CPU seconds, not wall seconds, so other people's jobs slowing the sweep down don't make
the grid look more expensive than it is: the plan prices N cores of your own.

Starting it again after a stop, a crash or a reboot continues where it was. While it
runs, don't edit snn_feedforward.py or optimize_snn.py in this folder: every run that
starts imports them afresh.

------------------------------------------------------------------------------
WHAT IT RUNS (the block below)
------------------------------------------------------------------------------
  cells       both environments x both encodings x 3 sizes, first-spike decoding,
              poprate with regular spikes (as in run_pipeline.py's GRID):
                MountainCar  (6,)/n2 d=42   (6,)/n4 d=66    (12,)/n10 d=276
                Acrobot      (6,)/n2 d=90   (6,)/n4 d=162   (12,)/n10 d=756
              the smallest and the largest genome of the grid in each environment, and 6/4
  optimizers  TwoPointsDE, DiagonalCMA, PSO, OnePlusOne, RandomSearch, NoisyRL3
  seeds       5. The optimizers of a seed start from the same weights, train on the same
              episode starts and are scored on the same report episodes
  budgets     LEVELS x d evaluations
  settings    the module defaults (W_LOW / W_HIGH, R_MAX, R_MIN, LAT_MIN, LAT_WINDOW), or
              a make_grid_settings.py file in SETTINGS_FILE

------------------------------------------------------------------------------
ONE RUN, OR ONE RUN PER BUDGET
------------------------------------------------------------------------------
An optimizer that ignores its declared budget runs once, to the largest level, and its
recommendation is read after every smaller level: the network a run with that budget
would recommend. An optimizer that uses its budget (NoisyRL3 hands over from its first
stage to its second at half of it) runs once per level in PER_RUN_LEVELS. Which is which
is tested when the sweep starts, not assumed: each optimizer runs on a cheap function with
budgets B and 4B, and it gets one run per level if its first B candidates or its
recommendation after B differ, or if reading its recommendation changes what follows.

------------------------------------------------------------------------------
THE SEARCH IS THE GRID'S
------------------------------------------------------------------------------
optimize_snn's make_objective(), initial_point() and make_parametrization(), and
train_weights()'s ask/tell loop: the same box, sigma, noise declaration, EVAL_EPISODES
and recommendation rule the grid uses, whatever optimize_snn.py in this folder sets. The
loop is written out here so that each level is saved as soon as it is reached; at start-up
it is checked against train_weights() on the cheap function, and the sweep refuses to run
if they recommend different points. Training episodes are seeded (train_seed): fresh at
every evaluation, as in the grid, but repeatable. Seed s runs as trial SWEEP_TRIAL_OFFSET
+ s, never a grid trial (0-4) or a tuner trial (1000+). Each level's recommendation is
scored on optimize_snn's report block of that trial (REPORT_EPISODES episodes): raw
reward, pass (>= -110 MountainCar, >= -100 Acrobot), random actions.

What it times: the search itself (ask, tell and the training episodes) up to each level,
apart from the compile and from each level's scoring, which are timed on their own. Each
is recorded in CPU seconds (what the plan prices) and in wall seconds (what --status
uses for the time left). A run is single-threaded, so on an idle core the two agree; on
a busy machine the wall time grows and the CPU time does not.

------------------------------------------------------------------------------
THE PLAN (--summary)
------------------------------------------------------------------------------
For each measured cell and optimizer the sweep gives T(k), the CPU seconds a run of k x d
evaluations takes (mean over seeds). A grid cell is priced from the measured cells of its
environment and encoding: the mean seconds per evaluation, T(k) / (k d), is interpolated
linearly in d, so the (6,6) and (10,4) topologies are priced by their number of weights.
A grid task costs that, plus its compile and the ~216 episodes optimize() scores around
the search. An optimizer the sweep did not run is priced as the mean of the six. The tasks
(cell x trial x optimizer) are packed longest first on N cores, as experiment_runner.py
hands them out, and the plan reports the largest k whose wall time fits --hours on N
free cores: first for every N in CORE_COUNTS, then in detail (one k for both
environments, and the largest MountainCar k for each Acrobot k) for --cores. The budget
is clipped to run_pipeline.py's [budget_min, budget_max]. The tuning stage
run_pipeline.py runs before the grid is priced the same way.

Results: OUT_DIR/runs/<run>.json, one file per run, written only by that run and after
every level, so --status and --summary can read a run that is still going. Starting the
sweep again skips the finished runs and runs the others from the start (they are seeded,
so a run cut short repeats the same levels); a run made with other settings is set aside
(.stale-*) and run again.
"""

import os

# One thread per process, before anything imports ANNarchy (as in experiment_runner.py).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import ast
import contextlib
import glob
import heapq
import io
import json
import math
import shutil
import signal
import subprocess
import sys
import time
import warnings

import numpy as np


# ============================ what to run ============================
ENVIRONMENTS = ['mountaincar', 'acrobot']
ENCODINGS = ['poprate', 'poplatency']
DECODING = 'first'                   # every cell, as in run_pipeline.py's GRID
GENERATOR = 'regular'                # poprate's spike generator, as in the GRID
SIZES = [([6], 2), ([6], 4), ([12], 10)]          # (hidden, n_per_dim) in every
                                                  # environment and encoding
OPTIMIZERS = ['TwoPointsDE', 'DiagonalCMA', 'PSO', 'OnePlusOne', 'RandomSearch',
              'NoisyRL3']
SEEDS = 5
LEVELS = {'mountaincar': (10, 20, 30, 40, 50, 60, 80, 100),   # evaluations per weight
          'acrobot':     (10, 20, 30, 40, 50, 60, 80, 100)}   # scored; the largest is
                                                              # how long a run goes
PER_RUN_LEVELS = {'mountaincar': (20, 40, 60, 100),   # optimizers that use their
                  'acrobot':     (20, 40, 60, 100)}   # budget: one run per level
SETTINGS_FILE = None        # make_grid_settings.py output for per-cell settings;
                            # None = the module defaults
EVAL_EPISODES = None        # training episodes per evaluation; None = optimize_snn's
REPORT_EPISODES = None      # report episodes per level; None = optimize_snn's
SWEEP_TRIAL_OFFSET = 3000   # seed s runs as trial 3000 + s
CORES = None                # runs at once. None: the sweep won't start without --cores N
LEAVE_FREE = None           # with a number M: start a run only while the load average
                            # leaves at least M cores idle (--leave-free)
NICE = 10                   # lower priority, so the server stays usable
CORE_COUNTS = (16, 32, 64, 96, 128, 192, 288)   # --dry-run and --summary show these
BY_CORES_FIXED_K = (10, 20, 30, 40)   # --summary: the largest MountainCar k when Acrobot
                                      # gets each of these
ATTEMPTS = 2                # a run that fails is started again once

# ---- the plan ----
HOURS = 24.0                     # the window the grid must fit in (--hours)
PIPELINE_FILE = 'run_pipeline.py'      # its GRID and tuning settings are read, not run
TUNER_FILE = 'tune_hyperparams.py'

OUT_DIR = 'budget_sweep'
POLL_SECONDS = 20
BLIND_TEST_BUDGET = 120     # B in the budget test (start-up)
# One training episode of an untrained network on the server, regular spikes (the
# runner's 0.018 / 0.065 s times the measured slowdown of regular spikes). --dry-run's
# estimate only; --status and --summary use the times the sweep measures.
EP_SECONDS = {'mountaincar': 0.024, 'acrobot': 0.107}

# --smoke: everything on one size and tiny budgets, in its own folder
SMOKE = dict(SIZES=[([6], 2)], SEEDS=1,
             LEVELS={'mountaincar': (0.5, 1.0), 'acrobot': (0.25, 0.5)},
             PER_RUN_LEVELS={'mountaincar': (0.5, 1.0), 'acrobot': (0.25, 0.5)},
             EVAL_EPISODES=2, REPORT_EPISODES=5, OUT_DIR='budget_sweep_smoke')

# run_pipeline.py's GRID, used only when PIPELINE_FILE can't be read
DEFAULT_GRID = dict(environments=['mountaincar', 'acrobot'],
                    encodings=['poprate', 'poplatency'], decoding='first',
                    topologies=[[6], [12], [6, 6], [10, 4]], n_per_dim=[2, 4, 6, 10],
                    trials=5, generator='regular', evals_per_dim=100, budget_min=1000,
                    budget_max=None)
# ======================================================================

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = {'mountaincar': -110.0, 'acrobot': -100.0}
SHORT = {'TwoPointsDE': 'TPDE', 'DiagonalCMA': 'DCMA', 'OnePlusOne': '1+1',
         'RandomSearch': 'RS', 'NoisyRL3': 'NRL3'}
_STOPPING = [False]


def out_path(*parts):
    return os.path.join(CODE_DIR, OUT_DIR, *parts)


def log(msg):
    print("%s  %s" % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


def fmt_secs(s):
    s = int(max(0, s))
    if s < 3600:
        return "%dm" % max(1, s // 60) if s >= 60 else "%ds" % s
    if s < 86400:
        return "%dh %02dm" % (s // 3600, (s % 3600) // 60)
    return "%dd %dh" % (s // 86400, (s % 86400) // 3600)


def _json_default(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    if hasattr(o, 'item'):
        return o.item()
    return str(o)


def atomic_json(path, obj):
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def short(name):
    return SHORT.get(name, name[:8])


# ---------------------------------------------------------------- the modules
@contextlib.contextmanager
def bare_argv():
    """ANNarchy reads sys.argv with its own parser (-c, -d, -v, --report, --num-threads,
    ...) when it is imported and again when a network compiles: it sees none of ours."""
    saved = sys.argv
    sys.argv = saved[:1]
    try:
        yield
    finally:
        sys.argv = saved


_MODULES = {}


def modules():
    """snn_feedforward and optimize_snn from this folder."""
    if not _MODULES:
        if CODE_DIR not in sys.path:
            sys.path.insert(0, CODE_DIR)
        with bare_argv(), contextlib.redirect_stdout(io.StringIO()):
            import snn_feedforward
            import optimize_snn
        _MODULES.update(snn=snn_feedforward, O=optimize_snn)
    return _MODULES['snn'], _MODULES['O']


def n_eval():
    return int(EVAL_EPISODES or modules()[1].EVAL_EPISODES)


def n_report():
    return int(REPORT_EPISODES or modules()[1].REPORT_EPISODES)


# ---------------------------------------------------------------- cells and settings
def topo(hidden):
    return 'x'.join(str(h) for h in hidden) or 'none'


def sweep_cells():
    return [dict(environment=env, encoding=enc, decoding=DECODING,
                 hidden_sizes=list(hidden), n_per_dim=int(n))
            for env in ENVIRONMENTS for enc in ENCODINGS for hidden, n in SIZES]


def cell_tag(c):
    return "%s-%s-%s-h%s-n%d" % (c['environment'], c['encoding'], c['decoding'],
                                 topo(c['hidden_sizes']), c['n_per_dim'])


def cell_label(c):
    return "%s/%s" % (topo(c['hidden_sizes']), c['n_per_dim'])


def genome_dim(env, hidden, encoding, n):
    snn, O = modules()
    return int(O.n_params(snn.weight_shapes(env, hidden, encoding, n)))


def cell_dim(c):
    return genome_dim(c['environment'], c['hidden_sizes'], c['encoding'], c['n_per_dim'])


def settings_key(c):
    """experiment_runner.settings_key() of the cell: the key make_grid_settings.py writes."""
    return "%s|%s|%s|%s|n%d" % (c['environment'], topo(c['hidden_sizes']), c['encoding'],
                                c['decoding'], c['n_per_dim'])


def load_settings():
    if not SETTINGS_FILE:
        return {}
    data = read_json(os.path.join(CODE_DIR, SETTINGS_FILE))
    if not data or 'cells' not in data:
        raise SystemExit("%s: not a make_grid_settings.py file" % SETTINGS_FILE)
    return data['cells']


def cell_settings(c, table):
    """What the cell runs with, as experiment_runner.py would run it: the module defaults
    with poprate on GENERATOR, or the cell's entry in SETTINGS_FILE."""
    snn, O = modules()
    gen = GENERATOR if c['encoding'] == 'poprate' else None
    s = table.get(settings_key(c))
    if not s:
        return dict(source='module defaults', w_low=float(O.W_LOW), w_high=float(O.W_HIGH),
                    r_max=float(snn.R_MAX), r_min=float(snn.R_MIN),
                    lat_min=float(snn.LAT_MIN), lat_window=float(snn.LAT_WINDOW),
                    generator=gen)
    w = float(s['w_scale'])
    if c['encoding'] == 'poprate' and s.get('generator'):
        gen = s['generator']
    return dict(source="%s: %s" % (os.path.basename(SETTINGS_FILE), s.get('source', '?')),
                w_low=-w, w_high=w, r_max=float(s['r_max']), r_min=float(s['r_min']),
                lat_min=float(s['lat_min']), lat_window=float(s['lat_window']),
                generator=gen)


def apply_settings(s):
    """Set a run's settings in this process's modules (a fresh process per run)."""
    snn, O = modules()
    O.W_LOW, O.W_HIGH = float(s['w_low']), float(s['w_high'])
    snn.R_MAX, snn.R_MIN = float(s['r_max']), float(s['r_min'])
    snn.LAT_MIN, snn.LAT_WINDOW = float(s['lat_min']), float(s['lat_window'])


def search_key(settings):
    """Everything that changes what a run computes, apart from the run itself: a stored
    run is reused only if this matches."""
    snn, O = modules()
    sigma = O.PARAM_SIGMA
    if sigma is not None and not isinstance(sigma, str):
        sigma = float(sigma)
    return dict(settings={k: v for k, v in settings.items() if k != 'source'},
                eval_episodes=n_eval(), report_episodes=n_report(),
                param_noisy=bool(O.PARAM_NOISY), param_sigma=sigma, seed=int(O.SEED),
                report_seed=int(O.REPORT_SEED), t_step=float(snn.T_STEP),
                trial_offset=int(SWEEP_TRIAL_OFFSET))


# ---------------------------------------------------------------- the search
def search(name, init, trial, budget, objective, marks, on_mark):
    """optimize_snn.train_weights()'s run, written out so on_mark(count, x) gets the
    recommendation after each count in `marks` as soon as it exists. budget_test() checks
    that it recommends exactly what train_weights() does."""
    import nevergrad as ng
    snn, O = modules()
    param, info = O.make_parametrization(name, init, trial)
    optimizer = ng.optimizers.registry[name](parametrization=param, budget=int(budget))
    wanted = set(int(c) for c in marks)
    n = 0
    for i in range(optimizer.budget):
        cand = optimizer.ask()
        optimizer.tell(cand, objective(*cand.args, **cand.kwargs))
        n += 1
        if i + 1 in wanted:
            on_mark(i + 1, np.array(optimizer.provide_recommendation().value, dtype=float,
                                    copy=True))
    rec = np.array(optimizer.provide_recommendation().value, dtype=float, copy=True)
    return rec, info, O.sub_optim(optimizer), n


@contextlib.contextmanager
def weight_box(s):
    snn, O = modules()
    saved = (O.W_LOW, O.W_HIGH)
    O.W_LOW, O.W_HIGH = float(s['w_low']), float(s['w_high'])
    try:
        yield
    finally:
        O.W_LOW, O.W_HIGH = saved


def budget_test(name, dim):
    """Does optimizer `name` ignore its declared budget? On a cheap deterministic function,
    through make_parametrization(): a run with budget B and one with 4B must ask the same
    first B candidates and recommend the same point after B, and reading the
    recommendation after B must not change the rest of the 4B run. Also checks search()
    against train_weights(). Runs under the weight box weight_box() has set."""
    import nevergrad as ng
    snn, O = modules()
    trial = SWEEP_TRIAL_OFFSET
    init = O.initial_point(dim, trial)
    centre = np.random.default_rng(12345).uniform(O.W_LOW, O.W_HIGH, dim)

    def f(x):
        return float(np.mean((np.asarray(x, dtype=float) - centre) ** 2))

    def plain(budget, read_at=()):
        param, _ = O.make_parametrization(name, init, trial)
        opt = ng.optimizers.registry[name](parametrization=param, budget=budget)
        xs, recs = [], {}
        for i in range(budget):
            cand = opt.ask()
            xs.append(np.array(cand.value, dtype=float, copy=True))
            opt.tell(cand, f(*cand.args, **cand.kwargs))
            if i + 1 in read_at:
                recs[i + 1] = np.array(opt.provide_recommendation().value, dtype=float,
                                       copy=True)
        return xs, recs, np.array(opt.provide_recommendation().value, dtype=float,
                                  copy=True)

    B = int(BLIND_TEST_BUDGET)
    state = np.random.get_state()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            x1, _, r1 = plain(B)
            x4, recs, r4 = plain(4 * B, read_at=(B,))
            _, _, r4_unread = plain(4 * B)
            marks = (B // 2, B)
            mine = {}
            rec_mine = search(name, init, trial, B, f, marks,
                              lambda c, x: mine.__setitem__(c, x))[0]
            ref = O.train_weights(name, init, trial, B, f, checkpoints=marks)
        finally:
            np.random.set_state(state)
    first = next((i + 1 for i, (a, b) in enumerate(zip(x1, x4))
                  if not np.array_equal(a, b)), None)
    blind = (first is None and np.array_equal(r1, recs[B])
             and np.array_equal(r4, r4_unread))
    same_loop = (np.array_equal(rec_mine, ref['rec'])
                 and all(np.array_equal(mine[c], ref['checkpoints'][c][0]) for c in marks))
    return dict(blind=bool(blind), first_difference=first, same_loop=bool(same_loop))


def budget_tests(cells_settings):
    """budget_test() for every optimizer and genome size, cached in OUT_DIR/budget_test.json."""
    import nevergrad as ng
    snn, O = modules()
    path = out_path('budget_test.json')
    cache = read_json(path, {}) or {}
    out, changed = {}, False
    for c, s in cells_settings:
        d = cell_dim(c)
        for name in OPTIMIZERS:
            key = "%s|d%d|box %g %g|noisy %s|sigma %s|B%d|nevergrad %s" % (
                name, d, s['w_low'], s['w_high'], O.PARAM_NOISY, O.PARAM_SIGMA,
                BLIND_TEST_BUDGET, ng.__version__)
            if key not in cache:
                with weight_box(s):
                    cache[key] = budget_test(name, d)
                changed = True
            out[(name, d, s['w_low'], s['w_high'])] = cache[key]
    if changed:
        os.makedirs(out_path(), exist_ok=True)
        atomic_json(path, cache)
    bad = sorted({k[0] for k, v in out.items() if not v['same_loop']})
    if bad:
        raise SystemExit("budget_sweep.search() no longer recommends what optimize_snn."
                         "train_weights() does for %s: optimize_snn.py has changed, so "
                         "update search() to follow train_weights()." % ", ".join(bad))
    return out


# ---------------------------------------------------------------- the runs
def make_tasks():
    """Every run of the sweep, in the order the CONFIG block lists them."""
    import nevergrad as ng
    unknown = [n for n in OPTIMIZERS if n not in ng.optimizers.registry]
    if unknown:
        raise SystemExit("not in Nevergrad's registry: %s" % ", ".join(unknown))
    table = load_settings()
    pairs = [(c, cell_settings(c, table)) for c in sweep_cells()]
    tests = budget_tests(pairs)
    tasks = []
    for c, s in pairs:
        d = cell_dim(c)
        env = c['environment']
        levels = sorted(float(x) for x in LEVELS[env])
        key = search_key(s)
        for name in OPTIMIZERS:
            if tests[(name, d, s['w_low'], s['w_high'])]['blind']:
                runs = [('checkpoints', levels[-1], levels, levels)]
            else:
                runs = [('per-level', float(L), [x for x in levels if x < L] + [float(L)],
                         [float(L)]) for L in sorted(PER_RUN_LEVELS[env])]
            for seed in range(SEEDS):
                for kind, run_k, marks, scored in runs:
                    tag = "%s-%s-s%d-%s%g" % (cell_tag(c), name, seed,
                                              'to' if kind == 'checkpoints' else 'at', run_k)
                    tasks.append(dict(
                        tag=tag, cell=c, optimizer=name, seed=seed,
                        trial=SWEEP_TRIAL_OFFSET + seed, kind=kind, run_k=run_k,
                        marks=marks, scored=scored, dim=d,
                        budget=int(round(run_k * d)), settings=s, key=key,
                        eval_episodes=n_eval(), report_episodes=n_report(),
                        result=out_path('runs', tag + '.json'),
                        spec=out_path('tasks', tag + '.json'),
                        log=out_path('logs', tag + '.log'),
                        build=out_path('annarchy', tag)))
    return tasks


def _same_task(r, t):
    a = r.get('task', {})
    return (r.get('key') == t['key'] and a.get('marks') == t['marks']
            and a.get('scored') == t['scored'] and a.get('budget') == t['budget']
            and a.get('trial') == t['trial'])


def finished(t):
    r = read_json(t['result'])
    return bool(r) and bool(r.get('done')) and _same_task(r, t)


def set_aside_stale(t):
    """Rename a stored run made with other settings, so it is run again."""
    r = read_json(t['result'])
    if r is not None and not _same_task(r, t):
        os.replace(t['result'], t['result'] + '.stale-' + time.strftime('%Y%m%d-%H%M%S'))
        return True
    return False


def measured_rates():
    """Seconds per evaluation measured so far, per (env, encoding, d, optimizer): the
    latest level of every stored run of this sweep."""
    rates = {}
    for path in glob.glob(out_path('runs', '*.json')):
        r = read_json(path)
        if not r or not r.get('rows'):
            continue
        t = r['task']
        row = r['rows'][-1]
        if row['evals'] > 0:
            key = (t['cell']['environment'], t['cell']['encoding'], int(t['dim']),
                   t['optimizer'])
            rates.setdefault(key, []).append(row['train_seconds'] / row['evals'])
    return {k: float(np.median(v)) for k, v in rates.items()}


def est_seconds(t, rates=None):
    """How long a run takes: measured seconds per evaluation if this sweep has any for the
    same cell and optimizer, else EP_SECONDS (an untrained network: an upper bound)."""
    c = t['cell']
    key = (c['environment'], c['encoding'], int(t['dim']), t['optimizer'])
    per_eval = (rates or {}).get(key)
    ep = EP_SECONDS.get(c['environment'], 0.05)
    if per_eval is None:
        per_eval = t['eval_episodes'] * ep
    return (t['budget'] * per_eval + len(t['scored']) * t['report_episodes'] * ep + 30.0)


def makespan(durations, workers):
    """Tasks handed out longest first, each to the first free worker."""
    free = [0.0] * max(1, min(int(workers), len(durations) or 1))
    end = 0.0
    for d in sorted(durations, reverse=True):
        t = heapq.heappop(free) + d
        end = max(end, t)
        heapq.heappush(free, t)
    return end


def children_cpu():
    """CPU seconds of this process's finished child processes (and theirs)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class EpisodeCounter:
    """Wraps snn_feedforward.evaluate() to count the training episodes and add up their raw
    reward. It asks evaluate() for the raw reward too and hands back exactly what the
    caller asked for, so the fitness the optimizer sees is unchanged."""

    def __init__(self, evaluate):
        self.evaluate, self.episodes, self.raw = evaluate, 0, 0.0

    def __call__(self, net, weights, episode_seed=None, return_raw=False):
        fitness, raw = self.evaluate(net, weights, episode_seed=episode_seed, return_raw=True)
        self.episodes += 1
        self.raw += float(raw)
        return (fitness, raw) if return_raw else fitness


def run_task(spec_path):
    """ONE run, in its own process: compile the cell, run the optimizer for the run's
    budget, score the recommendation at the scored levels, save after every level."""
    spec = read_json(spec_path)
    if spec is None:
        raise SystemExit("no run spec at %s" % spec_path)
    # --stop sends SIGTERM: leave through `finally`, so the compile folder is removed
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    snn, O = modules()
    apply_settings(spec['settings'])
    # the parametrization the launcher planned with (search_key): the same values as
    # optimize_snn.py's own unless a wrapper such as sigma_check.py set other ones
    O.PARAM_SIGMA = spec['key'].get('param_sigma', O.PARAM_SIGMA)
    O.PARAM_NOISY = spec['key'].get('param_noisy', O.PARAM_NOISY)
    c = spec['cell']
    env = c['environment']
    trial = int(spec['trial'])
    result = dict(task={k: spec[k] for k in ('tag', 'cell', 'optimizer', 'seed', 'trial',
                                             'kind', 'run_k', 'marks', 'scored', 'dim',
                                             'budget')},
                  key=spec['key'], rows=[], done=False,
                  started=time.strftime('%Y-%m-%d %H:%M:%S'), host=os.uname().nodename)
    build = spec['build'] + '-p%d' % os.getpid()
    t0, cpu0, kids0 = time.time(), time.process_time(), children_cpu()
    try:
        with bare_argv(), contextlib.redirect_stdout(io.StringIO()):
            net = snn.build_network(env, c['hidden_sizes'], c['encoding'], c['decoding'],
                                    build, n_per_dim=c['n_per_dim'],
                                    spike_generator=spec['settings'].get('generator'))
        shapes = net['shapes']
        dim = O.n_params(shapes)
        if dim != int(spec['dim']):
            raise RuntimeError("the network has %d weights, the plan says %d"
                               % (dim, spec['dim']))
        result['compile_seconds'] = round(time.time() - t0, 2)
        # the compiler runs as child processes (cmake, make, g++): their CPU counts too
        result['compile_cpu_seconds'] = round(time.process_time() - cpu0
                                              + children_cpu() - kids0, 2)
        O.check_scoring_loop(net, shapes)       # the report scores what evaluate() scores

        counter = EpisodeCounter(O.evaluate)
        O.evaluate = counter                    # make_objective() calls optimize_snn.evaluate
        np.random.seed(trial)                   # the decoder's random draws while training
        objective = O.make_objective(net, shapes, int(spec['eval_episodes']),
                                     train_seed=trial)
        init = O.initial_point(dim, trial)
        per_weight = {int(round(k * dim)): float(k) for k in spec['marks']}
        scored = set(int(round(k * dim)) for k in spec['scored'])
        report_seed = int(O.REPORT_SEED) + trial          # optimize()'s report block
        clock = dict(start=time.perf_counter(), paused=0.0,
                     cpu_start=time.process_time(), cpu_paused=0.0)

        def on_mark(count, x):
            t_mark, cpu_mark = time.perf_counter(), time.process_time()
            row = dict(per_weight=per_weight[count], evals=count,
                       train_seconds=round(t_mark - clock['start'] - clock['paused'], 3),
                       train_cpu_seconds=round(cpu_mark - clock['cpu_start']
                                               - clock['cpu_paused'], 3),
                       train_episodes=counter.episodes,
                       train_raw_mean=round(counter.raw / max(1, counter.episodes), 3))
            if count in scored:
                t_rep, cpu_rep = time.perf_counter(), time.process_time()
                r = O.report_fitness(net, O.flat_to_weights(x, shapes),
                                     int(spec['report_episodes']), report_seed)
                rewards = np.asarray(r['rewards'], dtype=float)
                row.update(raw=round(float(r['reward_mean']), 3),
                           raw_sem=round(float(rewards.std(ddof=1) / math.sqrt(len(rewards)))
                                         if len(rewards) > 1 else 0.0, 3),
                           shaped=round(float(r['mean']), 3),
                           goal_share=round(float(r['goal_share']), 4),
                           random_actions=round(float(r['random_actions']), 4),
                           silent=round(float(r['silent']), 4),
                           tied=round(float(r['tied']), 4),
                           passes=bool(r['reward_mean'] >= THRESHOLD[env]),
                           report_seconds=round(time.perf_counter() - t_rep, 3),
                           report_cpu_seconds=round(time.process_time() - cpu_rep, 3))
            result['rows'].append(row)
            result['wall_seconds'] = round(time.time() - t0, 1)
            atomic_json(spec['result'], result)
            clock['paused'] += time.perf_counter() - t_mark
            clock['cpu_paused'] += time.process_time() - cpu_mark

        atomic_json(spec['result'], result)
        rec, info, ran, n = search(spec['optimizer'], init, trial, int(spec['budget']),
                                   objective, sorted(per_weight), on_mark)
        result.update(done=True, param=info, sub_optim=ran, evaluations=n,
                      wall_seconds=round(time.time() - t0, 1),
                      finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        atomic_json(spec['result'], result)
        last = result['rows'][-1]
        print("%s: done in %s | raw %.1f at %g x d (%d evaluations)"
              % (spec['tag'], fmt_secs(time.time() - t0), last.get('raw', float('nan')),
                 last['per_weight'], last['evals']), flush=True)
    finally:
        shutil.rmtree(build, ignore_errors=True)


# ---------------------------------------------------------------- machine and processes
def physical_cores():
    """Physical cores this process may use (SMT siblings count once, a container's CPU
    quota respected): run_pipeline.py's rule."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpus = list(range(os.cpu_count() or 1))
    cores = set()
    for c in cpus:
        try:
            with open('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % c) as f:
                cores.add(f.read().strip())
        except OSError:
            cores.add(str(c))
    n = len(cores) or len(cpus) or 1
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()[:2]
        if quota != 'max':
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                    # a zombie child of ours is not running
        with open('/proc/%d/stat' % int(pid)) as f:
            if f.read().split(')')[-1].split()[0] == 'Z':
                return False
    except (OSError, IndexError):
        pass
    return True


def runs_this_script(pid):
    if not alive(pid):
        return False
    try:
        with open('/proc/%d/cmdline' % int(pid), 'rb') as f:
            args = [a.decode(errors='replace') for a in f.read().split(b'\0') if a]
    except OSError:
        return True
    return any(a.endswith('budget_sweep.py') for a in args)


def child_env():
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[v] = '1'
    bindir = os.path.dirname(sys.executable)        # the venv's cmake / nanobind for ANNarchy
    env['PATH'] = bindir + os.pathsep + env.get('PATH', '')
    if sys.prefix != getattr(sys, 'base_prefix', sys.prefix):
        env['VIRTUAL_ENV'] = sys.prefix
    return env


def start(t):
    """One run in its own session (so --stop stops it with everything it spawns), at lower
    priority, appending to its log."""
    for folder in ('tasks', 'runs', 'logs', 'annarchy'):
        os.makedirs(out_path(folder), exist_ok=True)
    atomic_json(t['spec'], t)
    cmd = ([shutil.which('nice'), '-n', str(NICE)] if NICE and shutil.which('nice') else []) \
        + [sys.executable, os.path.abspath(__file__), '--run-task', t['spec']]
    f = open(t['log'], 'a')
    f.write("\n===== %s  %s\n" % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(cmd)))
    f.flush()
    proc = subprocess.Popen(cmd, cwd=CODE_DIR, env=child_env(), stdout=f,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    f.close()
    return proc


def kill_group(pid, wait=20):
    if not alive(pid):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(int(pid), signal.SIGTERM)
    t0 = time.time()
    while alive(pid) and time.time() - t0 < wait:
        time.sleep(0.5)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(int(pid), signal.SIGKILL)


def tail(path, n=3):
    try:
        with open(path, errors='replace') as f:
            lines = [l.strip() for l in f.readlines()[-200:] if l.strip()]
    except OSError:
        return ''
    errors = [l for l in lines if l.split(':')[0].endswith(('Error', 'Exception', 'Exit'))]
    return " / ".join(errors[-1:] + lines[-n:])[-600:]


# ---------------------------------------------------------------- the launcher
def state_path():
    return out_path('state.json')


def take_lock():
    os.makedirs(out_path(), exist_ok=True)
    lock = out_path('launcher.pid')
    other = read_json(lock)
    if other and other.get('pid') != os.getpid() and runs_this_script(other.get('pid')):
        raise SystemExit("the sweep is already running here (pid %s, started %s). Use "
                         "--status to follow it, or --stop to stop it."
                         % (other['pid'], other.get('started')))
    atomic_json(lock, dict(pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S')))


def drop_lock():
    lock = out_path('launcher.pid')
    other = read_json(lock)
    if other and other.get('pid') == os.getpid():
        with contextlib.suppress(OSError):
            os.remove(lock)


def machine_report():
    """What this machine offers right now: cores, and how busy it is."""
    phys = physical_cores()
    try:
        threads = len(os.sched_getaffinity(0))
    except AttributeError:
        threads = os.cpu_count() or phys
    load = os.getloadavg()
    idle = max(0, int(round(phys - load[0])))
    return ("this machine: %d physical cores (%d hardware threads) open to this account; "
            "load average %.1f / %.1f / %.1f over 1 / 5 / 15 minutes, so about %d %s idle "
            "right now" % (phys, threads, load[0], load[1], load[2], idle,
                           "core is" if idle == 1 else "cores are"))


def cmd_run(cores, leave_free=None):
    take_lock()
    state = read_json(state_path(), {}) or {}
    st = state.setdefault('runs', {})
    state['launcher'] = dict(pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                             cores=cores, leave_free=leave_free)
    state['holding'] = None
    phys = physical_cores()
    log(machine_report())
    clean_builds()
    log("sweep: checking which optimizers ignore their budget, and the search loop")
    tasks = make_tasks()
    describe_tests(tasks)
    moved = [t['tag'] for t in tasks if set_aside_stale(t)]
    if moved:
        log("sweep: %d stored runs were made with other settings and are set aside "
            "(.stale-*), e.g. %s" % (len(moved), moved[0]))
    running = {}                                     # tag -> (task, Popen or None, pid)
    for t in tasks:                                  # runs left going by an earlier launcher
        s = st.get(t['tag'], {})
        if s.get('status') == 'running' and runs_this_script(s.get('pid')):
            running[t['tag']] = (t, None, s['pid'])
    todo = [t for t in tasks if t['tag'] not in running and not finished(t)]
    for t in tasks:
        if finished(t):
            st[t['tag']] = dict(status='done')
        elif t['tag'] not in running:
            st[t['tag']] = dict(status='waiting', attempts=0)
    rates = measured_rates()
    todo.sort(key=lambda t: -est_seconds(t, rates))   # longest first
    log("sweep: %d runs, %d done, %d to run, %d still running from before; at most %d at "
        "a time%s" % (len(tasks), len(tasks) - len(todo) - len(running), len(todo),
                      len(running), cores,
                      "" if leave_free is None else
                      ", and only while %d cores stay idle for others" % leave_free))
    atomic_json(state_path(), state)

    def may_start():
        """(start one more run now?, idle cores). With leave_free, the machine's 1-minute
        load average must leave leave_free cores idle after the new run. The load average
        catches up with a new process over about a minute, so each of our runs is counted
        in it only as far as it has had time to show up (1 - exp(-age / 60 s))."""
        if leave_free is None:
            return True, None
        now = time.time()
        ours = sum(1.0 - math.exp(-max(0.0, now - st[tag].get('started_ts', now)) / 60.0)
                   for tag in running)
        others = max(0.0, os.getloadavg()[0] - ours)
        idle = phys - others - len(running)
        return idle - 1 >= leave_free, idle

    def on_signal(signum, frame):
        _STOPPING[0] = True
        log("stop requested (signal %d): stopping every run this sweep started" % signum)
        for tag, (t, proc, pid) in list(running.items()):
            kill_group(pid)
            st[tag] = dict(status='waiting', attempts=st.get(tag, {}).get('attempts', 0))
        state['stage'] = 'stopped'
        atomic_json(state_path(), state)
        drop_lock()
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    state['stage'] = 'running'
    n_total, t_start = len(tasks), time.time()
    try:
        while todo or running:
            while todo and len(running) < cores:
                ok, idle = may_start()
                if not ok:
                    if not state.get('holding'):
                        log("sweep: about %.0f idle, %d cores to keep free: waiting before "
                            "starting more runs (%d going)" % (idle, leave_free, len(running)))
                        state['holding'] = ("waiting for idle cores since %s (%d kept free)"
                                            % (time.strftime('%a %H:%M'), leave_free))
                    break
                if state.get('holding'):
                    log("sweep: cores free again, starting runs")
                    state['holding'] = None
                t = todo.pop(0)
                s = st.setdefault(t['tag'], {})
                s['attempts'] = s.get('attempts', 0) + 1
                proc = start(t)
                running[t['tag']] = (t, proc, proc.pid)
                s.update(status='running', pid=proc.pid, started_ts=time.time())
            atomic_json(state_path(), state)
            time.sleep(POLL_SECONDS)
            for tag, (t, proc, pid) in list(running.items()):
                if (proc.poll() is None) if proc is not None else alive(pid):
                    continue
                del running[tag]
                s = st[tag]
                s.pop('pid', None)
                if finished(t):
                    s['status'] = 'done'
                    r = read_json(t['result'])
                    n_done = sum(1 for x in tasks
                                 if st.get(x['tag'], {}).get('status') == 'done')
                    last = r['rows'][-1]
                    log("[%d/%d] %s: raw %.1f at %g x d, %s"
                        % (n_done, n_total, tag, last.get('raw', float('nan')),
                           last['per_weight'], fmt_secs(r.get('wall_seconds', 0))))
                elif s.get('attempts', 0) < ATTEMPTS and not _STOPPING[0]:
                    s['status'] = 'waiting'
                    todo.insert(0, t)
                    log("%s: stopped without finishing (%s); starting it again"
                        % (tag, tail(t['log'])))
                else:
                    s['status'] = 'failed'
                    log("%s: FAILED %d times, giving up on it | %s"
                        % (tag, s.get('attempts', 0), tail(t['log'])))
            atomic_json(state_path(), state)
    finally:
        if not _STOPPING[0]:
            drop_lock()
    state['stage'] = 'done'
    atomic_json(state_path(), state)
    failed = [tag for tag, s in st.items() if s.get('status') == 'failed']
    log("sweep: finished in %s%s. Results: %s"
        % (fmt_secs(time.time() - t_start),
           "" if not failed else ", %d runs failed (start it again to retry them)"
           % len(failed), out_path('runs')))
    cmd_summary(HOURS, cores, None)


def describe_tests(tasks):
    kinds = {}
    for t in tasks:
        kinds.setdefault(t['optimizer'], set()).add(t['kind'])
    blind = [n for n in OPTIMIZERS if kinds.get(n) == {'checkpoints'}]
    aware = [n for n in OPTIMIZERS if 'per-level' in kinds.get(n, ())]
    if blind:
        log("budget test: %s ignore their budget: one run each, read at every level"
            % ", ".join(blind))
    if aware:
        log("budget test: %s use their budget: one run per level in PER_RUN_LEVELS"
            % ", ".join(aware))


def cmd_stop():
    lock = read_json(out_path('launcher.pid'))
    said = False
    if lock and runs_this_script(lock.get('pid')):
        os.kill(int(lock['pid']), signal.SIGTERM)
        t0 = time.time()
        while alive(lock['pid']) and time.time() - t0 < 60:
            time.sleep(0.5)
        print("stopped the sweep (pid %s) and the runs it had going" % lock['pid'])
        said = True
    state = read_json(state_path(), {}) or {}
    n = 0
    for s in state.get('runs', {}).values():
        if s.get('pid') and runs_this_script(s['pid']):
            kill_group(s['pid'])
            n += 1
    if n or not said:
        print("stopped %d run(s) still going" % n)
    print("Start the sweep again to continue: runs that were cut start over, and the levels "
          "they finished stay saved until then.")


def clean_builds():
    """Remove compile folders left by runs whose process is gone (a kill skips a run's own
    clean-up). Folders of live processes are left alone."""
    import re
    for folder in glob.glob(out_path('annarchy', '*-p*')):
        m = re.search(r'-p(\d+)$', folder)
        if m and not alive(int(m.group(1))):
            shutil.rmtree(folder, ignore_errors=True)


def cmd_status(cores):
    state = read_json(state_path(), {}) or {}
    lock = read_json(out_path('launcher.pid'))
    launcher = state.get('launcher') or {}
    live = bool(lock and runs_this_script(lock.get('pid')))
    if live:
        head = "running (pid %s, started %s, at most %s at a time%s)" % (
            lock['pid'], lock.get('started'), launcher.get('cores', '?'),
            "" if launcher.get('leave_free') is None else
            ", keeping %s cores free" % launcher['leave_free'])
    elif state:
        head = "NOT running (last stage: %s)" % state.get('stage', '?')
    else:
        head = "not started yet"
    print("budget sweep in %s: %s" % (out_path(), head))
    tasks = make_tasks()
    st = state.get('runs', {})
    rates = measured_rates()
    done = [t for t in tasks if finished(t)]
    running = [t for t in tasks if st.get(t['tag'], {}).get('status') == 'running'
               and runs_this_script(st[t['tag']].get('pid')) and t not in done]
    failed = [t for t in tasks if st.get(t['tag'], {}).get('status') == 'failed']
    waiting = [t for t in tasks if t not in done and t not in running and t not in failed]
    print("runs: %d/%d done, %d running, %d failed, %d waiting"
          % (len(done), len(tasks), len(running), len(failed), len(waiting)))
    if live and state.get('holding'):
        print("  %s" % state['holding'])
    print("  %s" % machine_report())
    left_running = []
    if running:
        print("  %-62s %12s %9s %9s" % ('running', 'level', 'elapsed', 'left'))
        now = time.time()
        for t in sorted(running, key=lambda t: t['tag']):
            r = read_json(t['result']) or {}
            rows = r.get('rows') or []
            elapsed = now - st[t['tag']].get('started_ts', now)
            if rows:
                last = rows[-1]
                per_eval = last['train_seconds'] / max(1, last['evals'])
                left = (t['budget'] - last['evals']) * per_eval
                level = "%g/%g x d" % (last['per_weight'], t['run_k'])
            else:
                left = max(0.0, est_seconds(t, rates) - elapsed)
                level = "-/%g x d" % t['run_k']
            left_running.append(left)
            print("  %-62s %12s %9s %9s" % (t['tag'], level, fmt_secs(elapsed),
                                            fmt_secs(left)))
    for t in failed:
        print("  FAILED %s (log: %s)" % (t['tag'], os.path.relpath(t['log'], CODE_DIR)))
    if waiting or running:
        free = [x for x in left_running] + [0.0] * max(0, cores - len(left_running))
        heapq.heapify(free)
        end = max(left_running) if left_running else 0.0
        for d in sorted((est_seconds(t, rates) for t in waiting), reverse=True):
            x = heapq.heappop(free) + d
            end = max(end, x)
            heapq.heappush(free, x)
        print("about %s left on %d cores (waiting runs priced by %s); finish around %s"
              % (fmt_secs(end), cores, "the times measured so far" if rates else
                 "EP_SECONDS, an upper bound", time.strftime('%a %d %b %H:%M',
                                                             time.localtime(time.time() + end))))
    print("logs: %s" % os.path.relpath(out_path('logs'), CODE_DIR))


def cmd_dry_run(cores):
    tasks = make_tasks()
    describe_tests(tasks)
    rates = measured_rates()
    print("\nbudget sweep: %d cells x %d optimizers x %d seeds -> %d runs (%d done)\n"
          % (len(sweep_cells()), len(OPTIMIZERS), SEEDS, len(tasks),
             sum(1 for t in tasks if finished(t))))
    print("  %-38s %5s  %-40s %9s %9s" % ('cell', 'd', 'budgets (evaluations)', 'runs',
                                          'longest'))
    for c in sweep_cells():
        mine = [t for t in tasks if t['cell'] == c]
        d = mine[0]['dim']
        levels = LEVELS[c['environment']]
        budgets = "%d ... %d" % (round(min(levels) * d), round(max(levels) * d))
        print("  %-38s %5d  %-40s %9d %9s" % (cell_tag(c), d, budgets, len(mine),
                                             fmt_secs(max(est_seconds(t, rates)
                                                          for t in mine))))
    durations = [est_seconds(t, rates) for t in tasks if not finished(t)]
    print("\n  ~%s core-hours, longest run %s. Priced by %s."
          % ("{:,.0f}".format(sum(durations) / 3600),
             fmt_secs(max(durations) if durations else 0),
             "the times measured so far" if rates else
             "EP_SECONDS %s s per episode of an untrained network: an upper bound, since "
             "trained networks end episodes sooner" % EP_SECONDS))
    counts = [int(cores)] if cores else sorted(set(CORE_COUNTS))
    print("  the sweep takes about:")
    for n in counts:
        print("    %4d cores  %s" % (n, fmt_secs(makespan(durations, n))))
    print("\n%s" % machine_report())


# ---------------------------------------------------------------- results
def load_results():
    out = []
    for path in sorted(glob.glob(out_path('runs', '*.json'))):
        r = read_json(path)
        if r and r.get('rows') is not None and 'task' in r:
            out.append(r)
    return out


def scores(results):
    """{(env, enc, hidden, n): {optimizer: {level: [row per seed]}}} of the scored levels;
    a per-level run gives the level of its own budget only."""
    table = {}
    for r in results:
        t = r['task']
        c = t['cell']
        ck = (c['environment'], c['encoding'], topo(c['hidden_sizes']), c['n_per_dim'],
              int(t['dim']))
        cell = table.setdefault(ck, {}).setdefault(t['optimizer'], {})
        for row in r['rows']:
            if 'raw' not in row:
                continue
            if t['kind'] == 'per-level' and float(row['per_weight']) != float(t['run_k']):
                continue
            cell.setdefault(float(row['per_weight']), []).append(row)
    return table


class CostModel:
    """CPU seconds of a run of k x d evaluations, from the sweep's times (wall seconds for
    a run saved before the sweep recorded CPU time)."""

    def __init__(self, results):
        grid_eval = float(modules()[1].EVAL_EPISODES)
        self.cells = {}            # (env, enc, d) -> {optimizer: dict(points, ...)}
        for r in results:
            t = r['task']
            c = t['cell']
            key = (c['environment'], c['encoding'], int(t['dim']))
            o = self.cells.setdefault(key, {}).setdefault(
                t['optimizer'], dict(points={}, compile=[], report_ep=[]))
            rows = r['rows']
            if t['kind'] == 'checkpoints':
                use = rows
            else:                  # its own budget only: earlier marks belong to a run
                use = [x for x in rows if float(x['per_weight']) == float(t['run_k'])]
            # priced at the grid's episodes per evaluation (the sweep's are the same unless
            # EVAL_EPISODES was changed, as in --smoke)
            scale = grid_eval / float(r['key'].get('eval_episodes') or grid_eval)
            for row in use:
                o['points'].setdefault(float(row['per_weight']), []).append(
                    float(row.get('train_cpu_seconds', row['train_seconds'])) * scale)
            comp = r.get('compile_cpu_seconds', r.get('compile_seconds'))
            if comp is not None:
                o['compile'].append(float(comp))
            n_rep = max(1, int(r['key'].get('report_episodes', 1)))
            for row in rows:
                rep = row.get('report_cpu_seconds', row.get('report_seconds'))
                if rep is not None:
                    o['report_ep'].append(float(rep) / n_rep)
        self.probed = sorted({n for v in self.cells.values() for n in v})

    def measured(self, key, name, k):
        """T(k) of one measured cell, interpolated in k (seeds averaged per level). Beyond
        the largest level it continues at the last level's rate (extrapolated=True)."""
        o = self.cells.get(key, {}).get(name)
        if not o or not o['points']:
            return None, False
        ks = sorted(o['points'])
        Ts = [float(np.mean(o['points'][x])) for x in ks]
        if k <= ks[0]:
            return Ts[0] * k / ks[0], False
        if k <= ks[-1]:
            return float(np.interp(k, ks, Ts)), False
        slope = ((Ts[-1] - Ts[-2]) / (ks[-1] - ks[-2])) if len(ks) > 1 else Ts[-1] / ks[-1]
        return Ts[-1] + slope * (k - ks[-1]), True

    def _timed(self, key, name=None):
        """Has cell `key` (optimizer `name`, or any) a measured time yet? A run that has
        started but not reached its first level has none."""
        opts = self.cells.get(key, {})
        if name is not None:
            return bool(opts.get(name, {}).get('points'))
        return any(o['points'] for o in opts.values())

    def _keys(self, env, enc):
        """The measured cells of this environment and encoding, or of the environment when
        that encoding has none yet."""
        keys = sorted(x for x in self.cells if x[0] == env and x[1] == enc and self._timed(x))
        return keys or sorted(x for x in self.cells if x[0] == env and self._timed(x))

    def ready(self, envs, encs):
        """The (environment, encoding) pairs the plan can't price yet."""
        return [(e, c) for e in envs for c in encs if not self._keys(e, c)]

    def train_seconds(self, env, enc, d, name, k):
        """T for any cell: the mean seconds per evaluation, T(k) / (k d), of the measured
        cells of this environment and encoding, interpolated linearly in d. An optimizer
        without a measured time (not in the sweep, or not past its first level yet) is
        priced as the mean of those that have one."""
        keys = self._keys(env, enc)
        names = [name] if any(self._timed(x, name) for x in keys) else self.probed
        pts, extra = [], False
        for x in keys:
            vals = []
            for nm in names:
                T, e = self.measured(x, nm, k)
                if T is not None:
                    vals.append(T / (k * x[2]))
                    extra = extra or e
            if vals:
                pts.append((x[2], float(np.mean(vals))))
        if not pts:
            raise SystemExit("no times measured for %s/%s yet" % (env, enc))
        ds, ms = zip(*pts)
        return float(np.interp(d, ds, ms)) * k * d, extra

    def _per_d(self, env, enc, field, default):
        pts = []
        for x in self._keys(env, enc):
            vals = [v for o in self.cells[x].values() for v in o[field]]
            if vals:
                pts.append((x[2], float(np.mean(vals))))
        if not pts:
            return lambda d: default
        ds, vs = zip(*pts)
        return lambda d: float(np.interp(d, ds, vs))

    def overhead(self, env, enc, d, episodes, compile_=True):
        """The compile (if compile_) plus `episodes` scored episodes of a trained network,
        both interpolated in d."""
        comp = self._per_d(env, enc, 'compile', 30.0)(d) if compile_ else 0.0
        return comp + episodes * self._per_d(env, enc, 'report_ep', 0.0)(d)


def read_constants(path, names):
    """Top-level `NAME = <literal>` (or `dict(...)` of literals) assignments of a Python
    file, read without importing it."""
    def literal(node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'dict' and not node.args):
            return {kw.arg: literal(kw.value) for kw in node.keywords}
        return ast.literal_eval(node)
    try:
        with open(path) as f:
            tree = ast.parse(f.read(), path)
    except (OSError, SyntaxError):
        return {}
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            with contextlib.suppress(ValueError, TypeError, SyntaxError):
                out[node.targets[0].id] = literal(node.value)
    return out


def flag_value(args, flag, default, kind=str):
    """The value after `flag` in an argument list, as tune_hyperparams.py would read it."""
    args = list(args or [])
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return kind(args[i + 1])
        if a.startswith(flag + '='):
            return kind(a.split('=', 1)[1])
    return default


def pipeline_setup():
    """run_pipeline.py's GRID and tuning settings, and tune_hyperparams.py's."""
    pipe = read_constants(os.path.join(CODE_DIR, PIPELINE_FILE),
                          {'GRID', 'TUNE_EVALS_PER_DIM', 'TUNER_ARGS', 'TUNE_N'})
    found = 'GRID' in pipe
    grid = dict(DEFAULT_GRID)
    grid.update(pipe.get('GRID', {}))
    tuner = read_constants(os.path.join(CODE_DIR, TUNER_FILE),
                           {'OUTER_BUDGET', 'K_INNER', 'TUNE_REPORT_EPISODES',
                            'CHECKPOINTS_PER_DIM', 'INNER_EVAL_EPISODES', 'INNER_OPTIMIZER'})
    targs = pipe.get('TUNER_ARGS', [])
    tune = None
    if tuner:
        tune = dict(
            per_dim=pipe.get('TUNE_EVALS_PER_DIM'),
            n=pipe.get('TUNE_N'),                   # None (or an older file): every cell
            outer=flag_value(targs, '--outer-budget', tuner.get('OUTER_BUDGET', 40), int),
            k_inner=flag_value(targs, '--k-inner', tuner.get('K_INNER', 4), int),
            optimizer=flag_value(targs, '--inner-optimizer',
                                 tuner.get('INNER_OPTIMIZER', 'RandomSearch')),
            eval_episodes=flag_value(targs, '--eval-episodes',
                                     tuner.get('INNER_EVAL_EPISODES', 10), int),
            report_episodes=flag_value(targs, '--report-episodes',
                                       tuner.get('TUNE_REPORT_EPISODES', 20), int),
            checkpoints=len(tuner.get('CHECKPOINTS_PER_DIM', (20, 50))))
    return grid, found, tune


def grid_cells(grid):
    out = []
    for env in grid['environments']:
        for hidden in grid['topologies']:
            for enc in grid['encodings']:
                for n in grid['n_per_dim']:
                    out.append((env, enc, list(hidden), int(n),
                                genome_dim(env, hidden, enc, n)))
    return out


def clip_budget(k, d, grid):
    b = int(round(k * d))
    if grid.get('budget_min') is not None:
        b = max(int(grid['budget_min']), b)
    if grid.get('budget_max') is not None:
        b = min(int(grid['budget_max']), b)
    return b


class Plan:
    """Wall time of the grid under a rule: the tasks priced by the CostModel, packed longest
    first on `cores` free cores (set .cores to price another number)."""

    def __init__(self, model, grid, names, cores):
        snn, O = modules()
        self.model, self.grid, self.names, self.cores = model, grid, names, int(cores)
        self.cells = grid_cells(grid)
        # what optimize() scores around the search: check (6), report block, N3 block,
        # the recommendation's training loss
        self.extra_episodes = 6 + 2 * int(O.REPORT_EPISODES) + int(O.EVAL_EPISODES)
        self.grid_eval = int(O.EVAL_EPISODES)
        self._cache = {}
        self._tasks = {}

    def task(self, env, enc, d, name, k):
        """(seconds, extrapolated) of one grid task with k evaluations per weight."""
        key = (env, enc, d, name, round(k, 6))
        if key not in self._cache:
            b = clip_budget(k, d, self.grid)
            T, extra = self.model.train_seconds(env, enc, d, name, b / float(d))
            self._cache[key] = (T + self.model.overhead(env, enc, d, self.extra_episodes),
                                extra)
        return self._cache[key]

    def _durations(self, k_env):
        key = tuple(sorted(k_env.items()))
        if key not in self._tasks:
            durations, extra, by_env = [], False, {}
            for env, enc, hidden, n, d in self.cells:
                for name in self.names:
                    s, e = self.task(env, enc, d, name, k_env[env])
                    durations += [s] * int(self.grid['trials'])
                    by_env[env] = by_env.get(env, 0.0) + s * int(self.grid['trials']) / 3600
                    extra = extra or e
            self._tasks[key] = (sorted(durations, reverse=True), extra, by_env)
        return self._tasks[key]

    def evaluate(self, k_env):
        """{wall, core_hours, by_env, longest, extrapolated} for k_env = {env: k}."""
        durations, extra, by_env = self._durations(k_env)
        return dict(wall=makespan(durations, self.cores), core_hours=sum(durations) / 3600,
                    by_env=by_env, longest=durations[0], extrapolated=extra,
                    tasks=len(durations))

    def largest(self, window, fixed=None, vary=None, k_max=100.0, step=1.0):
        """The largest k (in `step`s up to k_max) for the environments in `vary` (all if
        None) whose wall time fits `window`; `fixed` holds the other environments' k. The
        wall time grows with k, so the scan stops once it is well past the window."""
        envs = self.grid['environments']
        vary = list(vary or envs)
        best = None
        k = step
        while k <= k_max + 1e-9:
            k_env = dict(fixed or {})
            k_env.update({e: k for e in vary})
            r = self.evaluate(k_env)
            if r['wall'] <= window:
                best = (k, r)
            elif r['wall'] > 1.05 * window:
                break
            k += step
        return best


def tuning_cost(model, plan, tune, k_env):
    """(core-hours, wall lower bound) of run_pipeline.py's tuning stage: per tuned cell
    (those with TUNE_N neurons per observation, or all), (outer + 2) x K inner runs of the
    tuner's inner optimizer, at TUNE_EVALS_PER_DIM x d or, when that is None, at the
    cell's grid budget."""
    runs = (tune['outer'] + 2) * tune['k_inner']
    core = longest = 0.0
    for env, enc, hidden, n, d in plan.cells:
        if tune.get('n') is not None and int(n) != int(tune['n']):
            continue
        kt = float(tune['per_dim']) if tune['per_dim'] is not None else k_env[env]
        b = clip_budget(kt, d, plan.grid)
        T, _ = model.train_seconds(env, enc, d, tune['optimizer'], b / float(d))
        T *= tune['eval_episodes'] / float(plan.grid_eval)
        # a tuner worker compiles its cell once, then runs many inner runs
        T += model.overhead(env, enc, d, (tune['checkpoints'] + 1) * tune['report_episodes'],
                            compile_=False)
        core += runs * T
        longest = max(longest, T)
    return core / 3600, max(core / plan.cores, longest)


def solved_counts(tab, env, k):
    """Of the measured cells of `env`: in how many does some optimizer pass with its median,
    and RandomSearch, at the largest measured level <= k."""
    cells = [ck for ck in tab if ck[0] == env]
    some = rs = n = 0
    for ck in cells:
        levels = sorted({L for o in tab[ck].values() for L in o if L <= k + 1e-9})
        if not levels:
            continue
        L = levels[-1]
        n += 1
        best = [np.median([r['raw'] for r in o[L]]) for o in tab[ck].values() if L in o]
        if best and max(best) >= THRESHOLD[env]:
            some += 1
        if 'RandomSearch' in tab[ck] and L in tab[ck]['RandomSearch']:
            if np.median([r['raw'] for r in tab[ck]['RandomSearch'][L]]) >= THRESHOLD[env]:
                rs += 1
    return "%d/%d (RS %d)" % (some, n, rs) if n else "-"


def print_scores(results):
    tab = scores(results)
    rows_meta = {}
    for r in results:
        t = r['task']
        c = t['cell']
        ck = (c['environment'], c['encoding'], topo(c['hidden_sizes']), c['n_per_dim'],
              int(t['dim']))
        rows_meta.setdefault(ck, {}).setdefault(t['optimizer'], set()).add(t['kind'])
    print("=== SCORES: seeds whose recommendation passes / seeds scored (median raw reward "
          "on the report block) ===")
    pending = []
    for env in ENVIRONMENTS:
        for ck in sorted((x for x in tab if x[0] == env), key=lambda x: (x[1], x[4])):
            levels = sorted({L for o in tab[ck].values() for L in o})
            if not levels:                  # its runs have started, none has a score yet
                pending.append(ck)
                continue
            print("\n%s | %s | hidden %s, %d per observation | d=%d | pass: raw >= %g"
                  % (env, ck[1], ck[2], ck[3], ck[4], THRESHOLD[env]))
            print("  %-13s" % 'optimizer' + "".join("%11s" % ("%gxd" % L) for L in levels)
                  + "  cpu ms/eval")
            model = CostModel([r for r in results if (r['task']['cell']['environment'],
                                                      r['task']['cell']['encoding'],
                                                      int(r['task']['dim'])) ==
                               (ck[0], ck[1], ck[4])])
            for name in OPTIMIZERS:
                if name not in tab[ck]:
                    continue
                cells_ = []
                for L in levels:
                    rows = tab[ck][name].get(L)
                    if not rows:
                        cells_.append("%11s" % '-')
                        continue
                    raws = [x['raw'] for x in rows]
                    cells_.append("%11s" % ("%d/%d %4.0f" % (sum(x['passes'] for x in rows),
                                                            len(rows), np.median(raws))))
                T, _ = model.measured((ck[0], ck[1], ck[4]), name, levels[-1])
                ms = "%13.1f" % (1000 * T / (levels[-1] * ck[4])) if T else "%13s" % "-"
                note = "  (one run per budget)" if 'per-level' in rows_meta[ck][name] else ""
                print("  %-13s" % name + "".join(cells_) + ms + note)
    if pending:
        print("\nstarted, no score yet: %s" % ", ".join(
            "%s %s %s/%d" % (ck[0], ck[1], ck[2], ck[3]) for ck in pending))
    print("\n=== BEST OF THE SIX: median raw of the best optimizer at each budget ===")
    for env in ENVIRONMENTS:
        cks = sorted((x for x in tab if x[0] == env and any(tab[x].values())),
                     key=lambda x: (x[1], x[4]))
        if not cks:
            continue
        levels = sorted({L for ck in cks for o in tab[ck].values() for L in o})
        print("\n%s (pass: raw >= %g)" % (env, THRESHOLD[env]))
        print("  %-26s" % 'cell' + "".join("%12s" % ("%gxd" % L) for L in levels))
        for ck in cks:
            cells_ = []
            for L in levels:
                best = [(np.median([r['raw'] for r in o[L]]), n) for n, o in tab[ck].items()
                        if L in o]
                if not best:
                    cells_.append("%12s" % '-')
                    continue
                v, n = max(best)
                cells_.append("%12s" % ("%.0f %s" % (v, short(n))))
            print("  %-26s" % ("%s %s/%d d=%d" % (ck[1], ck[2], ck[3], ck[4]))
                  + "".join(cells_))
    return tab


def cmd_summary(hours, cores, grid_optimizers, k_max=None):
    results = load_results()
    if not results:
        print("no runs saved yet in %s" % out_path('runs'))
        return
    n_done = sum(1 for r in results if r.get('done'))
    print("%d runs saved (%d finished, %d partial) in %s\n"
          % (len(results), n_done, len(results) - n_done, out_path('runs')))
    tab = print_scores(results)

    model = CostModel(results)
    grid, found, tune = pipeline_setup()
    missing = model.ready(grid['environments'], grid['encodings'])
    if missing:
        print("\n=== GRID TIME: not yet. It needs a measured time from every environment, and "
              "%s has none: no run there has reached its first level. Ask again later. ==="
              % " and ".join(sorted({e for e, c in missing})))
        return
    names = list(OPTIMIZERS)
    if grid_optimizers:
        if str(grid_optimizers).isdigit():
            names = ['(mean of the six)'] * int(grid_optimizers)
        else:
            names = [x.strip() for x in str(grid_optimizers).split(',') if x.strip()]
    detail = int(cores or max(CORE_COUNTS))
    plan = Plan(model, grid, names, detail)
    window = float(hours) * 3600
    top = max(max(LEVELS[e]) for e in grid['environments'] if e in LEVELS)
    k_max = float(k_max or top)
    n_tasks = len(plan.cells) * int(grid['trials']) * len(names)
    print("\n=== GRID TIME: %s's GRID, %d cells x %d trials x %d optimizers = %d tasks, "
          "window %gh ===" % (PIPELINE_FILE if found else 'DEFAULT_GRID (no %s here)'
                              % PIPELINE_FILE, len(plan.cells), int(grid['trials']),
                              len(names), n_tasks, hours))
    print("optimizers priced: %s%s" % (", ".join(sorted(set(names))),
                                       "" if grid_optimizers else
                                       " (the sweep's six; --grid-optimizers to change)"))
    measured = {env: sorted({x[2] for x in model.cells if x[0] == env})
                for env in grid['environments']}
    print("cost: CPU time measured at d = %s, interpolated in d for the other cells; budget "
          "= clip(k x d, %s, %s); * = cost beyond the largest level measured. Wall times "
          "assume free cores, one task each."
          % ("; ".join("%s (%s)" % (", ".join(map(str, v)), e) for e, v in measured.items()),
             grid.get('budget_min'), grid.get('budget_max')))
    print("solved = measured cells where some optimizer's median passes at the largest "
          "level <= k (RS: RandomSearch's)")

    if tune:
        per = tune['per_dim']
        n_tuned = sum(1 for c in plan.cells
                      if tune.get('n') is None or int(c[3]) == int(tune['n']))
        print("tuning >= = %s's tuning stage before the grid, at least: (%d + 2) x %d %s runs "
              "per tuned cell at %s (TUNE_EVALS_PER_DIM = %s), %d of the %d cells tuned%s"
              % (PIPELINE_FILE, tune['outer'], tune['k_inner'], tune['optimizer'],
                 "%g x d" % per if per is not None else "its grid budget", per, n_tuned,
                 len(plan.cells), "" if tune.get('n') is None
                 else " (TUNE_N = %d)" % int(tune['n'])))
    tune_col = (lambda k_env: fmt_secs(tuning_cost(model, plan, tune, k_env)[1])) if tune \
        else (lambda k_env: '-')

    envs = grid['environments']
    name_of = {'mountaincar': 'MountainCar', 'acrobot': 'Acrobot'}
    lab = lambda e: name_of.get(e, e)
    pair = (('acrobot', 'mountaincar') if set(envs) == {'acrobot', 'mountaincar'}
            else (tuple(envs) if len(envs) == 2 else None))

    # ---- the answer for every number of cores ----
    counts = sorted(set(CORE_COUNTS) | ({int(cores)} if cores else set()))
    fixed_ks = [float(x) for x in BY_CORES_FIXED_K]
    print("\nLARGEST k THAT FITS %gh, BY THE FREE CORES THE GRID GETS" % hours)
    head = "  %6s %9s %10s %10s" % ('cores', 'one k', 'grid wall', 'tuning >=')
    if pair:
        head += "   " + "".join("%14s" % ("%s %g:" % (lab(pair[0]), ka)) for ka in fixed_ks)
    print(head)
    if pair:
        print("  %38s   %s" % ('', "".join("%14s" % ("%s k" % lab(pair[1])) for _ in fixed_ks)))
    for n in counts:
        plan.cores = n
        best = plan.largest(window, k_max=k_max)
        if best:
            k, r = best
            row = "  %6d %9s %10s %10s" % (
                n, ("%g" % k) + ("+" if k >= k_max else ""), fmt_secs(r['wall']),
                tune_col({e: k for e in envs}))
        else:
            row = "  %6d %9s %10s %10s" % (n, '-', '', '')
        if pair:
            cols = []
            for ka in fixed_ks:
                res = plan.largest(window, fixed={pair[0]: ka}, vary=[pair[1]], k_max=k_max)
                cols.append("%14s" % ((("%g" % res[0]) + ("+" if res[0] >= k_max else ""))
                                      if res else '-'))
            row += "   " + "".join(cols)
        print(row + ("   <- --cores" if cores and n == int(cores) else ""))
    print("  - = doesn't fit even at k = 1; + = stopped at k_max = %g (--k-max looks "
          "further, at extrapolated cost)" % k_max)
    plan.cores = detail

    ks = sorted(set([float(x) for e in envs for x in LEVELS.get(e, ())] +
                    [float(grid.get('evals_per_dim') or 0)]) - {0.0})
    print("\nON %d CORES%s" % (detail, "" if cores else
                              " (the largest count above; --cores N for yours)"))
    print("\nONE k FOR BOTH ENVIRONMENTS")
    print("  %5s %9s %13s %11s %s %10s   %s"
          % ('k', 'wall', 'longest task', 'core-hours',
             " ".join("%12s" % lab(e) for e in envs), 'tuning >=',
             " ".join("%-19s" % ("solved " + lab(e)) for e in envs)))
    for k in ks:
        k_env = {e: k for e in envs}
        r = plan.evaluate(k_env)
        cur = "  <- run_pipeline.py's GRID now" if k == float(grid.get('evals_per_dim') or -1) \
            else ""
        print("  %5g %9s %13s %11s %s %10s   %s%s%s"
              % (k, fmt_secs(r['wall']), fmt_secs(r['longest']),
                 "{:,.0f}".format(r['core_hours']),
                 " ".join("%12s" % "{:,.0f}".format(r['by_env'].get(e, 0)) for e in envs),
                 tune_col(k_env), " ".join("%-19s" % solved_counts(tab, e, k) for e in envs),
                 " *" if r['extrapolated'] else "", cur))
    best = plan.largest(window, k_max=k_max)
    if best:
        k, r = best
        print("  -> largest k that fits %gh: k = %g (wall %s, %s core-hours)%s"
              % (hours, k, fmt_secs(r['wall']), "{:,.0f}".format(r['core_hours']),
                 " -- the cap k_max = %g, so more would fit" % k_max if k >= k_max else ""))
    else:
        print("  -> no k from 1 up fits %gh" % hours)

    if pair:
        a, b = pair
        print("\nONE k PER ENVIRONMENT: the largest %s k for each %s k" % (lab(b), lab(a)))
        print("  %11s %15s %9s %11s %10s   %-17s %-17s"
              % (lab(a) + ' k', lab(b) + ' k', 'wall', 'core-hours', 'tuning >=',
                 'solved ' + lab(a), 'solved ' + lab(b)))
        for ka in [float(x) for x in LEVELS.get(a, ())]:
            res = plan.largest(window, fixed={a: ka}, vary=[b], k_max=k_max)
            if not res:
                print("  %11g %15s   does not fit even with %s at k = 1"
                      % (ka, '-', lab(b)))
                continue
            kb, r = res
            print("  %11g %15s %9s %11s %10s   %-17s %-17s%s"
                  % (ka, ("%g" % kb) + ("+" if kb >= k_max else ""), fmt_secs(r['wall']),
                     "{:,.0f}".format(r['core_hours']), tune_col({a: ka, b: kb}),
                     solved_counts(tab, a, ka), solved_counts(tab, b, kb),
                     " *" if r['extrapolated'] else ""))
        print("  + = stopped at k_max = %g (--k-max looks further, at extrapolated cost)"
              % k_max)


# ---------------------------------------------------------------- main
def main():
    global SIZES, SEEDS, LEVELS, PER_RUN_LEVELS, EVAL_EPISODES, REPORT_EPISODES, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help='runs and a time estimate only')
    ap.add_argument('--status', action='store_true', help='progress and time left')
    ap.add_argument('--stop', action='store_true', help='stop the sweep and its runs')
    ap.add_argument('--summary', action='store_true', help='score tables and the plan')
    ap.add_argument('--smoke', action='store_true',
                    help='the whole sweep on tiny budgets, in %s/' % SMOKE['OUT_DIR'])
    ap.add_argument('--hours', type=float, default=HOURS,
                    help='the window the grid must fit (default %g)' % HOURS)
    ap.add_argument('--cores', type=int, default=None,
                    help='runs at once (needed to start the sweep), and the cores the plan '
                         'details for the grid (default: CORES)')
    ap.add_argument('--leave-free', type=int, default=None,
                    help='start a run only while the load average leaves this many cores '
                         'idle for others (default: LEAVE_FREE, off)')
    ap.add_argument('--grid-optimizers', default=None,
                    help='the grid\'s optimizers for the plan: a number, or names '
                         'separated by commas (default: the sweep\'s six)')
    ap.add_argument('--k-max', type=float, default=None,
                    help='largest k the plan looks at (default: the largest level)')
    ap.add_argument('--run-task', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_task:
        run_task(args.run_task)
        return
    if args.smoke:
        SIZES, SEEDS = SMOKE['SIZES'], SMOKE['SEEDS']
        LEVELS, PER_RUN_LEVELS = SMOKE['LEVELS'], SMOKE['PER_RUN_LEVELS']
        EVAL_EPISODES, REPORT_EPISODES = SMOKE['EVAL_EPISODES'], SMOKE['REPORT_EPISODES']
        OUT_DIR = SMOKE['OUT_DIR']
    given = args.cores or CORES
    launched = ((read_json(state_path(), {}) or {}).get('launcher') or {}).get('cores')
    leave_free = args.leave_free if args.leave_free is not None else LEAVE_FREE
    if args.stop:
        cmd_stop()
    elif args.status:
        cmd_status(given or launched or physical_cores())
    elif args.dry_run:
        cmd_dry_run(given)                    # None: the sweep's time for every CORE_COUNTS
    elif args.summary:
        cmd_summary(args.hours, given or launched, args.grid_optimizers, args.k_max)
    else:
        if args.smoke and not given:
            given = min(4, physical_cores())
        if not given:
            raise SystemExit(
                "budget_sweep.py starts only when you say how many runs it may keep going "
                "at once: --cores N (one core each, at nice %d). Add --leave-free M to "
                "wait whenever fewer than M cores are idle. --dry-run shows how long the "
                "sweep takes on %s cores.\n%s"
                % (NICE, ", ".join(str(n) for n in sorted(set(CORE_COUNTS))),
                   machine_report()))
        cmd_run(int(given), leave_free)


if __name__ == '__main__':
    main()
