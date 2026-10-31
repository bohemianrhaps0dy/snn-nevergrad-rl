"""
Nevergrad tuning of the ENCODING hyperparameters of the feedforward SNN.

Sibling of meta_optimize.py. That file tunes the inner optimizer's own
hyperparameters (CMA scale, DE crossover, ...). This file tunes the encoding
parameters that every optimizer shares.

TUNED (3 dimensions for every encoding):

    w_scale       weight bound: the inner run searches the grid's box
                  [-w_scale, w_scale], so w_scale means exactly what W_LOW / W_HIGH
                  mean in optimize_snn.py (what experiment_runner.py sets per cell
                  from grid_settings.json) and --bound means in calibrate.py.
    r_max, r_min  rate bounds of the active bin (Hz)           -- poprate
    lat_min       latency of the active bin at r = 1 (ms)      -- poplatency
    lat_window    latency of the active bin at r = 0 (ms)      -- poplatency
                  (searched as lat_min + lat_span, see LATENCY BOUNDS)

The population encoders use their full bounds: poprate interpolates the active
bin over [r_min, r_max] and poplatency interpolates its spike time over
[lat_window, lat_min], so all of these are live.

FIXED, set by hand in the CELLS table below:

    hidden_sizes  hidden layer width(s)
    n_per_dim     neurons per input dimension (use even values: with an odd
                  count the middle velocity bin spans v = 0)
    generator     poprate spike generator, 'poisson' or 'regular'
    t_step        simulated ms per environment step -- pinned to 50.0

Because topology, n_per_dim and generator are fixed, every worker compiles the
cell's network exactly ONCE. A tuned w_scale belongs to its cell: it does not
transfer to other topologies or n_per_dim values, so tune each cell the grid runs.

------------------------------------------------------------------------------
THE INNER RUN IS THE GRID'S
------------------------------------------------------------------------------

Every inner run goes through optimize_snn.py's own code: make_objective() for the
training objective, initial_point() for the starting weights, and train_weights()
for make_parametrization() plus the ask/tell loop. So a config is scored on the
search the grid runs: the raw box [-w_scale, w_scale] with PARAM_SIGMA's step
(Nevergrad's 1.0 by default) and PARAM_NOISY's declaration, INNER_EVAL_EPISODES
training episodes per evaluation, and the optimizer's own recommendation as the
final network (N1 is a grid diagnostic and is not used here). Three differences,
all deliberate:

  - Training episodes are seeded (make_objective(train_seed=seed)): the starts are
    fresh at every evaluation, as in the grid, but inner run k of every config sees
    the same sequence, so configs are compared on the same draws.
  - The recommendation is scored on TUNE_REPORT_EPISODES seeded episodes, kept clear
    of the training episodes and of the grid's report blocks.
  - Inner run k uses trial TRIAL_OFFSET + seed in optimize_snn, so its starting
    weights and optimizer seed never coincide with a grid trial's.

Budget: INNER_EVALS_PER_DIM x d evaluations (100 x d, no upper cap), the grid's
per-weight formula. --inner-budget N replaces it with a fixed N for every cell.

------------------------------------------------------------------------------
IS 100 x d WASTED? (--budget-check)
------------------------------------------------------------------------------

Every inner run also reads its recommendation after CHECKPOINTS_PER_DIM x d
evaluations (20 and 50 x d by default) and scores it on the same report episodes
as the final one, so each run shows what the later evaluations added; the results
keep these under 'checkpoints'. --budget-check runs only the current settings of
each cell, K runs in parallel, prints that table with the paired difference
between the final recommendation and each checkpoint, and stops. If the 20 x d
and 100 x d rows match on a cell, the extra budget is wasted there. The answer
belongs to the inner optimizer: add --inner-optimizer to ask it of another one.

------------------------------------------------------------------------------
ONE CAVEAT BEFORE THE OUTPUT GOES IN THE THESIS
------------------------------------------------------------------------------

Tuning uses ONE inner optimizer (--inner-optimizer, default RandomSearch).
Whatever it finds is "the encoding that suits RandomSearch". RandomSearch adapts
nothing, so it gives no grid contestant a home-field advantage, but it measures
the best of many random networks, not what the grid's optimizers reach. Confirm
the adopted settings with a short run of the grid's optimizers (tuned vs current,
5 seeds) before the full grid.

------------------------------------------------------------------------------
PARALLEL RUNS
------------------------------------------------------------------------------

The outer optimizer proposes OUTER_BATCH configs at a time (TBPSA's population,
12 for 3 dimensions, by default), and every one of their K inner runs runs in its
own worker process: a batch of 12 configs at K = 4 is 48 inner runs at once. Each
worker is single-threaded, compiles the cell's network once into its own folder
(deleted at the end unless --keep-builds), and receives its config with every
task, because spawn workers never see module changes made in the main process.
Only the main process writes the checkpoint, after each complete batch, so
--resume continues from the last finished batch. It continues only a checkpoint
made with the same scoring settings (inner optimizer, budget, checkpoints,
episodes, K): one written by an earlier version of this tuner is never resumed,
because its scores came from a different inner run. If one cell's batches leave
cores idle, run cells side by side as separate processes (--env / --encoding).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

    python tune_hyperparams.py --dry-run         # budgets, batches, cost estimate only
    python tune_hyperparams.py --quick           # minutes, plumbing check
    python tune_hyperparams.py --budget-check --env acrobot   # is 100 x d wasted?
    python tune_hyperparams.py --env mountaincar --encoding poprate --decoding vote
    python tune_hyperparams.py --env mountaincar --generator poisson
    python tune_hyperparams.py --resume          # continue from checkpoint
    python tune_hyperparams.py --workers 1       # serial, for debugging
    python tune_hyperparams.py --out-dir tuning_100d   # keep earlier results apart

Results overwrite tuning/result-<cell>.json, the files make_grid_settings.py reads
by default; use --out-dir (and its --results glob) to keep an earlier run's.
"""

import os

# One thread per process, set before anything imports ANNarchy; spawn workers re-import
# this file, so they get it too (as in experiment_runner.py).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import contextlib
import glob
import json
import math
import shutil
import time

import numpy as np
import nevergrad as ng
from nevergrad.optimization import optimizerlib as O

import snn_feedforward as snn
import optimize_snn
from snn_feedforward import build_network, weight_shapes
from optimize_snn import n_params, flat_to_weights, W_HIGH


# ============================================================================
# CONFIGURATION
# ============================================================================

T_STEP_FIXED = 50.0        # pinned, not tuned

# --- cells to tune. hidden_sizes, n_per_dim and generator are set BY HAND. --
# The cell should be one the MAIN GRID actually runs, otherwise the tuned values
# are fitted to a network that never appears in the results. generator only
# matters for poprate (None = SPIKE_GENERATOR in snn_feedforward.py).
CELLS = [
    dict(environment='mountaincar', encoding='poprate', decoding='vote',
         hidden_sizes=[6], n_per_dim=4, generator='regular'),
    dict(environment='mountaincar', encoding='poplatency', decoding='first',
         hidden_sizes=[6], n_per_dim=4, generator=None),
    dict(environment='acrobot', encoding='poprate', decoding='vote',
         hidden_sizes=[6], n_per_dim=4, generator='regular'),
    dict(environment='acrobot', encoding='poplatency', decoding='first',
         hidden_sizes=[6], n_per_dim=4, generator=None),
]

# --- outer loop -------------------------------------------------------------
OUTER_BUDGET    = 40       # configurations to evaluate (space is 3 dims)
OUTER_OPTIMIZER = 'TBPSA'  # noise-aware; 'NGOpt' is a reasonable alternative
OUTER_BATCH     = None     # configs proposed and scored at once. None = TBPSA's
                           # population (12 for 3 dims); other outer optimizers get
                           # workers // K_INNER. A bigger batch fills more cores but
                           # leaves TBPSA fewer generations (40 configs at 48 = one).

# --- inner loop (cost driver) ----------------------------------------------
INNER_OPTIMIZER      = 'RandomSearch'   # see the caveat in the module docstring
INNER_EVALS_PER_DIM  = 100      # budget = 100 x d weight evaluations (d = genome size),
INNER_BUDGET_MIN     = 1000     # clipped to [MIN, MAX] like the grid's formula;
INNER_BUDGET_MAX     = None     # None = no upper cap
CHECKPOINTS_PER_DIM  = (20, 50)  # also score the recommendation after 20 x d and
                                 # 50 x d evaluations (see --budget-check)
# Training episodes per evaluation, as EVAL_EPISODES in optimize_snn.py. They are
# seeded and fresh at every evaluation (make_objective(train_seed=...)), so an inner
# run trains on as many different starts as a grid run does. If the run is too slow,
# lower OUTER_BUDGET or K_INNER instead of this.
INNER_EVAL_EPISODES  = 10
K_INNER              = 4      # independent inner runs averaged to score a config
TUNE_REPORT_EPISODES = 20     # report episodes per inner run (and per checkpoint)

# The encoders' `graded` mode. evaluate() encodes through drive_encoder_fast(),
# which implements 'residual' only, so it is pinned and recorded in the results.
GRADED = 'residual'

# --- seeding ----------------------------------------------------------------
SEED                = 0
HOLDOUT_OFFSET      = 10_000    # held-out seed block for the honest re-score
BUDGET_CHECK_OFFSET = 20_000    # seed block of --budget-check
REPORT_SEED_GAP     = 500_000   # keeps report episodes disjoint from training ones
TRIAL_OFFSET        = 1_000     # inner run with seed s is trial TRIAL_OFFSET + s in
                                # optimize_snn (starting weights, optimizer's seed)

# --- search space bounds ----------------------------------------------------
# w_scale, log-scaled. At the low end the network goes silent, at the high end
# the outputs saturate into ties; +-20, +-40 and +-60 were tested in D5.
W_SCALE_LO, W_SCALE_HI = 5.0, 100.0
R_MAX_LO,   R_MAX_HI   = 50.0, 500.0  # log-scaled Hz
R_MIN_FRAC_HI          = 0.5          # r_min = frac * r_max, so r_min < r_max

# LATENCY BOUNDS (fix F4). The latency code needs its spike times spread over
# several simulation steps: with dt = 1 ms, a range below about 1.5 ms puts every
# spike in the same step and the code disappears (the Oct 5 run ended at
# 0.5-1.1 ms). So lat_min starts at 1 ms and the window is searched as
# lat_min + lat_span with a span of at least LAT_SPAN_LO ms. Both bounds keep
# lat_window <= T_STEP_FIXED - 2 by construction, because ANNarchy drops spikes
# scheduled past the end of the step without a warning.
LAT_MIN_LO,  LAT_MIN_HI  = 1.0, 10.0
LAT_SPAN_LO = 5.0
LAT_SPAN_HI = T_STEP_FIXED - 2.0 - LAT_MIN_HI

# --- cost estimate only (seconds per episode on the server, as in experiment_runner.py)
EP_SECONDS = {'mountaincar': 0.018, 'acrobot': 0.065}

# --- housekeeping -----------------------------------------------------------
OUT_DIR = 'tuning'
BUILD_DIR = 'annarchy'      # worker w compiles into BUILD_DIR/tune-<cell>-w<pid>


# ============================================================================
# GLOBAL PATCHING
# ============================================================================
# evaluate() encodes through drive_encoder_fast(), which reads R_MIN, R_MAX,
# LAT_MIN, LAT_WINDOW and T_STEP from snn_feedforward's module globals every
# time it runs, and reads n_per_dim from the compiled input layer. The inner
# run's box comes from optimize_snn's W_LOW / W_HIGH, which make_parametrization()
# and initial_point() read when they run. So setting the module globals is
# enough -- in the process that runs the inner run, which is why every task
# carries its config to its worker.

_TUNED_GLOBALS = ('T_STEP', 'R_MAX', 'R_MIN', 'LAT_WINDOW', 'LAT_MIN')

DEFAULTS = {name: getattr(snn, name) for name in _TUNED_GLOBALS}


@contextlib.contextmanager
def snn_config(cfg):
    """Apply a resolved config to snn_feedforward and optimize_snn, then restore."""
    saved = {name: getattr(snn, name) for name in _TUNED_GLOBALS}
    saved_box = (optimize_snn.W_LOW, optimize_snn.W_HIGH)
    try:
        snn.T_STEP     = float(T_STEP_FIXED)
        snn.R_MAX      = float(cfg['r_max'])
        snn.R_MIN      = float(cfg['r_min'])
        snn.LAT_WINDOW = float(cfg['lat_window'])
        snn.LAT_MIN    = float(cfg['lat_min'])
        w = float(cfg['w_scale'])
        optimize_snn.W_LOW, optimize_snn.W_HIGH = -w, w
        yield
    finally:
        for name, value in saved.items():
            setattr(snn, name, value)
        optimize_snn.W_LOW, optimize_snn.W_HIGH = saved_box


# ============================================================================
# SEARCH SPACE
# ============================================================================

def active_params(encoding):
    """Which hyperparameters affect behaviour for this encoding."""
    p = ['w_scale']
    if encoding == 'poprate':
        p += ['r_max', 'r_min_frac']
    if encoding == 'poplatency':
        p += ['lat_min', 'lat_span']
    return p


def build_space(encoding):
    """Instrumentation over the active parameters for this encoding."""
    full = dict(
        w_scale   =ng.p.Log(lower=W_SCALE_LO, upper=W_SCALE_HI),
        r_max     =ng.p.Log(lower=R_MAX_LO, upper=R_MAX_HI),
        r_min_frac=ng.p.Scalar(lower=0.0, upper=R_MIN_FRAC_HI),
        lat_min   =ng.p.Scalar(lower=LAT_MIN_LO, upper=LAT_MIN_HI),
        lat_span  =ng.p.Scalar(lower=LAT_SPAN_LO, upper=LAT_SPAN_HI),
    )
    return ng.p.Instrumentation(**{k: full[k] for k in active_params(encoding)})


def resolve(theta, encoding):
    """Raw search-space point -> concrete, always-valid config.

    r_min is a FRACTION of r_max, and lat_window is lat_min + lat_span, so
    r_min < r_max and lat_min < lat_window hold by construction, and the bounds
    above keep lat_window inside the simulated step. A tuner that evaluates broken
    configurations without noticing produces a confidently wrong answer, so the
    constraints live in the parametrization rather than in a post-hoc check.

    Inactive parameters fall back to the module defaults so snn_config() always
    receives a complete config.
    """
    g = DEFAULTS
    r_max = float(theta.get('r_max', g['R_MAX']))
    r_min = float(theta.get('r_min_frac', g['R_MIN'] / g['R_MAX'])) * r_max
    if 'lat_min' in theta:
        lat_min = float(theta['lat_min'])
        lat_window = lat_min + float(theta['lat_span'])
    else:
        lat_min, lat_window = float(g['LAT_MIN']), float(g['LAT_WINDOW'])
    return dict(
        w_scale=float(theta.get('w_scale', abs(W_HIGH))),
        r_max=r_max,
        r_min=r_min,
        lat_window=lat_window,
        lat_min=lat_min,
        t_step=float(T_STEP_FIXED),
    )


def baseline_config():
    """The settings the code uses today: the weight bound W_HIGH from
    optimize_snn.py and the rates and latencies in snn_feedforward.py."""
    g = DEFAULTS
    return dict(
        w_scale=float(abs(W_HIGH)),
        r_max=float(g['R_MAX']),
        r_min=float(g['R_MIN']),
        lat_window=float(g['LAT_WINDOW']),
        lat_min=float(g['LAT_MIN']),
        t_step=float(T_STEP_FIXED),
    )


# ============================================================================
# BUDGET
# ============================================================================

def cell_dim(cell):
    """Genome size d of a cell, without building its network."""
    return n_params(weight_shapes(cell['environment'], cell['hidden_sizes'],
                                  cell['encoding'], cell['n_per_dim']))


def inner_budget(dim, args):
    """Weight evaluations per inner run: --inner-budget if given, otherwise
    --evals-per-dim x d clipped to [INNER_BUDGET_MIN, INNER_BUDGET_MAX]."""
    if args.inner_budget is not None:
        return int(args.inner_budget)
    budget = max(int(round(args.evals_per_dim * dim)), INNER_BUDGET_MIN)
    if INNER_BUDGET_MAX is not None:
        budget = min(budget, int(INNER_BUDGET_MAX))
    return budget


def checkpoint_counts(dim, budget, per_dim):
    """Evaluation counts at which to read the recommendation: per_dim x d, below the
    budget (the final recommendation is always scored)."""
    counts = sorted({int(round(p * dim)) for p in per_dim})
    return [c for c in counts if 0 < c < budget]


# ============================================================================
# INNER RUN AND CONFIG SCORING
# ============================================================================

def _report(net, flat, n_episodes, episode_seed):
    """optimize_snn.report_fitness() of one flat weight vector, the fields kept."""
    r = optimize_snn.report_fitness(net, flat_to_weights(flat, net['shapes']),
                                    n_episodes, episode_seed)
    return dict(mean=r['mean'], raw=r['reward_mean'], goal_share=r['goal_share'],
                random_actions=r['random_actions'], silent=r['silent'],
                tied=r['tied'])


def run_inner(net, seed, inner_optimizer, budget, checkpoints, eval_episodes,
              report_episodes):
    """One weight optimization under the config snn_config() has applied, through
    optimize_snn's own inner run. Returns the report of the recommended weights
    (shaped mean, raw mean, goal share, random-action shares) and, under
    'checkpoints', the same report of the recommendation after each evaluation count
    in `checkpoints`.

    Seeding, kept disjoint on purpose:
      training episodes  optimize_snn.training_seed(seed, i, j, ...), 10**12 and up,
                         fresh at every evaluation i
      report episodes    (seed + REPORT_SEED_GAP)*1000 + k, the same block for the
                         final recommendation and every checkpoint (so they pair)
      decoder draws      numpy's global generator, seeded with `seed` here
      starting weights,  trial TRIAL_OFFSET + seed in optimize_snn
      optimizer's seed
    so the reported score is never measured on an episode the inner loop trained
    against, and the optimism gap stays visible.
    """
    shapes = net['shapes']
    trial = TRIAL_OFFSET + int(seed)
    np.random.seed(int(seed))                          # pin the decoder's draws
    objective = optimize_snn.make_objective(net, shapes, int(eval_episodes),
                                            train_seed=int(seed))
    init = optimize_snn.initial_point(n_params(shapes), trial)
    run = optimize_snn.train_weights(inner_optimizer, init, trial, int(budget),
                                     objective, checkpoints=checkpoints)
    report_seed = int(seed) + REPORT_SEED_GAP
    out = _report(net, run['rec'], report_episodes, report_seed)
    out['checkpoints'] = {str(c): _report(net, x, report_episodes, report_seed)
                          for c, (x, _) in sorted(run['checkpoints'].items())}
    out.update(evaluations=run['evaluations'], param=run['param'],
               sub_optim=run['sub_optim'])
    return out


# ---------- workers: one compiled network per process ----------
_WORKER = {}


def _build(cell, compile_dir):
    """Compile the cell's network and check that the report scores what evaluate()
    scores."""
    with snn_config(baseline_config()):
        net = build_network(cell['environment'], cell['hidden_sizes'],
                            cell['encoding'], cell['decoding'], compile_dir,
                            n_per_dim=cell['n_per_dim'],
                            spike_generator=cell.get('generator'))
        n_obs = int(net['cfg']['n_entrada'])
        if net['encoder'].size != n_obs * cell['n_per_dim']:
            raise RuntimeError('asked for %d neurons per observation, the network has %d'
                               % (cell['n_per_dim'], net['encoder'].size // n_obs))
        optimize_snn.check_scoring_loop(net, net['shapes'])
    return net


def _worker_init(cell, build_root):
    """Once per worker process: compile this cell's network into the worker's own
    folder."""
    folder = os.path.join(build_root, "tune-%s-w%d" % (_cell_tag(cell), os.getpid()))
    _WORKER['net'] = _build(cell, folder)


def _task(job):
    """WORKER: one inner run. The config travels with the task: a spawn worker
    starts from the modules as imported, not as the main process left them."""
    cfg, seed, inner_optimizer, budget, checkpoints, eval_episodes, report_episodes = job
    with snn_config(cfg):
        return run_inner(_WORKER['net'], seed, inner_optimizer, budget, checkpoints,
                         eval_episodes, report_episodes)


class Runner:
    """Runs inner runs on `workers` spawn processes (one network each), or in this
    process when workers is 1. map() returns the results in job order."""

    def __init__(self, cell, workers, build_root=BUILD_DIR):
        self.cell, self.workers, self.build_root = cell, int(workers), build_root
        self.pool = None
        if self.workers <= 1:
            _worker_init(cell, build_root)
        else:
            import concurrent.futures as cf
            import multiprocessing as mp
            self.pool = cf.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context('spawn'),
                initializer=_worker_init, initargs=(cell, build_root))

    def map(self, jobs):
        if self.pool is None:
            return [_task(j) for j in jobs]
        futures = [self.pool.submit(_task, j) for j in jobs]
        return [f.result() for f in futures]

    def close(self, keep_builds=False):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        if not keep_builds:
            pattern = os.path.join(self.build_root,
                                   "tune-%s-w*" % _cell_tag(self.cell))
            for folder in glob.glob(pattern):
                shutil.rmtree(folder, ignore_errors=True)


def _aggregate(reps):
    """Mean final quality over K inner runs, and the same for every checkpoint."""
    def block(rows):
        vals = np.array([r['mean'] for r in rows], dtype=float)
        raws = np.array([r['raw'] for r in rows], dtype=float)
        return dict(mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)) if len(rows) > 1 else 0.0,
                    vals=vals.tolist(),
                    raw=float(raws.mean()), raw_vals=raws.tolist(),
                    goal_share=float(np.mean([r['goal_share'] for r in rows])),
                    random_actions=float(np.mean([r['random_actions'] for r in rows])),
                    silent=float(np.mean([r['silent'] for r in rows])),
                    tied=float(np.mean([r['tied'] for r in rows])))
    out = block(reps)
    out['checkpoints'] = {c: block([r['checkpoints'][c] for r in reps])
                          for c in reps[0]['checkpoints']}
    out.update(param=reps[0]['param'], sub_optim=reps[0]['sub_optim'])
    return out


def score_configs(runner, cfgs, base_seed, args, budget, checkpoints):
    """Score every config in cfgs on K inner runs (seeds base_seed + k), all of them at
    once. Returns one summary per config, in order."""
    K = int(args.k_inner)
    jobs = [(cfg, base_seed + k, args.inner_optimizer, int(budget), list(checkpoints),
             int(args.eval_episodes), int(args.report_episodes))
            for cfg in cfgs for k in range(K)]
    reps = runner.map(jobs)
    return [_aggregate(reps[i * K:(i + 1) * K]) for i in range(len(cfgs))]


# ============================================================================
# CHECKPOINTING
# ============================================================================

def _atomic_write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)          # single-writer, no partial file on crash


def _regular(cell):
    gen = cell.get('generator') or snn.SPIKE_GENERATOR
    return cell['encoding'] == 'poprate' and gen == 'regular'


def _cell_tag(cell):
    """Everything that defines the compiled network, so a --resume never picks up
    a checkpoint from a different cell."""
    return "%s-%s-%s-h%s-n%d%s" % (
        cell['environment'], cell['encoding'], cell['decoding'],
        "x".join(str(h) for h in cell['hidden_sizes']), cell['n_per_dim'],
        '-regular' if _regular(cell) else '')


def _fmt(cfg, keys):
    """One-line view of just the parameters this encoding actually uses."""
    out = ['w=%.1f' % cfg['w_scale']]
    if 'r_max' in keys:
        out.append('rmax=%.0f' % cfg['r_max'])
    if 'r_min_frac' in keys:
        out.append('rmin=%.0f' % cfg['r_min'])
    if 'lat_min' in keys:
        out.append('latmin=%.1f' % cfg['lat_min'])
        out.append('latwin=%.1f' % cfg['lat_window'])
    return ' '.join(out)


def _ck_label(count, dim):
    """'20d' for a whole number of evaluations per weight, else the count itself."""
    per = count / float(dim)
    if per >= 1 and abs(per - round(per)) < 0.05 * per:
        return "%dd" % int(round(per))
    return "%d evals" % count


def _fmt_ck(sc, checkpoints, dim):
    """Raw reward of the recommendation at each checkpoint: 'raw@20d -195.0/@50d -190.0'."""
    parts = ["@%s %.1f" % (_ck_label(c, dim), sc['checkpoints'][str(c)]['raw'])
             for c in checkpoints]
    return ("| raw" + "/".join(parts)) if parts else ""


def _fmt_secs(s):
    s = int(s)
    if s < 3600:
        return "%dm %ds" % (s // 60, s % 60)
    return "%dh %dm" % (s // 3600, (s % 3600) // 60)


# ============================================================================
# OUTER OPTIMIZER AND BATCHES
# ============================================================================

def default_batch(encoding, outer_optimizer, workers, K):
    """OUTER_BATCH when it is None: TBPSA's population for this space, or
    workers // K for any other outer optimizer."""
    if OUTER_BATCH is not None:
        return int(OUTER_BATCH)
    if outer_optimizer == 'TBPSA':
        probe = O.ParametrizedTBPSA(naive=True)(parametrization=build_space(encoding),
                                                budget=1)
        return int(probe.popsize.llambda)
    return max(1, int(workers) // int(K))


def make_outer(space, outer_optimizer, budget, batch):
    """The outer optimizer, proposing `batch` configs at a time. Its sampler is seeded,
    so a rerun proposes the same configs (a resumed run continues the pickled state)."""
    space.random_state = np.random.RandomState(SEED)
    if outer_optimizer == 'TBPSA':
        return O.ParametrizedTBPSA(naive=True)(parametrization=space, budget=int(budget),
                                               num_workers=int(batch))
    return ng.optimizers.registry[outer_optimizer](parametrization=space,
                                                   budget=int(budget),
                                                   num_workers=int(batch))


def _cpus():
    return os.cpu_count() or 1


def _run_key(args, budget, checkpoints):
    """What has to match for --resume to continue a checkpoint: everything that changes
    how a config is scored. Checkpoints written before this tuner used optimize_snn's
    inner run have no key, so they are never resumed."""
    return dict(inner_run='optimize_snn.train_weights', inner_optimizer=args.inner_optimizer,
                inner_budget=int(budget), checkpoints=[int(c) for c in checkpoints],
                eval_episodes=int(args.eval_episodes), k_inner=int(args.k_inner),
                report_episodes=int(args.report_episodes),
                outer_optimizer=args.outer_optimizer, seed=SEED,
                trial_offset=TRIAL_OFFSET)


# ============================================================================
# THE TUNING RUN
# ============================================================================

def tune(cell, args):
    tag = _cell_tag(cell)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt_json = os.path.join(out_dir, "tune-%s.json" % tag)
    ckpt_opt = os.path.join(out_dir, "tune-%s.opt.pkl" % tag)

    keys = active_params(cell['encoding'])
    space = build_space(cell['encoding'])
    dim = cell_dim(cell)
    budget = inner_budget(dim, args)
    checkpoints = checkpoint_counts(dim, budget, args.checkpoints)
    K = int(args.k_inner)
    batch = args.batch or default_batch(cell['encoding'], args.outer_optimizer,
                                        args.workers or _cpus(), K)
    workers = args.workers or min(_cpus(), batch * K)
    run_key = _run_key(args, budget, checkpoints)

    print("\n=== tuning %s ===" % tag)
    print("    hidden=%s n_per_dim=%d t_step=%.0f -> genome dim=%d"
          % (cell['hidden_sizes'], cell['n_per_dim'], T_STEP_FIXED, dim))
    print("    tuning: %s" % ", ".join(keys))
    print("    outer=%s budget=%d in batches of %d | inner=%s budget=%d (%.3g x d) K=%d, "
          "%d episodes per evaluation | %d workers"
          % (args.outer_optimizer, args.outer_budget, batch, args.inner_optimizer,
             budget, budget / float(dim), K, args.eval_episodes, workers))
    if checkpoints:
        print("    checkpoints: recommendation also scored after %s evaluations"
              % ", ".join("%d (%.3g x d)" % (c, c / float(dim)) for c in checkpoints))

    # ---- outer optimizer, resumed if asked and available -------------------
    history, outer = [], None
    if args.resume and os.path.exists(ckpt_opt):
        with open(ckpt_json) as f:
            saved = json.load(f)
        if saved.get('run') != run_key:
            raise SystemExit(
                "%s was made with different scoring settings:\n  saved: %s\n  now:   %s\n"
                "Its scores can't be mixed with new ones. Rerun without --resume to "
                "start over (that overwrites it), or use another --out-dir."
                % (ckpt_json, saved.get('run'), run_key))
        try:
            outer = ng.optimizers.base.Optimizer.load(ckpt_opt)
            history = saved.get('history', [])
            print("    [resume] %d/%d evaluations already done"
                  % (len(history), args.outer_budget))
        except Exception as exc:
            print("    [resume] could not load %s (%s); starting fresh"
                  % (ckpt_opt, exc))
            outer, history = None, []
    if outer is None:
        outer = make_outer(space, args.outer_optimizer, args.outer_budget, batch)

    t_build = time.time()
    runner = Runner(cell, workers)
    try:
        # ---- batches: ask, score in parallel, tell, checkpoint ------------
        while len(history) < int(args.outer_budget):
            n = min(batch, int(args.outer_budget) - len(history))
            cands = [outer.ask() for _ in range(n)]
            cfgs = [resolve(dict(c.kwargs), cell['encoding']) for c in cands]
            t0 = time.time()
            scores = score_configs(runner, cfgs, SEED, args, budget, checkpoints)
            secs = time.time() - t0
            for cand, cfg, sc in zip(cands, cfgs, scores):
                outer.tell(cand, -sc['mean'])        # minimize -> maximize fitness
                i = len(history)
                history.append(dict(i=i, config=cfg, batch_seconds=secs, **sc))
                print("  [%3d/%d] %-36s -> %8.2f +/- %6.2f | raw %7.1f | goal %3.0f%% "
                      "| random %4.1f%% %s"
                      % (i + 1, args.outer_budget, _fmt(cfg, keys), sc['mean'], sc['std'],
                         sc['raw'], 100 * sc['goal_share'], 100 * sc['random_actions'],
                         _fmt_ck(sc, checkpoints, dim)))
            print("          batch of %d configs x %d runs: %s" % (n, K, _fmt_secs(secs)),
                  flush=True)
            _atomic_write(ckpt_json, dict(cell=cell, active=keys, run=run_key,
                                          history=history))
            try:
                outer.dump(ckpt_opt)
            except Exception:
                pass                                # resume degrades, run does not

        # ---- best config, then an honest held-out re-score -----------------
        best_entry = max(history, key=lambda h: h['mean'])
        best_cfg = best_entry['config']
        search_score = best_entry['mean']

        # paired baseline: the current settings, SAME held-out seeds, SAME network --
        # so the gap is purely the tuned parameters. Both in one batch.
        base_cfg = baseline_config()
        hold, base = score_configs(runner, [best_cfg, base_cfg], SEED + HOLDOUT_OFFSET,
                                   args, budget, checkpoints)
    finally:
        runner.close(args.keep_builds)

    result = dict(
        cell=cell, active=keys, genome_dim=dim,
        best_config=best_cfg,
        search_score=search_score,
        holdout_score=hold['mean'], holdout_std=hold['std'], holdout_vals=hold['vals'],
        holdout_raw=hold['raw'], holdout_raw_vals=hold['raw_vals'],
        holdout_goal_share=hold['goal_share'],
        holdout_random_actions=hold['random_actions'],
        holdout_checkpoints=hold['checkpoints'],
        baseline_config=base_cfg,
        baseline_score=base['mean'], baseline_std=base['std'], baseline_vals=base['vals'],
        baseline_raw=base['raw'], baseline_raw_vals=base['raw_vals'],
        baseline_goal_share=base['goal_share'],
        baseline_random_actions=base['random_actions'],
        baseline_checkpoints=base['checkpoints'],
        improvement=hold['mean'] - base['mean'],
        overfit_gap=search_score - hold['mean'],
        settings=dict(t_step=T_STEP_FIXED,
                      outer_optimizer=args.outer_optimizer,
                      outer_budget=args.outer_budget,
                      outer_batch=batch,
                      inner_optimizer=args.inner_optimizer,
                      inner_budget=budget,
                      inner_evals_per_dim=(None if args.inner_budget is not None
                                           else args.evals_per_dim),
                      checkpoints=checkpoints,
                      inner_eval_episodes=int(args.eval_episodes),
                      training_episodes='seeded, fresh at every evaluation',
                      k_inner=K,
                      tune_report_episodes=int(args.report_episodes),
                      weight_box='optimize_snn: [-w_scale, w_scale]',
                      param=hold['param'],
                      trial_offset=TRIAL_OFFSET,
                      graded=GRADED,
                      generator=cell.get('generator') or snn.SPIKE_GENERATOR,
                      seed=SEED, workers=workers,
                      wall_seconds=round(time.time() - t_build, 1)),
        history=history,
    )
    _atomic_write(os.path.join(out_dir, "result-%s.json" % tag), result)

    print("\n  best config   : %s" % _fmt(best_cfg, keys))
    print("  weight bound  : +/-%.1f (W_LOW / W_HIGH in optimize_snn.py, --bound in calibrate.py)"
          % best_cfg['w_scale'])
    print("  search score  : %.3f" % search_score)
    print("  holdout score : %.3f +/- %.3f | raw %.1f | goal %.0f%% | random %.1f%%"
          % (hold['mean'], hold['std'], hold['raw'], 100 * hold['goal_share'],
             100 * hold['random_actions']))
    print("  baseline      : %.3f +/- %.3f | raw %.1f | goal %.0f%% | random %.1f%%  (%s)"
          % (base['mean'], base['std'], base['raw'], 100 * base['goal_share'],
             100 * base['random_actions'], _fmt(base_cfg, keys)))
    print("  improvement   : %+.3f%s"
          % (hold['mean'] - base['mean'],
             "" if hold['mean'] > base['mean'] else "   (tuning did NOT help)"))
    print("  overfit gap   : %+.3f%s"
          % (search_score - hold['mean'],
             "   (large gap = tuned to the search seeds, not the encoding)"
             if search_score - hold['mean'] > abs(hold['std']) else ""))
    return result


# ============================================================================
# BUDGET CHECK: DO THE LAST EVALUATIONS ADD ANYTHING?
# ============================================================================

def _paired_p(a, b):
    """Two-sided paired t-test p of a vs b; None when it can't be computed."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    if len(d) < 2 or np.allclose(d, d[0]):
        return None
    try:
        from scipy import stats
        p = float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        return None
    return None if p != p else p


def budget_check(cell, args):
    """The current settings of one cell, K inner runs in parallel, scored at every
    checkpoint and at the full budget on the same report episodes."""
    tag = _cell_tag(cell)
    dim = cell_dim(cell)
    budget = inner_budget(dim, args)
    checkpoints = checkpoint_counts(dim, budget, args.checkpoints)
    K = int(args.k_inner)
    workers = args.workers or min(_cpus(), K)
    cfg = baseline_config()
    keys = active_params(cell['encoding'])

    print("\n=== budget check %s ===" % tag)
    print("    genome dim=%d | %s, budget %d (%.3g x d), K=%d runs on %d workers | "
          "current settings: %s"
          % (dim, args.inner_optimizer, budget, budget / float(dim), K, workers,
             _fmt(cfg, keys)))
    t0 = time.time()
    runner = Runner(cell, workers)
    try:
        sc = score_configs(runner, [cfg], SEED + BUDGET_CHECK_OFFSET, args, budget,
                           checkpoints)[0]
    finally:
        runner.close(args.keep_builds)
    secs = time.time() - t0

    rows = [(c, sc['checkpoints'][str(c)]) for c in checkpoints] + [(budget, sc)]
    final_raw = sc['raw_vals']
    print("\n    %8s %8s %10s %9s %6s %7s   %s"
          % ('evals', 'x d', 'shaped', 'raw', 'goal', 'random',
             'final minus this row: raw, runs final better, paired p'))
    table = []
    for count, r in rows:
        line = "    %8d %8.3g %10.2f %9.2f %5.0f%% %6.1f%%" % (
            count, count / float(dim), r['mean'], r['raw'], 100 * r['goal_share'],
            100 * r['random_actions'])
        entry = dict(evaluations=count, per_weight=count / float(dim), shaped=r['mean'],
                     raw=r['raw'], raw_vals=r['raw_vals'], goal_share=r['goal_share'],
                     random_actions=r['random_actions'])
        if count != budget:
            diff = float(np.mean(np.asarray(final_raw) - np.asarray(r['raw_vals'])))
            better = int(np.sum(np.asarray(final_raw) > np.asarray(r['raw_vals'])))
            p = _paired_p(final_raw, r['raw_vals'])
            line += "   %+8.2f  %d/%d  %s" % (diff, better, K,
                                             '-' if p is None else '%.3f' % p)
            entry.update(final_minus_this_raw=diff, runs_final_better=better, p=p)
        table.append(entry)
        print(line)
    print("\n    %d runs in %s. Raw reward of the recommendation on %d report episodes "
          "per run; the rows pair run by run." % (K, _fmt_secs(secs), args.report_episodes))
    if K < 8:
        print("    With K=%d the paired p is weak; --k-inner 8 or more costs no extra "
              "time while there are free cores." % K)
    out = dict(cell=cell, genome_dim=dim, config=cfg, inner_optimizer=args.inner_optimizer,
               budget=budget, k_inner=K, eval_episodes=int(args.eval_episodes),
               report_episodes=int(args.report_episodes), param=sc['param'],
               seed_block=SEED + BUDGET_CHECK_OFFSET, seconds=round(secs, 1), rows=table)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "budget-%s-%s.json" % (tag, args.inner_optimizer))
    _atomic_write(path, out)
    print("    -> %s" % path)
    return out


# ============================================================================
# COST ESTIMATE
# ============================================================================

def print_cost(cell, args):
    dim = cell_dim(cell)
    budget = inner_budget(dim, args)
    checkpoints = checkpoint_counts(dim, budget, args.checkpoints)
    K = int(args.k_inner)
    per_run = (budget * int(args.eval_episodes)
               + (len(checkpoints) + 1) * int(args.report_episodes))
    ep = EP_SECONDS.get(cell['environment'], 0.03)
    run_s = per_run * ep
    print("    genome dim %d -> inner budget %d (%.3g x d)%s"
          % (dim, budget, budget / float(dim),
             ("; checkpoints at " + ", ".join("%d" % c for c in checkpoints))
             if checkpoints else ""))
    print("    %d episodes per inner run, ~%s at %.3f s per episode"
          % (per_run, _fmt_secs(run_s), ep))
    if args.budget_check:
        workers = args.workers or min(_cpus(), K)
        waves = int(math.ceil(K / float(workers)))
        print("    budget check: %d runs on %d workers -> ~%s"
              % (K, workers, _fmt_secs(waves * run_s)))
        return
    batch = args.batch or default_batch(cell['encoding'], args.outer_optimizer,
                                        args.workers or _cpus(), K)
    workers = args.workers or min(_cpus(), batch * K)
    sizes = [batch] * (int(args.outer_budget) // batch)
    if int(args.outer_budget) % batch:
        sizes.append(int(args.outer_budget) % batch)
    sizes.append(2)                                   # holdout + baseline
    waves = sum(int(math.ceil(s * K / float(workers))) for s in sizes)
    n_runs = sum(sizes) * K
    print("    %d inner runs: %d batches of up to %d configs x K=%d on %d workers -> "
          "~%s (serial ~%s)"
          % (n_runs, len(sizes), batch, K, workers, _fmt_secs(waves * run_s),
             _fmt_secs(n_runs * run_s)))


# ============================================================================
# MAIN
# ============================================================================

def main():
    if GRADED != 'residual':
        raise ValueError("drive_encoder_fast() implements graded='residual' only")

    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--env', choices=list(snn.ENVIRONMENTS) + ['all'],
                    default='all')
    ap.add_argument('--encoding', choices=list(snn.ENCODINGS), default=None)
    ap.add_argument('--decoding', choices=list(snn.DECODINGS), default=None)
    ap.add_argument('--hidden', default=None,
                    help='override hidden sizes, e.g. 6 or 6x6')
    ap.add_argument('--n-per-dim', type=int, default=None)
    ap.add_argument('--generator', choices=list(snn.SPIKE_GENERATORS), default=None,
                    help='override the poprate spike generator of every cell')
    ap.add_argument('--outer-budget', type=int, default=OUTER_BUDGET)
    ap.add_argument('--outer-optimizer', default=OUTER_OPTIMIZER)
    ap.add_argument('--batch', type=int, default=None,
                    help='configs scored at once (default: see OUTER_BATCH)')
    ap.add_argument('--inner-optimizer', default=INNER_OPTIMIZER)
    ap.add_argument('--evals-per-dim', type=float, default=INNER_EVALS_PER_DIM,
                    help='inner budget = this x d (default %g)' % INNER_EVALS_PER_DIM)
    ap.add_argument('--inner-budget', type=int, default=None,
                    help='fixed inner budget for every cell instead of --evals-per-dim')
    ap.add_argument('--checkpoints', type=float, nargs='*',
                    default=list(CHECKPOINTS_PER_DIM),
                    help='also score the recommendation after these x d evaluations')
    ap.add_argument('--k-inner', type=int, default=K_INNER)
    ap.add_argument('--eval-episodes', type=int, default=INNER_EVAL_EPISODES,
                    help='episodes per inner evaluation (see INNER_EVAL_EPISODES)')
    ap.add_argument('--report-episodes', type=int, default=TUNE_REPORT_EPISODES)
    ap.add_argument('--workers', type=int, default=None,
                    help='worker processes (default: one per inner run of a batch, '
                         'at most the number of cores; 1 = serial)')
    ap.add_argument('--budget-check', action='store_true',
                    help='score the current settings at every checkpoint and stop')
    ap.add_argument('--keep-builds', action='store_true',
                    help="keep the workers' ANNarchy builds")
    ap.add_argument('--out-dir', default=OUT_DIR,
                    help='checkpoints and results (default %s)' % OUT_DIR)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='print budgets, batches and cost estimate, run nothing')
    ap.add_argument('--quick', action='store_true',
                    help='tiny budgets to check the plumbing end to end')
    args = ap.parse_args()

    if args.quick:
        args.outer_budget, args.inner_budget, args.k_inner = 4, 20, 1
        args.report_episodes, args.checkpoints = 3, [0.1]

    cells = [dict(c) for c in CELLS
             if args.env == 'all' or c['environment'] == args.env]
    if not cells:
        cells = [dict(environment=args.env, encoding='poprate', decoding='vote',
                      hidden_sizes=[6], n_per_dim=4, generator='regular')]
    for c in cells:
        if args.encoding:
            c['encoding'] = args.encoding
        if args.decoding:
            c['decoding'] = args.decoding
        if args.hidden:
            c['hidden_sizes'] = [int(h) for h in args.hidden.lower().split('x')]
        if args.n_per_dim:
            c['n_per_dim'] = args.n_per_dim
        if args.generator:
            c['generator'] = args.generator
    unique = {}
    for c in cells:                      # overrides can make two cells identical
        unique.setdefault(_cell_tag(c), c)
    cells = list(unique.values())

    for c in cells:
        print("\n%s" % ("-" * 70))
        print("cell: %s  hidden=%s n_per_dim=%d t_step=%.0f"
              % (_cell_tag(c), c['hidden_sizes'], c['n_per_dim'], T_STEP_FIXED))
        print("    tuning: %s" % ", ".join(active_params(c['encoding'])))
        print_cost(c, args)

    if args.dry_run:
        print("\n[dry-run] nothing executed. Drop --dry-run to start.")
        return

    if args.budget_check:
        for c in cells:
            budget_check(c, args)
        return

    results = {}
    for c in cells:
        results[_cell_tag(c)] = tune(c, args)

    summary = {k: dict(best_config=r['best_config'],
                       tuned=r['holdout_score'], tuned_raw=r['holdout_raw'],
                       tuned_random_actions=r['holdout_random_actions'],
                       baseline=r['baseline_score'], baseline_raw=r['baseline_raw'],
                       baseline_random_actions=r['baseline_random_actions'],
                       improvement=r['improvement'],
                       overfit_gap=r['overfit_gap'])
               for k, r in results.items()}
    _atomic_write(os.path.join(args.out_dir, 'summary.json'), summary)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
