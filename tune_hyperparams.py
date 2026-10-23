"""
Nevergrad tuning of the ENCODING hyperparameters of the feedforward SNN.

Sibling of meta_optimize.py. That file tunes the inner optimizer's own
hyperparameters (CMA scale, DE crossover, ...). This file tunes the encoding
parameters that every optimizer shares.

TUNED (3 dimensions for every encoding):

    w_scale       weight bound. Nevergrad searches in [-1, 1] and every
                  projection is multiplied by the same w_scale before it reaches
                  the network, so w_scale means exactly what W_LOW / W_HIGH mean
                  in optimize_snn.py and --bound means in calibrate.py.
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

Because topology, n_per_dim and generator are fixed, the ANNarchy network
compiles exactly ONCE per cell. A tuned w_scale belongs to its cell: it does not
transfer to other topologies or n_per_dim values, so tune each cell the grid runs.

------------------------------------------------------------------------------
WHY THE WEIGHTS ARE SEARCHED IN A [-1, 1] CUBE
------------------------------------------------------------------------------

The inner optimizer searches a fixed [-1, 1] cube and w_scale multiplies every
weight on its way into the network. With both bounds set, DE and RandomSearch
draw their first candidates uniformly over the whole box, so for them this is
the same search as optimize_snn.py's raw [-W_HIGH, W_HIGH]. Mutation-based
optimizers (OnePlusOne, CMA, TBPSA) take steps of INNER_SIGMA in cube units here,
i.e. INNER_SIGMA * w_scale in weight units, which is NOT what they do in
optimize_snn.py (an absolute step of 1). Keep that in mind if you tune with one.

------------------------------------------------------------------------------
ONE CAVEAT BEFORE THE OUTPUT GOES IN THE THESIS
------------------------------------------------------------------------------

Tuning uses ONE inner optimizer (--inner-optimizer, default DE). Whatever it
finds is "the encoding that suits DE". If those settings then become the fixed
settings for the whole optimizer grid, DE gets a home-field advantage in the
comparison that is your actual contribution. Tuning with an optimizer that is
not a contestant in the headline claim avoids that.

The inner run is also much shorter than a grid run (INNER_BUDGET evaluations
against the grid's 1,320 to 2,000), so treat the winner as a shortlist and
confirm it with a 5-seed calibrate.py test before adopting it.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

    python tune_hyperparams.py --dry-run     # space + cost estimate only
    python tune_hyperparams.py --quick       # minutes, plumbing check
    python tune_hyperparams.py --env mountaincar --encoding poprate --decoding vote
    python tune_hyperparams.py --env mountaincar --generator poisson
    python tune_hyperparams.py --resume      # continue from checkpoint

Needs calibrate.py next to it: the report episodes go through
calibrate.scored_episode(), which returns evaluate()'s exact fitness and also
counts the steps whose output was silent or tied.
"""

import argparse
import json
import os
import time
import contextlib

import numpy as np
import nevergrad as ng
import gymnasium as gym
from nevergrad.optimization import optimizerlib as O

import snn_feedforward as snn
from snn_feedforward import build_network, evaluate
from optimize_snn import n_params, flat_to_weights, W_HIGH
from calibrate import scored_episode


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
OUTER_BUDGET    = 50       # configurations to evaluate (space is 3 dims)
OUTER_OPTIMIZER = 'TBPSA'  # noise-aware; 'NGOpt' is a reasonable alternative

# --- inner loop (cost driver) ----------------------------------------------
INNER_OPTIMIZER      = 'RandomSearch'   # see the caveat in the module docstring
INNER_BUDGET         = 300    # weight evaluations per inner run
# Episodes averaged per inner objective call. Every candidate of one inner run is
# scored on the SAME episodes (common random numbers), so with 1 episode the whole
# inner run would be tuned to a single start position. 10 matches EVAL_EPISODES
# in optimize_snn.py. If the run is too slow, lower OUTER_BUDGET or K_INNER
# instead of this.
INNER_EVAL_EPISODES  = 10
K_INNER              = 4      # independent inner runs averaged to score a config
TUNE_REPORT_EPISODES = 100     # report episodes per inner run
INNER_SIGMA          = 0.3    # mutation step, in units of the [-1, 1] cube

# The encoders' `graded` mode. evaluate() encodes through drive_encoder_fast(),
# which implements 'residual' only, so it is pinned and recorded in the results.
GRADED = 'residual'

# --- seeding ----------------------------------------------------------------
SEED            = 0
HOLDOUT_OFFSET  = 10_000      # held-out seed block for the honest re-score
REPORT_SEED_GAP = 500_000     # keeps report episodes disjoint from training ones

# --- search space bounds ----------------------------------------------------
# w_scale, log-scaled. At the low end the network goes silent, at the high end
# the outputs saturate into ties; +-20, +-40 and +-60 were tested in D5.
W_SCALE_LO, W_SCALE_HI = 10.0, 50.0
R_MAX_HI           = 400.0  # log-scaled Hz
R_MIN_LO, R_MIN_HI = 20.0, 150.0          
R_SPAN_LO          = 50.0                 # Hz, at least 3 spike counts across a bin
R_SPAN_HI          = R_MAX_HI - R_MIN_HI

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

# --- housekeeping -----------------------------------------------------------
OUT_DIR = 'tuning'


# ============================================================================
# GLOBAL PATCHING
# ============================================================================
# evaluate() encodes through drive_encoder_fast(), which reads R_MIN, R_MAX,
# LAT_MIN, LAT_WINDOW and T_STEP from snn_feedforward's module globals every
# time it runs, and reads n_per_dim from the compiled input layer. So setting the
# module globals is enough; the encoder functions with frozen default arguments
# (encode_poprate, encode_poplatency) are not on that path.

_TUNED_GLOBALS = ('T_STEP', 'R_MAX', 'R_MIN', 'LAT_WINDOW', 'LAT_MIN')

DEFAULTS = {name: getattr(snn, name) for name in _TUNED_GLOBALS}


@contextlib.contextmanager
def snn_config(cfg):
    """Apply a resolved config to snn_feedforward, then restore."""
    saved = {name: getattr(snn, name) for name in _TUNED_GLOBALS}
    try:
        snn.T_STEP     = float(T_STEP_FIXED)
        snn.R_MAX      = float(cfg['r_max'])
        snn.R_MIN      = float(cfg['r_min'])
        snn.LAT_WINDOW = float(cfg['lat_window'])
        snn.LAT_MIN    = float(cfg['lat_min'])
        yield
    finally:
        for name, value in saved.items():
            setattr(snn, name, value)


# ============================================================================
# SEARCH SPACE
# ============================================================================

def active_params(encoding):
    """Which hyperparameters affect behaviour for this encoding."""
    p = ['w_scale']
    if encoding == 'poprate':
        p += ['r_min', 'r_span']
    if encoding == 'poplatency':
        p += ['lat_min', 'lat_span']
    return p


def build_space(encoding):
    """Instrumentation over the active parameters for this encoding."""
    full = dict(
        w_scale   =ng.p.Log(lower=W_SCALE_LO, upper=W_SCALE_HI),
        r_min     =ng.p.Log(lower=R_MIN_LO, upper=R_MIN_HI),
        r_span    =ng.p.Scalar(lower=R_SPAN_LO, upper=R_SPAN_HI),
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
    if 'r_min' in theta:
        r_min = float(theta['r_min'])
        r_max = r_min + float(theta['r_span'])
    else:
        r_min, r_max = float(g['R_MIN']), float(g['R_MAX'])
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
# INNER RUN AND CONFIG SCORING
# ============================================================================

_SIGMA_CHECKED = []


def _check_sigma(param):
    """Confirm sigma actually took, once per run.

    Silencing the bounds warning by reordering is only correct if the sigma we
    asked for is the sigma the parameter ends up with. If set_mutation had no
    effect, the warning would be gone while the underlying problem -- mutations
    spanning the whole cube and bouncing off the walls -- would remain. So read
    it back rather than assuming.
    """
    if _SIGMA_CHECKED:
        return
    try:
        sigma = float(np.asarray(param.sigma.value).ravel()[0])
    except Exception as exc:
        print("[sigma] could not read back sigma (%s)" % exc)
        _SIGMA_CHECKED.append(None)
        return
    ratio = 2.0 / sigma
    print("[sigma] requested=%.3f actual=%.3f -> bounds are %.1f sigma apart%s"
          % (INNER_SIGMA, sigma, ratio,
             "" if ratio >= 3.0 else "   <-- still below nevergrad's target of 3"))
    _SIGMA_CHECKED.append(sigma)


def report(net, weights, n_episodes, episode_seed):
    """report_fitness() from optimize_snn.py, plus the decoder's random actions.

    Same episode seeds (episode_seed*1000 + k) and the same random draws as
    report_fitness(), because calibrate.scored_episode() returns evaluate()'s
    exact fitness. A random action is a silent window, or a tie under vote (a
    first-spike tie goes to the lowest index, so it is not random).
    """
    base = int(episode_seed) * 1000
    rows = [scored_episode(snn, gym, net, weights, base + k)
            for k in range(int(n_episodes))]
    raw = np.array([r[0] for r in rows], dtype=float)
    shaped = np.array([r[1] for r in rows], dtype=float)
    silent = np.array([r[2] / r[4] for r in rows], dtype=float)
    tied = np.array([r[3] / r[4] for r in rows], dtype=float)
    rand = silent + tied if net['decoding'] == 'vote' else silent
    return dict(mean=float(shaped.mean()), raw=float(raw.mean()),
                random_actions=float(rand.mean()), silent=float(silent.mean()),
                tied=float(tied.mean()))


def run_inner(net, cfg, seed, inner_optimizer, inner_budget):
    """One weight optimization under a fixed encoding config.

    The optimizer searches a normalized [-1, 1] cube; every weight is then
    multiplied by cfg['w_scale'] on the way into the network. Returns the report
    of the recommended weights (shaped mean, raw mean, random-action shares).

    Seeding, kept disjoint on purpose:
      training episodes use seeds  seed*1000 + j
      report episodes   use seeds  (seed + REPORT_SEED_GAP)*1000 + k
    so the reported score is never measured on an episode the inner loop trained
    against. Without the gap, the report would reuse exactly the training
    block and the optimism gap would be invisible.
    """
    shapes = net['shapes']
    dim = n_params(shapes)
    scale = float(cfg['w_scale'])

    rng = np.random.default_rng(seed)
    init = rng.uniform(-1.0, 1.0, size=dim)

    # ORDER MATTERS. ng.p.Array(init=...) starts at sigma=1.0, and set_bounds
    # runs nevergrad's "at least 3 sigma between bounds" check immediately, at
    # whatever sigma is set at that moment. Bounding first and mutating second
    # checks 2.0/1.0 = 2.0 sigma and warns; setting sigma first checks
    # 2.0/0.3 = 6.7 sigma and does not.
    param = ng.p.Array(init=init).set_mutation(sigma=INNER_SIGMA)
    param.set_bounds(-1.0, 1.0)
    _check_sigma(param)
    param.random_state = np.random.RandomState(seed)   # pin the inner sampler
    np.random.seed(seed)                               # pin the decoder's draws

    if inner_optimizer == 'NgIohTuned':
        param.real_world = True
        param.neural = True
        param.function.deterministic = False

    opt_cls = ng.optimizers.registry[inner_optimizer]
    optimizer = opt_cls(parametrization=param, budget=int(inner_budget))

    train_seeds = [seed * 1000 + j for j in range(INNER_EVAL_EPISODES)]

    def objective(flat):
        weights = flat_to_weights(np.asarray(flat) * scale, shapes)
        # common random numbers within this inner run: every candidate is judged
        # on the same episodes, so differences are signal rather than luck
        vals = [evaluate(net, weights, episode_seed=s) for s in train_seeds]
        return -float(np.mean(vals))

    rec = optimizer.minimize(objective)
    best = flat_to_weights(np.asarray(rec.value) * scale, shapes)
    return report(net, best, TUNE_REPORT_EPISODES, seed + REPORT_SEED_GAP)


def score_config(net, cfg, base_seed, K, inner_optimizer, inner_budget):
    """Mean final quality over K independent inner runs under this config."""
    with snn_config(cfg):
        reps = [run_inner(net, cfg, base_seed + k, inner_optimizer, int(inner_budget))
                for k in range(int(K))]
    vals = np.array([r['mean'] for r in reps], dtype=float)
    return dict(mean=float(vals.mean()),
                std=float(vals.std(ddof=1)) if K > 1 else 0.0,
                vals=vals.tolist(),
                raw=float(np.mean([r['raw'] for r in reps])),
                random_actions=float(np.mean([r['random_actions'] for r in reps])),
                silent=float(np.mean([r['silent'] for r in reps])),
                tied=float(np.mean([r['tied'] for r in reps])))


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
    if 'r_min' in keys:
        out.append('rmin=%.0f' % cfg['r_min'])
        out.append('rmax=%.0f' % cfg['r_max'])
    if 'lat_min' in keys:
        out.append('latmin=%.1f' % cfg['lat_min'])
        out.append('latwin=%.1f' % cfg['lat_window'])
    return ' '.join(out)


# ============================================================================
# THE TUNING RUN
# ============================================================================

def tune(cell, args):
    tag = _cell_tag(cell)
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_json = os.path.join(OUT_DIR, "tune-%s.json" % tag)
    ckpt_opt = os.path.join(OUT_DIR, "tune-%s.opt.pkl" % tag)

    keys = active_params(cell['encoding'])
    space = build_space(cell['encoding'])

    # topology, n_per_dim and generator are fixed for this cell, so this is the
    # ONLY compile the whole tuning run performs
    with snn_config(baseline_config()):
        t0 = time.time()
        net = build_network(cell['environment'], cell['hidden_sizes'],
                            cell['encoding'], cell['decoding'], "annarchy/tune-" + tag,
                            n_per_dim=cell['n_per_dim'],
                            spike_generator=cell.get('generator'))
    dim = n_params(net['shapes'])
    n_obs = int(net['cfg']['n_entrada'])
    if net['encoder'].size != n_obs * cell['n_per_dim']:
        raise RuntimeError('asked for %d neurons per observation, the network has %d'
                           % (cell['n_per_dim'], net['encoder'].size // n_obs))

    print("\n=== tuning %s ===" % tag)
    print("    hidden=%s n_per_dim=%d t_step=%.0f -> genome dim=%d (%.1fs compile)"
          % (cell['hidden_sizes'], cell['n_per_dim'], T_STEP_FIXED, dim,
             time.time() - t0))
    print("    tuning: %s" % ", ".join(keys))
    print("    outer=%s budget=%d | inner=%s budget=%d K=%d, %d episodes per evaluation"
          % (args.outer_optimizer, args.outer_budget, args.inner_optimizer,
             args.inner_budget, args.k_inner, INNER_EVAL_EPISODES))

    # ---- outer optimizer, resumed if asked and available -------------------
    history, start_i, outer = [], 0, None
    if args.resume and os.path.exists(ckpt_opt):
        try:
            outer = ng.optimizers.base.Optimizer.load(ckpt_opt)
            with open(ckpt_json) as f:
                history = json.load(f).get('history', [])
            start_i = len(history)
            print("    [resume] %d/%d evaluations already done"
                  % (start_i, args.outer_budget))
        except Exception as exc:
            print("    [resume] could not load %s (%s); starting fresh"
                  % (ckpt_opt, exc))
            outer, history, start_i = None, [], 0
    if outer is None:
        if args.outer_optimizer == 'TBPSA':
            outer = O.ParametrizedTBPSA(naive=True)(
                parametrization=space, budget=int(args.outer_budget))
        else:
            outer = ng.optimizers.registry[args.outer_optimizer](
                parametrization=space, budget=int(args.outer_budget))

    # ---- ask / tell loop, so we can checkpoint between evaluations ---------
    for i in range(start_i, int(args.outer_budget)):
        cand = outer.ask()
        cfg = resolve(dict(cand.kwargs), cell['encoding'])

        t0 = time.time()
        sc = score_config(net, cfg, SEED, args.k_inner,
                          args.inner_optimizer, args.inner_budget)
        outer.tell(cand, -sc['mean'])           # minimize -> maximize fitness

        history.append(dict(i=i, config=cfg, seconds=time.time() - t0, **sc))
        print("  [%3d/%d] %-36s -> %8.2f +/- %6.2f | raw %7.1f | random %4.1f%%  (%.0fs)"
              % (i + 1, args.outer_budget, _fmt(cfg, keys), sc['mean'], sc['std'],
                 sc['raw'], 100 * sc['random_actions'], time.time() - t0))

        _atomic_write(ckpt_json, dict(cell=cell, active=keys, history=history))
        try:
            outer.dump(ckpt_opt)
        except Exception:
            pass                                # resume degrades, run does not

    # ---- best config, then an honest held-out re-score ---------------------
    best_entry = max(history, key=lambda h: h['mean'])
    best_cfg = best_entry['config']
    search_score = best_entry['mean']

    holdout_base = SEED + HOLDOUT_OFFSET
    hold = score_config(net, best_cfg, holdout_base, args.k_inner,
                        args.inner_optimizer, args.inner_budget)

    # paired baseline: the current settings, SAME held-out seeds, SAME network --
    # so the gap is purely the tuned parameters
    base_cfg = baseline_config()
    base = score_config(net, base_cfg, holdout_base, args.k_inner,
                        args.inner_optimizer, args.inner_budget)

    result = dict(
        cell=cell, active=keys, genome_dim=dim,
        best_config=best_cfg,
        search_score=search_score,
        holdout_score=hold['mean'], holdout_std=hold['std'], holdout_vals=hold['vals'],
        holdout_raw=hold['raw'], holdout_random_actions=hold['random_actions'],
        baseline_config=base_cfg,
        baseline_score=base['mean'], baseline_std=base['std'], baseline_vals=base['vals'],
        baseline_raw=base['raw'], baseline_random_actions=base['random_actions'],
        improvement=hold['mean'] - base['mean'],
        overfit_gap=search_score - hold['mean'],
        settings=dict(t_step=T_STEP_FIXED,
                      outer_optimizer=args.outer_optimizer,
                      outer_budget=args.outer_budget,
                      inner_optimizer=args.inner_optimizer,
                      inner_budget=args.inner_budget,
                      inner_eval_episodes=INNER_EVAL_EPISODES,
                      k_inner=args.k_inner,
                      tune_report_episodes=TUNE_REPORT_EPISODES,
                      weight_search_cube=[-1.0, 1.0],
                      inner_sigma=INNER_SIGMA,
                      graded=GRADED,
                      generator=cell.get('generator') or snn.SPIKE_GENERATOR,
                      seed=SEED),
        history=history,
    )
    _atomic_write(os.path.join(OUT_DIR, "result-%s.json" % tag), result)

    print("\n  best config   : %s" % _fmt(best_cfg, keys))
    print("  weight bound  : +/-%.1f (W_LOW / W_HIGH in optimize_snn.py, --bound in calibrate.py)"
          % best_cfg['w_scale'])
    print("  search score  : %.3f" % search_score)
    print("  holdout score : %.3f +/- %.3f | raw %.1f | random %.1f%%"
          % (hold['mean'], hold['std'], hold['raw'], 100 * hold['random_actions']))
    print("  baseline      : %.3f +/- %.3f | raw %.1f | random %.1f%%  (%s)"
          % (base['mean'], base['std'], base['raw'], 100 * base['random_actions'],
             _fmt(base_cfg, keys)))
    print("  improvement   : %+.3f%s"
          % (hold['mean'] - base['mean'],
             "" if hold['mean'] > base['mean'] else "   (tuning did NOT help)"))
    print("  overfit gap   : %+.3f%s"
          % (search_score - hold['mean'],
             "   (large gap = tuned to the search seeds, not the encoding)"
             if search_score - hold['mean'] > abs(hold['std']) else ""))
    return result


# ============================================================================
# COST ESTIMATE
# ============================================================================

def print_cost(args):
    per_cfg = args.k_inner * (args.inner_budget * INNER_EVAL_EPISODES
                              + TUNE_REPORT_EPISODES)
    total = per_cfg * (args.outer_budget + 2)   # +2 for holdout and baseline
    print("    episodes: %d per config x %d configs (+holdout+baseline) = %d"
          % (per_cfg, args.outer_budget, total))
    print("    at 0.10 s/ep -> ~%.1f h | at 0.50 s/ep -> ~%.1f h"
          % (total * 0.10 / 3600.0, total * 0.50 / 3600.0))


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
    ap.add_argument('--inner-optimizer', default=INNER_OPTIMIZER)
    ap.add_argument('--inner-budget', type=int, default=INNER_BUDGET)
    ap.add_argument('--k-inner', type=int, default=K_INNER)
    ap.add_argument('--eval-episodes', type=int, default=INNER_EVAL_EPISODES,
                    help='episodes per inner evaluation (see INNER_EVAL_EPISODES)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='print space and cost estimate, run nothing')
    ap.add_argument('--quick', action='store_true',
                    help='tiny budgets to check the plumbing end to end')
    args = ap.parse_args()

    globals()['INNER_EVAL_EPISODES'] = int(args.eval_episodes)
    if args.quick:
        args.outer_budget, args.inner_budget, args.k_inner = 4, 20, 1
        globals()['TUNE_REPORT_EPISODES'] = 3

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
        print_cost(args)

    if args.dry_run:
        print("\n[dry-run] nothing executed. Drop --dry-run to start.")
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
    _atomic_write(os.path.join(OUT_DIR, 'summary.json'), summary)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
