"""
Nevergrad tuning of the ENCODING hyperparameters of the feedforward SNN.

Sibling of meta_optimize.py. That file tunes the inner optimizer's own
hyperparameters (CMA scale, DE crossover, ...). This file tunes the encoding
parameters that every optimizer shares.

TUNED (3 dimensions for every encoding):

    w_scale       weight scale factor. Nevergrad searches in [-1, 1] and the
                  vector is multiplied by w_scale before it reaches the
                  projections, so w_scale replaces the old W_LOW / W_HIGH.
    r_max, r_min  rate-coding bounds (Hz)         -- rate / poprate
    lat_window    latency for normalized 0.0 (ms) -- latency / poplatency
    lat_min       latency for normalized 1.0 (ms) -- latency / poplatency

The population encoders now use their full bounds, so nothing is masked out:
encode_poprate interpolates the active bin over [rate_min, rate_max] instead of
[0, rate_on], and encode_poplatency interpolates its spike time over
[lat_window, lat_min] instead of firing at a single fixed lat_on. Under the old
encoders r_min was dead for poprate and lat_window was dead for poplatency, and
both were dropped from the space. They are live now.

Note the two r_min semantics are NOT the same quantity. In encode_rate, r_min is
the floor for every input neuron, so nothing is ever silent. In encode_poprate,
rate_off=0.0 is the floor for inactive bins and rate_min is the floor for the
ACTIVE bin only. So poprate's r_min sets how strongly a just-barely-active bin
separates itself from its silent neighbours, which is a different mechanism from
rate's r_min even though both are tuned over the same range.

FIXED, set by hand in the CELLS table below:

    hidden_sizes  hidden layer width(s)
    n_per_dim     neurons per input dimension (population encodings)
    t_step        simulated ms per environment step -- pinned to 50.0

Because topology and n_per_dim are fixed, the ANNarchy network compiles exactly
ONCE per cell. The search space is 1-3 dimensions, so a modest outer budget with
heavy replication per configuration is the right trade: the score is noisy and
there is very little space to cover.

------------------------------------------------------------------------------
WHY THE WEIGHTS ARE NORMALIZED
------------------------------------------------------------------------------

Previously the inner optimizer searched directly in [-20, 20]. Nevergrad's
default step sizes are calibrated for a roughly unit-scale domain, so bounding
at +-20 quietly changes the effective step size of every optimizer, and changes
it by a different amount for each one. Searching a fixed [-1, 1] cube and
applying w_scale afterwards keeps the search geometry identical across
configurations, which means w_scale measures what you want it to measure -- the
useful synaptic magnitude -- and not the optimizer's step size.

This also makes the sigma/bounds sensitivity question cleaner: sigma now lives
in normalized units and w_scale is a separate, orthogonal knob.

------------------------------------------------------------------------------
ONE CAVEAT BEFORE THE OUTPUT GOES IN THE THESIS
------------------------------------------------------------------------------

Tuning uses ONE inner optimizer (--inner-optimizer, default DE). Whatever it
finds is "the encoding that suits DE". If those settings then become the fixed
settings for the whole optimizer grid, DE gets a home-field advantage in the
comparison that is your actual contribution. Tuning with an optimizer that is
not a contestant in the headline claim -- RandomSearch is a defensible, weak,
unbiased choice -- avoids that.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

    python tune_hyperparams.py --dry-run     # space + cost estimate only
    python tune_hyperparams.py --quick       # minutes, plumbing check
    python tune_hyperparams.py --env acrobot --encoding rate --decoding first
    python tune_hyperparams.py --resume      # continue from checkpoint
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import nevergrad as ng
from nevergrad.optimization import optimizerlib as O

import snn_feedforward as snn
from snn_feedforward import build_network, evaluate
from optimize_snn import n_params, flat_to_weights, report_fitness


# ============================================================================
# CONFIGURATION
# ============================================================================

T_STEP_FIXED = 50.0        # pinned, not tuned

# --- cells to tune. hidden_sizes and n_per_dim are set BY HAND here. --------
CELLS = [
    dict(environment='mountaincar', encoding='poprate', decoding='vote',
         hidden_sizes=[10], n_per_dim=5),
    #dict(environment='acrobot', encoding='poplatency', decoding='first',
        # hidden_sizes=[20], n_per_dim=5),
]

# --- outer loop -------------------------------------------------------------
OUTER_BUDGET    = 100       # configurations to evaluate (space is 1-3 dims)
OUTER_OPTIMIZER = 'NgIohTuned'  # noise-aware; 'NGOpt' is a reasonable alternative

# --- inner loop (cost driver) ----------------------------------------------
INNER_OPTIMIZER      = 'RandomSearch'   # see the caveat in the module docstring
INNER_BUDGET         = 2000    # weight evaluations per inner run
INNER_EVAL_EPISODES  = 10      # episodes averaged per inner objective call
K_INNER              = 1      # independent inner runs averaged to score a config
TUNE_REPORT_EPISODES = 100     # report episodes per inner run
INNER_SIGMA          = 0.3    # mutation step, in units of the [-1, 1] cube

# The new encoders take a `graded` mode. It is PINNED here rather than tuned,
# and passed explicitly so it lands in the results JSON instead of being
# inherited invisibly from the encoder's default. It is a categorical knob that
# changes what the population code represents, so if you want it in the search
# add `graded=ng.p.Choice([...])` to build_space and thread it through resolve()
# -- but be aware that makes the encoding comparison a best-of-modes comparison.
GRADED = 'residual'

# --- seeding ----------------------------------------------------------------
SEED            = 0
HOLDOUT_OFFSET  = 10_000      # held-out seed block for the honest re-score
REPORT_SEED_GAP = 500_000     # keeps report episodes disjoint from training ones

# --- search space bounds ----------------------------------------------------
# Log-scaled. The current hardcoded setting is effectively 20, so this brackets
# it by ~10x down and ~4x up. The FLOOR matters: Izhikevich RS has a rheobase
# around I=4, so a w_scale much below ~2 means one presynaptic spike cannot
# bring a postsynaptic neuron anywhere near threshold. Those configurations
# produce silent networks, land on the censored fitness floor, and are
# indistinguishable from each other -- the outer optimizer would spend its
# budget mapping a flat dead region. Confirm the real floor for your neuron
# parameters with a random-draw firing screen rather than trusting this number.
W_SCALE_LO, W_SCALE_HI = 1.0, 40.0
R_MAX_LO,   R_MAX_HI   = 50.0, 400.0  # log-scaled Hz
R_MIN_FRAC_HI          = 0.3          # r_min = frac * r_max, so r_min < r_max
LAT_WINDOW_LO          = 1.0
LAT_MIN_FRAC           = (0.02, 0.60)         # of lat_window
LAT_BURST_TAIL = 40.0

# --- housekeeping -----------------------------------------------------------
OUT_DIR = 'tuning'


# ============================================================================
# GLOBAL PATCHING
# ============================================================================
# snn_feedforward reads T_STEP, R_MAX, R_MIN, LAT_WINDOW, LAT_MIN and
# NEURONS_PER_DIM from module globals INSIDE function bodies, so rebinding the
# module attribute is enough -- for those functions.
#
# encode_poprate and encode_poplatency are the exception. They take n_per_dim,
# rate_min, rate_max, lat_min and lat_window as DEFAULT ARGUMENTS, which Python
# binds once at import. Setting snn.R_MAX changes what encode_rate does but NOT
# what encode_poprate does, and setting snn.NEURONS_PER_DIM changes the input
# layer width build_network creates but NOT the width the encoder emits -- a
# silent desync between the compiled net and the spike array. The new encoders
# made this MORE dangerous, not less: poprate now has two frozen rate defaults
# instead of one, and poplatency has two frozen latency defaults instead of one,
# so a shim that forgot to pass them would tune parameters that never move.
#
# So drive_encoder is replaced with a version that reads the globals at call
# time and passes them explicitly. Permanent fix if you would rather not shim:
# default those arguments to None and resolve inside the body.

_TUNED_GLOBALS = ('T_STEP', 'R_MAX', 'R_MIN', 'LAT_WINDOW', 'LAT_MIN',
                  'NEURONS_PER_DIM')

DEFAULTS = {name: getattr(snn, name) for name in _TUNED_GLOBALS}


def _drive_encoder_dynamic(encoding, encoder, obs, obs_low, obs_high,
                           t_offset=0.0, squash=None):
    """drive_encoder, but every encoder parameter is read at call time."""
    if encoding == 'rate':
        encoder.rates = snn.encode_rate(obs, obs_low, obs_high, squash)
    elif encoding == 'latency':
        encoder.spike_times = snn.encode_latency(obs, t_offset, obs_low,
                                                 obs_high, squash)
    elif encoding == 'poprate':
        encoder.rates = snn.encode_poprate(
            obs, obs_low, obs_high, squash,
            n_per_dim=snn.NEURONS_PER_DIM,
            rate_min=snn.R_MIN, rate_max=snn.R_MAX, rate_off=0.0,
            graded=GRADED)
    elif encoding == 'poplatency':
        encoder.spike_times = snn.encode_poplatency(
            obs, t_offset, obs_low, obs_high, squash,
            n_per_dim=snn.NEURONS_PER_DIM,
            lat_min=snn.LAT_MIN, lat_window=snn.LAT_WINDOW,
            graded=GRADED)
    else:
        raise ValueError("encoding must be one of %s" % (snn.ENCODINGS,))


@contextlib.contextmanager
def snn_config(cfg, n_per_dim):
    """Apply a resolved config to snn_feedforward, then restore."""
    saved = {name: getattr(snn, name) for name in _TUNED_GLOBALS}
    saved_drive = snn.drive_encoder
    try:
        snn.T_STEP          = float(T_STEP_FIXED)
        snn.R_MAX           = float(cfg['r_max'])
        snn.R_MIN           = float(cfg['r_min'])
        snn.LAT_WINDOW      = float(cfg['lat_window'])
        snn.LAT_MIN         = float(cfg['lat_min'])
        snn.NEURONS_PER_DIM = int(n_per_dim)
        snn.drive_encoder   = _drive_encoder_dynamic
        yield
    finally:
        for name, value in saved.items():
            setattr(snn, name, value)
        snn.drive_encoder = saved_drive


# ============================================================================
# SEARCH SPACE
# ============================================================================

def active_params(encoding):
    """Which hyperparameters affect behaviour for this encoding.

    With the new encoders nothing is masked out: poprate interpolates the active
    bin over [rate_min, rate_max], and poplatency interpolates its spike time
    over [lat_window, lat_min]. Under the old encoders poprate ignored its floor
    and poplatency ignored its window, and both were dropped here. Every
    encoding is 3-dimensional now.
    """
    p = ['w_scale']
    if encoding in ('rate', 'poprate'):
        p += ['r_max', 'r_min_frac']
    if encoding in ('latency', 'poplatency'):
        p += ['lat_window', 'lat_min_frac']
    return p


def build_space(encoding):
    """Instrumentation over the active parameters for this encoding."""
    full = dict(
        w_scale     =ng.p.Log(lower=W_SCALE_LO, upper=W_SCALE_HI),
        r_max       =ng.p.Log(lower=R_MAX_LO, upper=R_MAX_HI),
        r_min_frac  =ng.p.Scalar(lower=0.0, upper=R_MIN_FRAC_HI),
        lat_window  =ng.p.Scalar(lower=LAT_WINDOW_LO,
                                 upper=LAT_BURST_TAIL),
        lat_min_frac=ng.p.Scalar(lower=LAT_MIN_FRAC[0], upper=LAT_MIN_FRAC[1]),
    )
    return ng.p.Instrumentation(**{k: full[k] for k in active_params(encoding)})


def resolve(theta, encoding):
    """Raw search-space point -> concrete, always-valid config.

    r_min and lat_min are parametrized as FRACTIONS of r_max and lat_window, so
    r_min < r_max and lat_min < lat_window hold by construction. Sampled
    independently, the outer optimizer would regularly propose lat_min >
    lat_window, or a lat_window past the end of the simulated step -- ANNarchy
    drops out-of-window spikes silently, the same failure mode as the burst
    t_offset bug. A tuner that evaluates broken configurations without noticing
    produces a confidently wrong answer, so the constraint lives in the
    parametrization rather than in a post-hoc check.

    This matters more with the new encoders than it did before. Under the old
    encode_poplatency, lat_window was ignored entirely, so an inconsistent
    lat_min/lat_window pair was harmless. Now poplatency interpolates between
    them, so the ordering is load-bearing for two encodings instead of one.

    Inactive parameters fall back to the module defaults so snn_config() always
    receives a complete config.
    """
    g = DEFAULTS

    r_max = float(theta.get('r_max', g['R_MAX']))
    r_min = float(theta.get('r_min_frac', g['R_MIN'] / g['R_MAX'])) * r_max

    lat_window = float(theta.get('lat_window',
                                 min(g['LAT_WINDOW'], LAT_BURST_TAIL)))
    if 'lat_min_frac' in theta:
        lat_min = float(np.clip(theta['lat_min_frac'] * lat_window,
                                0.5, max(lat_window - 0.5, 0.5)))
    else:
        lat_min = float(min(g['LAT_MIN'], max(lat_window - 0.5, 0.5)))

    return dict(
        w_scale=float(theta.get('w_scale', 20.0)),
        r_max=r_max,
        r_min=r_min,
        lat_window=lat_window,
        lat_min=lat_min,
        t_step=float(T_STEP_FIXED),
    )


def baseline_config():
    """The settings currently hardcoded in snn_feedforward / optimize_snn.

    W_LOW=-20, W_HIGH=+20 on the raw weights is exactly w_scale=20 on a [-1, 1]
    search cube, so the baseline is directly comparable to a tuned w_scale.
    """
    from optimize_snn import W_HIGH
    g = DEFAULTS
    return dict(
        w_scale=float(W_HIGH),
        r_max=float(g['R_MAX']),
        r_min=float(g['R_MIN']),
        lat_window=float(g['LAT_WINDOW']),
        lat_min=float(g['LAT_MIN']),
        t_step=float(T_STEP_FIXED),
    )


# ============================================================================
# INNER RUN AND CONFIG SCORING
# ============================================================================

def run_inner(net, cfg, seed, inner_optimizer, inner_budget):
    """One weight optimization under a fixed encoding config.

    The optimizer searches a normalized [-1, 1] cube; cfg['w_scale'] multiplies
    the vector on its way into the projections. Returns the mean report fitness
    of the recommended weights.

    Seeding, kept disjoint on purpose:
      training episodes use seeds  seed*1000 + j
      report episodes   use seeds  (seed + REPORT_SEED_GAP)*1000 + k
    so the reported score is never measured on an episode the inner loop trained
    against. Without the gap, report_fitness would reuse exactly the training
    block and the optimism gap would be invisible.
    """
    shapes = net['shapes']
    dim = n_params(shapes)
    scale = float(cfg['w_scale'])

    rng = np.random.default_rng(seed)
    init = rng.uniform(-1.0, 1.0, size=dim)
    param = ng.p.Array(init=init).set_mutation(sigma=INNER_SIGMA)
    param.set_bounds(-1.0, 1.0)
    # ng.p.Array carries sigma=1.0 by default. On the old [-20, 20] domain that
    # was a 1/40-of-range step; on this [-1, 1] cube it spans the whole domain
    # and mutations bounce off the bounds constantly. Setting it explicitly is
    # what actually makes the search geometry identical across configurations --
    # normalizing the cube alone does not. INNER_SIGMA is now a free knob, so it
    # belongs in the sigma/bounds sensitivity check rather than being inherited
    # silently. Print param.sigma.value once to see what your nevergrad version
    # does here before assuming this line changed anything.
    
    param.random_state = np.random.RandomState(seed)   # pin the inner sampler
    np.random.seed(seed)                               # pin the silence stream

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
    return report_fitness(net, best, TUNE_REPORT_EPISODES,
                          episode_seed=seed + REPORT_SEED_GAP)['mean']


def score_config(net, cell, cfg, base_seed, K, inner_optimizer, inner_budget):
    """Mean final quality over K independent inner runs under this config."""
    with snn_config(cfg, cell['n_per_dim']):
        vals = np.array([run_inner(net, cfg, base_seed + k, inner_optimizer,
                                   int(inner_budget))
                         for k in range(int(K))], dtype=float)
    return (float(vals.mean()),
            float(vals.std(ddof=1)) if K > 1 else 0.0,
            vals)


# ============================================================================
# CHECKPOINTING
# ============================================================================

def _atomic_write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)          # single-writer, no partial file on crash


def _cell_tag(cell):
    return "%s-%s-%s" % (cell['environment'], cell['encoding'], cell['decoding'])


def _fmt(cfg, keys):
    """One-line view of just the parameters this encoding actually uses."""
    out = ['w=%.2f' % cfg['w_scale']]
    if 'r_max' in keys:
        out.append('rmax=%.0f' % cfg['r_max'])
    if 'r_min_frac' in keys:
        out.append('rmin=%.0f' % cfg['r_min'])
    if 'lat_window' in keys:
        out.append('latwin=%.1f' % cfg['lat_window'])
    if 'lat_min_frac' in keys:
        out.append('latmin=%.1f' % cfg['lat_min'])
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
    space.real_world = True
    space.hptuning=True
    space.function.deterministic = False
    # topology and n_per_dim are fixed for this cell, so this is the ONLY
    # compile the whole tuning run performs
    compile_dir = ("annarchy/tune-%s-%s-%s-%s-n%d"
                   % (cell['environment'], cell['encoding'], cell['decoding'],
                      "x".join(str(h) for h in cell['hidden_sizes']),
                      cell['n_per_dim']))
    with snn_config(baseline_config(), cell['n_per_dim']):
        t0 = time.time()
        net = build_network(cell['environment'], cell['hidden_sizes'],
                            cell['encoding'], cell['decoding'], compile_dir)
    dim = n_params(net['shapes'])

    print("\n=== tuning %s ===" % tag)
    print("    hidden=%s n_per_dim=%d t_step=%.0f -> genome dim=%d (%.1fs compile)"
          % (cell['hidden_sizes'], cell['n_per_dim'], T_STEP_FIXED, dim,
             time.time() - t0))
    print("    tuning: %s" % ", ".join(keys))
    print("    outer=%s budget=%d | inner=%s budget=%d K=%d"
          % (args.outer_optimizer, args.outer_budget, args.inner_optimizer,
             args.inner_budget, args.k_inner))

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
        mean, std, vals = score_config(net, cell, cfg, SEED, args.k_inner,
                                       args.inner_optimizer, args.inner_budget)
        outer.tell(cand, -mean)                 # minimize -> maximize fitness

        history.append(dict(i=i, config=cfg, mean=mean, std=std,
                            vals=vals.tolist(), seconds=time.time() - t0))
        print("  [%3d/%d] %-42s -> %8.2f +/- %6.2f  (%.0fs)"
              % (i + 1, args.outer_budget, _fmt(cfg, keys), mean, std,
                 time.time() - t0))

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
    hmean, hstd, hvals = score_config(net, cell, best_cfg, holdout_base,
                                      args.k_inner, args.inner_optimizer,
                                      args.inner_budget)

    # paired baseline: the current hardcoded settings, SAME held-out seeds, SAME
    # network -- so the gap is purely the encoding parameters
    base_cfg = baseline_config()
    bmean, bstd, bvals = score_config(net, cell, base_cfg, holdout_base,
                                      args.k_inner, args.inner_optimizer,
                                      args.inner_budget)

    result = dict(
        cell=cell, active=keys, genome_dim=dim,
        best_config=best_cfg,
        search_score=search_score,
        holdout_score=hmean, holdout_std=hstd, holdout_vals=hvals.tolist(),
        baseline_config=base_cfg,
        baseline_score=bmean, baseline_std=bstd, baseline_vals=bvals.tolist(),
        improvement=hmean - bmean,
        overfit_gap=search_score - hmean,
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
                      seed=SEED),
        history=history,
    )
    _atomic_write(os.path.join(OUT_DIR, "result-%s.json" % tag), result)

    print("\n  best config   : %s" % _fmt(best_cfg, keys))
    print("  effective weight range: [%.2f, %.2f]"
          % (-best_cfg['w_scale'], best_cfg['w_scale']))
    print("  search score  : %.3f" % search_score)
    print("  holdout score : %.3f +/- %.3f" % (hmean, hstd))
    print("  baseline      : %.3f +/- %.3f" % (bmean, bstd))
    print("  improvement   : %+.3f%s"
          % (hmean - bmean, "" if hmean > bmean else "   (tuning did NOT help)"))
    print("  overfit gap   : %+.3f%s"
          % (search_score - hmean,
             "   (large gap = tuned to the search seeds, not the encoding)"
             if search_score - hmean > abs(hstd) else ""))
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=list(snn.ENVIRONMENTS) + ['all'],
                    default='all')
    ap.add_argument('--encoding', choices=list(snn.ENCODINGS), default=None)
    ap.add_argument('--decoding', choices=list(snn.DECODINGS), default=None)
    ap.add_argument('--hidden', default=None,
                    help='override hidden sizes, e.g. 20 or 20x10')
    ap.add_argument('--n-per-dim', type=int, default=None)
    ap.add_argument('--outer-budget', type=int, default=OUTER_BUDGET)
    ap.add_argument('--outer-optimizer', default=OUTER_OPTIMIZER)
    ap.add_argument('--inner-optimizer', default=INNER_OPTIMIZER)
    ap.add_argument('--inner-budget', type=int, default=INNER_BUDGET)
    ap.add_argument('--k-inner', type=int, default=K_INNER)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='print space and cost estimate, run nothing')
    ap.add_argument('--quick', action='store_true',
                    help='tiny budgets to check the plumbing end to end')
    args = ap.parse_args()

    if args.quick:
        args.outer_budget, args.inner_budget, args.k_inner = 4, 20, 1
        globals()['TUNE_REPORT_EPISODES'] = 3

    cells = [dict(c) for c in CELLS
             if args.env == 'all' or c['environment'] == args.env]
    if not cells:
        cells = [dict(environment=args.env, encoding='rate', decoding='first',
                      hidden_sizes=[20], n_per_dim=5)]
    for c in cells:
        if args.encoding:
            c['encoding'] = args.encoding
        if args.decoding:
            c['decoding'] = args.decoding
        if args.hidden:
            c['hidden_sizes'] = [int(h) for h in args.hidden.lower().split('x')]
        if args.n_per_dim:
            c['n_per_dim'] = args.n_per_dim

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
                       tuned=r['holdout_score'],
                       baseline=r['baseline_score'],
                       improvement=r['improvement'],
                       overfit_gap=r['overfit_gap'])
               for k, r in results.items()}
    _atomic_write(os.path.join(OUT_DIR, 'summary.json'), summary)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
