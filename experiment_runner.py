"""
experiment_runner.py -- one control point for the whole SNN study.

Configures and drives the modules without reimplementing them:
  snn_feedforward.py  the network (encoders / decoders / topology / genome sizing)
  optimize_snn.py     plain comparison: many optimizers, one config, fixed budget

It runs the Cartesian grid
    environment x topology x encoding x decoding x n_per_dim x trial x optimizer
calling optimize_snn.optimize() once per TASK. A task is one cell, one trial and ONE
optimizer, so the optimizers of a cell run side by side and the longest task is a
single optimizer's run. Every optimizer of a trial still starts from the same weights,
is scored on the same report block and gets the same budget: optimize() takes those
from the trial number, not from the other optimizers.

Thin orchestrator on purpose: the science lives in the modules; this only sequences
calls, scales budget, checkpoints, and summarizes. It depends only on their stable
entry points (optimize(), weight_shapes(), n_params()), so your local edits to the
modules are respected.

DEFINE the experiment in the CONFIG block, or in a JSON file given with --config (its
keys are the CONFIG names in lower case, e.g. {"optimizers": ["TwoPointsDE"],
"n_per_dim": [2, 4]}). CONTROL the run with flags:
  python experiment_runner.py --dry-run     # list cells + cost estimate, run nothing
  python experiment_runner.py               # run the grid (parallel, resumable)
  python experiment_runner.py --workers 8   # N parallel workers (default os.cpu_count())
  python experiment_runner.py --workers 1   # one task at a time (for debugging)
  python experiment_runner.py --settings grid_settings.json   # per-cell tuned settings
  python experiment_runner.py --config grid_config.json       # override the CONFIG block
  python experiment_runner.py --optimizers TwoPointsDE,NoisyRL3   # override OPTIMIZERS
  python experiment_runner.py --force       # ignore checkpoint; recompute every task
  python experiment_runner.py --summarize   # print summary from an existing results file
  python experiment_runner.py --out FILE    # results file (default below)

Parallelism: every task runs in its own fresh process ('spawn'), forced SINGLE-THREADED
(OMP_NUM_THREADS=1) so the parallelism is across tasks, not nested OpenMP threads inside
each tiny sim. A task that crashes (even a segfault) takes only itself down: it is
retried once, then recorded as failed, and the other tasks carry on. Each task compiles
its network in its own folder under annarchy/tasks/, removed when the task ends.

Resumability: only the main process writes the results file, after every task. A re-run
skips every (cell, trial, optimizer) already in it with the same budget and settings
(unless --force), so an interrupted sweep continues where it stopped, and optimizers
added to OPTIMIZERS later run into the same file. Per cell, the results hold one entry
per optimizer; the cell's status is 'ok' once every optimizer in OPTIMIZERS is there.
"""

import os
import json
import time
import shutil
import pickle
import signal
import argparse
import traceback

# Cap native (OpenMP / BLAS) threads to 1 PER PROCESS, before anything imports
# ANNarchy. Parallelism here is across tasks (worker processes); letting each
# worker's ANNarchy also fan out over all cores via OpenMP would oversubscribe
# (workers x cores threads) and run slower than serial. setdefault so an explicit
# external OMP_NUM_THREADS still wins. Runs on every import, including the fresh
# interpreter each 'spawn' worker starts, so it is set before ANNarchy loads.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')


# ===================== CONFIG: define the experiment here =====================
ENVIRONMENTS = ['mountaincar', 'acrobot']
ENCODINGS    = ['poprate']
TOPOLOGIES   = [[6]]          # genome scales with input width too
N_PER_DIM    = [4] # neurons per observation dim, swept for the population
OPTIMIZERS   = ['TwoPointsDE'] # any subset optimize_snn understands
TRIALS       = 5                                # repeats per cell (different trial seed)
WORKERS      = 0                                # parallel worker processes; 0 = auto
                                                # (os.cpu_count()), 1 = serial

# decoding coverage: 'full' = every encoding x DECODERS_ALL; 'natural' =
# latency/poplatency -> first, rate-like -> vote.
DECODING_MODE = 'natural'                          # 'full' | 'natural'
DECODERS_ALL  = ('first',)

# budget: 'fixed' = same BUDGET for every cell (the study's fixed-budget design);
# 'per_dim' = EVALS_PER_DIM * genome_dim, clamped to [BUDGET_MIN, BUDGET_MAX]
# (None = no bound on that side). 'per_dim' gives every topology the same number of
# evaluations per weight.
BUDGET_MODE   = 'fixed'                          # 'fixed' | 'per_dim'
BUDGET        = 7000
EVALS_PER_DIM = 1
BUDGET_MIN, BUDGET_MAX = 200, 4000

# cost-estimate only (measure your own machine); does not affect the actual run.
EP_SECONDS      = {'mountaincar': 0.018, 'acrobot': 0.065}
COMPILE_SECONDS = 6.0                            # rough per-task ANNarchy compile
REPORT_EP_EST   = 100                             # optimize_snn's REPORT_EPISODES (est)

RESULTS_PATH = 'experiment_results.json'
SAVE_WEIGHTS = True                            # also pickle each optimizer's network

# Per-cell weight bound, rates, latencies and spike generator, from
# make_grid_settings.py (which reads tune_hyperparams.py's results). None = the module
# defaults for every cell: W_LOW / W_HIGH in optimize_snn.py and R_MAX / R_MIN /
# LAT_MIN / LAT_WINDOW / SPIKE_GENERATOR in snn_feedforward.py. A cell missing from the
# file also runs on the module defaults. --settings FILE overrides this. Each result
# records what it ran with.
SETTINGS_FILE = None
# ============================================================================

# --config FILE: JSON keys -> the CONFIG names above
CONFIG_KEYS = dict(environments='ENVIRONMENTS', encodings='ENCODINGS',
                   topologies='TOPOLOGIES', n_per_dim='N_PER_DIM',
                   optimizers='OPTIMIZERS', trials='TRIALS', workers='WORKERS',
                   decoding_mode='DECODING_MODE', decoders_all='DECODERS_ALL',
                   budget_mode='BUDGET_MODE', budget='BUDGET',
                   evals_per_dim='EVALS_PER_DIM', budget_min='BUDGET_MIN',
                   budget_max='BUDGET_MAX', ep_seconds='EP_SECONDS',
                   results_path='RESULTS_PATH', save_weights='SAVE_WEIGHTS',
                   settings_file='SETTINGS_FILE')
TASK_ATTEMPTS = 2          # a failed or crashed task is run at most this many times
TASK_DIR = os.path.join('annarchy', 'tasks')    # per-task compile folders


# ---------------- grid helpers (pure python) ----------------
def topo_str(hidden):
    return 'x'.join(str(h) for h in hidden) or 'none'


def decoders_for(encoding):
    if DECODING_MODE == 'natural':
        return ['first'] if encoding in ('latency', 'poplatency') else ['vote']
    return list(DECODERS_ALL)

def n_str(n):
    return '-' if n is None else str(n)

def iter_cells():
    """Yield (env, hidden, encoding, decoding, n_per_dim, trial) for the whole grid."""
    for env in ENVIRONMENTS:
        for hidden in TOPOLOGIES:
            for enc in ENCODINGS:
                for dec in decoders_for(enc):
                    for n in N_PER_DIM:
                        for trial in range(TRIALS):
                            yield env, hidden, enc, dec, n, trial


def iter_tasks():
    """Yield (env, hidden, encoding, decoding, n_per_dim, trial, optimizer)."""
    for cell in iter_cells():
        for name in OPTIMIZERS:
            yield cell + (name,)


def cell_key(env, hidden, enc, dec, n, trial):
    return f"{env}|{topo_str(hidden)}|{enc}|{dec}|n{n_str(n)}|t{trial}"


def settings_key(env, hidden, enc, dec, n):
    """cell_key() without the trial: the key make_grid_settings.py writes."""
    return f"{env}|{topo_str(hidden)}|{enc}|{dec}|n{n_str(n)}"


def load_settings(path):
    """{settings_key: settings} from a make_grid_settings.py file; {} for None."""
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)['cells']


def settings_source(settings, env, hidden, enc, dec, n):
    s = settings.get(settings_key(env, hidden, enc, dec, n))
    return s['source'] if s else 'module defaults'


def apply_config(path):
    """Override the CONFIG block with a JSON file's values (keys in CONFIG_KEYS)."""
    with open(path) as f:
        cfg = json.load(f)
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        raise SystemExit("%s: unknown keys %s (known: %s)"
                         % (path, unknown, ", ".join(sorted(CONFIG_KEYS))))
    g = globals()
    for k, v in cfg.items():
        g[CONFIG_KEYS[k]] = tuple(v) if k == 'decoders_all' else v


def genome_dim(env, hidden, encoding, n_per_dim=None):
    """Genome size for a cell, via the real weight_shapes/n_params."""
    from snn_feedforward import weight_shapes
    from optimize_snn import n_params
    return int(n_params(weight_shapes(env, hidden, encoding, n_per_dim)))


def budget_for(env, hidden, encoding, n_per_dim=None):
    if BUDGET_MODE == 'per_dim':
        d = genome_dim(env, hidden, encoding, n_per_dim)
        b = int(round(EVALS_PER_DIM * d))
        if BUDGET_MIN is not None:
            b = max(int(BUDGET_MIN), b)
        if BUDGET_MAX is not None:
            b = min(int(BUDGET_MAX), b)
        return b
    return int(BUDGET)


def _fmt_secs(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


# ---------------- results store ----------------
def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        r.setdefault('cells', {}); r.setdefault('meta', {})
        return r
    return dict(config=_config_snapshot(), cells={}, meta={})


def _config_snapshot():
    return dict(environments=ENVIRONMENTS, encodings=ENCODINGS,
                topologies=TOPOLOGIES, n_per_dim=N_PER_DIM,
                optimizers=OPTIMIZERS, trials=TRIALS,
                decoding_mode=DECODING_MODE, decoders_all=list(DECODERS_ALL),
                budget_mode=BUDGET_MODE, budget=BUDGET, evals_per_dim=EVALS_PER_DIM,
                budget_min=BUDGET_MIN, budget_max=BUDGET_MAX,
                settings_file=SETTINGS_FILE, tasks='cell x trial x optimizer')


def _json_default(o):
    if hasattr(o, 'tolist'):        # numpy array or numpy scalar -> plain Python
        return o.tolist()
    if hasattr(o, 'item'):
        return o.item()
    return str(o)


def save_results(path, results):
    results['config'] = _config_snapshot()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    os.replace(tmp, path)           # atomic: a crash mid-write can't corrupt it


def _weights_file(weights_dir, key):
    return os.path.join(weights_dir, key.replace('|', '__') + '.pkl')


def save_weights(weights_dir, key, name, weights):
    """Add one optimizer's network to its cell's pickle ({optimizer: weights}). Only
    the main process writes these, so they never race."""
    os.makedirs(weights_dir, exist_ok=True)
    path = _weights_file(weights_dir, key)
    data = {}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    data[name] = weights
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp, path)


# ---------------- workers (top-level -> picklable for 'spawn') ----------------
def _worker_init():
    """Force ANNarchy single-threaded so parallelism stays ACROSS tasks (worker
    processes), not nested OpenMP threads inside each sim."""
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import ANNarchy as ann
        ann.Network().setup(num_threads=1)
    except Exception:
        pass

def _episode_list(values):
    """Per-episode array -> plain list of floats for the JSON (None if absent)."""
    return None if values is None else [float(v) for v in values]


_MODULE_DEFAULTS = None


def _apply_settings(s):
    """Set one cell's weight bound, rates, latencies and spike generator INSIDE its
    worker. Workers start a fresh interpreter ('spawn'), so module changes made in the
    main process never reach them. optimize(), build_network() and the encoder read
    these module globals when they run.

    Every call first puts back the values the modules had when they were imported, so
    a cell without settings never inherits another cell's. Returns what the cell
    actually runs with, for the results file."""
    global _MODULE_DEFAULTS
    import snn_feedforward as snn
    import optimize_snn as O
    if _MODULE_DEFAULTS is None:            # first cell in this process: as imported
        _MODULE_DEFAULTS = dict(W_LOW=O.W_LOW, W_HIGH=O.W_HIGH, R_MAX=snn.R_MAX,
                                R_MIN=snn.R_MIN, LAT_MIN=snn.LAT_MIN,
                                LAT_WINDOW=snn.LAT_WINDOW,
                                SPIKE_GENERATOR=getattr(snn, 'SPIKE_GENERATOR', None))
    d = _MODULE_DEFAULTS
    O.W_LOW, O.W_HIGH = d['W_LOW'], d['W_HIGH']
    snn.R_MAX, snn.R_MIN = d['R_MAX'], d['R_MIN']
    snn.LAT_MIN, snn.LAT_WINDOW = d['LAT_MIN'], d['LAT_WINDOW']
    if d['SPIKE_GENERATOR'] is not None:
        snn.SPIKE_GENERATOR = d['SPIKE_GENERATOR']
    if s:
        w = float(s['w_scale'])
        O.W_LOW, O.W_HIGH = -w, w
        snn.R_MAX, snn.R_MIN = float(s['r_max']), float(s['r_min'])
        snn.LAT_MIN, snn.LAT_WINDOW = float(s['lat_min']), float(s['lat_window'])
        if s.get('generator'):
            if s['generator'] not in getattr(snn, 'SPIKE_GENERATORS', (s['generator'],)):
                raise ValueError("unknown spike generator %r" % s['generator'])
            snn.SPIKE_GENERATOR = s['generator']
    return dict(source=s['source'] if s else 'module defaults',
                w_low=O.W_LOW, w_high=O.W_HIGH, r_max=snn.R_MAX, r_min=snn.R_MIN,
                lat_min=snn.LAT_MIN, lat_window=snn.LAT_WINDOW,
                spike_generator=getattr(snn, 'SPIKE_GENERATOR', None))


def _run_one_task(task):
    """WORKER: build + optimize ONE cell x trial with ONE optimizer. Returns
    (cell_key, optimizer, cell_info, result, weights); catches its own errors."""
    (env, hidden, enc, dec, n_per_dim, trial, name, budget, settings, task_root) = task
    t0 = time.time()
    key = cell_key(env, hidden, enc, dec, n_per_dim, trial)
    info, weights, work = {}, None, None
    here = os.getcwd()
    try:
        import optimize_snn                # (and snn_feedforward) before the chdir below
        info['genome_dim'] = genome_dim(env, hidden, enc, n_per_dim)
        info['settings'] = _apply_settings(settings)
        # optimize() compiles into ./annarchy/opt-<cell>-t<trial>: the optimizers of a
        # cell run at the same time, so each task works in a folder of its own
        work = os.path.join(task_root, "%s__%s__p%d"
                            % (key.replace('|', '__'), name, os.getpid()))
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        os.chdir(work)
        res = optimize_snn.optimize(environment=env, hidden_sizes=hidden, encoding=enc,
                                    decoding=dec, optimizers=[name], budget=budget,
                                    trial=trial, n_per_dim=n_per_dim)
        r = res[name]
        out = {k: v for k, v in r.items() if k not in ('weights', 'fitnesses', 'rewards')}
        out.update(rewards=_episode_list(r.get('rewards')),
                   fitnesses=_episode_list(r.get('fitnesses')),
                   budget=budget, settings=info['settings'], status='ok')
        weights = r['weights']
    except Exception as e:
        out = dict(status='failed', error=f"{type(e).__name__}: {e}",
                   tb=traceback.format_exc(), budget=budget)
    finally:
        os.chdir(here)
        if work:
            shutil.rmtree(work, ignore_errors=True)
    out['seconds'] = round(time.time() - t0, 1)
    return key, name, info, out, weights


def _child(task, conn):
    """Entry point of one task's process: run it, send the result back."""
    _worker_init()
    try:
        res = _run_one_task(task)
    except BaseException as e:                    # e.g. sys.exit() inside a library
        res = (cell_key(*task[:6]), task[6], {},
               dict(status='failed', error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()), None)
    conn.send(res)
    conn.close()


def run_isolated(tasks, workers, on_result):
    """Run every task in its own fresh 'spawn' process, `workers` at a time, in order.
    A process that dies without a result (segfault, kill) is reported as a failure of
    its task only. on_result(task, result_tuple) runs in this process."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    ctx = mp.get_context('spawn')
    pending, running = list(tasks), {}
    try:
        while pending or running:
            while pending and len(running) < max(1, workers):
                task = pending.pop(0)
                recv, send = ctx.Pipe(duplex=False)
                p = ctx.Process(target=_child, args=(task, send), daemon=False)
                p.start()
                send.close()
                running[p.sentinel] = (task, p, recv)
            ready = wait(list(running) + [r for _, _, r in running.values()], timeout=30)
            for obj in ready:
                hit = [s for s, (t, p, r) in running.items() if obj in (s, r)]
                if not hit:
                    continue
                s = hit[0]
                task, p, recv = running.pop(s)
                res = None
                try:
                    if recv.poll(0.1 if obj is recv else 2.0):
                        res = recv.recv()
                except (EOFError, OSError):
                    res = None
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
                    p.join()
                if res is None:
                    code = p.exitcode
                    why = (("crashed with %s" % signal.Signals(-code).name)
                           if code is not None and code < 0 else "exited with code %s" % code)
                    res = (cell_key(*task[:6]), task[6], {},
                           dict(status='failed', error='worker process ' + why), None)
                on_result(task, res)
    except KeyboardInterrupt:
        for task, p, recv in running.values():
            p.terminate()
        raise


# ---------------- the grid run ----------------
def _expected_settings(s):
    """What _apply_settings(s) will report, computed in this process (same modules)."""
    import snn_feedforward as snn
    import optimize_snn as O
    out = dict(source=s['source'] if s else 'module defaults',
               w_low=O.W_LOW, w_high=O.W_HIGH, r_max=snn.R_MAX, r_min=snn.R_MIN,
               lat_min=snn.LAT_MIN, lat_window=snn.LAT_WINDOW,
               spike_generator=getattr(snn, 'SPIKE_GENERATOR', None))
    if s:
        w = float(s['w_scale'])
        out.update(w_low=-w, w_high=w, r_max=float(s['r_max']), r_min=float(s['r_min']),
                   lat_min=float(s['lat_min']), lat_window=float(s['lat_window']))
        if s.get('generator'):
            out['spike_generator'] = s['generator']
    return out


def _same_settings(a, b):
    if not a or not b:
        return True                      # older results without the record: trust them
    for k in ('w_low', 'w_high', 'r_max', 'r_min', 'lat_min', 'lat_window'):
        if k in a and k in b and abs(float(a[k]) - float(b[k])) > 1e-9:
            return False
    return a.get('spike_generator', b.get('spike_generator')) == \
        b.get('spike_generator', a.get('spike_generator'))


def _done(cell, name, budget, expected):
    """Is optimizer `name` of this cell already in the results with the same budget and
    settings? Entries from older runs without a per-optimizer record are judged by the
    cell's budget and settings."""
    entry = (cell or {}).get('optimizers', {}).get(name)
    if entry is None:
        return False, None
    had_budget = entry.get('budget', cell.get('budget'))
    if had_budget is not None and int(had_budget) != int(budget):
        return False, "budget %s, now %s" % (had_budget, budget)
    if not _same_settings(entry.get('settings', cell.get('settings')), expected):
        return False, "other settings"
    return True, None


def refresh_cell(rec):
    """A cell's status, best optimizer and seconds, from the optimizers it holds."""
    opts = rec.setdefault('optimizers', {})
    errors = rec.get('errors') or {}
    missing = [n for n in OPTIMIZERS if n not in opts]
    if not missing:
        rec['status'] = 'ok'
    elif any(n in errors for n in missing):
        rec['status'] = 'failed'
    else:
        rec['status'] = 'partial'
    if rec['status'] == 'failed':
        rec['error'] = "; ".join("%s: %s" % (n, errors[n]['error'])
                                 for n in missing if n in errors)
    else:
        rec.pop('error', None)
    if opts:
        best = max(opts, key=lambda k: opts[k]['fitness'])
        rec['best_optimizer'], rec['best_fitness'] = best, opts[best]['fitness']
    per_opt = [o['seconds'] for o in opts.values() if 'seconds' in o]
    if per_opt:
        rec['seconds'] = round(sum(per_opt), 1)


def clean_task_dirs(task_root):
    """Remove compile folders left behind by tasks whose process is gone (a crash or a
    kill skips the task's own clean-up). Folders of live processes are left alone, so
    another runner working in the same folder is not disturbed."""
    import re
    if not os.path.isdir(task_root):
        return
    for name in os.listdir(task_root):
        m = re.search(r'__p(\d+)$', name)
        if not m:
            continue
        try:
            os.kill(int(m.group(1)), 0)
            continue                                # still running
        except ProcessLookupError:
            shutil.rmtree(os.path.join(task_root, name), ignore_errors=True)
        except PermissionError:
            continue                                # someone else's live process


def task_seconds_estimate(env, budget):
    import optimize_snn as O
    episodes = (budget + 1) * int(O.EVAL_EPISODES) + 2 * REPORT_EP_EST + 30
    return episodes * EP_SECONDS.get(env, 0.03) + COMPILE_SECONDS


def run_grid(results, path, force, workers, settings):
    tasks_all = list(iter_tasks())
    todo, redo = [], []
    budgets = {}
    for (env, hidden, enc, dec, n, trial, name) in tasks_all:
        ck = (env, topo_str(hidden), enc, n)
        if ck not in budgets:
            budgets[ck] = budget_for(env, hidden, enc, n)
        budget = budgets[ck]
        s = settings.get(settings_key(env, hidden, enc, dec, n))
        if not force:
            done, why = _done(results['cells'].get(cell_key(env, hidden, enc, dec, n, trial)),
                              name, budget, _expected_settings(s))
            if done:
                continue
            if why:
                redo.append((cell_key(env, hidden, enc, dec, n, trial), name, why))
        todo.append((env, hidden, enc, dec, n, trial, name, budget, s))
    n_cells = len(list(iter_cells()))
    print(f"[grid] {n_cells} cells x {len(OPTIMIZERS)} optimizers = {len(tasks_all)} tasks; "
          f"{len(todo)} to run, {len(tasks_all) - len(todo)} already done; "
          f"optimizers={OPTIMIZERS}")
    for key, name, why in redo[:10]:
        print(f"[grid] rerun {key} {name}: the stored result had {why}")
    if len(redo) > 10:
        print(f"[grid] ... and {len(redo) - 10} more stored results to redo")
    sources = [settings_source(settings, *t[:5]) for t in todo]
    print(f"[grid] settings ({SETTINGS_FILE or 'no settings file'}): "
          + (", ".join(f"{sources.count(s)} {s}" for s in sorted(set(sources))) or '-'))
    for c in iter_cells():                         # statuses under the current OPTIMIZERS
        rec = results['cells'].get(cell_key(*c))
        if rec and 'optimizers' in rec:
            refresh_cell(rec)
    save_results(path, results)
    if not todo:
        return
    # longest-processing-time-first: heaviest tasks start early so they don't
    # straggle at the tail
    todo.sort(key=lambda t: task_seconds_estimate(t[0], t[7]), reverse=True)
    weights_dir = os.path.splitext(path)[0] + '_weights'
    task_root = os.path.abspath(TASK_DIR)
    clean_task_dirs(task_root)
    os.makedirs(task_root, exist_ok=True)
    tasks = [t + (task_root,) for t in todo]
    attempts = {}
    count = [0]
    n_total = len(tasks)
    retry = []
    print(f"[grid] {n_total} tasks on {workers} workers (one fresh process each, "
          f"1 thread)\n", flush=True)

    def on_result(task, res):
        key, name, info, out, weights = res
        env, hidden, enc, dec, n, trial = task[:6]
        tid = (key, name)
        attempts[tid] = attempts.get(tid, 0) + 1
        if out.get('status') != 'ok' and attempts[tid] < TASK_ATTEMPTS:
            print(f"[grid] {key} {name} failed ({out.get('error')}); trying again later",
                  flush=True)
            retry.append(task)
            return
        count[0] += 1
        rec = results['cells'].setdefault(key, dict(env=env, hidden=hidden, encoding=enc,
                                                    decoding=dec, n_per_dim=n,
                                                    trial=trial, optimizers={}))
        rec['budget'] = task[7]
        if info.get('genome_dim') is not None:
            rec['genome_dim'] = info['genome_dim']
        if out.get('status') == 'ok':
            out = {k: v for k, v in out.items() if k != 'status'}
            rec.setdefault('optimizers', {})[name] = out
            if info.get('settings') is not None:
                rec['settings'] = info['settings']
            (rec.get('errors') or {}).pop(name, None)
            if SAVE_WEIGHTS and weights is not None:
                save_weights(weights_dir, key, name, weights)
            line = (f"fit={out['fitness']:.3f} raw={out['reward_mean']:.2f} "
                    f"goal={100 * out['goal_share']:.0f}%")
        else:
            rec.setdefault('errors', {})[name] = dict(error=out.get('error'),
                                                     tb=out.get('tb'))
            line = f"FAILED: {out.get('error')}"
        if not rec.get('errors'):
            rec.pop('errors', None)
        refresh_cell(rec)
        save_results(path, results)                 # single writer
        print(f"[grid {count[0]}/{n_total}] {key} {name}  dim={rec.get('genome_dim', '-')}  "
              f"{line}  ({_fmt_secs(out.get('seconds', 0))})", flush=True)

    run_isolated(tasks, workers, on_result)
    while retry:                                    # second attempts, after the rest
        again, retry[:] = list(retry), []
        run_isolated(again, workers, on_result)


# ---------------- dry run / cost estimate ----------------
def _makespan(durations, workers):
    """Tasks handed out in this order, each to the first free worker."""
    import heapq
    free = [0.0] * max(1, min(workers, len(durations) or 1))
    end = 0.0
    for d in durations:
        t = heapq.heappop(free) + d
        end = max(end, t)
        heapq.heappush(free, t)
    return end


def dry_run(workers, settings):
    cells = list(iter_cells())
    print(f"=== DRY RUN ===  {len(cells)} cells x {len(OPTIMIZERS)} optimizers, "
          f"optimizers={OPTIMIZERS} (budget_mode={BUDGET_MODE}, workers={workers}, "
          f"settings={SETTINGS_FILE or 'module defaults'})\n")
    hdr = (f"{'env':<12}{'topology':<10}{'encoding':<12}{'dec':<7}{'n/dim':>6}"
           f"{'dim':>6}{'budget':>8}{'task':>10}  settings")
    print(hdr); print('-' * len(hdr))
    durations = []
    for env, hidden, enc, dec, n, trial in cells:
        d = genome_dim(env, hidden, enc, n)
        b = budget_for(env, hidden, enc, n)
        sec = task_seconds_estimate(env, b)
        durations += [sec] * len(OPTIMIZERS)
        if trial == 0:                          # one line per cell (trials identical cost)
            print(f"{env:<12}{topo_str(hidden):<10}{enc:<12}{dec:<7}{n_str(n):>6}"
                  f"{d:>6}{b:>8}{_fmt_secs(sec):>10}  "
                  f"{settings_source(settings, env, hidden, enc, dec, n)}")
    print('-' * len(hdr))
    total = sum(durations)
    wall = _makespan(sorted(durations, reverse=True), workers)
    print(f"grid total: {len(durations)} tasks, ~{total / 3600:,.0f} core-hours "
          f"(EP_SECONDS = {EP_SECONDS})")
    print(f"  wall ~{_fmt_secs(wall)} on {workers} workers, longest task first "
          f"(longest task ~{_fmt_secs(max(durations) if durations else 0)})\n")


# ---------------- summary ----------------
def summarize(results):
    cells = results.get('cells', {})
    ok = {k: v for k, v in cells.items() if v.get('status') == 'ok'}
    fail = {k: v for k, v in cells.items() if v.get('status') == 'failed'}
    part = {k: v for k, v in cells.items() if v.get('status') == 'partial'}
    if not cells:
        print("(no cell results yet)")
    else:
        print(f"\n=== SUMMARY ===  {len(ok)} ok, {len(part)} partial, {len(fail)} failed, "
              f"{len(cells)} total cells")
        wins = {}
        for env in sorted({v['env'] for v in ok.values()}):
            rows = [v for v in ok.values() if v['env'] == env]
            rows.sort(key=lambda v: v['best_fitness'], reverse=True)
            print(f"\n-- {env} -- (ranked by best optimizer fitness)")
            hd = (f"{'topology':<10}{'encoding':<12}{'dec':<7}{'n/dim':>5}  "
                  f"{'best opt':<12}{'fitness':>10}{'dim':>6}{'bud':>7}")
            print(hd); print('-' * len(hd))
            for v in rows:
                print(f"{topo_str(v['hidden']):<10}{v['encoding']:<12}"
                      f"{v['decoding']:<7}{n_str(v.get('n_per_dim')):>5}  "
                      f"{v.get('best_optimizer',''):<12}"
                      f"{v['best_fitness']:>10.3f}{v.get('genome_dim', 0):>6}"
                      f"{v['budget']:>7}")
                if (
                    (v.get('best_fitness') <= -200 and v.get('env') == 'mountaincar')
                    or (v.get('best_fitness') <= -500 and v.get('env') == 'acrobot')
                ):
                    continue
                else:
                    wins[v.get('best_optimizer')] = wins.get(v.get('best_optimizer'), 0) + 1
            top = rows[0]
            print(f"   best {env} config: {top['encoding']}/{top['decoding']} "
                  f"topo={topo_str(top['hidden'])} "
                  f"n_per_dim={n_str(top.get('n_per_dim'))} "
                  f"via {top.get('best_optimizer')} "
                  f"-> {top['best_fitness']:.3f}")
        if wins:
            print("\noptimizer wins across cells: "
                  + ", ".join(f"{k}={n}" for k, n in
                              sorted(wins.items(), key=lambda x: -x[1])))
        if fail:
            print("\nfailed cells:")
            for k, v in fail.items():
                print(f"  {k}: {v.get('error')}")
    print()


# ---------------- entry point ----------------
def main():
    global SETTINGS_FILE, OPTIMIZERS
    ap = argparse.ArgumentParser(description="SNN study orchestrator", allow_abbrev=False)
    ap.add_argument('--dry-run', action='store_true',
                    help="list cells + cost estimate, run nothing")
    ap.add_argument('--force', action='store_true',
                    help="ignore checkpoint; recompute every task")
    ap.add_argument('--summarize', action='store_true',
                    help="print summary from the results file and exit")
    ap.add_argument('--workers', type=int, default=None,
                    help="parallel worker processes (default: WORKERS config; "
                         "0/unset = os.cpu_count(); 1 = one at a time)")
    ap.add_argument('--config', default=None,
                    help="JSON file overriding the CONFIG block (lower-case keys)")
    ap.add_argument('--optimizers', default=None,
                    help="comma-separated list overriding OPTIMIZERS")
    ap.add_argument('--out', default=None, help="results JSON path (default RESULTS_PATH)")
    ap.add_argument('--settings', default=None,
                    help="per-cell settings from make_grid_settings.py "
                         "(default: SETTINGS_FILE in the config)")
    args = ap.parse_args()

    if args.config:
        apply_config(args.config)
    if args.optimizers:
        OPTIMIZERS = [o.strip() for o in args.optimizers.split(',') if o.strip()]
    if args.settings is not None:
        SETTINGS_FILE = args.settings          # recorded in the results' config
    out = args.out or RESULTS_PATH

    import nevergrad as ng
    unknown = [o for o in OPTIMIZERS if o not in ng.optimizers.registry]
    if unknown or not OPTIMIZERS:
        raise SystemExit("not in Nevergrad's registry: %s" % (unknown or 'no optimizers'))

    settings = load_settings(SETTINGS_FILE)
    workers = args.workers if args.workers is not None else WORKERS
    workers = workers if workers and workers > 0 else (os.cpu_count() or 1)

    if args.dry_run:
        dry_run(workers, settings)
        return

    results = load_results(out)

    if args.summarize:
        summarize(results)
        return

    def _stop(signum, frame):                  # kill -TERM: stop the task processes too
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    try:                                       # ~2 open files per running task: lift the
        import resource                        # usual soft limit of 1024 (no root needed)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 4 * workers + 256
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (want if hard == resource.RLIM_INFINITY else min(want, hard),
                                hard))
    except (ImportError, ValueError, OSError):
        pass
    run_grid(results, out, args.force, workers, settings)

    summarize(results)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
