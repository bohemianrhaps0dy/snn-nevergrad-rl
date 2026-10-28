"""
run_pipeline.py -- tune every cell of the grid, then run the grid on the tuned settings,
unattended. Put it next to your code (snn_feedforward.py, optimize_snn.py,
tune_hyperparams.py, make_grid_settings.py, experiment_runner.py) and start it there.

  1. TUNING    tune_hyperparams.py once per grid cell (the cells in GRID below), as many
               cells side by side as the cores hold, longest cells first. A cell writes
               TUNING_DIR/result-<cell>.json when it ends. A finished cell is skipped, a
               cell that was interrupted continues from its last finished batch, and a
               cell whose run fails is started again (ATTEMPTS runs in total).
  2. SETTINGS  make_grid_settings.py --always-tuned: every cell gets its tuned values,
               with no held-out test (the comparison stays in the file for the record).
  3. GRID      experiment_runner.py over the same cells with those settings, the budget
               rule in GRID and the optimizers listed in OPTIMIZERS_FILE. That list is read
               when the grid starts, so it can change while the tuning runs; names added
               later run on their own when the pipeline is started again.

The tuner's inner runs get the grid's own budget for their cell (TUNE_EVALS_PER_DIM =
None), so each setting is scored on the search the grid will run.

    python run_pipeline.py --dry-run    # cells, budgets and a time estimate; runs nothing
    python run_pipeline.py --smoke      # a few minutes on tiny budgets, in pipeline_smoke/
    mkdir -p logs && nohup python run_pipeline.py > logs/pipeline.log 2>&1 &
    python run_pipeline.py --status     # progress and time left, from any shell
    python run_pipeline.py --stop       # stop it and everything it started
    python run_pipeline.py --tuning-only   # stop after the tuning and the settings file

Starting it again after a stop, a crash or a reboot continues where it was. While it
runs, don't edit snn_feedforward.py, optimize_snn.py or tune_hyperparams.py in this
folder: every cell that starts imports them afresh. Experiment in a copy of the folder.
"""

import argparse
import contextlib
import heapq
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

# ============================ what to run ============================
GRID = dict(
    environments=['mountaincar', 'acrobot'],
    encodings=['poprate', 'poplatency'],
    decoding='first',                     # every cell, both encodings
    topologies=[[6], [12], [6, 6], [10, 4]],
    n_per_dim=[2, 4, 6, 10],
    trials=5,
    generator='regular',                  # poprate's spike generator, in tuning and grid
    evals_per_dim=100,                    # grid budget = evals_per_dim x d, clipped to
    budget_min=1000,                      # [budget_min, budget_max]; None = no bound
    budget_max=None,
)
TUNE_EVALS_PER_DIM = None   # the tuner's inner budget per weight (same clip). None = the
                            # grid's budget, so tuning scores the grid's own search.
                            # 20 makes the tuning about 5x cheaper.
TUNER_ARGS = []             # extra tune_hyperparams.py flags for every cell
OPTIMIZERS_FILE = 'optimizers.txt'        # one Nevergrad name per line, read at grid start
DEFAULT_OPTIMIZERS = ['TwoPointsDE', 'NoisyRL3']   # written there if the file is missing
CORES = None                # worker processes at once; None = this machine's physical cores
NICE = 10                   # run everything at lower priority, so the server stays usable
ATTEMPTS = 3                # runs per tuning cell before it is given up

# ---- where things go (relative to this file's folder) ----
TUNING_DIR = 'tuning_grid'
SETTINGS_FILE = 'grid_settings_tuned.json'
RESULTS_FILE = 'experiment_results_grid.json'
STATE_DIR = 'pipeline'
LOG_DIR = 'logs'

POLL_SECONDS = 20
EP_SECONDS = {'mountaincar': 0.018, 'acrobot': 0.065}   # first estimate; --status uses
                                                        # the times the tuning measures

# --smoke: the whole pipeline on 4 cells and tiny budgets, in its own folder
SMOKE = dict(
    GRID=dict(environments=['mountaincar', 'acrobot'], encodings=['poprate', 'poplatency'],
              decoding='first', topologies=[[6]], n_per_dim=[2], trials=1,
              generator='regular', evals_per_dim=0.5, budget_min=10, budget_max=40),
    TUNER_ARGS=['--quick'],
    ROOT='pipeline_smoke',
)
# ======================================================================

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = CODE_DIR               # every path above is relative to the code folder
T = None                     # tune_hyperparams, imported by load_modules()
_STOPPING = [False]


def p(*parts):
    return os.path.join(ROOT, *parts)


def log(msg):
    print("%s  %s" % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


def fmt_secs(s):
    s = int(max(0, s))
    if s < 3600:
        return "%dm" % max(1, s // 60) if s >= 60 else "%ds" % s
    if s < 86400:
        return "%dh %02dm" % (s // 3600, (s % 3600) // 60)
    return "%dd %dh" % (s // 86400, (s % 86400) // 3600)


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- machine
def physical_cores():
    """Physical cores this process may use (SMT siblings count once, CPU quota of a
    container respected)."""
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


def cmdline(pid):
    try:
        with open('/proc/%d/cmdline' % int(pid), 'rb') as f:
            return [a.decode(errors='replace') for a in f.read().split(b'\0') if a]
    except OSError:
        return None


def runs_script(pid, script):
    """Is process `pid` alive and running a Python script named `script`?"""
    if not alive(pid):
        return False
    have = cmdline(pid)
    return True if have is None else any(a.endswith(script) for a in have)


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


def start(cmd, logfile):
    """Start cmd in its own session (so --stop can stop it with everything it spawns),
    at lower priority, appending to logfile. Returns the Popen."""
    full = ([shutil.which('nice'), '-n', str(NICE)] if NICE and shutil.which('nice') else []) \
        + [sys.executable] + cmd
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    out = open(logfile, 'a')
    out.write("\n===== %s  %s\n" % (time.strftime('%Y-%m-%d %H:%M:%S'),
                                     ' '.join(full)))
    out.flush()
    proc = subprocess.Popen(full, cwd=ROOT, env=child_env(), stdout=out,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    out.close()
    return proc


def kill_group(pid, wait=30):
    """SIGTERM the process group led by pid, SIGKILL whatever is left after `wait` s."""
    if not alive(pid):
        return
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    t0 = time.time()
    while alive(pid) and time.time() - t0 < wait:
        time.sleep(0.5)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(int(pid), signal.SIGKILL)


# ---------------------------------------------------------------- modules and cells
def load_modules():
    """Import tune_hyperparams (and through it the network modules) the way the tuner
    will see them. ANNarchy reads sys.argv when it is imported, so it gets none of ours."""
    global T
    if T is None:
        saved = sys.argv
        sys.argv = [saved[0]]
        sys.path.insert(0, CODE_DIR)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                import tune_hyperparams as tune_module
        finally:
            sys.argv = saved
        T = tune_module
    return T


def preflight():
    """Problems that would stop the pipeline days from now, found before it starts: the
    files it runs must be the versions it was written for."""
    problems = []
    for name in ('tune_hyperparams.py', 'make_grid_settings.py', 'experiment_runner.py',
                 'optimize_snn.py', 'snn_feedforward.py'):
        if not os.path.exists(os.path.join(CODE_DIR, name)):
            problems.append("%s is missing from %s" % (name, CODE_DIR))
    if problems:
        return problems
    T = load_modules()
    need = ('_cell_tag', 'cell_dim', 'default_batch', 'checkpoint_counts', '_run_key',
            'K_INNER', 'OUTER_BUDGET', 'INNER_OPTIMIZER', 'CHECKPOINTS_PER_DIM')
    if [a for a in need if not hasattr(T, a)]:
        problems.append("tune_hyperparams.py is older than the one this pipeline runs (the "
                        "Oct 27 version, whose inner runs go through optimize_snn's "
                        "train_weights)")
    if not all(hasattr(T.optimize_snn, a) for a in ('train_weights', 'make_objective',
                                                     'initial_point')):
        problems.append("optimize_snn.py has no train_weights()/make_objective(): use the "
                        "Oct 27 version that tune_hyperparams.py needs")
    for name, flag in (('make_grid_settings.py', '--always-tuned'),
                       ('experiment_runner.py', "'--config'")):
        with open(os.path.join(CODE_DIR, name)) as f:
            if flag not in f.read():
                problems.append("%s has no %s option: copy the new %s here"
                                % (name, flag.strip("'"), name))
    return problems


def topo(hidden):
    return 'x'.join(str(h) for h in hidden) or 'none'


def clip_budget(per, d, lo, hi):
    b = int(round(per * d))
    if lo is not None:
        b = max(int(lo), b)
    if hi is not None:
        b = min(int(hi), b)
    return b


def tuner_shape():
    """The tuner settings that decide a cell's batches: its defaults, then TUNER_ARGS."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--outer-budget', type=int, default=T.OUTER_BUDGET)
    ap.add_argument('--outer-optimizer', default=T.OUTER_OPTIMIZER)
    ap.add_argument('--inner-optimizer', default=T.INNER_OPTIMIZER)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--k-inner', type=int, default=T.K_INNER)
    ap.add_argument('--eval-episodes', type=int, default=T.INNER_EVAL_EPISODES)
    ap.add_argument('--report-episodes', type=int, default=T.TUNE_REPORT_EPISODES)
    ap.add_argument('--checkpoints', type=float, nargs='*', default=list(T.CHECKPOINTS_PER_DIM))
    ap.add_argument('--quick', action='store_true')
    a, _ = ap.parse_known_args(TUNER_ARGS)
    a.inner_budget = None
    if a.quick:                                     # what the tuner's --quick sets
        a.outer_budget, a.inner_budget, a.k_inner = 4, 20, 1
        a.report_episodes, a.checkpoints = 3, [0.1]
    return a


class Cell:
    """One tuning cell: what to run and what it costs."""

    def __init__(self, cell, shape, cores):
        self.cell = cell
        self.tag = T._cell_tag(cell)
        self.env = cell['environment']
        self.dim = T.cell_dim(cell)
        self.grid_budget = clip_budget(GRID['evals_per_dim'], self.dim,
                                       GRID['budget_min'], GRID['budget_max'])
        per = GRID['evals_per_dim'] if TUNE_EVALS_PER_DIM is None else TUNE_EVALS_PER_DIM
        self.budget = (shape.inner_budget if shape.inner_budget is not None else
                       clip_budget(per, self.dim, GRID['budget_min'], GRID['budget_max']))
        self.K = shape.k_inner
        self.outer = shape.outer_budget
        self.batch = shape.batch or T.default_batch(cell['encoding'], shape.outer_optimizer,
                                                    cores, self.K)
        self.workers = max(1, min(self.batch * self.K, cores))
        checkpoints = T.checkpoint_counts(self.dim, self.budget, shape.checkpoints)
        self.run_episodes = (self.budget * shape.eval_episodes
                             + (len(checkpoints) + 1) * shape.report_episodes)
        self.run_key = T._run_key(shape, self.budget, checkpoints)   # what --resume checks
        self.shape = dict(inner_budget=self.budget, k_inner=self.K,
                          inner_eval_episodes=shape.eval_episodes,
                          tune_report_episodes=shape.report_episodes,
                          inner_optimizer=shape.inner_optimizer,
                          outer_optimizer=shape.outer_optimizer, outer_budget=self.outer)
        sizes = [self.batch] * (self.outer // self.batch)
        if self.outer % self.batch:
            sizes.append(self.outer % self.batch)
        self.sizes = sizes + [2]                    # the last one: held-out + baseline
        self.result = p(TUNING_DIR, 'result-%s.json' % self.tag)
        self.ckpt = p(TUNING_DIR, 'tune-%s.json' % self.tag)
        self.log = p(LOG_DIR, 'tune-%s.log' % self.tag)
        self.cmd = [os.path.join(CODE_DIR, 'tune_hyperparams.py'),
                    '--env', self.env, '--encoding', cell['encoding'],
                    '--decoding', cell['decoding'], '--hidden', topo(cell['hidden_sizes']),
                    '--n-per-dim', str(cell['n_per_dim'])]
        if cell['encoding'] == 'poprate' and cell.get('generator'):
            self.cmd += ['--generator', cell['generator']]
        self.cmd += ['--inner-budget', str(self.budget), '--out-dir', p(TUNING_DIR),
                     '--workers', str(self.workers), '--resume'] + list(TUNER_ARGS)

    def waves(self, size):
        return int(math.ceil(size * self.K / float(self.workers)))

    def run_seconds(self, ep_s):
        return self.run_episodes * ep_s.get(self.env, 0.05)

    def batches_done(self):
        """Batches finished so far, from the tuner's checkpoint."""
        if self.finished():
            return len(self.sizes)
        h = (read_json(self.ckpt) or {}).get('history', [])
        done, i = 0, 0
        for size in self.sizes[:-1]:
            if i + size <= len(h):
                done, i = done + 1, i + size
        return done

    def usage(self):
        """Worker processes this cell keeps busy now (never goes up as the cell runs)."""
        b = self.batches_done()
        if b >= len(self.sizes):
            return 0
        return min(self.sizes[b] * self.K, self.workers)

    def seconds_left(self, ep_s, since_ckpt=0.0):
        b = self.batches_done()
        left = sum(self.waves(s) for s in self.sizes[b:]) * self.run_seconds(ep_s)
        return max(0.0, left - since_ckpt)

    def finished(self):
        """A result made with this cell's tuning settings exists."""
        r = read_json(self.result)
        return bool(r) and self._matches(r)

    def _matches(self, r):
        s = r.get('settings', {})
        return (all(s.get(k) == v for k, v in self.shape.items())
                and r.get('cell', {}).get('n_per_dim') == self.cell['n_per_dim'])

    def set_aside_stale(self):
        """Rename a result or checkpoint made with other tuning settings (an earlier
        budget rule, K, ...), so the tuner starts this cell over instead of refusing to
        resume, and make_grid_settings.py never reads the old result. Returns what was
        moved."""
        moved = []
        stamp = time.strftime('%Y%m%d-%H%M%S')
        r = read_json(self.result)
        if r is not None and not self._matches(r):
            os.replace(self.result, self.result + '.stale-' + stamp)
            moved.append(os.path.basename(self.result))
        ck = read_json(self.ckpt)
        if ck is not None and ck.get('run') != self.run_key:
            for f in (self.ckpt, self.ckpt[:-len('.json')] + '.opt.pkl'):
                if os.path.exists(f):
                    os.replace(f, f + '.stale-' + stamp)
                    moved.append(os.path.basename(f))
        return moved


def make_cells(cores):
    load_modules()
    shape = tuner_shape()
    cells = []
    for env in GRID['environments']:
        for hidden in GRID['topologies']:
            for enc in GRID['encodings']:
                for n in GRID['n_per_dim']:
                    cells.append(Cell(dict(environment=env, encoding=enc,
                                           decoding=GRID['decoding'],
                                           hidden_sizes=list(hidden), n_per_dim=int(n),
                                           generator=(GRID['generator'] if enc == 'poprate'
                                                      else None)), shape, cores))
    return cells


def settings_key(cell):
    c = cell.cell
    return "%s|%s|%s|%s|n%d" % (c['environment'], topo(c['hidden_sizes']), c['encoding'],
                                c['decoding'], c['n_per_dim'])


# ---------------------------------------------------------------- optimizers
def read_optimizers(create=False):
    path = os.path.join(CODE_DIR, OPTIMIZERS_FILE)    # the real list, also for --smoke
    if not os.path.exists(path):
        if not create:
            return list(DEFAULT_OPTIMIZERS), path
        with open(path, 'w') as f:
            f.write("# Optimizers for the grid, one Nevergrad name per line (# starts a "
                    "comment).\n# Read when the grid starts: edit it while the tuning "
                    "runs. Names added after\n# the grid has run are run when the pipeline "
                    "is started again.\n" + "\n".join(DEFAULT_OPTIMIZERS) + "\n")
    names = []
    with open(path) as f:
        for line in f:
            for x in line.split('#')[0].replace(',', ' ').split():
                if x not in names:
                    names.append(x)
    return names, path


def check_optimizers(names, path):
    import nevergrad as ng
    bad = [n for n in names if n not in ng.optimizers.registry]
    if not names:
        return "%s lists no optimizers" % path
    if bad:
        return "%s: not in Nevergrad's registry: %s" % (path, ", ".join(bad))
    return None


# ---------------------------------------------------------------- estimates
def grid_task_seconds(cells, n_opt, ep_s):
    """One duration per grid task (cell x trial x optimizer): an upper bound, episodes
    of random networks; trained ones end sooner."""
    import optimize_snn
    out = []
    for c in cells:
        eps = (c.grid_budget + 1) * int(optimize_snn.EVAL_EPISODES) + 230
        out += [eps * ep_s.get(c.env, 0.05) + 30] * (GRID['trials'] * n_opt)
    return out


def makespan(durations, workers):
    free = [0.0] * max(1, min(workers, len(durations) or 1))
    end = 0.0
    for d in sorted(durations, reverse=True):
        t = heapq.heappop(free) + d
        end = max(end, t)
        heapq.heappush(free, t)
    return end


def simulate_tuning(cells, cores, ep_s, running=(), now=None):
    """Wall time until every tuning cell is done, admitting cells as run_tuning() does.
    `running`: cells already started (their remaining batches count from their
    checkpoint)."""
    now = time.time() if now is None else now
    events, used, t = [], 0, 0.0
    seq = 0

    def push(c, b, start_t, first=None):
        nonlocal used, seq
        if b >= len(c.sizes):
            return
        dur = c.waves(c.sizes[b]) * c.run_seconds(ep_s) if first is None else first
        seq += 1
        heapq.heappush(events, (start_t + dur, seq, c, b))
        used += min(c.sizes[b] * c.K, c.workers)

    for c in running:
        b = c.batches_done()
        if b >= len(c.sizes):
            continue
        since = max(0.0, now - _mtime(c.ckpt, c))
        full = c.waves(c.sizes[b]) * c.run_seconds(ep_s)
        push(c, b, 0.0, first=max(60.0, full - since))
    waiting = sorted([c for c in cells if c not in running and not c.finished()],
                     key=lambda c: -c.seconds_left(ep_s))
    while waiting or events:
        while waiting and (used + waiting[0].workers <= cores or not events):
            push(waiting.pop(0), 0, t)
        t, _, c, b = heapq.heappop(events)
        used -= min(c.sizes[b] * c.K, c.workers)
        push(c, b + 1, t)
    return t


_STARTED = {}


def _mtime(path, cell):
    """When the cell's current batch started: its checkpoint's last write, or the cell's
    start if it has no checkpoint yet."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return _STARTED.get(cell.tag, time.time())


def measured_ep_seconds(cells):
    """Seconds per training episode the tuning has measured so far, per environment
    (median over finished batches); EP_SECONDS where nothing is measured yet."""
    per_env = {}
    for c in cells:
        h = (read_json(c.ckpt) or {}).get('history', [])
        i = 0
        for size in c.sizes[:-1]:
            if i + size > len(h):
                break
            secs = h[i].get('batch_seconds')
            if secs:
                per_env.setdefault(c.env, []).append(
                    secs / (c.waves(size) * c.run_episodes))
            i += size
    out = dict(EP_SECONDS)
    for env, vals in per_env.items():
        out[env] = sorted(vals)[len(vals) // 2]
    return out, {env: len(v) for env, v in per_env.items()}


# ---------------------------------------------------------------- state
def state_path():
    return p(STATE_DIR, 'state.json')


def load_state():
    return read_json(state_path(), {}) or {}


def save_state(state):
    os.makedirs(p(STATE_DIR), exist_ok=True)
    atomic_json(state_path(), state)


def take_lock():
    os.makedirs(p(STATE_DIR), exist_ok=True)
    lock = p(STATE_DIR, 'launcher.pid')
    other = read_json(lock)
    if other and other.get('pid') != os.getpid() and runs_script(other.get('pid'),
                                                                  'run_pipeline.py'):
        raise SystemExit("the pipeline is already running here (pid %s, started %s). "
                         "Use --status to follow it, or --stop to stop it."
                         % (other['pid'], other.get('started')))
    atomic_json(lock, dict(pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                           host=os.uname().nodename))


def drop_lock():
    lock = p(STATE_DIR, 'launcher.pid')
    other = read_json(lock)
    if other and other.get('pid') == os.getpid():
        with contextlib.suppress(OSError):
            os.remove(lock)


# ---------------------------------------------------------------- stage 1: tuning
def run_tuning(cells, cores, state):
    st = state.setdefault('cells', {})
    running = {}                     # tag -> (cell, Popen or None if adopted, pid)
    for c in cells:                  # tuners left running by an earlier launcher
        s = st.get(c.tag, {})
        if s.get('status') == 'running' and runs_script(s.get('pid'), 'tune_hyperparams.py'):
            running[c.tag] = (c, None, s['pid'])
            _STARTED[c.tag] = s.get('started_ts', time.time())
            log("tuning %s: still running from before (pid %s), following it"
                % (c.tag, s['pid']))
    waiting = []
    for c in cells:
        if c.tag in running:
            continue
        s = st.setdefault(c.tag, {})
        s['attempts'] = 0                           # attempts count within one launch
        if c.finished():
            s['status'] = 'done'
        else:
            s['status'] = 'waiting'
            waiting.append(c)
    ep_s, _ = measured_ep_seconds(cells)
    waiting.sort(key=lambda c: -c.seconds_left(ep_s))     # longest first
    n_done = sum(1 for c in cells if st[c.tag]['status'] == 'done')
    log("tuning: %d cells, %d done, %d to run, %d still running; %d cores"
        % (len(cells), n_done, len(waiting), len(running), cores))
    save_state(state)

    while waiting or running:
        if _STOPPING[0]:
            return False
        used = sum(c.usage() for c, _, _ in running.values())
        while waiting and (used + waiting[0].workers <= cores or not running):
            c = waiting.pop(0)
            s = st[c.tag]
            s['attempts'] = s.get('attempts', 0) + 1
            moved = c.set_aside_stale()
            if moved:
                log("tuning %s: made with other tuning settings, set aside (.stale-*): %s"
                    % (c.tag, ", ".join(moved)))
            proc = start(c.cmd, c.log)
            _STARTED[c.tag] = time.time()
            running[c.tag] = (c, proc, proc.pid)
            s.update(status='running', pid=proc.pid, started=time.strftime('%Y-%m-%d %H:%M:%S'),
                     started_ts=time.time(), log=os.path.relpath(c.log, ROOT))
            used += c.workers
            log("tuning %s: started (d=%d, inner budget %d, %d workers, attempt %d/%d)"
                % (c.tag, c.dim, c.budget, c.workers, s['attempts'], ATTEMPTS))
            save_state(state)
        time.sleep(POLL_SECONDS)
        for tag, (c, proc, pid) in list(running.items()):
            if proc is not None and proc.poll() is None:
                continue
            if proc is None and alive(pid):
                continue
            del running[tag]
            s = st[tag]
            s.pop('pid', None)
            secs = time.time() - s.get('started_ts', time.time())
            if c.finished():
                r = read_json(c.result)
                s.update(status='done', finished=time.strftime('%Y-%m-%d %H:%M:%S'))
                bc = r['best_config']
                if c.cell['encoding'] == 'poprate':
                    tuned = "w=%.1f rmax=%.0f rmin=%.0f" % (bc['w_scale'], bc['r_max'],
                                                          bc['r_min'])
                else:
                    tuned = "w=%.1f latmin=%.1f latwin=%.1f" % (bc['w_scale'], bc['lat_min'],
                                                              bc['lat_window'])
                log("tuning %s: done in %s | tuned %s | held-out raw %.1f vs current %.1f"
                    % (tag, fmt_secs(secs), tuned, r['holdout_raw'], r['baseline_raw']))
            else:
                tail = _tail(c.log, 4)
                if s.get('attempts', 0) < ATTEMPTS and not _STOPPING[0]:
                    s['status'] = 'waiting'
                    waiting.insert(0, c)                  # resumes from its checkpoint
                    log("tuning %s: stopped without a result after %s (code %s); starting "
                        "it again | %s" % (tag, fmt_secs(secs),
                                           None if proc is None else proc.returncode, tail))
                else:
                    s['status'] = 'failed'
                    log("tuning %s: FAILED %d times, giving up on it (see %s) | %s"
                        % (tag, s.get('attempts', 0), s.get('log'), tail))
            save_state(state)
            n_done = sum(1 for x in cells if st[x.tag]['status'] == 'done')
            ep_s, n_meas = measured_ep_seconds(cells)
            left = simulate_tuning(cells, cores, ep_s,
                                   running=[x for x, _, _ in running.values()])
            log("tuning: %d/%d cells done, %d running, %d waiting; about %s left"
                % (n_done, len(cells), len(running), len(waiting), fmt_secs(left)))
    return True


def _tail(path, n):
    """The end of a log: its last exception line first, then the last n lines."""
    try:
        with open(path, errors='replace') as f:
            lines = [l.strip() for l in f.readlines()[-300:] if l.strip()]
    except OSError:
        return ''
    errors = [l for l in lines if l.split(':')[0].endswith(('Error', 'Exception', 'Exit',
                                                            'Interrupt'))]
    return " / ".join(errors[-1:] + lines[-n:])[-700:]


# ---------------------------------------------------------------- stage 2: settings
def run_settings(cells):
    cmd = [os.path.join(CODE_DIR, 'make_grid_settings.py'), '--always-tuned',
           '--results', p(TUNING_DIR, 'result-*.json'), '--out', p(SETTINGS_FILE)]
    proc = start(cmd, p(LOG_DIR, 'settings.log'))
    proc.wait()
    if proc.returncode != 0:
        log("settings: make_grid_settings.py failed (code %s) | %s"
            % (proc.returncode, _tail(p(LOG_DIR, 'settings.log'), 3)))
        return None
    cells_in = (read_json(p(SETTINGS_FILE)) or {}).get('cells', {})
    missing = [settings_key(c) for c in cells if settings_key(c) not in cells_in]
    log("settings: %s has tuned values for %d of the grid's %d cells"
        % (os.path.relpath(p(SETTINGS_FILE), ROOT), len(cells) - len(missing), len(cells)))
    for k in missing:
        log("settings: WARNING %s has no tuning result; the grid runs it on the current "
            "module settings" % k)
    return missing


# ---------------------------------------------------------------- stage 3: grid
def grid_config():
    return dict(environments=GRID['environments'], encodings=GRID['encodings'],
                topologies=GRID['topologies'], n_per_dim=GRID['n_per_dim'],
                trials=GRID['trials'], decoding_mode='full',
                decoders_all=[GRID['decoding']], budget_mode='per_dim',
                evals_per_dim=GRID['evals_per_dim'], budget_min=GRID['budget_min'],
                budget_max=GRID['budget_max'], save_weights=True,
                results_path=p(RESULTS_FILE))


def run_grid(cells, cores, state):
    """Run the grid until every optimizer in OPTIMIZERS_FILE is done: names added to the
    file while the grid runs get a second pass (only their tasks run)."""
    ran = []
    while True:
        names, path = read_optimizers()
        problem = check_optimizers(names, path)
        if problem:
            log("grid: NOT started: %s. Fix the file and start the pipeline again (the "
                "tuning is not repeated)." % problem)
            return False
        if ran and set(names) <= set(ran):
            return True
        if ran:
            log("grid: %s now also lists %s: running those too"
                % (os.path.relpath(path, ROOT), ", ".join(n for n in names if n not in ran)))
        ok = _run_grid_once(cells, cores, state, names)
        if not ok:
            return False
        ran = list(state['grid'].get('optimizers', names))


def _run_grid_once(cells, cores, state, names):
    cfg_path = p(STATE_DIR, 'grid_config.json')
    atomic_json(cfg_path, grid_config())
    cmd = [os.path.join(CODE_DIR, 'experiment_runner.py'), '--config', cfg_path,
           '--settings', p(SETTINGS_FILE), '--out', p(RESULTS_FILE),
           '--optimizers', ','.join(names), '--workers', str(cores)]
    g = state.setdefault('grid', {})
    if g.get('status') == 'running' and runs_script(g.get('pid'), 'experiment_runner.py'):
        log("grid: still running from before (pid %s), following it" % g['pid'])
        proc, pid = None, g['pid']
        names = g.get('optimizers', names)
    else:
        proc = start(cmd, p(LOG_DIR, 'grid.log'))
        pid = proc.pid
        ep_s, _ = measured_ep_seconds(cells)
        tasks = grid_task_seconds(cells, len(names), ep_s)
        log("grid: started with %s (%d tasks; at most about %s on %d workers)"
            % (", ".join(names), len(tasks), fmt_secs(makespan(tasks, cores)), cores))
        g.update(status='running', pid=pid, optimizers=names,
                 started=time.strftime('%Y-%m-%d %H:%M:%S'), started_ts=time.time(),
                 tasks=len(cells) * GRID['trials'] * len(names))
    save_state(state)
    while True:
        if _STOPPING[0]:
            return False
        if proc is not None and proc.poll() is not None:
            code = proc.returncode
            break
        if proc is None and not alive(pid):
            code = None
            break
        time.sleep(POLL_SECONDS)
    g.pop('pid', None)
    res = read_json(p(RESULTS_FILE), {}) or {}
    done, failed = grid_progress(res, cells, names)
    total = len(cells) * GRID['trials'] * len(names)
    ok = code == 0 or (code is None and done + failed == total)
    g.update(status='done' if ok else 'stopped', finished=time.strftime('%Y-%m-%d %H:%M:%S'),
             tasks=total, failed=failed)
    save_state(state)
    log("grid: %s (exit code %s): %d/%d tasks done, %d failed -> %s%s"
        % ('finished' if ok else 'ended early', code, done, total, failed,
           os.path.relpath(p(RESULTS_FILE), ROOT),
           "" if not failed else " (failed tasks are retried when the pipeline is started "
           "again; see %s)" % os.path.relpath(p(LOG_DIR, 'grid.log'), ROOT)))
    return ok


def grid_progress(res, cells, names):
    done = failed = 0
    budgets = {settings_key(c): c.grid_budget for c in cells}
    for key, rec in (res.get('cells') or {}).items():
        skey = key.rsplit('|', 1)[0]
        if skey not in budgets:
            continue
        for n in names:
            o = rec.get('optimizers', {}).get(n)
            if o is not None and o.get('budget', rec.get('budget')) == budgets[skey]:
                done += 1
            elif n in (rec.get('errors') or {}):
                failed += 1
    return done, failed


# ---------------------------------------------------------------- commands
def cmd_run(args, cores):
    take_lock()
    state = load_state()
    state['launcher'] = dict(pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                             cores=cores, root=ROOT)
    names, path = read_optimizers(create=True)
    problems = preflight()
    problem = check_optimizers(names, path)
    if problem:
        problems.append(problem)
    if problems:
        drop_lock()
        raise SystemExit("not starting:\n  " + "\n  ".join(problems))

    def on_signal(signum, frame):
        _STOPPING[0] = True
        log("stop requested (signal %d): stopping everything this pipeline started" % signum)
        st = load_state()
        for s in st.get('cells', {}).values():
            if s.get('pid'):
                kill_group(s['pid'])
                s.pop('pid', None)
                s['status'] = 'waiting'
        if st.get('grid', {}).get('pid'):
            kill_group(st['grid']['pid'])
            st['grid'].pop('pid', None)
            st['grid']['status'] = 'stopped'
        st['stage'] = 'stopped'
        save_state(st)
        drop_lock()
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        cells = make_cells(cores)
        log("pipeline: %d tuning cells, then the grid with %s x %d trials; %d cores; "
            "results in %s" % (len(cells), ", ".join(names), GRID['trials'], cores, ROOT))
        state['stage'] = 'tuning'
        save_state(state)
        if not run_tuning(cells, cores, state):
            return
        failed = [c.tag for c in cells if state['cells'][c.tag]['status'] == 'failed']
        if failed:
            log("tuning: %d cells failed: %s" % (len(failed), ", ".join(failed)))
        state['stage'] = 'settings'
        save_state(state)
        missing = run_settings(cells)
        if missing is None:
            return
        if args.tuning_only:
            state['stage'] = 'tuned'
            save_state(state)
            log("pipeline: tuning done (--tuning-only); start it again without the flag "
                "for the grid")
            return
        state['stage'] = 'grid'
        save_state(state)
        ok = run_grid(cells, cores, state)
        state['stage'] = 'done' if ok else 'grid stopped'
        save_state(state)
        if ok:
            log("pipeline: DONE. Results: %s, networks: %s_weights/, settings: %s, "
                "tuning: %s/%s" % (RESULTS_FILE, os.path.splitext(RESULTS_FILE)[0],
                                  SETTINGS_FILE, TUNING_DIR,
                                  "" if not missing else
                                  " | %d cell(s) ran on the current settings because their "
                                  "tuning failed: start the pipeline again to retry them"
                                  % len(missing)))
    finally:
        if not _STOPPING[0]:
            drop_lock()


def cmd_status(cores):
    state = load_state()
    lock = read_json(p(STATE_DIR, 'launcher.pid'))
    cells = make_cells(cores)
    st = state.get('cells', {})
    if lock and runs_script(lock.get('pid'), 'run_pipeline.py'):
        head = "running (pid %s, started %s)" % (lock['pid'], lock.get('started'))
    elif state:
        head = "NOT running (last stage: %s)" % state.get('stage', '?')
    else:
        head = "not started yet"
    print("pipeline in %s: %s" % (ROOT, head))
    ep_s, n_meas = measured_ep_seconds(cells)
    finished = [c for c in cells if c.finished()]
    running = [c for c in cells if st.get(c.tag, {}).get('status') == 'running'
               and alive(st[c.tag].get('pid')) and not c.finished()]
    failed = [c for c in cells if st.get(c.tag, {}).get('status') == 'failed']
    for c in running:
        _STARTED[c.tag] = st[c.tag].get('started_ts', time.time())
    print("tuning: %d/%d cells done, %d running, %d failed, %d waiting"
          % (len(finished), len(cells), len(running), len(failed),
             len(cells) - len(finished) - len(running) - len(failed)))
    if running:
        print("  %-48s %9s %13s %10s" % ('running', 'configs', 'this batch', 'cell left'))
        now = time.time()
        for c in sorted(running, key=lambda c: c.tag):
            h = len((read_json(c.ckpt) or {}).get('history', []))
            since = now - _mtime(c.ckpt, c) if os.path.exists(c.ckpt) else \
                now - st[c.tag].get('started_ts', now)
            print("  %-48s %4d/%-4d %13s %10s"
                  % (c.tag, min(h, c.outer), c.outer, fmt_secs(since),
                     fmt_secs(c.seconds_left(ep_s, since))))
    for c in failed:
        print("  FAILED %s (log: %s)" % (c.tag, st[c.tag].get('log')))
    meas = ", ".join("%s %.3f s%s" % (e, ep_s[e], " (measured over %d batches)" % n_meas[e]
                                      if n_meas.get(e) else " (estimate)")
                     for e in GRID['environments'])
    print("  seconds per training episode: %s" % meas)
    names, path = read_optimizers()
    g = state.get('grid', {})
    tasks = grid_task_seconds(cells, max(1, len(names)), ep_s)
    grid_wall = makespan(tasks, cores)
    if len(finished) < len(cells):
        left = simulate_tuning(cells, cores, ep_s, running=running)
        print("  tuning left: about %s" % fmt_secs(left))
        print("grid: not started; will run %s (from %s) x %d trials: %d tasks, about %s"
              % (", ".join(names), os.path.relpath(path, ROOT), GRID['trials'],
                 len(tasks), fmt_secs(grid_wall)))
        print("finish: around %s (upper bound: trained networks end episodes sooner)"
              % time.strftime('%a %d %b %H:%M', time.localtime(time.time() + left + grid_wall)))
    else:
        res = read_json(p(RESULTS_FILE), {}) or {}
        gnames = g.get('optimizers', names)
        done, nfail = grid_progress(res, cells, gnames)
        total = len(cells) * GRID['trials'] * len(gnames)
        status = g.get('status', 'not started')
        if status == 'running' and not alive(g.get('pid')):
            status = 'not running'
        print("grid: %s, %s: %d/%d tasks done, %d failed"
              % (status, ", ".join(gnames), done, total, nfail))
        if g.get('started_ts') and done < total and status == 'running':
            el = time.time() - g['started_ts']
            print("  running for %s; at most about %s in all" % (fmt_secs(el), fmt_secs(grid_wall)))
    print("logs: %s/ (pipeline.log, tune-<cell>.log, grid.log)" % os.path.relpath(p(LOG_DIR), ROOT))


def cmd_stop():
    lock = read_json(p(STATE_DIR, 'launcher.pid'))
    state = load_state()
    if lock and runs_script(lock.get('pid'), 'run_pipeline.py'):
        os.kill(int(lock['pid']), signal.SIGTERM)
        t0 = time.time()
        while alive(lock['pid']) and time.time() - t0 < 90:
            time.sleep(0.5)
        print("stopped the launcher (pid %s)" % lock['pid'])
    pids = [s['pid'] for s in state.get('cells', {}).values() if s.get('pid')]
    if state.get('grid', {}).get('pid'):
        pids.append(state['grid']['pid'])
    n = 0
    for pid in pids:
        if alive(pid):
            kill_group(pid)
            n += 1
    print("stopped %d running tuner/grid process group(s). Start the pipeline again to "
          "continue where it stopped." % n)


def cmd_dry_run(cores):
    problems = preflight()
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        if any('missing' in x or 'older' in x for x in problems):
            return
    cells = make_cells(cores)
    names, path = read_optimizers()
    problem = check_optimizers(names, path)
    shape = tuner_shape()
    print("pipeline dry run in %s (%d cores, physical; CORES=%s)\n" % (ROOT, cores, CORES))
    if shape.inner_budget is not None:
        inner = "%d evaluations (--quick)" % shape.inner_budget
    elif TUNE_EVALS_PER_DIM is None:
        inner = "the grid's budget for the cell"
    else:
        inner = "%g x d" % TUNE_EVALS_PER_DIM
    print("TUNING: %d cells | tune_hyperparams.py: %s over %d configs in batches of up "
          "to %d, K=%d inner runs each, %d training episodes per evaluation; inner budget "
          "= %s" % (len(cells), shape.outer_optimizer, shape.outer_budget, cells[0].batch,
                    shape.k_inner, shape.eval_episodes, inner))
    print("  %-44s %5s %8s %8s %9s %10s %8s" % ('cell', 'd', 'inner', 'grid', 'one run',
                                               'cell wall', 'state'))
    for c in sorted(cells, key=lambda c: -c.seconds_left(EP_SECONDS)):
        print("  %-44s %5d %8d %8d %9s %10s %8s"
              % (c.tag, c.dim, c.budget, c.grid_budget, fmt_secs(c.run_seconds(EP_SECONDS)),
                 fmt_secs(c.seconds_left(EP_SECONDS)),
                 'done' if c.finished() else ('started' if os.path.exists(c.ckpt) else '-')))
    core_h = sum(sum(s * c.K for s in c.sizes) * c.run_seconds(EP_SECONDS) for c in cells) / 3600
    tune_wall = simulate_tuning(cells, cores, EP_SECONDS)
    tasks = grid_task_seconds(cells, max(1, len(names)), EP_SECONDS)
    grid_wall = makespan(tasks, cores)
    print("\n  tuning: ~%s core-hours -> about %s on %d cores (longest cells first, %d "
          "workers per cell)" % ("{:,.0f}".format(core_h), fmt_secs(tune_wall), cores,
                                  cells[0].workers))
    print("GRID: %d cells x %d trials x %d optimizers (%s, from %s%s) = %d tasks, "
          "budget %g x d in [%s, %s]"
          % (len(cells), GRID['trials'], len(names), ", ".join(names),
             os.path.relpath(path, ROOT), "" if os.path.exists(path) else ", not written yet",
             len(tasks), GRID['evals_per_dim'], GRID['budget_min'], GRID['budget_max']))
    print("  ~%s core-hours -> at most about %s on %d workers (longest task %s)"
          % ("{:,.0f}".format(sum(tasks) / 3600), fmt_secs(grid_wall), cores,
             fmt_secs(max(tasks))))
    print("\nTOTAL: about %s (episode times %s s: the runner's EP_SECONDS; --status "
          "replaces them with measured ones)" % (fmt_secs(tune_wall + grid_wall), EP_SECONDS))
    if problem:
        print("\nPROBLEM: %s" % problem)


def main():
    global GRID, TUNER_ARGS, ROOT
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help='plan and time estimate only')
    ap.add_argument('--status', action='store_true', help='progress and time left')
    ap.add_argument('--stop', action='store_true', help='stop the pipeline and its runs')
    ap.add_argument('--smoke', action='store_true',
                    help='the whole pipeline on tiny budgets, in pipeline_smoke/')
    ap.add_argument('--tuning-only', action='store_true',
                    help='stop after the tuning and the settings file')
    ap.add_argument('--cores', type=int, default=None,
                    help='worker processes at once (default: CORES, or the physical cores)')
    args = ap.parse_args()

    if args.smoke:
        GRID = dict(SMOKE['GRID'])
        TUNER_ARGS = list(SMOKE['TUNER_ARGS'])
        ROOT = os.path.join(CODE_DIR, SMOKE['ROOT'])
        os.makedirs(ROOT, exist_ok=True)
    cores = args.cores or CORES or physical_cores()
    if (args.status or args.dry_run) and not args.cores:
        launched = (load_state().get('launcher') or {}).get('cores')
        cores = launched or cores               # the estimate the running pipeline makes

    if args.status:
        cmd_status(cores)
    elif args.stop:
        cmd_stop()
    elif args.dry_run:
        cmd_dry_run(cores)
    else:
        os.makedirs(p(LOG_DIR), exist_ok=True)
        cmd_run(args, cores)


if __name__ == '__main__':
    main()
