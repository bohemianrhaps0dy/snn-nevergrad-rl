"""
experiment_runner.py -- one control point for the whole SNN study.

Configures and drives the three modules without reimplementing them:
  snn_feedforward.py  the network (encoders / decoders / topology / genome sizing)
  optimize_snn.py     plain comparison: many optimizers, one config, fixed budget
  meta_optimize.py    OPTIONAL: tune one inner optimizer family's hyperparameters

It runs the Cartesian grid
    environment x encoding x decoding x topology x optimizer
by calling optimize_snn.optimize() once per (env, topology, encoding, decoding)
cell -- optimize() sweeps the OPTIMIZERS internally -- and OPTIONALLY runs the
meta-optimization phase for chosen (environment, family) pairs.

Thin-orchestrator on purpose: the science lives in the three modules; this only
sequences calls, scales budget, checkpoints, and summarizes, so it stays correct
as those modules evolve. It depends only on their stable entry points
(optimize(), meta_optimize(), weight_shapes(), n_params()), not on their internal
constants, so your local edits (reset-in-evaluate, R_MIN, latency burst, ...) are
respected.

DEFINE the experiment in the CONFIG block. CONTROL the run with flags:
  python experiment_runner.py --dry-run     # list cells + cost estimate, run nothing
  python experiment_runner.py               # run the grid (parallel, resumable)
  python experiment_runner.py --workers 8   # N parallel workers (default os.cpu_count())
  python experiment_runner.py --workers 1   # serial (for debugging)
  python experiment_runner.py --meta        # also run the meta phase
  python experiment_runner.py --meta-only   # only the meta phase
  python experiment_runner.py --force       # ignore checkpoint; recompute every cell
  python experiment_runner.py --summarize   # print summary from an existing results file
  python experiment_runner.py --out FILE    # results file (default below)

Parallelism: each worker is a separate process that trains one cell's network at a
time ("one worker, one network"), via the 'spawn' start method so every worker
compiles ANNarchy in a fresh interpreter. Each worker is forced SINGLE-THREADED
(OMP_NUM_THREADS=1) so the parallelism is across cells, not nested OpenMP threads
inside each tiny sim (which would oversubscribe and be slower than serial). Only
the MAIN process writes the results file, so checkpoints never race; combined with
resume, an interrupted or even crashed run continues where it stopped.

Resumability: results are written after every cell; a re-run skips cells already
present (unless --force), so an interrupted multi-hour sweep continues.
"""

import os
import sys
import json
import time
import argparse
import itertools
import traceback

# Cap native (OpenMP / BLAS) threads to 1 PER PROCESS, before anything imports
# ANNarchy. Parallelism here is across cells (worker processes); letting each
# worker's ANNarchy also fan out over all cores via OpenMP would oversubscribe
# (workers x cores threads) and run slower than serial. setdefault so an explicit
# external OMP_NUM_THREADS still wins. Runs on every import, including the fresh
# interpreter each 'spawn' worker starts, so it is set before ANNarchy loads.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')


# ===================== CONFIG: define the experiment here =====================
ENVIRONMENTS = ['mountaincar', 'acrobot']
ENCODINGS    = ['poprate','poplatency']
TOPOLOGIES   = [[10]]          # genome scales with input width too
OPTIMIZERS   = ['TBPSA'] # any subset optimize_snn understands
TRIALS       = 1                                # repeats per cell (different trial seed)
WORKERS      = 0                                # parallel worker processes; 0 = auto
                                                # (os.cpu_count()), 1 = serial

# decoding coverage: 'full' = every encoding x both decoders (honours "combine the
# decodings"); 'natural' = latency/poplatency -> first, rate-like -> vote (halves
# the cells and drops the known-mushy latency+vote / rate+first pairings).
DECODING_MODE = 'natural'                          # 'full' | 'natural'
DECODERS_ALL  = ('vote', 'first')

# budget: 'fixed' = same BUDGET for every cell (the study's fixed-budget design);
# 'per_dim' = EVALS_PER_DIM * genome_dim, clamped to [BUDGET_MIN, BUDGET_MAX].
# 'per_dim' equalises search difficulty across the ~10x genome gap between
# single-neuron and population encodings -- it removes the budget-starvation
# confound (poprate at dim 230 vs latency at dim 25), at the price of unequal
# sample budgets. Use 'fixed' for the headline "families at a fixed budget"
# result; use 'per_dim' to ask "best each encoding can do, budget-fair".
BUDGET_MODE   = 'fixed'                          # 'fixed' | 'per_dim'
BUDGET        = 5000
EVALS_PER_DIM = 1
BUDGET_MIN, BUDGET_MAX = 200, 4000

# ---- OPTIONAL meta-optimization phase (runs only with --meta / --meta-only) ----
META_FAMILIES     = ['cma']                     # inner families to tune: cma|de|pso
META_OUTER_BUDGET = 50
META_K            = 15
META_INNER_BUDGET = 200
# representative config to TUNE ON, per environment (overrides meta_optimize's own
# REPRESENTATIVE). Pick a config the diagnostics call sound -- tuning an optimizer
# against an unsolvable config makes "improvement over default" meaningless.
META_CONFIGS = {
    'mountaincar': dict(hidden_sizes=[8], encoding='poprate', decoding='vote'),
    'acrobot':     dict(hidden_sizes=[8], encoding='latency', decoding='first'),
}

# cost-estimate only (measure your own machine); does not affect the actual run.
EP_SECONDS      = {'mountaincar': 0.018, 'acrobot': 0.065}
COMPILE_SECONDS = 6.0                            # rough per-cell ANNarchy compile
REPORT_EP_EST   = 100                             # optimize_snn's REPORT_EPISODES (est)

RESULTS_PATH = 'experiment_results.json'
SAVE_WEIGHTS = False                            # also pickle best-per-cell weights
# ============================================================================


# ---------------- grid helpers (pure python) ----------------
def topo_str(hidden):
    return 'x'.join(str(h) for h in hidden) or 'none'


def decoders_for(encoding):
    if DECODING_MODE == 'natural':
        return ['first'] if encoding in ('latency', 'poplatency') else ['vote']
    return list(DECODERS_ALL)


def iter_cells():
    """Yield (env, hidden, encoding, decoding, trial) for the whole grid."""
    for env in ENVIRONMENTS:
        for hidden in TOPOLOGIES:
            for enc in ENCODINGS:
                for dec in decoders_for(enc):
                    for trial in range(TRIALS):
                        yield env, hidden, enc, dec, trial


def cell_key(env, hidden, enc, dec, trial):
    return f"{env}|{topo_str(hidden)}|{enc}|{dec}|t{trial}"


def genome_dim(env, hidden, encoding):
    """Genome size for a cell, via the real weight_shapes/n_params. Falls back
    gracefully if a diverged weight_shapes lacks the `encoding` arg."""
    from snn_feedforward import weight_shapes
    from optimize_snn import n_params
    try:
        shapes = weight_shapes(env, hidden, encoding)
    except TypeError:
        shapes = weight_shapes(env, hidden)     # older signature
    return int(n_params(shapes))


def budget_for(env, hidden, encoding):
    if BUDGET_MODE == 'per_dim':
        d = genome_dim(env, hidden, encoding)
        return int(min(BUDGET_MAX, max(BUDGET_MIN, EVALS_PER_DIM * d)))
    return BUDGET


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
                topologies=TOPOLOGIES, optimizers=OPTIMIZERS, trials=TRIALS,
                decoding_mode=DECODING_MODE, budget_mode=BUDGET_MODE,
                budget=BUDGET, evals_per_dim=EVALS_PER_DIM)


def _json_default(o):
    if hasattr(o, 'item'):          # numpy scalar
        return o.item()
    if hasattr(o, 'tolist'):        # numpy array
        return o.tolist()
    return str(o)


def save_results(path, results):
    results['config'] = _config_snapshot()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    os.replace(tmp, path)           # atomic: a crash mid-write can't corrupt it


# ---------------- workers (top-level -> picklable for 'spawn') ----------------
def _worker_init():
    """Runs once per worker process. Force ANNarchy single-threaded so parallelism
    stays ACROSS cells (worker processes), not nested OpenMP threads inside each
    sim (which would oversubscribe and run slower than serial)."""
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        import ANNarchy as ann
        ann.setup(num_threads=1)
    except Exception:
        pass


def _cell_genome(env, hidden, enc):
    from snn_feedforward import weight_shapes
    from optimize_snn import n_params
    try:
        shapes = weight_shapes(env, hidden, enc)
    except TypeError:
        shapes = weight_shapes(env, hidden)
    return int(n_params(shapes))


def _run_one_cell(task):
    """WORKER: build + optimize ONE cell (one network). Pure, picklable; returns
    (cell_key, record). Catches its own errors so one bad cell can't kill the pool."""
    env, hidden, enc, dec, trial, budget, optimizers, save_weights, weights_dir = task
    t0 = time.time()
    rec = dict(env=env, hidden=hidden, encoding=enc, decoding=dec, trial=trial,
               budget=budget)
    try:
        rec['genome_dim'] = _cell_genome(env, hidden, enc)
        import optimize_snn as O
        res = O.optimize(environment=env, hidden_sizes=hidden, encoding=enc,
                         decoding=dec, optimizers=optimizers, budget=budget,
                         trial=trial)
        rec['optimizers'] = {
            name: dict(fitness=r['fitness'], std=r['std'], sem=r['sem'],
                       opt_loss=r.get('opt_loss'), n_episodes=r.get('n_episodes'))
            for name, r in res.items()}
        best = max(res, key=lambda k: res[k]['fitness'])
        rec.update(status='ok', best_optimizer=best,
                   best_fitness=res[best]['fitness'])
        if save_weights:                 # save per-cell (no big arrays through IPC)
            import pickle
            os.makedirs(weights_dir, exist_ok=True)
            fn = cell_key(env, hidden, enc, dec, trial).replace('|', '__') + '.pkl'
            with open(os.path.join(weights_dir, fn), 'wb') as f:
                pickle.dump({n: r['weights'] for n, r in res.items()}, f)
    except Exception as e:
        rec.update(status='failed', error=f"{type(e).__name__}: {e}",
                   tb=traceback.format_exc())
    rec['seconds'] = round(time.time() - t0, 1)
    return cell_key(env, hidden, enc, dec, trial), rec


def _meta_key(t):
    env, family, rep = t[0], t[1], t[2]
    return (f"meta|{env}|{family}|{topo_str(rep['hidden_sizes'])}|"
            f"{rep['encoding']}|{rep['decoding']}")


def _run_one_meta(task):
    """WORKER: one (env, family) meta-optimization. Sets the representative and
    inner budget in ITS OWN process (module-global mutation is process-local, so
    concurrent meta tasks with different configs don't clash)."""
    env, family, rep, outer_budget, K, inner_budget = task
    t0 = time.time()
    entry = dict(env=env, family=family, representative=rep)
    try:
        import meta_optimize as M
        M.INNER_BUDGET = inner_budget
        M.REPRESENTATIVE[env] = rep
        r = M.meta_optimize(env, family=family, outer_budget=outer_budget, K=K)
        entry.update(best_theta=r['best_theta'],
                     tuned=r['holdout_score'], tuned_std=r.get('holdout_std'),
                     default=r['baseline_score'], default_std=r.get('baseline_std'),
                     improvement=r['improvement'], status='ok')
    except Exception as e:
        entry.update(status='failed', error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc())
    entry['seconds'] = round(time.time() - t0, 1)
    return _meta_key(task), entry


def _print_cell(prefix, key, rec):
    if rec.get('status') == 'ok':
        extra = (f"best={rec.get('best_optimizer')} fit={rec['best_fitness']:.3f}"
                 if 'best_fitness' in rec else
                 f"tuned={rec.get('tuned')} impr={rec.get('improvement')}")
        print(f"{prefix} {key}  dim={rec.get('genome_dim','-')}  {extra}  "
              f"({_fmt_secs(rec.get('seconds', 0))})", flush=True)
    else:
        print(f"{prefix} {key}  FAILED: {rec.get('error')}", flush=True)


def _resolve_workers(workers):
    if workers is None:
        workers = WORKERS
    return workers if workers and workers > 0 else (os.cpu_count() or 1)


def _dispatch(tasks, worker_fn, key_of, store, results, path, workers, label):
    """Run worker_fn over tasks; serial if workers<=1 else on a spawn pool. The
    MAIN process is the ONLY writer of results/checkpoint, so there is no write
    race; key_of(task) gives a stable key even if a worker process dies hard."""
    n = len(tasks)
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            _, rec = worker_fn(t)
            store[key_of(t)] = rec
            save_results(path, results)
            _print_cell(f"[{label} {i}/{n}]", key_of(t), rec)
        return
    import concurrent.futures as cf
    import multiprocessing as mp
    ctx = mp.get_context('spawn')        # fresh interpreter per worker -> no fork hazards
    print(f"[{label}] {n} tasks on {workers} workers (spawn, 1 thread each)\n",
          flush=True)
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                initializer=_worker_init) as ex:
        futs = {ex.submit(worker_fn, t): t for t in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            t = futs[fut]
            key = key_of(t)
            try:
                _, rec = fut.result()
            except Exception as e:       # worker crashed (e.g. native segfault)
                rec = dict(status='failed', error=f"worker crashed: {e}")
            store[key] = rec
            save_results(path, results)  # single writer
            _print_cell(f"[{label} {i}/{n}]", key, rec)


# ---------------- the grid run ----------------
def run_grid(results, path, force, workers):
    cells = list(iter_cells())
    todo = []
    for env, hidden, enc, dec, trial in cells:
        prev = results['cells'].get(cell_key(env, hidden, enc, dec, trial))
        if prev and prev.get('status') == 'ok' and not force:
            continue
        todo.append((env, hidden, enc, dec, trial))
    print(f"[grid] {len(cells)} cells; {len(todo)} to run, "
          f"{len(cells) - len(todo)} already done; optimizers={OPTIMIZERS}")
    if not todo:
        return
    # longest-processing-time-first: heaviest cells start early so they don't
    # straggle at the tail under dynamic dispatch (better load balance).
    todo.sort(key=lambda c: budget_for(c[0], c[1], c[2]) * EP_SECONDS.get(c[0], 0.03),
              reverse=True)
    weights_dir = os.path.splitext(path)[0] + '_weights'
    tasks = [(env, hidden, enc, dec, trial, budget_for(env, hidden, enc),
              OPTIMIZERS, SAVE_WEIGHTS, weights_dir)
             for (env, hidden, enc, dec, trial) in todo]
    _dispatch(tasks, _run_one_cell, lambda t: cell_key(*t[:5]),
              results['cells'], results, path, workers, 'grid')


# ---------------- the optional meta phase ----------------
def run_meta(results, path, force, workers):
    tasks = []
    for env in [e for e in ENVIRONMENTS if e in META_CONFIGS]:
        rep = META_CONFIGS[env]
        for family in META_FAMILIES:
            t = (env, family, rep, META_OUTER_BUDGET, META_K, META_INNER_BUDGET)
            if _meta_key(t) in results['meta'] and not force:
                continue
            tasks.append(t)
    print(f"\n[meta] {len(tasks)} (env,family) task(s) to run "
          f"(outer={META_OUTER_BUDGET}, K={META_K}, inner={META_INNER_BUDGET})")
    if not tasks:
        return
    # meta tasks are few and huge; never spawn more workers than tasks.
    mw = min(workers, len(tasks)) if workers > 1 else workers
    _dispatch(tasks, _run_one_meta, _meta_key,
              results['meta'], results, path, mw, 'meta')


# ---------------- dry run / cost estimate ----------------
def dry_run(workers):
    cells = list(iter_cells())
    print(f"=== DRY RUN ===  {len(cells)} cells, optimizers={OPTIMIZERS} "
          f"(budget_mode={BUDGET_MODE}, workers={workers})\n")
    hdr = f"{'env':<12}{'topology':<10}{'encoding':<12}{'dec':<7}{'dim':>6}{'budget':>8}{'episodes':>10}"
    print(hdr); print('-' * len(hdr))
    total_ep = 0.0
    total_sec = 0.0
    for env, hidden, enc, dec, trial in cells:
        d = genome_dim(env, hidden, enc)
        b = budget_for(env, hidden, enc)
        ep = len(OPTIMIZERS) * (b + REPORT_EP_EST)
        total_ep += ep
        total_sec += ep * EP_SECONDS.get(env, 0.03) + COMPILE_SECONDS
        if trial == 0:                          # one line per cell (trials identical cost)
            print(f"{env:<12}{topo_str(hidden):<10}{enc:<12}{dec:<7}"
                  f"{d:>6}{b:>8}{ep:>10}")
    print('-' * len(hdr))
    par = total_sec / max(1, workers)
    print(f"grid total: ~{int(total_ep):,} optimization+report episodes")
    print(f"  serial wall ~{_fmt_secs(total_sec)}  ->  ~{_fmt_secs(par)} on "
          f"{workers} workers (ideal; compile bursts + imbalance add overhead)\n")

    # meta estimate
    if META_CONFIGS:
        per_call = (META_OUTER_BUDGET * META_K * META_INNER_BUDGET
                    + 2 * META_K * META_INNER_BUDGET)     # search + holdout + baseline
        n_calls = len([e for e in ENVIRONMENTS if e in META_CONFIGS]) * len(META_FAMILIES)
        m_ep = per_call * n_calls
        m_sec = sum(per_call * EP_SECONDS.get(e, 0.03)
                    for e in ENVIRONMENTS if e in META_CONFIGS) * len(META_FAMILIES)
        m_par = m_sec / max(1, min(workers, n_calls))
        print(f"meta (only with --meta): {n_calls} call(s) x ~{per_call:,} "
              f"inner episodes = ~{m_ep:,} episodes")
        print(f"  serial wall ~{_fmt_secs(m_sec)} -> ~{_fmt_secs(m_par)} on "
              f"{min(workers, n_calls)} workers (only {n_calls} parallel task(s))")
        print("  NOTE: meta is ~1-2 orders of magnitude costlier than the grid; "
              "each call is\n  outer_budget x K x inner_budget inner episodes, and "
              "only parallelizes across\n  (env,family) tasks here. Keep "
              "META_FAMILIES / META_CONFIGS small.\n")


# ---------------- summary ----------------
def summarize(results):
    cells = results.get('cells', {})
    ok = {k: v for k, v in cells.items() if v.get('status') == 'ok'}
    fail = {k: v for k, v in cells.items() if v.get('status') == 'failed'}
    if not cells:
        print("(no cell results yet)")
    else:
        print(f"\n=== SUMMARY ===  {len(ok)} ok, {len(fail)} failed, "
              f"{len(cells)} total cells")
        wins = {}
        for env in sorted({v['env'] for v in ok.values()}):
            rows = [v for v in ok.values() if v['env'] == env]
            rows.sort(key=lambda v: v['best_fitness'], reverse=True)
            print(f"\n-- {env} -- (ranked by best optimizer fitness)")
            hd = (f"{'topology':<10}{'encoding':<12}{'dec':<7}{'best opt':<10}"
                  f"{'fitness':>10}{'dim':>6}{'bud':>6}")
            print(hd); print('-' * len(hd))
            for v in rows:
                print(f"{topo_str(v['hidden']):<10}{v['encoding']:<12}"
                      f"{v['decoding']:<7}{v.get('best_optimizer',''):<10}"
                      f"{v['best_fitness']:>10.3f}{v['genome_dim']:>6}"
                      f"{v['budget']:>6}")
                if (
                    (v.get('best_fitness') <= -200 and v.get('env') == 'mountaincar')
                    or (v.get('best_fitness') <= -500 and v.get('env') == 'acrobot')
                ):
                    continue
                else:
                    wins[v.get('best_optimizer')] = wins.get(v.get('best_optimizer'), 0) + 1
            top = rows[0]
            print(f"   best {env} config: {top['encoding']}/{top['decoding']} "
                  f"topo={topo_str(top['hidden'])} via {top.get('best_optimizer')} "
                  f"-> {top['best_fitness']:.3f}")
        if wins:
            print("\noptimizer wins across cells: "
                  + ", ".join(f"{k}={n}" for k, n in
                              sorted(wins.items(), key=lambda x: -x[1])))
        if fail:
            print("\nfailed cells:")
            for k, v in fail.items():
                print(f"  {k}: {v.get('error')}")

    meta = results.get('meta', {})
    if meta:
        print("\n=== META-OPTIMIZATION ===")
        for k, v in meta.items():
            if v.get('status') == 'ok':
                print(f"  {k}: tuned={v['tuned']:.3f} default={v['default']:.3f} "
                      f"improvement={v['improvement']:+.3f}  theta={v['best_theta']}")
            else:
                print(f"  {k}: FAILED {v.get('error')}")
    print()


# ---------------- entry point ----------------
def main():
    ap = argparse.ArgumentParser(description="SNN study orchestrator")
    ap.add_argument('--dry-run', action='store_true',
                    help="list cells + cost estimate, run nothing")
    ap.add_argument('--meta', action='store_true', help="also run the meta phase")
    ap.add_argument('--meta-only', action='store_true', help="only the meta phase")
    ap.add_argument('--force', action='store_true',
                    help="ignore checkpoint; recompute every cell")
    ap.add_argument('--summarize', action='store_true',
                    help="print summary from the results file and exit")
    ap.add_argument('--workers', type=int, default=None,
                    help="parallel worker processes (default: WORKERS config; "
                         "0/unset = os.cpu_count(); 1 = serial)")
    ap.add_argument('--out', default=RESULTS_PATH, help="results JSON path")
    args = ap.parse_args()

    workers = _resolve_workers(args.workers)

    if args.dry_run:
        dry_run(workers)
        return

    results = load_results(args.out)

    if args.summarize:
        summarize(results)
        return

    if not args.meta_only:
        run_grid(results, args.out, args.force, workers)
    if args.meta or args.meta_only:
        run_meta(results, args.out, args.force, workers)

    summarize(results)
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
