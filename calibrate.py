"""
calibrate.py -- measure which budget (evaluations per weight) lets SOME optimizers pass
Gym's threshold while others don't.

Every episode goes through snn_feedforward.evaluate() -- the fast loop with F1-F4 -- so a
calibration scores networks exactly the way your optimizations do. Before optimizing, each
run checks that its reporting loop returns the same fitness as evaluate(), bit for bit.

One run per (optimizer, seed), up to the largest budget. At each checkpoint the best network
so far is scored the way Gym does (mean RAW reward of 100 fresh episodes; pass if >= -110 on
MountainCar, >= -100 on Acrobot), so one run gives the score at every smaller budget too.

    python calibrate.py --env mountaincar --per-weight 5 10 20 40 80 \
                        --optimizers TwoPointsDE CMA PSO OnePlusOne RandomSearch --seeds 5
    python calibrate.py --env mountaincar --n-per-dim 4 --encoding poplatency --decoding first \
                        --per-weight 10 20 40 --optimizers TwoPointsDE RandomSearch --seeds 5

How to read the table it prints (passed seeds per optimizer at each budget):
  - too small : nobody passes                      -> raise the budget (or the cell is too hard)
  - too large : RandomSearch passes, or everybody  -> the comparison no longer discriminates
  - usable    : the best optimizers pass in most seeds, RandomSearch in none
Choose ONE evaluations-per-weight value for the whole study from the usable range, then give
every optimizer in an experiment the same budget = that value * number of weights
(budget_plan.py prints the budget and the compute cost of every experiment).

Each checkpoint in the JSON also stores the share of steps with a silent output, with a tie,
and with a random action (vote: silent or tie; first spike: silent only, because a
first-spike tie goes to the lowest index).

Notes
  - Checkpoints are only exact for optimizers that ignore the declared budget (DE, TwoPointsDE,
    CMA, PSO, OnePlusOne, RandomSearch). NGOpt / NgIohTuned pick their method FROM the budget,
    so run those once per budget (--per-weight with a single value).
  - Per seed, every optimizer gets the same start: the same init vector (used by the optimizers
    that start from it: OnePlusOne, the CMA family) and the same random_state (used by DE and
    RandomSearch, which with both bounds set draw their first candidates uniformly over the
    whole box and never evaluate the init). An all-zero init is a silent network and a flat
    landscape for the optimizers that start from it.
  - --rescore-top 5: at each checkpoint the 5 best-scoring candidates are re-scored on 30 fresh
    episodes and the best of them is reported. Without it the reported network is the one with
    the luckiest single score, and that gets worse as the budget grows.
  - --n-per-dim sets the neurons per observation. Use even values: with an odd count the
    middle velocity bin spans v = 0 and hides the sign of small velocities.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import time, json, argparse
import numpy as np

THRESHOLD = {'mountaincar': -110.0, 'acrobot': -100.0}


def _outcome(decoding, spikes, n_out):
    """'decisive', 'tie' or 'silent' for one window. Draws no random numbers."""
    if decoding == 'vote':
        counts = [len(spikes.get(a, ())) for a in range(n_out)]
        top = max(counts)
        if top == 0:
            return 'silent'
        winners = sum(1 for c in counts if c == top)
    else:
        firsts = [min(spikes[a]) for a in range(n_out) if len(spikes.get(a, ()))]
        if not firsts:
            return 'silent'
        earliest = min(firsts)
        winners = sum(1 for t in firsts if t == earliest)
    return 'decisive' if winners == 1 else 'tie'


def scored_episode(snn, gym, nh, W, seed):
    """One episode through evaluate()'s own loop, counting what the decoder saw.

    Returns (raw reward, shaped fitness, silent steps, tied steps, steps). The shaped
    fitness is exactly what snn.evaluate(nh, W, episode_seed=seed) returns: the counting
    only reads the spikes, and decode_action_fast() makes the same random draws.
    """
    snn.set_weights(nh, W)
    cfg, net = nh['cfg'], nh['net']
    n_out = int(cfg['n_salida'])
    fast = snn._fast(nh)
    spike_monitor = fast['spike_monitor']
    env = fast['env']
    if env is None:
        env = fast['env'] = gym.make(cfg['gym_id'])
    obs, _ = env.reset(seed=int(seed))
    net.reset()
    paused = [m for m in net.get_monitors() if m is not spike_monitor]
    for m in paused:
        m.pause()
    total, best, done = 0.0, cfg['progress'](obs), False
    steps = silent = tied = 0
    try:
        while not done:
            snn.drive_encoder_fast(nh, obs)
            net.simulate(snn.T_STEP)
            sp = spike_monitor.get('spike')
            outcome = _outcome(nh['decoding'], sp, n_out)
            silent += outcome == 'silent'
            tied += outcome == 'tie'
            action = snn.decode_action_fast(nh['decoding'], sp, n_out)
            obs, r, te, tr, _ = env.step(action)
            total += r
            steps += 1
            done = te or tr
            best = max(best, cfg['progress'](obs))
            snn._reset_step(nh)
    finally:
        for m in paused:
            m.resume()
    return total, total + cfg['shaping'] * best, silent, tied, steps


def check_against_evaluate(snn, gym, nh, split, dim, bound, seed, n_sets=3):
    """scored_episode() and evaluate() must agree bit for bit. Leaves numpy's global
    random state as it found it, so the optimization draws what it would have drawn."""
    state = np.random.get_state()
    rng = np.random.default_rng(777 + seed)
    try:
        for i in range(n_sets):
            W = split(rng.uniform(-bound, bound, dim))
            np.random.seed(4242 + i)
            f_ref = snn.evaluate(nh, W, episode_seed=900 + i)
            np.random.seed(4242 + i)
            f_new = scored_episode(snn, gym, nh, W, 900 + i)[1]
            if f_ref != f_new:
                raise RuntimeError("calibrate.py's reporting loop no longer matches "
                                   "snn_feedforward.evaluate(): %r vs %r" % (f_new, f_ref))
    finally:
        np.random.set_state(state)


def job(a):
    import nevergrad as ng
    import gymnasium as gym
    import snn_feedforward as snn
    snn.NEURONS_PER_DIM = a['n_per_dim']        # older build_network reads the global
    if a.get('generator') is not None:
        if not hasattr(snn, 'SPIKE_GENERATORS'):
            raise RuntimeError("this snn_feedforward.py has no spike generator option;"
                               " use the version with regular spikes")
        snn.SPIKE_GENERATOR = a['generator']
    regular = a['encoding'] == 'poprate' and getattr(snn, 'SPIKE_GENERATOR', 'poisson') == 'regular'
    np.random.seed(a['seed'])
    tag = '%s-%s-%s-%s-h%s-%s-s%d%s' % (a['env'], a['encoding'], a['decoding'], a['optimizer'],
                                        'x'.join(map(str, a['hidden'])), a['n_per_dim'], a['seed'],
                                        '-regular' if regular else '')
    nh = snn.build_network(a['env'], a['hidden'], a['encoding'], a['decoding'], 'annarchy/cal-' + tag,
                           n_per_dim=a['n_per_dim'])
    n_obs = int(nh['cfg']['n_entrada'])
    if nh['encoder'].size != n_obs * a['n_per_dim']:
        raise RuntimeError('asked for %d neurons per observation, the network has %d'
                           % (a['n_per_dim'], nh['encoder'].size // n_obs))
    shapes = nh['shapes']
    sizes = [p * q for p, q in shapes]
    dim = sum(sizes)
    split = lambda flat: [np.asarray(flat[sum(sizes[:k]):sum(sizes[:k + 1])]).reshape(shapes[k])
                          for k in range(len(shapes))]
    check_against_evaluate(snn, gym, nh, split, dim, a['bound'], a['seed'])
    checks = sorted({max(1, int(round(k * dim))) for k in a['per_weight']}) if a['per_weight'] else [a['budget']]
    budget = checks[-1]
    init = np.random.default_rng(a['seed']).uniform(-a['bound'], a['bound'], dim)   # shared start per seed
    param = ng.p.Array(init=init).set_bounds(-a['bound'], a['bound'])
    param.random_state = np.random.RandomState(a['seed'])
    opt = ng.optimizers.registry[a['optimizer']](parametrization=param, budget=budget)
    rng = np.random.default_rng(1000 + a['seed'])
    rr = np.random.default_rng(5_000_000 + a['seed'])
    hist, rows, t0 = [], [], time.time()
    for i in range(1, budget + 1):
        cand = opt.ask()
        flat = np.array(cand.value, copy=True)
        f = float(np.mean([snn.evaluate(nh, split(flat), episode_seed=s)
                           for s in rng.integers(0, 2**31, a['episodes'])]))
        opt.tell(cand, -f)
        if a['rescore_top']:
            hist.append((f, flat))
            hist = sorted(hist, key=lambda t: -t[0])[:a['rescore_top']]
        if i in checks:
            pick = np.asarray(opt.provide_recommendation().value)
            if a['rescore_top']:
                sc = [np.mean([snn.evaluate(nh, split(w), episode_seed=s)
                               for s in rr.integers(0, 2**31, 30)])
                      for _, w in hist]
                pick = hist[int(np.argmax(sc))][1]
            rep = [scored_episode(snn, gym, nh, split(pick), s) for s in rr.integers(0, 2**31, 100)]
            raw = float(np.mean([x[0] for x in rep]))
            drawn = (lambda x: x[2] + x[3]) if a['decoding'] == 'vote' else (lambda x: x[2])
            rows.append(dict(evals=i, per_weight=round(i / dim, 1), raw=raw,
                             passes=bool(raw >= THRESHOLD[a['env']]),
                             random_actions=float(np.mean([drawn(x) / x[4] for x in rep])),
                             silent=float(np.mean([x[2] / x[4] for x in rep])),
                             tied=float(np.mean([x[3] / x[4] for x in rep]))))
    return dict(a, dim=dim, rows=rows, minutes=(time.time() - t0) / 60)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='mountaincar', choices=list(THRESHOLD))
    ap.add_argument('--optimizers', nargs='+', default=['TwoPointsDE', 'CMA', 'RandomSearch'])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--per-weight', type=float, nargs='+', default=None,
                    help='checkpoints in evaluations per weight, e.g. 5 10 20 40 (last = run budget)')
    ap.add_argument('--budget', type=int, default=1000, help='absolute budget, if --per-weight is not given')
    ap.add_argument('--episodes', type=int, default=10, help='episodes averaged per candidate')
    ap.add_argument('--rescore-top', type=int, default=5, help='0 = report the optimizer\'s own recommendation')
    ap.add_argument('--hidden', default='6', help='e.g. 6 or 6x6')
    ap.add_argument('--encoding', default='poprate')
    ap.add_argument('--decoding', default='vote')
    ap.add_argument('--n-per-dim', type=int, default=None, help='default: 10 MountainCar, 4 Acrobot')
    ap.add_argument('--bound', type=float, default=60.0, help='weights searched in [-bound, bound]')
    ap.add_argument('--generator', default=None, choices=('poisson', 'regular'),
                    help='poprate spike generator (default: SPIKE_GENERATOR in snn_feedforward.py)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', default='calibration.json')
    args = ap.parse_args()
    n = args.n_per_dim or (10 if args.env == 'mountaincar' else 4)
    jobs = [dict(env=args.env, optimizer=o, seed=s, budget=args.budget, per_weight=args.per_weight,
                 episodes=args.episodes, rescore_top=args.rescore_top,
                 hidden=[int(h) for h in args.hidden.split('x')], encoding=args.encoding,
                 decoding=args.decoding, n_per_dim=n, bound=args.bound, generator=args.generator)
            for o in args.optimizers for s in range(args.seeds)]
    import concurrent.futures as cf, multiprocessing as mp
    results = []
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                mp_context=mp.get_context('spawn')) as ex:
        for r in ex.map(job, jobs):
            results.append(r)
            print('%-13s seed %d  %s  (%.0f min)' % (r['optimizer'], r['seed'], '  '.join(
                '%d evals: %.1f%s, %.0f%% random' % (x['evals'], x['raw'], ' PASS' if x['passes'] else '',
                                                     100 * x['random_actions']) for x in r['rows']),
                r['minutes']), flush=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    dim = results[0]['dim']
    evals = [x['evals'] for x in results[0]['rows']]
    print('\n%s (threshold %.0f), %s/%s%s, hidden %s, %d neurons per observation, %d weights, bound %.0f'
          % (args.env, THRESHOLD[args.env], args.encoding, args.decoding,
             ' (%s spikes)' % args.generator if args.generator and args.encoding == 'poprate' else '',
             args.hidden, n, dim, args.bound))
    print('seeds passed (median raw reward) at each budget')
    print('%-13s' % 'optimizer' + ''.join('%18s' % ('%d (%.0f/weight)' % (e, e / dim)) for e in evals))
    for o in args.optimizers:
        g = [r for r in results if r['optimizer'] == o]
        cells_ = []
        for j in range(len(evals)):
            raws = [r['rows'][j]['raw'] for r in g]
            cells_.append('%d/%d (%.0f)' % (sum(r['rows'][j]['passes'] for r in g), len(g), np.median(raws)))
        print('%-13s' % o + ''.join('%18s' % c for c in cells_))
