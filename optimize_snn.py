"""
Weight optimization for the feedforward SNN using Nevergrad.

How each optimizer's final network is chosen and scored (test plan items in brackets):

  The final network is the optimizer's own recommendation, scored on a block of
  REPORT_EPISODES fresh episodes (the report block): the recommendation rule is part of
  each optimizer, so the study reports it as is.
  [N1] Diagnostic only, off by default (N1_TOP = 0). With N1_TOP > 0, the N1_TOP best
       different weight vectors by training score (a vector evaluated more than once is
       ranked by the mean of its scores), plus the recommendation, are re-scored on
       N1_EPISODES fresh episodes (the selection block), and the best of them is
       reported instead. The recommendation's own scores stay in rec_*.
  [D1] Random actions. Every scored episode counts the steps whose output was silent or
       tied. A silent output always gives a random action; a tie does only under the vote
       decoder (a first-spike tie goes to the lowest index). Raw reward and the share of
       episodes that reach the goal are recorded too.
  [P2] Start check, MountainCar only: the first action from x = -0.58 and x = -0.42 at
       rest. 'random (silent)' or 'random (tie)' means the decoder would draw.
  [N3] The optimizer's recommendation after 20 x d evaluations (d = number of weights),
       scored on the report block too, so topologies can be compared at the same number
       of evaluations per weight. None when the budget is smaller. The run uses
       Nevergrad's ask/tell loop instead of minimize() to read the recommendation at that
       point; the loop makes exactly the same calls, and reading the recommendation
       doesn't change what the optimizer does next.

One inner run, shared with tune_hyperparams.py:
  make_objective()  the training objective: minus the mean fitness over EVAL_EPISODES
                    training episodes
  initial_point()   the trial's starting weights, uniform in [W_LOW, W_HIGH]
  train_weights()   make_parametrization() plus the ask/tell loop above, returning the
                    recommendation (and the recommendation at any checkpoints)
optimize() runs every optimizer through these, and tune_hyperparams.py runs its inner
optimizations through the same three, so a tuned setting is scored on exactly the search
the grid runs: same box, same sigma, same noise flag, same recommendation rule.

Episodes:
  training    unseeded random starts (unchanged). make_objective(train_seed=t), used by
              tune_hyperparams.py only, gives episode j of evaluation i the start
              TRAIN_SEED_BASE + t * 10**9 + i * n + j (n episodes per evaluation): fresh
              at every evaluation, but the same sequence for everything run with seed t
  selection   seeds (SELECT_SEED + trial) * 1000 + k   (N1_TOP > 0 only)
  report      seeds (REPORT_SEED + trial) * 1000 + k
Within a trial every candidate and every optimizer is scored on the same starts and the
same decoder draws (numpy's global generator is seeded per scored episode and restored
afterwards). The scored episodes run on their own environment, so their seeded resets
never touch the one the optimizers train on. At the start of every optimize() call the
scoring loop is checked against snn_feedforward.evaluate(), bit for bit.
"""

import numpy as np
import nevergrad as ng
import gymnasium as gym

import snn_feedforward as snn
from snn_feedforward import build_network, evaluate, weight_shapes


# ---------- optimization configuration ----------
OPTIMIZERS = ["NGOpt", "CMA", "DE", "PSO", "OnePlusOne", "TBPSA"]
BUDGET  = 200      # function evaluations per optimizer (main cost knob)
W_LOW  = -40.0     # weight search lower bound
W_HIGH =  40.0     # weight search upper bound
SEED    = 0        # trial t starts from default_rng(SEED + t), shared by its optimizers
REPORT_EPISODES = 100   # report block: episodes scored for the FINAL report only
                        # (not during optimization)
EVAL_EPISODES = 10     # number of episodes to measure weights

# ---------- final pick and extra logging (test plan N1, D1, P2, N3) ----------
N1_TOP        = 0      # 0 = report the optimizer's recommendation (the study's rule);
                       # > 0 = diagnostic: re-score that many best candidates as well
N1_EPISODES   = 100    # selection block: episodes per candidate (N1_TOP > 0 only)
N3_PER_WEIGHT = 20     # also score the recommendation after 20 x d evaluations
REPORT_SEED   = 7000   # report block seeds    (REPORT_SEED + trial) * 1000 + k
SELECT_SEED   = 8000   # selection block seeds (SELECT_SEED + trial) * 1000 + k
CHECK_SEED    = 9_000_000          # episodes used by check_scoring_loop()
TRAIN_SEED_BASE = 10 ** 12         # seeded training episodes (make_objective(train_seed=...),
                                   # tune_hyperparams.py only): far above every block above
START_CHECK_X = (-0.58, -0.42)     # P2 start positions, MountainCar at rest
MC_ACTIONS    = ('left', 'none', 'right')

# ---------- parametrization: what every optimizer is given ----------
# Nevergrad doesn't measure noise and doesn't know what a weight unit is: both come from
# the parametrization, and both change what several optimizers do (checked in Nevergrad
# 1.0.12 with identical losses):
#   PARAM_NOISY  True declares the objective noisy (param.function.deterministic = False)
#                for every optimizer. 'CMA' then runs OldCMA instead of the elitist
#                CMAbounded, 'NGOpt' switches to the chain NoisyRL3 runs, and OnePlusOne
#                and PSO recommend by Nevergrad's pessimistic estimate instead of the best
#                single score. TBPSA, DE, TwoPointsDE, RandomSearch and NoisyRL3 ignore it.
#   PARAM_SIGMA  the mutation step, in weight units. None keeps Nevergrad's default of
#                1.0; 'range/6' uses (W_HIGH - W_LOW) / 6, the step Nevergrad sets itself
#                when an Array gets its bounds in the constructor; a number sets it
#                directly. The steps of CMA, NGOpt, TBPSA, OnePlusOne and NoisyRL3, and
#                PSO's starting speeds, scale with it. DE, TwoPointsDE and RandomSearch
#                sample the whole box and step by differences, so they don't.
# The defaults reproduce the runs made before these settings existed.
PARAM_NOISY = True
PARAM_SIGMA = 'range/6'


# ---------- flatten / unflatten ----------
def n_params(shapes):
    """Total parameter count for a list of (pre, post) weight shapes."""
    return int(sum(p * q for p, q in shapes))


def flat_to_weights(flat, shapes):
    """Split a flat parameter vector into the list of weight matrices."""
    weights, idx = [], 0
    for (p, q) in shapes:
        size = p * q
        weights.append(np.asarray(flat[idx:idx + size],
                                  dtype=float).reshape(p, q))
        idx += size
    return weights


# ---------- scoring loop (report only, AFTER optimization) ----------
def _outcome(decoding, spikes, n_out):
    """'decisive', 'tie' or 'silent' for one 50 ms window. Draws no random numbers."""
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


def _decided_action(decoding, spikes, n_out):
    """The action the decoder takes without a random draw, or None if it would draw."""
    if decoding == 'vote':
        counts = [len(spikes.get(a, ())) for a in range(n_out)]
        top = max(counts)
        winners = [a for a in range(n_out) if counts[a] == top]
        return winners[0] if top > 0 and len(winners) == 1 else None
    best, best_t = None, None
    for a in range(n_out):
        times = spikes.get(a, ())
        if len(times):
            t = min(times)
            if best_t is None or t < best_t:      # strict <: lowest index wins ties
                best, best_t = a, t
    return best


def _scoring_env(net_handle):
    """A second environment for scored episodes. Seeded resets re-seed an environment's
    generator, so scoring on the training environment would make the unseeded training
    starts that follow repeat in every run."""
    env = net_handle.get('scoring_env')
    if env is None:
        env = net_handle['scoring_env'] = gym.make(net_handle['cfg']['gym_id'])
    return env


def scored_episode(net_handle, weights, episode_seed=None):
    """One episode through evaluate()'s own loop, counting what the decoder saw.

    The fitness is exactly what evaluate(net_handle, weights, episode_seed) returns
    (check_scoring_loop() verifies it): the counting only reads the spikes, and
    decode_action_fast() makes the same random draws. Returns a dict with the shaped
    fitness, the raw reward, whether the goal was reached, and the silent and tied steps.
    """
    snn.set_weights(net_handle, weights)
    cfg, net = net_handle['cfg'], net_handle['net']
    decoding = net_handle['decoding']
    n_out = int(cfg['n_salida'])
    spike_monitor = snn._fast(net_handle)['spike_monitor']
    env = _scoring_env(net_handle)
    if episode_seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=int(episode_seed))

    net.reset()
    paused = [m for m in net.get_monitors() if m is not spike_monitor]
    for m in paused:
        m.pause()
    total_reward = 0.0
    best_progress = cfg['progress'](obs)
    n_steps = silent = tied = 0
    done = goal = False
    try:
        while not done:
            snn.drive_encoder_fast(net_handle, obs)
            net.simulate(snn.T_STEP)
            spikes = spike_monitor.get('spike')
            outcome = _outcome(decoding, spikes, n_out)
            silent += outcome == 'silent'
            tied += outcome == 'tie'
            action = snn.decode_action_fast(decoding, spikes, n_out)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            best_progress = max(best_progress, cfg['progress'](obs))
            n_steps += 1
            goal = bool(terminated)
            done = terminated or truncated
            snn._reset_step(net_handle)
    finally:
        for m in paused:
            m.resume()
    return dict(fitness=total_reward + cfg['shaping'] * best_progress,
                raw=total_reward, goal=goal, silent=silent, tied=tied, steps=n_steps)


def _fresh_training_starts(net_handle):
    """evaluate() with a seed re-seeds the environment the optimizers train on. Re-seed
    it from the operating system, so the unseeded training starts stay random."""
    env = snn._fast(net_handle)['env']
    if env is not None:
        env.reset(seed=int(np.random.SeedSequence().generate_state(1)[0]))


def check_scoring_loop(net_handle, shapes, n_sets=3):
    """scored_episode() must return evaluate()'s fitness and raw reward bit for bit.

    Leaves numpy's global random state as it found it, so the optimization draws what it
    would have drawn, and gives the training environment fresh random starts afterwards.
    """
    state = np.random.get_state()
    rng = np.random.default_rng(CHECK_SEED)
    try:
        for i in range(n_sets):
            W = flat_to_weights(rng.uniform(W_LOW, W_HIGH, n_params(shapes)), shapes)
            np.random.seed(4242 + i)
            f_ref, r_ref = evaluate(net_handle, W, episode_seed=CHECK_SEED + i,
                                    return_raw=True)
            np.random.seed(4242 + i)
            ep = scored_episode(net_handle, W, CHECK_SEED + i)
            if (ep['fitness'], ep['raw']) != (f_ref, r_ref):
                raise RuntimeError(
                    "optimize_snn.scored_episode() no longer matches "
                    "snn_feedforward.evaluate(): fitness %r vs %r, raw %r vs %r. Update "
                    "scored_episode() to follow evaluate()'s loop."
                    % (ep['fitness'], f_ref, ep['raw'], r_ref))
    finally:
        np.random.set_state(state)
        _fresh_training_starts(net_handle)


def report_fitness(net_handle, weights, n_episodes=REPORT_EPISODES,
                   episode_seed=None):
    """Score fixed weights over n INDEPENDENT episodes -- report only.

    episode_seed controls reproducibility of the report episodes:
      - None (default): n fully random, independent episodes.
      - int: episode k uses seed (episode_seed*1000 + k) for its start, and numpy's
        global generator is seeded with the same number for the decoder's random draws
        (and restored afterwards). The n episodes are DISTINCT from one another (so
        variance is still measured), the whole block is REPEATABLE, and two networks
        scored on the same block see the same starts and draws.

    Returns mean, sample std (ddof=1) and standard error of the mean of the shaped
    fitness, the per-episode fitnesses and raw rewards, and:
      goal_share      share of episodes that reach the goal
      silent, tied    mean share of steps with a silent or a tied output
      random_actions  mean share of steps with a random action: silent + tied under
                      the vote decoder, silent only under first spike
    """
    n = int(n_episodes)
    if episode_seed is None:
        eps = [scored_episode(net_handle, weights) for _ in range(n)]
    else:
        base = int(episode_seed) * 1000  # spread blocks so they don't overlap
        state = np.random.get_state()
        try:
            eps = []
            for k in range(n):
                np.random.seed(base + k)
                eps.append(scored_episode(net_handle, weights, base + k))
        finally:
            np.random.set_state(state)
    fits    = np.array([e['fitness'] for e in eps], dtype=float)
    rewards = np.array([e['raw'] for e in eps], dtype=float)
    silent  = np.array([e['silent'] / e['steps'] for e in eps], dtype=float)
    tied    = np.array([e['tied'] / e['steps'] for e in eps], dtype=float)
    drawn   = silent + tied if net_handle['decoding'] == 'vote' else silent
    mean = float(fits.mean())
    std  = float(fits.std(ddof=1)) if n > 1 else 0.0
    sem  = std / np.sqrt(n) if n > 1 else 0.0
    return dict(mean=mean, std=std, sem=sem, n=n, fitnesses=fits,
                rewards=rewards, reward_mean=float(rewards.mean()),
                goal_share=float(np.mean([e['goal'] for e in eps])),
                random_actions=float(drawn.mean()),
                silent=float(silent.mean()), tied=float(tied.mean()))


def start_check(net_handle, weights):
    """[P2] MountainCar: the first action from each x in START_CHECK_X, at rest.

    Runs the first 50 ms window of an episode exactly as evaluate() does. Returns
    {'-0.58': 'right', ...}; 'random (silent)' or 'random (tie)' when the decoder
    would draw.
    """
    snn.set_weights(net_handle, weights)
    cfg, net = net_handle['cfg'], net_handle['net']
    decoding = net_handle['decoding']
    n_out = int(cfg['n_salida'])
    spike_monitor = snn._fast(net_handle)['spike_monitor']
    paused = [m for m in net.get_monitors() if m is not spike_monitor]
    for m in paused:
        m.pause()
    out = {}
    try:
        for x0 in START_CHECK_X:
            net.reset()
            snn.drive_encoder_fast(net_handle, np.array([x0, 0.0], dtype=np.float32))
            net.simulate(snn.T_STEP)
            spikes = spike_monitor.get('spike')
            a = _decided_action(decoding, spikes, n_out)
            out['%.2f' % x0] = (MC_ACTIONS[a] if a is not None
                                else 'random (%s)' % _outcome(decoding, spikes, n_out))
            snn._reset_step(net_handle)
    finally:
        for m in paused:
            m.resume()
    return out


# ---------- candidate tracking during the run ----------
class _Tracker:
    """Wraps the objective while the optimizer runs and records, for every DIFFERENT
    weight vector it evaluates, the mean of its training scores. train(x) gives that
    record for one vector (the recommendation's training score); best(k) hands back the
    k best different vectors for the N1 diagnostic, so the vectors themselves are kept
    only when `top` > 0.

    Ranking different vectors matters for optimizers that re-evaluate a point, such as
    NoisyRL3, whose noisy (1+1) half scores its current best again and again: ranked
    by single evaluations, its top 5 were usually five copies of one vector."""

    def __init__(self, objective, top):
        self.objective, self.top = objective, int(top)
        self.n = 0
        self.seen = {}      # vector bytes -> [sum of training fitness, evaluations,
                            #                  first evaluation number, vector or None]

    def __call__(self, flat):
        loss = self.objective(flat)
        self.n += 1
        x = np.array(flat, dtype=float, copy=True)
        key = x.tobytes()
        entry = self.seen.get(key)
        if entry is None:
            self.seen[key] = [-float(loss), 1, self.n, x if self.top > 0 else None]
        else:
            entry[0] -= float(loss)
            entry[1] += 1
        return loss

    def train(self, x):
        """(mean training fitness, evaluations) of vector x; (None, 0) if never evaluated."""
        entry = self.seen.get(np.asarray(x, dtype=float).tobytes())
        return (entry[0] / entry[1], entry[1]) if entry else (None, 0)

    def best(self, k):
        """The k best different vectors: (mean training fitness, evaluations, vector)."""
        ranked = sorted(self.seen.values(), key=lambda e: (-e[0] / e[1], e[2]))
        return [(e[0] / e[1], e[1], e[3]) for e in ranked[:k]]


def _candidates(rec_flat, tracker):
    """[N1] The best different vectors by mean training score, then the recommendation
    unless it is one of them. Returns a list of [label, vector, mean training fitness
    (None if the recommendation was never evaluated), training evaluations]."""
    cands, index = [], {}
    for train, count, x in tracker.best(tracker.top):
        index[x.tobytes()] = len(cands)
        cands.append(['top%d' % (len(cands) + 1), x, train, count])
    key = rec_flat.tobytes()
    if key in index:
        cands[index[key]][0] += '=rec'
    else:
        cands.append(['rec', rec_flat] + list(tracker.train(rec_flat)))
    return cands


# ---------- parametrization ----------
def param_sigma():
    """PARAM_SIGMA in weight units, or None for Nevergrad's default (1.0)."""
    if PARAM_SIGMA is None:
        return None
    if PARAM_SIGMA == 'range/6':
        return (W_HIGH - W_LOW) / 6.0
    return float(PARAM_SIGMA)


def make_parametrization(name, init, trial):
    """The search space optimizer `name` is given: the box [W_LOW, W_HIGH] starting at
    init, with PARAM_SIGMA's step and PARAM_NOISY's declaration. Returns (param, info),
    info being what the results record.

    With the defaults this is exactly what optimize() built before. The bounds stay in
    set_bounds(): ng.p.Array(init=..., lower=..., upper=...) would give the same sigma
    as 'range/6' but switch off the box-wide sampling DE, PSO and RandomSearch start
    from when an init is given.
    """
    sigma = param_sigma()
    param = ng.p.Array(init=init.copy())
    if sigma is not None:
        param.set_mutation(sigma=sigma)   # before set_bounds, which checks 3 sigma fit
    param.set_bounds(W_LOW, W_HIGH)
    noisy = bool(PARAM_NOISY)
    param.function.deterministic = False
    param.real_world = True
    param.neural = True    
    param.random_state = np.random.RandomState(SEED + int(trial))
    info = dict(noisy=noisy, sigma=float(np.ravel(param.sigma.value)[0]),
                sigma_rule=('nevergrad default' if PARAM_SIGMA is None
                            else str(PARAM_SIGMA)))
    return param, info


def sub_optim(optimizer):
    """What a selector such as CMA, NGOpt or NgIohTuned actually ran; None for the
    others. Read after the run, so asking changes nothing."""
    try:
        return optimizer._info().get('sub-optim')
    except Exception:
        return None


# ---------- the inner run, shared with tune_hyperparams.py ----------
def training_seed(train_seed, evaluation, j, n_episodes):
    """Episode seed of training episode j of evaluation `evaluation` (both from 0) in a
    run seeded with train_seed. See make_objective()."""
    offset = int(evaluation) * int(n_episodes) + int(j)
    if offset >= 10 ** 9:
        raise ValueError("more than 10**9 seeded training episodes in one run")
    return TRAIN_SEED_BASE + int(train_seed) * 10 ** 9 + offset


def make_objective(net, shapes, n_episodes=None, train_seed=None):
    """The objective every optimizer minimizes: minus the mean fitness of a weight
    vector over n_episodes training episodes (EVAL_EPISODES when None).

    train_seed None: unseeded random starts at every evaluation, the grid's regime.
    train_seed int:  evaluation i (counted by this objective, from 0) runs the episodes
                     training_seed(train_seed, i, j, n_episodes) for j < n_episodes.
                     The starts are fresh at every evaluation, as in the grid, but
                     anything run with the same train_seed sees the same starts at the
                     same evaluation, so two tuning configs are compared on the same
                     draws. TRAIN_SEED_BASE keeps them clear of every scored block.
    The decoder's random draws come from numpy's global generator in both cases.
    """
    n = EVAL_EPISODES if n_episodes is None else int(n_episodes)
    if n < 1:
        raise ValueError("n_episodes must be more than 0")
    calls = [0]

    def objective(flat):
        weights = flat_to_weights(flat, shapes)
        fitness = 0
        for j in range(n):
            if train_seed is None:
                fitness = fitness + -float(evaluate(net, weights))
            else:
                seed = training_seed(train_seed, calls[0], j, n)
                fitness = fitness + -float(evaluate(net, weights, episode_seed=seed))
        calls[0] += 1
        return fitness / n

    return objective


def initial_point(dim, trial):
    """The starting weights every optimizer of trial `trial` gets: one uniform draw in
    [W_LOW, W_HIGH] from default_rng(SEED + trial)."""
    rng = np.random.default_rng(SEED + int(trial))
    return rng.uniform(W_LOW, W_HIGH, size=int(dim))


def train_weights(name, init, trial, budget, objective, checkpoints=(), top=0):
    """Run optimizer `name` on the weights: make_parametrization(name, init, trial), then
    Nevergrad's ask/tell loop for its budget -- what optimizer.minimize(objective) does,
    written out so the recommendation can be read after the evaluation counts in
    `checkpoints` (reading it doesn't change what the optimizer does next).

    Returns a dict:
      rec          the final recommendation (flat vector)
      rec_train    (mean training fitness, evaluations) of rec; (None, 0) if never evaluated
      checkpoints  {count: (recommendation after `count` evaluations, its rec_train)},
                   for every count the budget reaches
      tracker      the _Tracker (keeps the `top` best vectors for the N1 diagnostic)
      param        make_parametrization()'s info: sigma, noise declared or not
      sub_optim    what a selector (CMA, NGOpt, ...) ran
      evaluations  objective calls made
    """
    param, param_info = make_parametrization(name, init, trial)
    opt_cls = ng.optimizers.registry[name]
    optimizer = opt_cls(parametrization=param, budget=budget)

    tracker = _Tracker(objective, top)
    wanted = set(int(c) for c in checkpoints)
    snaps = {}
    for i in range(optimizer.budget):
        cand = optimizer.ask()
        optimizer.tell(cand, tracker(*cand.args, **cand.kwargs))
        if i + 1 in wanted:
            x = np.array(optimizer.provide_recommendation().value, dtype=float,
                         copy=True)
            snaps[i + 1] = (x, tracker.train(x))
    rec_flat = np.array(optimizer.provide_recommendation().value, dtype=float,
                        copy=True)
    rec_train = tracker.train(rec_flat)
    ran = sub_optim(optimizer)
    return dict(rec=rec_flat, rec_train=rec_train, checkpoints=snaps, tracker=tracker,
                param=param_info, sub_optim=ran, evaluations=tracker.n)


# ---------- run multiple optimizers ----------
def optimize(environment, hidden_sizes, encoding, decoding,
             optimizers=OPTIMIZERS, budget=BUDGET, trial=0, n_per_dim=None):
    """Compile the SNN once, then run each optimizer on its weights.

    Returns {optimizer_name: dict}. The final network is the optimizer's recommendation
    (the N1 pick only when N1_TOP > 0), scored on the report block:
      fitness, std, sem, n_episodes, fitnesses   shaped fitness (mean over REPORT_EPISODES)
      rewards, reward_mean                       raw reward
      goal_share, random_actions, silent, tied   [D1]
      start_check                                [P2] MountainCar only, else None
      rec_train, rec_train_evals                 the recommendation's mean training score
                                                 and how often it was evaluated (None, 0
                                                 for a point never evaluated, e.g. CMA's mean)
      n3                                         [N3] dict, or None if budget < 20 x d
      opt_loss                                   the recommendation's loss on
                                                 EVAL_EPISODES more training episodes
      pick, selection, rec_fitness, rec_raw,     [N1] diagnostic; with N1_TOP = 0, pick is
      rec_goal_share, rec_random_actions         'rec', selection is empty and rec_* repeat
                                                 the values above
      param                                      the parametrization: sigma in weight
                                                 units, noise declared or not
      sub_optim                                  what a selector (CMA, NGOpt, ...) ran
      evaluations, weights
    """
    if EVAL_EPISODES < 1:
            raise ValueError("EVAL_EPISODES must be more than 0")
    # one compile for the whole comparison; dir is unique per configuration
    compile_dir = ("annarchy/opt-%s-%s-%s-%s-t%d"
                   % (environment, encoding, decoding,
                      "x".join(str(h) for h in hidden_sizes), trial))
    if n_per_dim is not None:
        compile_dir += "-n%d" % int(n_per_dim)
    net = build_network(environment, hidden_sizes,
                         encoding, decoding, compile_dir, n_per_dim=n_per_dim)

    shapes = net['shapes']
    dim = n_params(shapes)
    check_scoring_loop(net, shapes)    # the report must score what evaluate() scores

    # EVAL_EPISODES unseeded training episodes per evaluation (only proj.w updates)
    objective = make_objective(net, shapes)

    # one random start per trial, shared by every optimizer of that trial
    init = initial_point(dim, trial)

    n3_at = int(round(N3_PER_WEIGHT * dim))
    select_seed = SELECT_SEED + int(trial)
    report_seed = REPORT_SEED + int(trial)

    results = {}
    for name in optimizers:
        # the shared inner run (also tune_hyperparams.py's): make_parametrization() and
        # the ask/tell loop, with the recommendation read after n3_at evaluations [N3]
        run = train_weights(name, init, trial, budget, objective,
                            checkpoints=(n3_at,), top=N1_TOP)
        rec_flat, rec_train, tracker = run['rec'], run['rec_train'], run['tracker']
        n3_flat, n3_train = run['checkpoints'].get(n3_at, (None, (None, 0)))
        param_info, ran = run['param'], run['sub_optim']

        opt_loss = objective(rec_flat)

        # [N1] diagnostic: re-score the candidates on the selection block, keep the best
        cands = (_candidates(rec_flat, tracker) if N1_TOP > 0
                 else [['rec', rec_flat] + list(rec_train)])
        selection = []
        if len(cands) > 1:
            for label, x, train, count in cands:
                s = report_fitness(net, flat_to_weights(x, shapes), N1_EPISODES,
                                   select_seed)
                selection.append(dict(candidate=label, train=train, train_evals=count,
                                      select=s['mean'], select_raw=s['reward_mean']))
            k = int(np.argmax([s['select'] for s in selection]))
        else:
            k = 0
        pick, pick_flat = cands[k][0], cands[k][1]
        best_weights = flat_to_weights(pick_flat, shapes)

        # report block: the final network, the recommendation, the N3 recommendation
        scored = {}

        def rescore(x):
            key = x.tobytes()
            if key not in scored:
                scored[key] = report_fitness(net, flat_to_weights(x, shapes),
                                             REPORT_EPISODES, report_seed)
            return scored[key]

        rep = rescore(pick_flat)
        rec = rescore(rec_flat)
        n3 = None
        if n3_flat is not None:
            r3 = rescore(n3_flat)
            n3 = dict(evals=n3_at, per_weight=N3_PER_WEIGHT,
                      train=n3_train[0], train_evals=n3_train[1],
                      fitness=r3['mean'], raw=r3['reward_mean'],
                      goal_share=r3['goal_share'],
                      random_actions=r3['random_actions'])
        start = (start_check(net, best_weights) if environment == 'mountaincar'
                 else None)

        results[name] = dict(fitness=rep['mean'], std=rep['std'], sem=rep['sem'],
                             n_episodes=rep['n'], fitnesses=rep['fitnesses'],
                             rewards=rep['rewards'],
                             reward_mean=rep['reward_mean'],
                             goal_share=rep['goal_share'],
                             random_actions=rep['random_actions'],
                             silent=rep['silent'], tied=rep['tied'],
                             start_check=start,
                             rec_train=rec_train[0], rec_train_evals=rec_train[1],
                             pick=pick, selection=selection,
                             rec_fitness=rec['mean'], rec_raw=rec['reward_mean'],
                             rec_goal_share=rec['goal_share'],
                             rec_random_actions=rec['random_actions'],
                             n3=n3, evaluations=tracker.n,
                             opt_loss=opt_loss, param=param_info, sub_optim=ran,
                             weights=best_weights)

        print("%-12s budget=%d  mean=%.3f  std=%.3f  sem=%.3f  raw=%.2f  "
              "(n=%d, opt=%.3f)"
              % (name, budget, rep['mean'], rep['std'], rep['sem'],
                 rep['reward_mean'], rep['n'], -opt_loss), flush=True)
        if decoding == 'vote':
            drawn = ("random %.1f%% (silent %.1f%%, tied %.1f%%)"
                     % (100 * rep['random_actions'], 100 * rep['silent'],
                        100 * rep['tied']))
        else:
            drawn = ("random %.1f%% (silent; first-spike ties %.1f%% go to the "
                     "lowest index)" % (100 * rep['random_actions'], 100 * rep['tied']))
        line = "%-12s goal %.0f%%  %s" % ('', 100 * rep['goal_share'], drawn)
        if N1_TOP > 0:
            line += ("  |  pick %s, recommendation: mean %.3f raw %.2f"
                     % (pick, rec['mean'], rec['reward_mean']))
        print(line, flush=True)
        if n3 is not None:
            extra = ["recommendation after %d evaluations: mean %.3f raw %.2f"
                     % (n3_at, n3['fitness'], n3['raw'])]
        else:
            extra = ["budget below %g x %d = %d evaluations, no N3 score"
                     % (N3_PER_WEIGHT, dim, n3_at)]
        if start is not None:
            extra.append("start: " + ", ".join("%s %s" % kv for kv in start.items()))
        extra.append("sigma %.3g%s%s" % (param_info['sigma'],
                                          ", declared noisy" if param_info['noisy'] else "",
                                          ", ran %s" % ran if ran else ""))
        print("%-12s %s" % ('', "  |  ".join(extra)), flush=True)

    return results


if __name__ == "__main__":
    # one experiment cell: environment x topology x encoding x decoding.
    # encoding options:
    #   single-neuron : "rate" | "latency"
    #   population    : "poprate" | "poplatency"
    results = optimize(
        environment="mountaincar",   # or "acrobot", "mountaincar"
        hidden_sizes=[6],           # [20], [20,20], [20,10]
        encoding="poprate",          # "poplatency" or "poprate"
        decoding="vote",             # "vote" or "first"
        optimizers=["DE"],
        budget=1000,
    )
    best = max(results, key=lambda k: results[k]["fitness"])
    print("\nbest optimizer:", best,
          "-> fitness", round(results[best]["fitness"], 3))
    # results[best]["weights"] is the list of trained weight matrices.
