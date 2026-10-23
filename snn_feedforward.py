"""
Feedforward Izhikevich SNN for Gymnasium control tasks (ANNarchy 5.0 API).
"""

import numpy as np
import gymnasium as gym
import ANNarchy as ann


# ---------- shared encoder/decoder configuration ----------
T_STEP   = 50.0    # ms simulated per environment step

R_MAX = 300.0      # rate coding: Hz max
R_MIN = 100.0      # rate coding: Hz floor


LAT_WINDOW = 30.0  # latency coding: ms latency for normalized value 1.0
LAT_MIN    = 1.0   # latency coding: ms latency for normalized value 0.0


NEURONS_PER_DIM = 4 # neurons x dim for pop encoding (default; build_network(n_per_dim=...)
                    # sets it per network, and every later step reads it from the network)

# How poprate's active bin turns its rate into spikes (default; build_network(spike_generator=...)
# sets it per network):
#   'poisson' -> PoissonPopulation, a random draw at every time step. Every window starts with
#                a reset that re-seeds the generator, so one rate always gives the same
#                irregular train, and each input neuron has its own uneven rate-to-count map.
#   'regular' -> SpikeSourceArray with evenly spaced spikes at the rate: one spike in the middle
#                of each inter-spike interval, (j + 0.5) * 1000 / rate ms, so round(rate * T_STEP
#                / 1000) spikes per window, the same for every neuron. Rates below
#                500 / T_STEP Hz (10 Hz at 50 ms) give no spike.
SPIKE_GENERATOR  = 'poisson'
SPIKE_GENERATORS = ('poisson', 'regular')

# POP_ENCODINGS share the population (one-hot place-code) input layer of width
# n_entrada*NEURONS_PER_DIM. They differ ONLY in how the active bin is activated:
#   poprate    -> Poisson firing at R_MAX across the window (closest to popcurrent)
#   poplatency -> a single early spike at LAT_MIN (weakest; EPSP may decay early)
ENCODINGS     = ('poprate', 'poplatency')
DECODINGS     = ('vote', 'first')


# ---------- per-environment configuration ----------
def _mc_progress(obs):
    """MountainCar progress: normalized total mechanical energy.

    Replaces raw position. Car height is sin(3*x) (gym's own _height shape), so
    potential energy ~ sin(3*x) and kinetic energy ~ v**2. Each is normalized to
    ~[0,1] over its reachable range and summed, so neither term dominates.

    """

    x, v = float(obs[0]), float(obs[1])
    height_norm = (np.sin(3.0 * x) + 1.0) / 2.0   # sin(3x) in [-1,1] -> [0,1]
    #speed_norm  = abs(v) / 0.07                    # |v|, bound 0.07 -> [0,1]
    return height_norm #+ speed_norm


def _acro_progress(obs):
    """Acrobot progress: height of the free tip = -cos(t1) - cos(t1+t2)."""
    c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
    cos_12 = c1 * c2 - s1 * s2
    return float(-c1 - cos_12)

ENVIRONMENTS = {
    'mountaincar': dict(
        gym_id='MountainCar-v0',
        obs_low =np.array([-1.2, -0.07]),
        obs_high=np.array([ 0.6,  0.07]),
        n_entrada=2, n_salida=3,
        progress=_mc_progress, shaping=50.0,
    ),
    'acrobot': dict(
        gym_id='Acrobot-v1',
        obs_low =np.array([-1., -1., -1., -1., -4 * np.pi, -9 * np.pi]),
        obs_high=np.array([ 1.,  1.,  1.,  1.,  4 * np.pi,  9 * np.pi]),
        # tanh-squash the two velocity dims only (np.nan = linear). scale =
        # bound/2.6 so +-4pi / +-9pi land near +-0.99 instead of saturating
        # flat; the four cos/sin dims are already in [-1,1] and well-spread, so
        # they stay linear.
        squash=np.array([np.nan, np.nan, np.nan, np.nan,
                         4 * np.pi / 2.6, 9 * np.pi / 2.6]),
        n_entrada=6, n_salida=3,
        progress=_acro_progress, shaping=10.0,
    ),
}


def _normalize(obs, obs_low, obs_high, squash=None):
    """Observation -> per-dimension value in [0, 1] for the given bounds.

    If `squash` is given (per-dim array; np.nan = linear, positive = tanh scale
    s), the flagged dims use (tanh(obs/s)+1)/2 instead of the linear map. This
    keeps rarely-hit extremes (e.g. Acrobot's +-4pi/+-9pi velocities) from
    compressing every value toward 0.5. Shared by BOTH encoders, so the
    encoding comparison stays fair.
    """
    obs = np.asarray(obs, dtype=float)
    norm = (obs - obs_low) / (obs_high - obs_low)          # linear path
    if squash is not None:
        s = np.asarray(squash, dtype=float)
        mask = np.isfinite(s) & (s > 0)                    # only the tanh dims
        norm[mask] = (np.tanh(obs[mask] / s[mask]) + 1.0) / 2.0
    return np.clip(norm, 0.0, 1.0)


# ================= ENCODERS (input) =================

def _bin_and_residual(u, n_per_dim):
    """Active bin index k and within-bin position r in [0, 1]."""
    n = int(n_per_dim)
    k = min(int(u * n), n - 1)
    return k, float(np.clip(u * n - k, 0.0, 1.0))


def _grade(u, r, mode):
    """Scalar in [0,1] driving the active neuron's intensity."""
    if mode == 'residual':
        return r          # coarse-fine: bin = region, intensity = position in it
    return 1.0            # 'none': original hard-on behaviour


def encode_poprate(obs, obs_low, obs_high, squash=None,
                   n_per_dim=NEURONS_PER_DIM, rate_min=R_MIN,
                   rate_max=R_MAX, rate_off=0.0, graded='residual'):
    """
    Returns a per-input-neuron rate array of length len(obs)*n_per_dim
    """
    norm = _normalize(obs, obs_low, obs_high, squash)
    n = int(n_per_dim)
    rates = np.full(len(norm) * n, float(rate_off), dtype=float)
    for d, u in enumerate(norm):
        k, r = _bin_and_residual(u, n)
        rates[d * n + k] = rate_min + (rate_max - rate_min) * _grade(u, r, graded)
    return rates



def encode_poplatency(obs, t_offset, obs_low, obs_high, squash=None,
                      n_per_dim=NEURONS_PER_DIM,
                      lat_min=LAT_MIN, lat_window=LAT_WINDOW,
                      graded='residual'):
    """Population latency coding: active bin spikes at a graded latency.

    Latency runs from lat_window (r=0) down to lat_min (r=1), same mapping as
    encode_latency. Every spike carries t_offset -- do not drop it on the
    burst offsets.
    """
    norm = _normalize(obs, obs_low, obs_high, squash)
    n = int(n_per_dim)
    spike_times = [[] for _ in range(len(norm) * n)]
    for d, u in enumerate(norm):
        k, r = _bin_and_residual(u, n)
        lat = lat_window - _grade(u, r, graded) * (lat_window - lat_min)
        spike_times[d * n + k] = [float(t_offset + lat)]
    return spike_times


def _regular_train(rate):
    """Evenly spaced spike times (ms from the start of the window) at `rate` Hz.

    One spike in the middle of each inter-spike interval: (j + 0.5) * 1000 / rate for
    j = 0, 1, ... while inside the T_STEP window, i.e. round(rate * T_STEP / 1000) spikes.
    """
    if rate <= 0.0:
        return []
    isi = 1000.0 / rate
    times = []
    j = 0
    while (j + 0.5) * isi < T_STEP:
        times.append((j + 0.5) * isi)
        j += 1
    return times


def encode_poprate_regular(obs, t_offset, obs_low, obs_high, squash=None,
                           n_per_dim=NEURONS_PER_DIM, rate_min=None,
                           rate_max=None, graded='residual'):
    """encode_poprate() with regular spike trains instead of Poisson rates.

    The active bin gets the same rate as in encode_poprate() and fires the evenly spaced
    train of _regular_train(); every other input neuron stays silent. Returns spike times
    in ms (one list per input neuron), shifted by t_offset. rate_min / rate_max default
    to R_MIN / R_MAX as they are when the function is called.
    """
    rate_min = R_MIN if rate_min is None else rate_min
    rate_max = R_MAX if rate_max is None else rate_max
    norm = _normalize(obs, obs_low, obs_high, squash)
    n = int(n_per_dim)
    spike_times = [[] for _ in range(len(norm) * n)]
    for d, u in enumerate(norm):
        k, r = _bin_and_residual(u, n)
        rate = rate_min + (rate_max - rate_min) * _grade(u, r, graded)
        spike_times[d * n + k] = [t_offset + t for t in _regular_train(rate)]
    return spike_times


def drive_encoder(encoding, encoder, obs, obs_low, obs_high, t_offset=0.0,
                  squash=None):
    """Push one observation into the input layer for the upcoming step.

    A SpikeSourceArray input (poplatency, or poprate with regular spikes) is reset after
    its new pattern is assigned, so the spikes start in this window (F1). ANNarchy's
    reset() makes spike times relative to the current time, so for those inputs the
    times are computed from 0 and t_offset is not added.
    """
    if encoding == 'poprate' and isinstance(encoder, ann.SpikeSourceArray):
        encoder.spike_times = encode_poprate_regular(obs, 0.0, obs_low, obs_high, squash,
                                                     n_per_dim=encoder.size // len(obs))
        encoder.reset()
    elif encoding == 'poprate':
        encoder.rates = encode_poprate(obs, obs_low, obs_high, squash,
                                       n_per_dim=encoder.size // len(obs))
    elif encoding == 'poplatency':
        encoder.spike_times = encode_poplatency(obs, 0.0, obs_low,
                                                obs_high, squash,
                                                n_per_dim=encoder.size // len(obs))
        encoder.reset()
    else:
        raise ValueError("encoding must be one of %s" % (ENCODINGS,))


# ================= DECODERS (output / action) =================
def decode_vote(spikes, n_salida):
    """Spike voting: action = output neuron with the most spikes this window.

    Ties resolve to random index.
    """
    counts = [len(spikes.get(a, [])) for a in range(int(n_salida))]
    if max(counts) == 0:
        return int(np.random.randint(int(n_salida)))   # silent -> random
    max_val = np.max(counts)
    winners = np.where(counts == max_val)[0]
    return int(np.random.choice(winners))


def decode_first_spike(spikes, n_salida): #this
    """First spike: action = output neuron whose earliest spike comes soonest.

    If the output is SILENT (no neuron spikes), a uniformly random action is
    taken -- intentional stochasticity for the noisy-objective comparison.
    """
    first = []
    for a in range(int(n_salida)):
        times = spikes.get(a, [])
        first.append(min(times) if len(times) else np.inf)
    if all(np.isinf(t) for t in first):
        return int(np.random.randint(int(n_salida)))   # silent -> random
    return int(np.argmin(first))


def decode_action(decoding, spikes, n_salida):
    """Single dispatch point for action selection."""
    if decoding == 'vote':
        return decode_vote(spikes, n_salida)
    elif decoding == 'first':
        return decode_first_spike(spikes, n_salida)
    else:
        raise ValueError("decoding must be 'vote' or 'first'")


# ================= NETWORK: BUILD ONCE =================
def weight_shapes(environment, hidden_sizes, encoding='poprate', n_per_dim=None):
    """Expected (pre, post) shape of each weight matrix. Use to size the genome.

    For the POPULATION encodings (poprate / poplatency) the input
    layer is place-coded, so it has n_entrada*n_per_dim neurons (not
    n_entrada). n_per_dim=None means NEURONS_PER_DIM.
    """
    cfg = ENVIRONMENTS[environment]
    n = NEURONS_PER_DIM if n_per_dim is None else int(n_per_dim)
    n_in = cfg['n_entrada'] * n
    sizes = [n_in] + [int(h) for h in hidden_sizes] + [cfg['n_salida']]
    return [(sizes[k], sizes[k + 1]) for k in range(len(sizes) - 1)]


def build_network(environment, hidden_sizes, encoding, decoding, compile_dir
                  , n_per_dim=None, spike_generator='regular'):
    """Build and COMPILE the network once (ANNarchy 5.0 Network instance).

    Projections start with dense (all-to-all) zero-weight connectivity;
    real weight values are written later by set_weights().
    spike_generator ('poisson' or 'regular', None = SPIKE_GENERATOR) only affects
    poprate. A 'regular' network has a different input population, so compile it in
    its own directory.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError("environment must be one of %s" % list(ENVIRONMENTS))
    if encoding not in ENCODINGS:
        raise ValueError("encoding must be one of %s" % (ENCODINGS,))
    if decoding not in DECODINGS:
        raise ValueError("decoding must be one of %s" % (DECODINGS,))
    generator = SPIKE_GENERATOR if spike_generator is None else spike_generator
    if generator not in SPIKE_GENERATORS:
        raise ValueError("spike_generator must be one of %s" % (SPIKE_GENERATORS,))

    cfg = ENVIRONMENTS[environment]
    n_entrada, n_salida = cfg['n_entrada'], cfg['n_salida']

    # each call gets its own Network -- no global state, no clear() needed
    net = ann.Network(seed=42)

    # input layer. Population encodings widen it to the place-code width
    # n_entrada*n_per_dim (n_per_dim=None means NEURONS_PER_DIM)
    n = NEURONS_PER_DIM if n_per_dim is None else int(n_per_dim)
    if n < 1:
        raise ValueError("n_per_dim must be 1 or more, got %r" % (n_per_dim,))
    n_in = n_entrada * n
    if encoding == 'poprate' and generator == 'poisson':
        encoder = net.create(
            ann.PoissonPopulation(geometry=n_in, rates=0.0))
    else:                       # poplatency, or poprate with regular spikes
        encoder = net.create(
            ann.SpikeSourceArray(spike_times=[[] for _ in range(n_in)]))

    # hidden + output
    hidden = [net.create(geometry=int(h), neuron=ann.Izhikevich)
              for h in hidden_sizes]
    output = net.create(geometry=n_salida, neuron=ann.Izhikevich)
    layers = [encoder] + hidden + [output]

    # feedforward projections: layer k -> layer k+1
    projections = []
    for k in range(len(layers) - 1):
        pre_n, post_n = layers[k].size, layers[k + 1].size
        proj = net.connect(pre=layers[k], post=layers[k + 1], target='exc')
        proj.from_matrix(np.zeros((post_n, pre_n)))   # (post, pre)
        projections.append(proj)

    monitor = net.monitor(output, ['spike', 'v'])
    net.compile(directory=compile_dir, clean=False, silent=True)

    return dict(net=net, encoder=encoder, output=output,

                monitor=monitor,
                projections=projections,
                shapes=weight_shapes(environment, hidden_sizes, encoding, n),
                n_per_dim=n,
                spike_generator=generator if encoding == 'poprate' else None,
                cfg=cfg, encoding=encoding, decoding=decoding)

def set_weights(net_handle, weights, _verified=[False]):
    """Write a list of weight matrices into the compiled projections.

    weights[k] has shape (pre, post). Updates happen in place, no recompile.
    """
    if len(weights) != len(net_handle['projections']):
        raise ValueError("expected %d weight matrices, got %d"
                          % (len(net_handle['projections']), len(weights)))
    for k, proj in enumerate(net_handle['projections']):
        W = np.asarray(weights[k], dtype=float)
        if W.shape != net_handle['shapes'][k]:
            raise ValueError("weights[%d] shape %s, expected %s"
                             % (k, W.shape, net_handle['shapes'][k]))
        # dendrite r collects synapses INTO post-neuron r, indexed by pre rank
        for post_rank in range(W.shape[1]):
            proj.dendrite(post_rank).w = W[:, post_rank]

"""
# ================= NETWORK: EVALUATE MANY =================
def evaluate(net_handle, weights, episode_seed=None):
    ""Set weights, run ONE episode, return scalar fitness.

    Fitness = total_reward.

    episode_seed controls the Gymnasium reset:
      - None (default): env.reset() with NO seed -> a fully random, independent
        episode every call.
      - int: env.reset(seed=episode_seed) -> a repeatable episode.
    ""
    set_weights(net_handle, weights)
    cfg = net_handle['cfg']
    net = net_handle['net']
    encoder = net_handle['encoder']
    monitor = net_handle['monitor']

    env = gym.make(cfg['gym_id'])
    if episode_seed is None:
        obs, _ = env.reset()                  # fully random, independent episode
    else:
        obs, _ = env.reset(seed=int(episode_seed))   # repeatable episode
    net.reset()
    total_reward = 0.0
    best_progress = cfg['progress'](obs)
    n_steps = 0
    done = False
    while not done:
        drive_encoder(net_handle['encoding'], encoder, obs,
                      cfg['obs_low'], cfg['obs_high'],
                      t_offset=net.time, squash=cfg.get('squash'))

        net.simulate(T_STEP)

        spikes = monitor.get('spike')

        action = decode_action(net_handle['decoding'], spikes,
                               cfg['n_salida'])

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        best_progress = max(best_progress, cfg['progress'](obs))
        n_steps += 1
        done = terminated or truncated
        net.reset()  # clear neuron state (weights preserved)

    env.close()
    fitness = total_reward + cfg['shaping'] * best_progress
    return (fitness, total_reward) if return_raw else fitness
"""
# ================= FASTER EPISODE LOOP (changes 3, 5, 6, 7, 8) =================
# evaluate() below returns what evaluate_reference() returns: same weights, same
# episode seed, same numpy seed -> the same fitness, bit for bit.
#
#   [3] reset each population's VARIABLES, not its constant parameters
#   [5] record the output spikes and nothing else while an episode runs
#   [6] encoder arithmetic in plain Python floats
#   [7] decoder in plain Python (same random draws)
#   [8] one Gym environment per network instead of gym.make() per episode
#
# Only documented ANNarchy calls are used -- no cyInstance, no net.instance:
#   Population.variables, Population.reset(attributes=...), attribute assignment
#   (encoder.rates = ..., encoder.spike_times = ...), Network.monitor(),
#   Monitor.get() / pause() / resume(), Network.reset(populations=, monitors=).

# Encodings drive_encoder_fast() computes itself; anything else goes through
# drive_encoder() exactly as before. Removing an entry sends that encoding back
# to the original path. NOTE: for 'poplatency' the fast path emits the pattern
# of the CURRENT observation (the original loop emits the previous one).
FAST_ENCODINGS = ('rate', 'poprate', 'poplatency')


def _fast(net_handle):
    """Per-network constants for evaluate(): built on first use, cached in the handle."""
    fast = net_handle.get('fast')
    if fast is not None:
        return fast
    net, encoder = net_handle['net'], net_handle['encoder']
    cfg = net_handle['cfg']
    low = np.asarray(cfg['obs_low'], dtype=float)
    span = np.asarray(cfg['obs_high'], dtype=float) - low
    fast = dict(
        # [3] populations with dynamics (everything except the input layer) and
        #     the names of their variables
        state=[(p, list(p.variables)) for p in net.get_populations() if p is not encoder],
        # [6] is this encoding on the fast path? (decided once per network)
        #     A SpikeSourceArray on the fast path is reset when its pattern is
        #     assigned, so it needs no reset at the end of the step.
        fast_path=net_handle['encoding'] in FAST_ENCODINGS,
        input_resets_itself=(net_handle['encoding'] in FAST_ENCODINGS
                             and isinstance(encoder, ann.SpikeSourceArray)),
        # poprate with regular spikes: the input layer is a SpikeSourceArray
        regular=(net_handle['encoding'] == 'poprate'
                 and isinstance(encoder, ann.SpikeSourceArray)),
        # [5] a second monitor on the output layer that records spikes only.
        #     Monitors may be created after compile(). Your own monitor (spikes
        #     + v) stays as it is and is paused while an episode runs.
        spike_monitor=net.monitor(net_handle['output'], ['spike']),
        # [6] observation scaling as plain floats
        low=[float(x) for x in low],
        span=[float(x) for x in span],
        # [6] neurons per observation, read from the compiled input layer so the
        #     encoder always matches this network (not the NEURONS_PER_DIM global)
        n=encoder.size // len(low),
        squash_mask=None, squash_scale=None, squash_dims=(),
        # [8]
        env=None,
    )
    squash = cfg.get('squash')
    if squash is not None:
        s = np.asarray(squash, dtype=float)
        mask = np.isfinite(s) & (s > 0)
        fast['squash_mask'] = mask
        fast['squash_scale'] = s[mask]
        fast['squash_dims'] = set(int(i) for i in np.flatnonzero(mask))
    net_handle['fast'] = fast
    return fast


# ---------- [6] encoder ----------
def _normalize_fast(fast, obs):
    """_normalize() on plain floats. numpy on 2-6 numbers is mostly call overhead.

    The tanh dimensions go through the same numpy expression as _normalize(),
    so they produce the same bits.
    """
    low, span = fast['low'], fast['span']
    out = []
    if fast['squash_mask'] is None:
        for i in range(len(low)):
            u = (float(obs[i]) - low[i]) / span[i]
            out.append(0.0 if u < 0.0 else (1.0 if u > 1.0 else u))
        return out
    o = np.asarray(obs, dtype=float)
    squashed = ((np.tanh(o[fast['squash_mask']] / fast['squash_scale']) + 1.0)
                / 2.0).tolist()
    j = 0
    for i in range(len(low)):
        if i in fast['squash_dims']:
            u = squashed[j]
            j += 1
        else:
            u = (float(o[i]) - low[i]) / span[i]
        out.append(0.0 if u < 0.0 else (1.0 if u > 1.0 else u))
    return out


def drive_encoder_fast(net_handle, obs):
    """drive_encoder() for FAST_ENCODINGS: same numbers, computed in plain Python,
    then handed to the input population the documented way."""
    fast = _fast(net_handle)
    encoding = net_handle['encoding']
    encoder = net_handle['encoder']
    if not fast['fast_path']:                               # original path, unchanged
        cfg = net_handle['cfg']
        drive_encoder(encoding, encoder, obs, cfg['obs_low'],
                      cfg['obs_high'], t_offset=net_handle['net'].time,
                      squash=cfg.get('squash'))
        return

    norm = _normalize_fast(fast, obs)
    if encoding == 'rate':                                  # encode_rate()
        encoder.rates = [R_MIN + (R_MAX - R_MIN) * u for u in norm]
        return

    n = fast['n']                                           # this network's bins per observation
    if encoding == 'poprate' and fast['regular']:           # encode_poprate_regular()
        times = [[] for _ in range(len(norm) * n)]
        for d, u in enumerate(norm):
            k = min(int(u * n), n - 1)
            r = u * n - k
            r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
            times[d * n + k] = _regular_train(R_MIN + (R_MAX - R_MIN) * r)
        encoder.spike_times = times
        encoder.reset()                                     # spikes start in this window (F1)
        return
    if encoding == 'poprate':                               # encode_poprate()
        rates = [0.0] * (len(norm) * n)
        for d, u in enumerate(norm):
            k = min(int(u * n), n - 1)                      # _bin_and_residual()
            r = u * n - k
            r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
            rates[d * n + k] = R_MIN + (R_MAX - R_MIN) * r  # graded='residual'
        encoder.rates = rates                               # one rate per input neuron
        return

    times = [[] for _ in range(len(norm) * n)]              # encode_poplatency()
    for d, u in enumerate(norm):
        k = min(int(u * n), n - 1)
        r = u * n - k
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        times[d * n + k] = [LAT_WINDOW - r * (LAT_WINDOW - LAT_MIN)]   # ms from now
    encoder.spike_times = times
    encoder.reset()      # documented: "sets the spike times relative to the current
                         # time". Without it the pattern starts one window late.


# ---------- [7] decoder ----------
def decode_action_fast(decoding, spikes, n_salida):
    """decode_action() without numpy on 3 numbers. Same action, same random draws."""



    """
    if decoding == 'vote':
        counts = [len(spikes.get(a, ())) for a in range(n)]
        top = max(counts)
        if top == 0:
            random += 1
        winners = [a for a in range(n) if counts[a] == top]
        if len(winners) == 1:
            random += 1         # np.random.choice of ONE item draws nothing
    if decoding == 'first':
        best, best_t = -1, None
        for a in range(n):
            times = spikes.get(a, ())
            if len(times):
                t = min(times)
                if best_t is None or t < best_t:            # strict <: lowest index wins ties
                    best, best_t = a, t
        if best < 0 random += 1
    """
    n = int(n_salida)
    if decoding == 'vote':
        counts = [len(spikes.get(a, ())) for a in range(n)]
        top = max(counts)
        if top == 0:
            return int(np.random.randint(n))                # silent -> random
        winners = [a for a in range(n) if counts[a] == top]
        if len(winners) == 1:
            return winners[0]         # np.random.choice of ONE item draws nothing
        return int(np.random.choice(np.array(winners)))     # tie -> same draw as before
    if decoding == 'first':
        best, best_t = -1, None
        for a in range(n):
            times = spikes.get(a, ())
            if len(times):
                t = min(times)
                if best_t is None or t < best_t:            # strict <: lowest index wins ties
                    best, best_t = a, t
        return best if best >= 0 else int(np.random.randint(n))   # silent -> random
    return decode_action(decoding, spikes, n)


# ---------- [3] + [5] per-step reset ----------
def _reset_step(net_handle):
    """What net.reset() does between two windows, minus the wasted work."""
    fast = _fast(net_handle)
    # [3] net.reset() rewrites all 14 attributes of every Izhikevich population.
    #     8 of them are parameters nothing in the loop changes. Reset the variables.
    for pop, variables in fast['state']:
        pop.reset(attributes=variables)
    if not fast['input_resets_itself']:
        net_handle['encoder'].reset()
    # Clock back to 0 and RNG reseeded, as net.reset() does. [5]: monitors=False,
    # because the one monitor still recording was emptied by get('spike').
    net_handle['net'].reset(populations=False, monitors=False)


def evaluate(net_handle, weights, episode_seed=None, return_raw=False):
    """Set weights, run ONE episode, return scalar fitness.

    Same contract and same result as evaluate_reference(); see the block
    comment above for what changed.
    """
    set_weights(net_handle, weights)
    cfg = net_handle['cfg']
    net = net_handle['net']
    fast = _fast(net_handle)
    spike_monitor = fast['spike_monitor']

    # [8] one environment per network; reset(seed=...) gives the same start
    #     a fresh gym.make() would
    env = fast['env']
    if env is None:
        env = fast['env'] = gym.make(cfg['gym_id'])
    if episode_seed is None:
        obs, _ = env.reset()                  # fully random, independent episode
    else:
        obs, _ = env.reset(seed=int(episode_seed))   # repeatable episode

    net.reset()
    # [5] while the episode runs, record the output spikes and nothing else
    #     ('v' was recorded on every step and thrown away unread at every reset)
    paused = [m for m in net.get_monitors() if m is not spike_monitor]
    for m in paused:
        m.pause()

    total_reward = 0.0
    best_progress = cfg['progress'](obs)
    n_steps = 0
    done = False
    try:
        while not done:
            drive_encoder_fast(net_handle, obs)                       # [6]

            net.simulate(T_STEP)                                      # not touched

            spikes = spike_monitor.get('spike')                       # [5] returns the spikes
                                                                      #     and empties the buffer
            """
            random actions counter for test
            """

            action = decode_action_fast(net_handle['decoding'], spikes,
                                        cfg['n_salida'])              # [7]

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            best_progress = max(best_progress, cfg['progress'](obs))
            n_steps += 1
            done = terminated or truncated
            _reset_step(net_handle)                                   # [3] + [5]
    finally:
        for m in paused:                      # hand your monitors back recording
            m.resume()
    fitness = total_reward + cfg['shaping'] * best_progress
    return (fitness, total_reward) if return_raw else fitness
