"""
Feedforward Izhikevich SNN for Gymnasium control tasks (ANNarchy 5.0 API).
"""

import numpy as np
import gymnasium as gym
import ANNarchy as ann


# ---------- shared encoder/decoder configuration ----------
T_STEP   = 50.0    # ms simulated per environment step

R_MAX = 300.0      # rate coding: Hz max
R_MIN = 10.0       # rate coding: Hz floor
              

LAT_WINDOW = 30.0  # latency coding: ms latency for normalized value 1.0
LAT_MIN    = 1.0   # latency coding: ms latency for normalized value 0.0


NEURONS_PER_DIM = 5 # neurons x dim for pop encoding

# POP_ENCODINGS share the population (one-hot place-code) input layer of width
# n_entrada*NEURONS_PER_DIM. They differ ONLY in how the active bin is activated:
#   poprate    -> Poisson firing at R_MAX across the window (closest to popcurrent)
#   poplatency -> a single early spike at LAT_MIN (weakest; EPSP may decay early)
POP_ENCODINGS = ('poprate', 'poplatency')
ENCODINGS     = ('rate', 'latency') + POP_ENCODINGS
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
def encode_rate(obs, obs_low, obs_high, squash=None):
    """Rate coding: normalized observation -> Poisson firing rates (Hz)."""
    return R_MIN + (R_MAX - R_MIN) * _normalize(obs, obs_low, obs_high, squash)


def encode_latency(obs, t_offset, obs_low, obs_high, squash=None):
    """Latency coding: normalized observation -> one spike time per dimension."""
    norm = _normalize(obs, obs_low, obs_high, squash)
    latency = LAT_WINDOW - norm * (LAT_WINDOW - LAT_MIN)
    return [float(t_offset + lat) for lat in latency]

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


def drive_encoder(encoding, encoder, obs, obs_low, obs_high, t_offset=0.0,
                  squash=None):
    """Push one observation into the input layer for the upcoming step."""
    if encoding == 'rate':
        encoder.rates = encode_rate(obs, obs_low, obs_high, squash)
    elif encoding == 'latency':
        encoder.spike_times = encode_latency(obs, t_offset, obs_low, obs_high,
                                             squash)
    elif encoding == 'poprate':
        encoder.rates = encode_poprate(obs, obs_low, obs_high, squash)
    elif encoding == 'poplatency':
        encoder.spike_times = encode_poplatency(obs, t_offset, obs_low,
                                                obs_high, squash)
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
def weight_shapes(environment, hidden_sizes, encoding='rate'):
    """Expected (pre, post) shape of each weight matrix. Use to size the genome.

    For the POPULATION encodings (poprate / poplatency) the input
    layer is place-coded, so it has n_entrada*NEURONS_PER_DIM neurons (not
    n_entrada)
    """
    cfg = ENVIRONMENTS[environment]
    n_in = (cfg['n_entrada'] * NEURONS_PER_DIM if encoding in POP_ENCODINGS
            else cfg['n_entrada'])
    sizes = [n_in] + [int(h) for h in hidden_sizes] + [cfg['n_salida']]
    return [(sizes[k], sizes[k + 1]) for k in range(len(sizes) - 1)]


def build_network(environment, hidden_sizes, encoding, decoding, compile_dir):
    """Build and COMPILE the network once (ANNarchy 5.0 Network instance).

    Projections start with dense (all-to-all) zero-weight connectivity;
    real weight values are written later by set_weights().
    """
    if environment not in ENVIRONMENTS:
        raise ValueError("environment must be one of %s" % list(ENVIRONMENTS))
    if encoding not in ENCODINGS:
        raise ValueError("encoding must be one of %s" % (ENCODINGS,))
    if decoding not in DECODINGS:
        raise ValueError("decoding must be one of %s" % (DECODINGS,))

    cfg = ENVIRONMENTS[environment]
    n_entrada, n_salida = cfg['n_entrada'], cfg['n_salida']

    # each call gets its own Network -- no global state, no clear() needed
    net = ann.Network()

    # input layer. Population encodings (POP_ENCODINGS) widen it to the place-code
    # width n_entrada*NEURONS_PER_DIM; 
    n_in = (n_entrada * NEURONS_PER_DIM if encoding in POP_ENCODINGS
            else n_entrada)
    if encoding in ('rate', 'poprate'):
        encoder = net.create(
            ann.PoissonPopulation(geometry=n_in, rates=0.0))
    elif encoding in ('latency', 'poplatency'):
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
                shapes=weight_shapes(environment, hidden_sizes, encoding),
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

# ================= NETWORK: EVALUATE MANY =================
def evaluate(net_handle, weights, episode_seed=None):
    """Set weights, run ONE episode, return scalar fitness.

    Fitness = total_reward.

    episode_seed controls the Gymnasium reset:
      - None (default): env.reset() with NO seed -> a fully random, independent
        episode every call.
      - int: env.reset(seed=episode_seed) -> a repeatable episode. 
    """
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
        monitor.reset()
        net.reset()  # clear neuron state (weights preserved)

    env.close()
    return (total_reward + cfg['shaping'] * best_progress)

