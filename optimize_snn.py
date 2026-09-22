"""
Weight optimization for the feedforward SNN using Nevergrad.
"""

import numpy as np
import nevergrad as ng
from snn_feedforward import build_network, evaluate, weight_shapes


# ---------- optimization configuration ----------
OPTIMIZERS = ["NGOpt", "CMA", "DE", "PSO", "OnePlusOne", "TBPSA"]
BUDGET  = 200      # function evaluations per optimizer (main cost knob)
W_LOW  = -20.0     # weight search lower bound
W_HIGH =  20.0 #this    # weight search upper bound (asymmetric, excitatory-skewed;
                   # set from another NN's hyperparameters -- experimental)
SEED    = 0        # seeds the shared initial point (fair across optimizers)
REPORT_EPISODES = 100   # episodes averaged for the FINAL report only (not during
                       # optimization). Independent draws per optimizer.
EVAL_EPISODES = 1     # number of episodes to measure weights

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


# ---------- final-report scoring (averaged, AFTER optimization) ----------
def report_fitness(net_handle, weights, n_episodes=REPORT_EPISODES,
                   episode_seed=None):
    """Score fixed weights over n INDEPENDENT episodes -- report only.

    episode_seed controls reproducibility of the report episodes:
      - None (default): each episode calls evaluate() with no seed -> n fully
        random, independent episodes.
      - int: episode k uses seed (episode_seed*N_LARGE + k) -> the n episodes are
        DISTINCT from one another (so variance is still measured) but the whole
        block is REPEATABLE.

    Returns mean, sample std (ddof=1), standard error of the mean, and the raw
    per-episode fitnesses.
    """
    n = int(n_episodes)
    if episode_seed is None:
        fits = np.array([evaluate(net_handle, weights) for _ in range(n)],
                        dtype=float)
    else:
        base = int(episode_seed) * 1000  # spread blocks so they don't overlap
        fits = np.array([evaluate(net_handle, weights, episode_seed=base + k)
                         for k in range(n)], dtype=float)
    mean = float(fits.mean())
    std  = float(fits.std(ddof=1)) if n > 1 else 0.0
    sem  = std / np.sqrt(n) if n > 1 else 0.0
    return dict(mean=mean, std=std, sem=sem, n=n, fitnesses=fits)


# ---------- run multiple optimizers ----------
def optimize(environment, hidden_sizes, encoding, decoding,
             optimizers=OPTIMIZERS, budget=BUDGET, trial=0):
    """Compile the SNN once, then run each optimizer on its weights.

    Returns {optimizer_name: {'fitness'(=mean over REPORT_EPISODES), 'std',
    'sem', 'n_episodes', 'fitnesses', 'opt_loss', 'weights'}}.

    Reproducibility: the report uses independent draws from the free-running
    global RNG.
    """
    if EVAL_EPISODES < 1:
            raise ValueError("EVAL_EPISODES must be more than 0")
    # one compile for the whole comparison; dir is unique per configuration
    compile_dir = ("annarchy/opt-%s-%s-%s-%s-t%d"
                   % (environment, encoding, decoding,
                      "x".join(str(h) for h in hidden_sizes), trial))
    net = build_network(environment, hidden_sizes,
                         encoding, decoding, compile_dir)
    shapes = net['shapes']
    dim = n_params(shapes)

    def objective(flat):
        # only proj.w updates
        #this 10
        fitness = 0
        i = 0
        while i < EVAL_EPISODES:
            # testing multiple monitors for net activity
            # -float(evaluate(net, flat_to_weights(flat, shapes)))
            fitness = fitness + -float(evaluate(net, flat_to_weights(flat, shapes)))
            i = i + 1
        return fitness / EVAL_EPISODES
        
    # shared random start -> every optimizer begins from the same point
    rng = np.random.default_rng(SEED)
    init = rng.uniform(W_LOW, W_HIGH, size=dim)

    results = {}
    for name in optimizers:
        if name == 'NgIohTuned':
            param = (ng.p.Array(init=init.copy())
                   .set_bounds(W_LOW, W_HIGH))
            param.real_world = True
            param.neural = True
            param.function.deterministic = False
            #NoisyRL2
        else:
            param = (ng.p.Array(init=init.copy())
                   .set_bounds(W_LOW, W_HIGH))
        opt_cls = ng.optimizers.registry[name]
        optimizer = opt_cls(parametrization=param, budget=budget)

        recommendation = optimizer.minimize(objective)
        best_flat = recommendation.value
        best_weights = flat_to_weights(best_flat, shapes)

        opt_loss = objective(best_flat)

        # averaged score over REPORT_EPISODES independent episodes (report only)
        rep = report_fitness(net, best_weights, REPORT_EPISODES)

        results[name] = dict(fitness=rep['mean'], std=rep['std'], sem=rep['sem'],
                             n_episodes=rep['n'], fitnesses=rep['fitnesses'],
                             opt_loss=opt_loss, weights=best_weights)
        print("%-12s budget=%d  mean=%.3f  std=%.3f  sem=%.3f  (n=%d, opt=%.3f)"
              % (name, budget, rep['mean'], rep['std'], rep['sem'],
                 rep['n'], -opt_loss))

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
