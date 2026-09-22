"""
visualize_results.py -- figures from experiment_runner.py's results JSON.

Reads experiment_results.json and writes publication-style figures. The central
plots are BOX PLOTS, but a box needs a distribution and the runner stores one
mean fitness per (cell, optimizer) -- so each box aggregates ACROSS configs:
"every fitness value <encoding> reached, over all topologies / decodings /
optimizers" is one box. That answers the study's questions (which encoding wins,
which optimizer is most consistent, how topology matters) from stored data.

Everything is faceted BY ENVIRONMENT: MountainCar and Acrobot use different
shaped-fitness scales, so their boxes must never share an axis.

Higher shaped fitness = better (it is raw_reward + shaping*progress, so still
negative; less negative is better).

Figures written (one per env, side by side):
  fitness_by_encoding.png   box: best_fitness per cell, grouped by encoding   [headline]
  fitness_by_optimizer.png  box: each optimizer's fitness across cells
  fitness_by_topology.png   box: best_fitness per cell, grouped by topology
  encoding_x_optimizer.png  heatmap: mean fitness over the two factors (interactions)
  genome_vs_fitness.png     scatter: genome dim vs best_fitness, coloured by encoding
  meta_tuned_vs_default.png bar: meta-opt tuned vs default (only if meta results exist)
  episode_box_best.png      box: the 30 report episodes of the best config per env
                            (only if you kept per-episode 'fitnesses' -- see note)

NOTE on episode-level boxes: to get the true per-config episode distribution (the
most literal "box plot of a run"), keep the raw episodes in the runner -- in
experiment_runner._run_one_cell add  fitnesses=r.get('fitnesses')  to each
optimizer dict. This script uses them if present and silently skips that one
figure if not.

Run:  python visualize_results.py                 # reads experiment_results.json
      python visualize_results.py --results F --outdir figures
Requires: pip install matplotlib
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')                       # headless: write files, no display
import matplotlib.pyplot as plt


ENCODING_ORDER = ['rate', 'latency', 'popcurrent', 'poprate', 'poplatency']
_TAB = plt.cm.tab10.colors


def topo_str(hidden):
    return 'x'.join(str(h) for h in hidden) or 'none'


# ---------------- load & flatten ----------------
def load_records(path):
    with open(path) as f:
        R = json.load(f)
    per_opt, per_cell = [], []              # one row per (cell,optimizer); one per cell
    for c in R.get('cells', {}).values():
        if c.get('status') != 'ok':
            continue
        base = dict(env=c['env'], encoding=c['encoding'], decoding=c['decoding'],
                    topology=topo_str(c.get('hidden', [])),
                    genome_dim=c.get('genome_dim'), budget=c.get('budget'))
        for name, o in c.get('optimizers', {}).items():
            per_opt.append(dict(base, optimizer=name, fitness=o['fitness'],
                                std=o.get('std'), fitnesses=o.get('fitnesses')))
        if 'best_fitness' in c:
            per_cell.append(dict(base, optimizer=c.get('best_optimizer'),
                                 fitness=c['best_fitness']))
    return R, per_opt, per_cell


def _order_groups(records, key):
    present = {r[key] for r in records}
    ordered = [g for g in ENCODING_ORDER if g in present]     # fixed order for encodings
    return ordered + sorted(present - set(ordered))            # then anything else, sorted


def _envs(records):
    return sorted({r['env'] for r in records})


# ---------------- box plot (faceted by env) ----------------
def box_by(records, group_key, value_key, title, fname, outdir, ylabel):
    envs = _envs(records)
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 5),
                             squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = [r for r in records if r['env'] == env and r[value_key] is not None]
        groups = _order_groups(sub, group_key)
        data = [[r[value_key] for r in sub if r[group_key] == g] for g in groups]
        colors = [_TAB[i % len(_TAB)] for i in range(len(groups))]
        bp = ax.boxplot(data, positions=range(len(groups)), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col); patch.set_alpha(0.45)
        for i, vals in enumerate(data):
            if not vals:
                continue
            jit = np.random.default_rng(i).normal(i, 0.06, size=len(vals))
            ax.scatter(jit, vals, s=16, color=colors[i], edgecolor='black',
                       linewidth=0.3, zorder=3)
            ax.text(i, 0.01, f"n={len(vals)}", transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color='dimgray')
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.set_title(env, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(title, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, fname, outdir)


# ---------------- heatmap encoding x optimizer ----------------
def heatmap_enc_opt(per_opt, outdir):
    envs = _envs(per_opt)
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 4.5),
                             squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = [r for r in per_opt if r['env'] == env]
        encs = _order_groups(sub, 'encoding')
        opts = sorted({r['optimizer'] for r in sub})
        M = np.full((len(encs), len(opts)), np.nan)
        for i, e in enumerate(encs):
            for j, o in enumerate(opts):
                v = [r['fitness'] for r in sub
                     if r['encoding'] == e and r['optimizer'] == o]
                if v:
                    M[i, j] = float(np.mean(v))
        im = ax.imshow(M, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(opts))); ax.set_xticklabels(opts, rotation=30, ha='right')
        ax.set_yticks(range(len(encs))); ax.set_yticklabels(encs)
        for i in range(len(encs)):
            for j in range(len(opts)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha='center', va='center',
                            fontsize=8, color='white')
        ax.set_title(env, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='mean fitness')
    fig.suptitle('mean fitness: encoding x optimizer', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'encoding_x_optimizer.png', outdir)


# ---------------- scatter genome dim vs fitness ----------------
def scatter_dim(per_cell, outdir):
    envs = _envs(per_cell)
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 4.5),
                             squeeze=False)
    encs = _order_groups(per_cell, 'encoding')
    cmap = {e: _TAB[i % len(_TAB)] for i, e in enumerate(encs)}
    for ax, env in zip(axes[0], envs):
        sub = [r for r in per_cell if r['env'] == env and r['genome_dim'] is not None]
        for e in encs:
            pts = [(r['genome_dim'], r['fitness']) for r in sub if r['encoding'] == e]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=30, color=cmap[e], edgecolor='black',
                           linewidth=0.3, label=e, alpha=0.8)
        ax.set_xlabel('genome dim'); ax.set_ylabel('best fitness')
        ax.set_title(env, fontweight='bold'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title='encoding')
    fig.suptitle('genome size vs best fitness (budget-confound check)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'genome_vs_fitness.png', outdir)


# ---------------- meta tuned vs default ----------------
def meta_bar(R, outdir):
    meta = {k: v for k, v in R.get('meta', {}).items() if v.get('status') == 'ok'}
    if not meta:
        return
    keys = list(meta)
    labels = [k.replace('meta|', '') for k in keys]
    tuned = [meta[k]['tuned'] for k in keys]
    default = [meta[k]['default'] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(keys)), 4.5))
    ax.bar(x - w / 2, default, w, label='default', color=_TAB[0], alpha=0.8)
    ax.bar(x + w / 2, tuned, w, label='tuned', color=_TAB[2], alpha=0.8)
    for i, k in enumerate(keys):
        ax.text(i, max(tuned[i], default[i]), f"{meta[k]['improvement']:+.1f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('holdout fitness'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('meta-optimization: tuned vs default (label = improvement)',
                 fontweight='bold')
    ax.legend()
    fig.tight_layout()
    _save(fig, 'meta_tuned_vs_default.png', outdir)


# ---------------- episode-distribution box (only if raw fitnesses kept) ----------------
def episode_box_best(per_opt, outdir):
    have = [r for r in per_opt if r.get('fitnesses')]
    if not have:
        return                              # runner didn't keep per-episode arrays
    envs = _envs(have)
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 5),
                             squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = [r for r in have if r['env'] == env]
        # top few (cell,optimizer) by mean fitness
        top = sorted(sub, key=lambda r: r['fitness'], reverse=True)[:6]
        data = [list(r['fitnesses']) for r in top]
        labels = [f"{r['encoding']}/{r['decoding']}\n{r['topology']} {r['optimizer']}"
                  for r in top]
        bp = ax.boxplot(data, positions=range(len(top)), widths=0.6,
                        patch_artist=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor(_TAB[2]); patch.set_alpha(0.45)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax.set_title(env, fontweight='bold'); ax.set_ylabel('per-episode fitness')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('report-episode distribution, top configs per env',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'episode_box_best.png', outdir)


def _save(fig, fname, outdir):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="visualize experiment_runner results")
    ap.add_argument('--results', default='experiment_results.json')
    ap.add_argument('--outdir', default='figures')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    R, per_opt, per_cell = load_records(args.results)
    if not per_opt:
        print("no completed cells in results; run the experiment first.")
        return
    print(f"loaded {len(per_cell)} cells / {len(per_opt)} (cell,optimizer) rows "
          f"from {args.results}\nwriting figures to {args.outdir}/ ...")

    box_by(per_cell, 'encoding', 'fitness',
           'best fitness by ENCODING (higher = better)',
           'fitness_by_encoding.png', args.outdir, 'best fitness')
    box_by(per_opt, 'optimizer', 'fitness',
           'fitness by OPTIMIZER (all cells)',
           'fitness_by_optimizer.png', args.outdir, 'fitness')
    box_by(per_cell, 'topology', 'fitness',
           'best fitness by TOPOLOGY',
           'fitness_by_topology.png', args.outdir, 'best fitness')
    heatmap_enc_opt(per_opt, args.outdir)
    scatter_dim(per_cell, args.outdir)
    meta_bar(R, args.outdir)
    episode_box_best(per_opt, args.outdir)
    print("done.")


if __name__ == "__main__":
    main()
