"""
drop_old_runs.py -- remove the grid results made before the sigma / noise fix, for the
optimizers whose search the fix changes, so the next pipeline start runs them again.

TwoPointsDE, RandomSearch and ScrHammersleySearchPlusMiddlePoint are kept: they ask the same
candidates and recommend the same network under either parametrization (checked to
rounding error), so their old runs are already the fixed runs.

Run it in the pipeline folder with the grid STOPPED (the running grid rewrites the results
file after every task and would put the removed entries back):

    python drop_old_runs.py            # shows what it would remove, changes nothing
    python drop_old_runs.py --apply    # backs up the results file, then removes them
"""
import json
import os
import shutil
import subprocess
import sys
import time
import collections as C

RESULTS = 'experiment_results_grid.json'
CHANGED = {'CMA', 'PSO', 'TBPSA', 'NgIohTuned'}     # optimizers the fix changes
KEEP_RULE = 'range/6'                               # what the fixed runs record

apply = '--apply' in sys.argv

running = subprocess.run(['pgrep', '-af', 'run_pipeline.py|experiment_runner.py'],
                         capture_output=True, text=True).stdout.strip().splitlines()
running = [l for l in running if str(os.getpid()) not in l.split()[0] and 'pgrep' not in l]
if running and apply:
    sys.exit("the grid is still running (stop it first):\n  " + "\n  ".join(running))

data = json.load(open(RESULTS))
gone = C.Counter()
for key, cell in data['cells'].items():
    opts = cell.get('optimizers') or {}
    removed = 0
    for name in list(opts):
        p = opts[name].get('param') or {}
        if name in CHANGED and not (p.get('sigma_rule') == KEEP_RULE and p.get('noisy')):
            gone[(cell['env'], name, int(cell['n_per_dim']))] += 1
            removed += 1
            if apply:
                del opts[name]
    if apply and removed:              # the cell is missing runs again
        cell['status'] = 'partial'
        if opts:
            best = max(opts, key=lambda k: opts[k]['fitness'])
            cell['best_optimizer'], cell['best_fitness'] = best, opts[best]['fitness']
        else:
            cell.pop('best_optimizer', None)
            cell.pop('best_fitness', None)

print("runs made before the fix (env, optimizer, n_per_dim: runs):")
for k in sorted(gone):
    print("  %-12s %-11s n%-3d %d" % (k[0], k[1], k[2], gone[k]))
print("total: %d runs" % sum(gone.values()))

if not apply:
    print("\nnothing changed; run with --apply (grid stopped) to remove them")
    sys.exit()
backup = RESULTS + '.before_drop_' + time.strftime('%Y%m%d_%H%M%S')
shutil.copy2(RESULTS, backup)
tmp = RESULTS + '.tmp'
with open(tmp, 'w') as f:
    json.dump(data, f, indent=1)
os.replace(tmp, RESULTS)
print("\nremoved; the old file is in %s. Start the pipeline again with the same command: "
      "it runs only these." % backup)
