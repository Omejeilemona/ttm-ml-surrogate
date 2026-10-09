"""
checks/convergence_check.py
===========================
Is peak Tl converged in time step and grid spacing for the 100 ps runs used
in the data set?

The v1 convergence tests (dt 2 -> 1 fs, dz 1 -> 0.5 nm) were run to 20 ps.
The data set uses t_end = 100 ps and the adaptive time step of v1.1
(dt = 2 fs during the pulse, growing to dt_max = 50 fs afterwards), so the
check is repeated here under the data-set conditions.

Reference runs:
  - time:  dt = 1 fs, dt_max = 10 fs   (default: 2 fs, 50 fs)
  - grid:  dz = 0.5 nm                 (default: 1 nm)

Result (2026-10-09, ttm_solver v1.1, t_end = 100 ps)
---------------------------------------------------
| F_abs (J/m^2) | tp (fs) | Peak Tl default (K) | dTl finer time step (K) | dTl finer grid (K) |
|---------------|---------|---------------------|-------------------------|--------------------|
|  300          |  100    |  596.03             | -0.02                   | +0.00              |
| 1500          |  100    | 1496.77             | +0.05                   | +0.00              |
| 3000          |  100    | 2557.41             | +0.32                   | +0.00              |
|  300          | 1000    |  613.30             | +0.10                   | +0.00              |
| 1500          | 1000    | 1548.53             | +0.49                   | +0.00              |
| 3000          | 1000    | 2644.94             | +0.97                   | +0.01              |

Melted flag identical in all variants. Energy error: default -0.025 % to
+0.0005 %; finer time step -0.006 % to 0 %.

Conclusion: peak Tl is converged to better than 1 K (0.04 %) in time step
and 0.01 K in grid spacing. Together with the thickness check (<= 3.3 K,
0.13 %), the solver's numerical uncertainty on peak Tl is ~1-3 K.

Run with:  conda run -n ttm-ml python checks/convergence_check.py
"""

import os
import sys
import time

# ttm_solver.py lives in the project root, one folder above this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttm_solver import run_ttm

T_END = 100e-12                                # same as generate_data.py
CASES = [(300.0, 100e-15), (1500.0, 100e-15), (3000.0, 100e-15),
         (300.0, 1e-12), (1500.0, 1e-12), (3000.0, 1e-12)]   # (F_abs J/m^2, tp s)
VARIANTS = [("default (dt 2 fs, dt_max 50 fs, dz 1 nm)", {}),
            ("finer time step (dt 1 fs, dt_max 10 fs)", {"dt": 1e-15, "dt_max": 10e-15}),
            ("finer grid (dz 0.5 nm)", {"dz": 0.5e-9})]

print("| F_abs (J/m^2) | tp (fs) | Variant | Peak Tl (K) | dTl vs default (K) | "
      "Melted | Energy error (%) | Runtime (s) |")
print("|---|---|---|---|---|---|---|---|")
for F_abs, tp in CASES:
    Tl_default = None
    for name, opts in VARIANTS:
        t0 = time.perf_counter()
        res = run_ttm(F_abs, tp, t_end=T_END, **opts)
        runtime = time.perf_counter() - t0
        if Tl_default is None:
            Tl_default = res["Tl_surf_max"]
        print(f"| {F_abs:.0f} | {tp * 1e15:.0f} | {name} | {res['Tl_surf_max']:.2f} | "
              f"{res['Tl_surf_max'] - Tl_default:+.2f} | {int(res['melt_depth'] > 0)} | "
              f"{res['energy_error'] * 100:+.4f} | {runtime:.1f} |")
