"""
checks/thickness_check.py
=========================
Does the 1 um sample thickness affect the 100 ps runs used for the data set?

Why this check is needed
------------------------
Electrons carry heat with diffusivity D = ke / Ce. With ke = k0*Te/Tl and
Ce = gamma*Te, Te cancels:  D = k0 / (gamma*Tl) ~ 0.015 m^2/s at Tl = 300 K.
After a time t heat has spread over a diffusion length ~ sqrt(D*t):
0.55 um at 20 ps, but 1.2 um at 100 ps. So at 100 ps the heat reaches the
insulated back face of a 1 um sample, is reflected, and could keep the
surface hotter. At 20 ps (the v1 convergence test) it could not.

Method: tp = 100 fs; F_abs = 300, 1500, 3000 J/m^2; L = 1, 2, 4 um;
t_end = 100 ps. L = 4 um is the reference (effectively infinitely thick).

Result (2026-10-09, ttm_solver v1.1)
------------------------------------
| F_abs (J/m^2) | L (um) | Peak Tl (K) | dTl vs 4 um | t peak Tl (ps) | Melted | Back Te at 100 ps (K) |
|---------------|--------|-------------|-------------|----------------|--------|-----------------------|
|  300          | 1      |  596.0      | +0.00 %     | 17.1           | 0      | 330                   |
|  300          | 2      |  596.0      | +0.00 %     | 17.1           | 0      | 300                   |
|  300          | 4      |  596.0      |   ref       | 17.1           | 0      | 300                   |
| 1500          | 1      | 1496.8      | +0.07 %     | 34.3           | 1      | 543                   |
| 1500          | 2      | 1495.7      | +0.00 %     | 34.2           | 1      | 301                   |
| 1500          | 4      | 1495.7      |   ref       | 34.2           | 1      | 300                   |
| 3000          | 1      | 2557.4      | +0.13 %     | 48.7           | 1      | 825                   |
| 3000          | 2      | 2554.1      | +0.00 %     | 48.4           | 1      | 304                   |
| 3000          | 4      | 2554.1      |   ref       | 48.4           | 1      | 300                   |

Conclusion: heat does reach the back of the 1 um sample within 100 ps, but
only after the surface Tl has peaked (17-49 ps). Peak Tl changes by at most
0.13 %; peak time and melted flag do not change. Keep L = 1 um.

Recheck when: Ce(Te) is tabulated (v2 physics), t_end goes beyond 100 ps,
or melt depth / bulk temperatures become ML targets.

Run with:  conda run -n ttm-ml python checks/thickness_check.py
"""

import os
import sys
import time
import numpy as np

# ttm_solver.py lives in the project root, one folder above this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttm_solver import run_ttm

TP = 100e-15                         # pulse duration (FWHM), s
T_END = 100e-12                      # same simulated time as generate_data.py
FLUENCES = (300.0, 1500.0, 3000.0)   # absorbed fluence, J/m^2
THICKNESSES = (1e-6, 2e-6, 4e-6)     # sample thickness L, m
L_REF = 4e-6                         # reference thickness

print("| F_abs (J/m^2) | L (um) | Peak Tl (K) | dTl vs 4 um (%) | "
      "t peak Tl (ps) | Melted | Back Te at 100 ps (K) | Runtime (s) |")
print("|---|---|---|---|---|---|---|---|")
for F_abs in FLUENCES:
    runs = {}
    for L in THICKNESSES:
        t0 = time.perf_counter()
        res = run_ttm(F_abs, TP, L=L, t_end=T_END)
        runs[L] = (res, time.perf_counter() - t0)

    Tl_ref = runs[L_REF][0]["Tl_surf_max"]
    for L in THICKNESSES:
        res, runtime = runs[L]
        t_peak = res["t"][np.argmax(res["Tl_surf"])]   # time of peak surface Tl
        dTl = (res["Tl_surf_max"] - Tl_ref) / Tl_ref * 100.0
        print(f"| {F_abs:.0f} | {L * 1e6:.0f} | {res['Tl_surf_max']:.1f} | "
              f"{dTl:+.2f} | {t_peak * 1e12:.1f} | {int(res['melt_depth'] > 0)} | "
              f"{res['Te_final'][-1]:.0f} | {runtime:.2f} |")
