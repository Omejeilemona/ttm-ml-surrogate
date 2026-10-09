"""
generate_data.py
================
Build the training data set for the ML surrogate: run the TTM solver at many
(absorbed fluence, pulse duration) points and save the results to a CSV file.

Sampling design
---------------
- Inputs are sampled in LOG space: log10(F_abs) and log10(tp) are uniform.
  Both ranges span more than a factor of 40, and the physics responds to
  ratios (doubling the fluence matters at 50 J/m^2 as much as at 1500 J/m^2),
  so every decade gets the same number of points.
- Points are placed with a Latin hypercube: each input axis is cut into
  N equal strips (in log space) and every strip contains exactly one point.
  This covers the input space evenly with few samples, unlike plain random
  sampling, which leaves gaps and clumps.
- The random seed is fixed, so the same points are produced on every run.

Outputs (columns of data/data.csv, SI units)
--------------------------------------------
  F_abs        absorbed fluence, J/m^2
  tp           pulse duration (FWHM), s
  Te_peak      peak surface electron temperature, K
  Tl_peak      peak surface lattice temperature, K      (regression target)
  melted       1 if the lattice reached Tm anywhere, else 0 (classification)
  energy_error relative energy-conservation error of the run (quality check)
  t_Tl_peak    time at which Tl_peak was reached, s (checks t_end is long enough)

Run with:  conda run -n ttm-ml python generate_data.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # write PNG files only, never open windows
import matplotlib.pyplot as plt
from scipy.stats import qmc

from ttm_solver import run_ttm, GOLD


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
N_SAMPLES = 500                # number of solver runs (~2 s each)
SEED = 42                      # fixed seed -> reproducible sample points

F_MIN, F_MAX = 50.0, 3000.0    # absorbed fluence range, J/m^2 (5-300 mJ/cm^2)
TP_MIN, TP_MAX = 50e-15, 2e-12 # pulse duration range (FWHM), s

# Simulated time. Peak surface Tl is reached when the surface electrons have
# cooled down to the lattice temperature; over the whole input range this
# happens by ~55 ps (slowest case: F_abs = 3000 J/m^2, tp = 2 ps), so 100 ps
# leaves a safety margin of about 2x.
T_END = 100e-12

OUT_DIR = "data"
CSV_PATH = os.path.join(OUT_DIR, "data.csv")
PLOT_PATH = "sampling.png"


# ---------------------------------------------------------------------------
# 2. Latin hypercube sample in log space
# ---------------------------------------------------------------------------
def latin_hypercube_log(n, seed):
    """
    Return n (F_abs, tp) pairs, Latin-hypercube distributed in log10 space.

    qmc.LatinHypercube gives points in the unit square [0, 1)^2 with one
    point per strip along each axis. qmc.scale stretches them linearly onto
    [log10(min), log10(max)] for each input; 10**x converts back to SI values.
    """
    sampler = qmc.LatinHypercube(d=2, rng=np.random.default_rng(seed))
    unit = sampler.random(n)                            # shape (n, 2), in [0, 1)
    log_lo = [np.log10(F_MIN), np.log10(TP_MIN)]
    log_hi = [np.log10(F_MAX), np.log10(TP_MAX)]
    logs = qmc.scale(unit, log_lo, log_hi)              # shape (n, 2), log10 values
    return 10.0 ** logs[:, 0], 10.0 ** logs[:, 1]


# ---------------------------------------------------------------------------
# 3. Run the solver at every sample point
# ---------------------------------------------------------------------------
def run_all(F_list, tp_list):
    rows = []
    t_start = time.perf_counter()
    for i, (F_abs, tp) in enumerate(zip(F_list, tp_list), start=1):
        res = run_ttm(F_abs, tp, t_end=T_END)

        # Time at which the surface lattice temperature peaked
        t_Tl_peak = res["t"][np.argmax(res["Tl_surf"])]

        rows.append({
            "F_abs": F_abs,
            "tp": tp,
            "Te_peak": res["Te_surf_max"],
            "Tl_peak": res["Tl_surf_max"],
            # melt_depth > 0 means some cell's lattice reached Tm
            "melted": int(res["melt_depth"] > 0.0),
            "energy_error": res["energy_error"],
            "t_Tl_peak": t_Tl_peak,
        })
        print(f"[{i:3d}/{len(F_list)}] F_abs = {F_abs:7.1f} J/m^2, "
              f"tp = {tp * 1e15:7.1f} fs -> Te_peak {res['Te_surf_max']:7.0f} K, "
              f"Tl_peak {res['Tl_surf_max']:6.0f} K, melted {rows[-1]['melted']}, "
              f"E err {res['energy_error'] * 100:+.3f} %")

    print(f"Total solver time: {time.perf_counter() - t_start:.1f} s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Quality checks on the finished data set
# ---------------------------------------------------------------------------
def check_quality(df):
    # Peak reached too close to the end of the run -> it may have been cut off
    late = df["t_Tl_peak"] > 0.8 * T_END
    if late.any():
        print(f"WARNING: {late.sum()} run(s) peaked after 80 % of t_end; "
              f"increase T_END.")
    print(f"Max |energy error|: {df['energy_error'].abs().max() * 100:.3f} %")
    print(f"Latest Tl peak: {df['t_Tl_peak'].max() * 1e12:.1f} ps "
          f"(t_end = {T_END * 1e12:.0f} ps)")
    print(f"Melted: {df['melted'].sum()} of {len(df)} samples")


# ---------------------------------------------------------------------------
# 5. Plot the sample points on log axes, coloured by melted
# ---------------------------------------------------------------------------
def plot_samples(df, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Faint lines at the Latin hypercube strip boundaries: each row and each
    # column of this grid should contain exactly one point. Only drawn for
    # small samples; for large N the lines merge into a grey background.
    n = len(df)
    if n <= 50:
        for x in np.logspace(np.log10(F_MIN), np.log10(F_MAX), n + 1):
            ax.axvline(x, color="0.9", lw=0.6, zorder=0)
        for y in np.logspace(np.log10(TP_MIN), np.log10(TP_MAX), n + 1):
            ax.axhline(y * 1e15, color="0.9", lw=0.6, zorder=0)

    for flag, colour, label in [(0, "tab:blue", "not melted"),
                                (1, "tab:red", "melted ($T_l \\geq T_m$)")]:
        sel = df["melted"] == flag
        ax.scatter(df.loc[sel, "F_abs"], df.loc[sel, "tp"] * 1e15,
                   c=colour, label=label, s=40, edgecolor="k", lw=0.5, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(TP_MIN * 1e15, TP_MAX * 1e15)
    ax.set_xlabel("Absorbed fluence $F_{abs}$ (J/m$^2$)")
    ax.set_ylabel("Pulse duration $t_p$ (fs)")
    ax.set_title(f"Latin hypercube sample, N = {n}, Au "
                 f"($T_m$ = {GOLD['Tm']:.0f} K)")
    # Legend below the axes so it can never hide a sample point
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    F_list, tp_list = latin_hypercube_log(N_SAMPLES, SEED)
    df = run_all(F_list, tp_list)
    check_quality(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Data saved to {CSV_PATH} ({len(df)} rows)")

    plot_samples(df, PLOT_PATH)
