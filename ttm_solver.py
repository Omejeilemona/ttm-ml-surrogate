"""
ttm_solver.py  (version 1.1: adaptive time steps)
==================================================
1D two-temperature model (TTM) for ultrafast laser heating of a metal (gold).

    Ce(Te) dTe/dt = d/dz[ ke dTe/dz ] - G (Te - Tl) + S(z, t)     (electrons)
    Cl     dTl/dt =                   + G (Te - Tl)              (lattice)

    Ce = gamma * Te              (free-electron heat capacity)
    ke = k0 * Te / Tl            (low-temperature electron conductivity)

Input energy is the ABSORBED fluence F_abs (J/m^2). Converting from incident
fluence needs a reflectivity R:  F_abs = (1 - R) * F_incident.

Version 1 simplifications (to be improved later):
  - constant gamma and G (valid for Te up to a few thousand K in gold)
  - no latent heat of melting, no ballistic electron transport
  - no lattice heat conduction (negligible on ps timescales)
  - no heat loss at the surface (insulated boundaries)

Version 1.1 (speed-up): the time step is small (dt) during the laser pulse
and grows geometrically to dt_max afterwards, when temperatures change
slowly. Same physics and same discretisation as version 1.

All quantities are in SI units (m, s, K, J, W).
"""

import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf


# ---------------------------------------------------------------------------
# 1. Material parameters for gold (typical literature values)
#    References: Hohlfeld et al., Chem. Phys. 251, 237 (2000);
#                Lin, Zhigilei & Celli, Phys. Rev. B 77, 075133 (2008)
# ---------------------------------------------------------------------------
GOLD = {
    "gamma": 71.0,      # electron heat capacity coefficient, J m^-3 K^-2
    "G": 2.2e16,        # electron-phonon coupling, W m^-3 K^-1 (reported ~2-3e16)
    "Cl": 2.5e6,        # lattice heat capacity, J m^-3 K^-1
    "k0": 318.0,        # electron thermal conductivity at 300 K, W m^-1 K^-1
    "delta": 15e-9,     # optical penetration depth at 800 nm, m
    "Tm": 1337.0,       # melting temperature, K
}


# ---------------------------------------------------------------------------
# 2. Laser source
# ---------------------------------------------------------------------------
def pulse_energy_fraction(t_start, t_end, t_peak, tp):
    """
    Fraction of the total pulse energy delivered between t_start and t_end.

    The pulse is Gaussian in time with full width at half maximum tp.
    Integrating the Gaussian exactly (with erf) instead of sampling it at one
    instant means the solver deposits exactly F_abs, whatever the time step.
    """
    c = 2.0 * np.sqrt(np.log(2.0)) / tp
    return 0.5 * (erf(c * (t_end - t_peak)) - erf(c * (t_start - t_peak)))


# ---------------------------------------------------------------------------
# 3. Time grid: small steps during the pulse, growing steps afterwards
# ---------------------------------------------------------------------------
def make_time_grid(t_end, t_fine_end, dt, dt_max, growth):
    """
    Build the list of time points t_0 = 0 < t_1 < ... < t_n = t_end.

    - From 0 to t_fine_end (the end of the pulse) the step is dt.
    - Afterwards each step is `growth` times longer than the previous one,
      up to a maximum of dt_max. Growing gradually (e.g. x1.05 per step)
      rather than jumping avoids a sudden drop in accuracy just after
      the pulse, when electrons are still cooling quickly.
    - The final step is shortened so the grid ends exactly at t_end.

    With dt_max = dt this gives the uniform grid of version 1.
    """
    times = [0.0]
    t, h = 0.0, dt
    while t < t_end * (1.0 - 1e-12):
        if t >= t_fine_end:
            h = min(h * growth, dt_max)   # grow the step after the pulse
        t = min(t + h, t_end)             # never step past t_end
        times.append(t)
    return np.array(times)


# ---------------------------------------------------------------------------
# 4. The solver
# ---------------------------------------------------------------------------
def run_ttm(F_abs, tp, params=GOLD, L=1.0e-6, dz=1.0e-9, dt=2.0e-15,
            dt_max=50e-15, growth=1.05, t_end=20e-12, T0=300.0, n_iter=2):
    """
    Solve the 1D TTM for one laser pulse.

    Parameters
    ----------
    F_abs : absorbed fluence, J/m^2   (1 J/m^2 = 0.1 mJ/cm^2)
    tp    : pulse duration (FWHM), s
    L     : sample thickness, m   (must be much larger than the heated depth)
    dz    : grid spacing, m
    dt    : time step during the pulse, s
    dt_max: largest time step allowed after the pulse, s (dt_max = dt
            reproduces the constant-step solver of version 1)
    growth: factor by which the step grows each step after the pulse
    t_end : simulated time, s
    T0    : initial temperature, K
    n_iter: iterations per time step for the heat-capacity update (2 is enough)

    Returns a dictionary with time histories, final profiles and summary numbers.
    """
    gamma, G, Cl, k0 = params["gamma"], params["G"], params["Cl"], params["k0"]
    delta, Tm = params["delta"], params["Tm"]

    # --- Spatial grid: N cells of width dz, temperatures at cell centres ---
    N = int(round(L / dz))
    z = (np.arange(N) + 0.5) * dz

    # --- Depth profile of absorbed energy: exp(-z/delta), normalised on the
    #     grid so that sum(w) * dz = 1 exactly (exact energy bookkeeping) ---
    w = np.exp(-z / delta)
    w /= w.sum() * dz

    # --- Initial conditions ---
    Te = np.full(N, T0)
    Tl = np.full(N, T0)

    # Pulse peak placed 3 FWHM after t = 0 so the pulse starts from ~zero
    t_peak = 3.0 * tp

    # Time grid. The pulse is over (all but ~1e-12 of its energy delivered)
    # at t_peak + 3 FWHM; only after that do the steps start to grow.
    times = make_time_grid(t_end, t_peak + 3.0 * tp, dt, dt_max, growth)
    n_steps = len(times) - 1

    # Storage for surface temperatures (every step) and the maximum lattice
    # temperature reached at each depth (used for the melt-depth estimate)
    t_hist = np.zeros(n_steps + 1)
    Te_surf = np.zeros(n_steps + 1)
    Tl_surf = np.zeros(n_steps + 1)
    Te_surf[0], Tl_surf[0] = T0, T0
    Tl_max = Tl.copy()

    # Total energy at the start (electrons: gamma*Te^2/2, lattice: Cl*Tl)
    def total_energy(Te, Tl):
        return np.sum(0.5 * gamma * Te**2 + Cl * Tl) * dz

    E0 = total_energy(Te, Tl)

    ab = np.zeros((3, N))   # banded matrix storage for solve_banded

    for n in range(n_steps):
        t_old, t_new = times[n], times[n + 1]
        dt = t_new - t_old          # length of THIS step (no longer constant)

        # Implicit electron-lattice coupling (see explanation in the guide):
        # eliminating Tl_new gives an effective coupling G_eff = G / (1 + a).
        # Both depend on dt, so they are recomputed every step.
        a = dt * G / Cl
        G_eff = G / (1.0 + a)

        # Conductivity evaluated at the old temperatures ("lagged")
        ke = k0 * Te / Tl
        k_face = 0.5 * (ke[:-1] + ke[1:])        # conductivity between cells
        A = k_face / dz**2                       # length N-1

        # Heat flux coefficients to the left and right of each cell.
        # Zero at the outer faces = insulated surface and back side.
        A_left = np.concatenate(([0.0], A))
        A_right = np.concatenate((A, [0.0]))

        # Absorbed power density averaged over this time step, W/m^3
        Q = F_abs * w * pulse_energy_fraction(t_old, t_new, t_peak, tp) / dt

        # Solve the tridiagonal system  M @ Te_new = rhs.
        # Ce = gamma * (Te_old + Te_new) / 2 makes the electron energy change
        # exact (gamma*Te^2/2), but Te_new is unknown, so we iterate:
        # first guess Te_new = Te_old, then re-solve with the improved guess.
        Te_guess = Te
        for _ in range(n_iter):
            Ce = gamma * 0.5 * (Te + Te_guess)
            ab[0, 1:] = -A                                   # upper diagonal
            ab[1, :] = Ce / dt + A_left + A_right + G_eff    # main diagonal
            ab[2, :-1] = -A                                  # lower diagonal
            rhs = Ce / dt * Te + G_eff * Tl + Q
            Te_guess = solve_banded((1, 1), ab, rhs)

        Te = Te_guess
        Tl = (Tl + a * Te) / (1.0 + a)

        t_hist[n + 1] = t_new
        Te_surf[n + 1] = Te[0]
        Tl_surf[n + 1] = Tl[0]
        np.maximum(Tl_max, Tl, out=Tl_max)

    # --- Energy check: energy gained should equal energy deposited ---
    E_dep = F_abs * pulse_energy_fraction(0.0, t_end, t_peak, tp)
    energy_error = (total_energy(Te, Tl) - E0 - E_dep) / E_dep

    # --- Melt-depth estimate: deepest cell whose lattice ever exceeded Tm.
    #     No latent heat in v1, so this OVERESTIMATES the true melt depth. ---
    exceeded = Tl_max >= Tm
    melt_depth = z[exceeded].max() + 0.5 * dz if exceeded.any() else 0.0

    return {
        "t": t_hist, "Te_surf": Te_surf, "Tl_surf": Tl_surf,
        "z": z, "Te_final": Te, "Tl_final": Tl, "Tl_max": Tl_max,
        "Te_surf_max": Te_surf.max(), "Tl_surf_max": Tl_surf.max(),
        "melt_depth": melt_depth, "energy_error": energy_error,
    }


# ---------------------------------------------------------------------------
# 5. Example run: executed only when this file is run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    F_abs = 300.0       # J/m^2  (= 30 mJ/cm^2 absorbed)
    tp = 100e-15        # 100 fs FWHM

    t0 = time.perf_counter()
    res = run_ttm(F_abs, tp)
    runtime = time.perf_counter() - t0

    print(f"Absorbed fluence      : {F_abs:.0f} J/m^2 ({F_abs / 10:.1f} mJ/cm^2)")
    print(f"Pulse duration (FWHM) : {tp * 1e15:.0f} fs")
    print(f"Peak surface Te       : {res['Te_surf_max']:.0f} K")
    print(f"Peak surface Tl       : {res['Tl_surf_max']:.0f} K")
    print(f"Depth above Tm        : {res['melt_depth'] * 1e9:.1f} nm  (no latent heat)")
    print(f"Energy error          : {res['energy_error'] * 100:.3f} %")
    print(f"Runtime               : {runtime:.2f} s")

    # Plot surface temperatures against time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["t"] * 1e12, res["Te_surf"], label="Electrons, $T_e$")
    ax.plot(res["t"] * 1e12, res["Tl_surf"], label="Lattice, $T_l$")
    ax.axhline(GOLD["Tm"], color="gray", ls="--", lw=1, label="$T_m$ (Au)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title(f"Au, $F_{{abs}}$ = {F_abs / 10:.0f} mJ/cm$^2$, "
                 f"$t_p$ = {tp * 1e15:.0f} fs")
    ax.legend()
    fig.tight_layout()
    fig.savefig("ttm_example.png", dpi=150)
    print("Plot saved to ttm_example.png")