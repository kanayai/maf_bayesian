"""FE functional-form exercise.

Standalone diagnostic (NOT a run-bundle experiment). Uses the 100 FE simulation
runs per direction to learn how the extension responds to the *controllable*
inputs (load ``P`` and loading angle ``alpha``) at *fixed* values of the
*uncontrollable* material parameters (``E1, E2, v12, v23, G12``). This is the
controllable/uncontrollable split made in the paper.

Because the FE design is a 100-point Latin-hypercube where all seven inputs vary
at once, we cannot simply filter rows at a fixed theta. Instead we fit an
anisotropic Gaussian-process surrogate over all seven inputs (one per
direction), then *slice* the fitted surface at the config-centred nominal theta
and sweep the controllables.

Outputs four figures to ``figures/fe_functional_form/`` (git-ignored):
  1. normal extension vs angle   (per-uncontrollable low/nominal/high sweep)
  2. shear  extension vs angle   (per-uncontrollable low/nominal/high sweep)
  3. extension vs load           (uncontrollable envelope + GP interval)
  4. extension vs load + experiment

Run: ``uv run python scripts/fe_functional_form.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "simulation"
EXP_DATA = REPO / "data" / "experimental" / "full_experimental_data"
OUTDIR = REPO / "figures" / "fe_functional_form"

# Input column names in fitting order: 2 controllable, 5 uncontrollable.
INPUT_NAMES = ["P", "alpha", "E1", "E2", "v12", "v23", "G12"]
THETA_NAMES = ["E1", "E2", "v12", "v23", "G12"]

# Nominal (config-centred) uncontrollable values, from configs/default_config.py
# theta.reparam means.
NOMINAL_THETA = {
    "E1": 154900.0,
    "E2": 10285.0,
    "v12": 0.33,
    "v23": 0.435,
    "G12": 5115.0,
}

E1_REFERENCE_THETAS = {
    "Irene E1": {"E1": 148800.0, "color": "tab:purple", "linestyle": "--"},
    "Hexcel E1": {"E1": 161000.0, "color": "tab:green", "linestyle": "-."},
}

# Physical labels. v = normal extension, h = shear extension.
DIRECTIONS = {"v": "Normal extension", "h": "Shear extension"}

# Representative fixed loads for the vs-angle slices (kN); one row of panels each.
FIXED_LOADS = [5.0, 10.0]
# Experimental angles for the vs-load slices (degrees).
EXP_ANGLES_DEG = [45.0, 90.0, 135.0]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_direction(direction: str):
    """Return (X, y) for one FE direction.

    X: (100, 7) raw inputs [P, alpha(rad), E1, E2, v12, v23, G12].
    y: (100,)   extension, averaged across the three FE sensor columns.
    """
    d = DATA / direction
    load_angle = np.loadtxt(d / "input_load_angle_sim.txt", delimiter=",")  # (100,2)
    theta = np.loadtxt(d / "input_theta_sim.txt", delimiter=",")  # (100,5)
    ext = np.loadtxt(d / "data_extension_sim.txt", delimiter=",").mean(axis=1)  # (100,)
    X = np.column_stack([load_angle, theta])
    return X, ext


def load_experimental(direction, angle_deg):
    """Return list of (load, extension) arrays, one per specimen, for a given
    direction ('v'/'h') and nominal angle. Extension is averaged across the
    three DIC sensor columns, matching the FE convention. Loads with NaN or
    significantly negative values are dropped."""
    specimens = []
    for angle_file in sorted(EXP_DATA.glob(f"input_load_angle_exp_{angle_deg:g}_{direction}_*")):
        ext_file = EXP_DATA / angle_file.name.replace("input_load_angle", "data_extension")
        if not ext_file.exists():
            continue
        la = np.loadtxt(angle_file, delimiter=",")  # (N, 2): load, angle(rad)
        ext = np.loadtxt(ext_file, delimiter=",")  # (N, 3)
        good = (~np.isnan(la).any(axis=1)) & (~np.isnan(ext).any(axis=1)) & (la[:, 0] >= -1e-6)
        if good.any():
            specimens.append((la[good, 0], ext[good].mean(axis=1)))
    return specimens


# --------------------------------------------------------------------------- #
# Anisotropic GP (squared-exponential), fit by max marginal likelihood
# --------------------------------------------------------------------------- #


def _sqdist(Xa, Xb):
    """Pairwise squared Euclidean distance between standardised inputs."""
    a2 = jnp.sum(Xa**2, axis=1)[:, None]
    b2 = jnp.sum(Xb**2, axis=1)[None, :]
    return jnp.maximum(a2 + b2 - 2.0 * Xa @ Xb.T, 0.0)


class GPSurrogate:
    """Zero-mean anisotropic RBF GP over standardised inputs and outputs.

    Hyperparameters (all optimised in log space): 7 lengthscales, signal
    variance, noise variance.
    """

    def __init__(self, X, y):
        self.x_mean = X.mean(0)
        self.x_std = X.std(0)
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.Xs = jnp.asarray((X - self.x_mean) / self.x_std)
        self.ys = jnp.asarray((y - self.y_mean) / self.y_std)
        self.n, self.d = self.Xs.shape
        self.params = None  # filled by fit()

    def _kernel(self, Xa, Xb, log_ls, log_sf2):
        ls = jnp.exp(log_ls)
        sf2 = jnp.exp(log_sf2)
        r2 = _sqdist(Xa / ls, Xb / ls)
        return sf2 * jnp.exp(-0.5 * r2)

    def _nlml(self, p):
        log_ls, log_sf2, log_sn2 = p[: self.d], p[self.d], p[self.d + 1]
        K = self._kernel(self.Xs, self.Xs, log_ls, log_sf2)
        K = K + (jnp.exp(log_sn2) + 1e-8) * jnp.eye(self.n)
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), self.ys)
        ll = -0.5 * self.ys @ alpha
        ll -= jnp.sum(jnp.log(jnp.diag(L)))
        ll -= 0.5 * self.n * jnp.log(2.0 * jnp.pi)
        return -ll

    def fit(self):
        val_grad = jax.jit(jax.value_and_grad(self._nlml))

        def obj(p_np):
            v, g = val_grad(jnp.asarray(p_np))
            return float(v), np.asarray(g, dtype=np.float64)

        # init: lengthscale ~1 (inputs standardised), signal var ~1, noise ~0.01
        p0 = np.concatenate([np.zeros(self.d), [0.0], [np.log(1e-2)]])
        best = None
        for jitter in [0.0, 0.3, -0.3]:
            res = minimize(
                obj, p0 + jitter, jac=True, method="L-BFGS-B",
                options={"maxiter": 500},
            )
            if best is None or res.fun < best.fun:
                best = res
        self.params = jnp.asarray(best.x)
        # Precompute predictive cache.
        log_ls, log_sf2, log_sn2 = (
            self.params[: self.d], self.params[self.d], self.params[self.d + 1]
        )
        K = self._kernel(self.Xs, self.Xs, log_ls, log_sf2)
        self._K = K + (jnp.exp(log_sn2) + 1e-8) * jnp.eye(self.n)
        self._L = jnp.linalg.cholesky(self._K)
        self._alpha = jax.scipy.linalg.cho_solve((self._L, True), self.ys)
        return self

    def predict(self, Xnew):
        """Posterior mean at raw inputs Xnew (m, 7), returned in physical units."""
        Xs = jnp.asarray((np.asarray(Xnew) - self.x_mean) / self.x_std)
        log_ls, log_sf2 = self.params[: self.d], self.params[self.d]
        Ks = self._kernel(Xs, self.Xs, log_ls, log_sf2)
        mean_s = Ks @ self._alpha
        return np.asarray(mean_s) * self.y_std + self.y_mean

    def predict_interval(self, Xnew, level=0.95, include_noise=True):
        """Normal GP interval at raw inputs Xnew, returned in physical units.

        If include_noise is True, this is the posterior predictive interval for
        a new FE output. If False, it is the latent-mean uncertainty interval.
        """
        Xs = jnp.asarray((np.asarray(Xnew) - self.x_mean) / self.x_std)
        log_ls, log_sf2, log_sn2 = (
            self.params[: self.d], self.params[self.d], self.params[self.d + 1]
        )
        Ks = self._kernel(Xs, self.Xs, log_ls, log_sf2)
        mean_s = Ks @ self._alpha
        v = jax.scipy.linalg.solve_triangular(self._L, Ks.T, lower=True)
        var_s = jnp.exp(log_sf2) - jnp.sum(v**2, axis=0)
        if include_noise:
            var_s = var_s + jnp.exp(log_sn2)
        sd = np.sqrt(np.maximum(np.asarray(var_s), 0.0)) * self.y_std
        mean = np.asarray(mean_s) * self.y_std + self.y_mean
        z = norm.ppf(0.5 + level / 2.0)
        return mean, mean - z * sd, mean + z * sd

    def loo_r2(self):
        """Closed-form GP leave-one-out CV R^2 (in standardised space == same R^2)."""
        Kinv = jnp.linalg.inv(self._K)
        mu_loo = self.ys - (Kinv @ self.ys) / jnp.diag(Kinv)
        y = np.asarray(self.ys)
        pred = np.asarray(mu_loo)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Input-row builders (slicing the fitted surface at nominal / swept theta)
# --------------------------------------------------------------------------- #


def make_rows(P, alpha_rad, theta):
    """Build (m, 7) input rows. P and alpha_rad broadcast; theta is a dict."""
    P = np.atleast_1d(P).astype(float)
    alpha_rad = np.atleast_1d(alpha_rad).astype(float)
    m = max(P.size, alpha_rad.size)
    P = np.broadcast_to(P, (m,))
    alpha_rad = np.broadcast_to(alpha_rad, (m,))
    cols = [P, alpha_rad] + [np.full(m, theta[t]) for t in THETA_NAMES]
    return np.column_stack(cols)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_vs_angle(gp, direction, ranges, alpha_grid_deg):
    """Grid of panels: one row per fixed load, 5 columns (one per uncontrollable).

    Each panel shows extension vs angle with the swept uncontrollable at its
    FE-range low / nominal / high value.
    """
    alpha_rad = np.radians(alpha_grid_deg)
    nrows = len(FIXED_LOADS)
    fig, axes = plt.subplots(
        nrows, 5, figsize=(20, 4.2 * nrows), sharex=True, squeeze=False
    )
    for r, load in enumerate(FIXED_LOADS):
        for ax, tname in zip(axes[r], THETA_NAMES):
            lo, hi = ranges[tname]
            for level, val, colour in [
                (f"low ({lo:.3g})", lo, "tab:blue"),
                (f"nominal ({NOMINAL_THETA[tname]:.3g})", NOMINAL_THETA[tname], "black"),
                (f"high ({hi:.3g})", hi, "tab:red"),
            ]:
                theta = dict(NOMINAL_THETA)
                theta[tname] = val
                y = gp.predict(make_rows(load, alpha_rad, theta))
                # Axes swapped: extension on x, angle on y.
                ax.plot(y, alpha_grid_deg, color=colour, lw=1.8, label=level)
            ax.axvline(0.0, color="grey", lw=0.6, ls=":")
            # Experimental angles are fixed values of alpha -> horizontal lines.
            for ang in EXP_ANGLES_DEG:
                ax.axhline(ang, color="darkgreen", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(f"P = {load:g} kN | vary {tname}")
            ax.set_ylabel("loading angle (deg)")
        axes[r][0].legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("extension (mm)")
    fig.suptitle(
        f"{DIRECTIONS[direction]} vs angle "
        f"(rows: P = {', '.join(f'{l:g}' for l in FIXED_LOADS)} kN; "
        f"theta fixed at nominal except the swept parameter)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTDIR / f"{direction}_vs_angle.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_vs_load(gps, ranges_by_dir, P_grid, overlay_experiment=False,
                 n_env=300, seed=0):
    """Combined 2x3 figure: rows = normal (v, top) / shear (h, bottom);
    columns = experimental angles. Axes swapped: extension on x, load on y.

    If overlay_experiment is True, the measured experimental load-extension
    points (sensors averaged) are scattered on each panel.
    """
    rng = np.random.default_rng(seed)
    row_dirs = ["v", "h"]  # normal on top, shear on bottom
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex="row", squeeze=False)
    for r, direction in enumerate(row_dirs):
        gp = gps[direction]
        ranges = ranges_by_dir[direction]
        theta_samples = {
            t: rng.uniform(ranges[t][0], ranges[t][1], size=n_env) for t in THETA_NAMES
        }
        for ax, ang in zip(axes[r], EXP_ANGLES_DEG):
            arad = np.radians(ang)
            X_nom = make_rows(P_grid, arad, NOMINAL_THETA)
            y_nom, gp_lo, gp_hi = gp.predict_interval(
                X_nom, level=0.95, include_noise=True
            )
            preds = np.empty((n_env, P_grid.size))
            for k in range(n_env):
                theta_k = {t: theta_samples[t][k] for t in THETA_NAMES}
                preds[k] = gp.predict(make_rows(P_grid, arad, theta_k))
            lo = np.percentile(preds, 5, axis=0)
            hi = np.percentile(preds, 95, axis=0)
            # Axes swapped: extension on x, load on y.
            ax.fill_betweenx(P_grid, lo, hi, color="tab:orange", alpha=0.25,
                             label="FE 5-95% over uncontrollables")
            ax.fill_betweenx(P_grid, gp_lo, gp_hi, color="tab:cyan", alpha=0.28,
                             label="GP 95% predictive at nominal theta")
            ax.plot(y_nom, P_grid, color="black", lw=0.6, label="FE nominal theta")
            for label, spec in E1_REFERENCE_THETAS.items():
                theta_ref = dict(NOMINAL_THETA)
                theta_ref["E1"] = spec["E1"]
                y_ref = gp.predict(make_rows(P_grid, arad, theta_ref))
                ax.plot(
                    y_ref,
                    P_grid,
                    color=spec["color"],
                    ls=spec["linestyle"],
                    lw=1.2,
                    label=f"FE {label}",
                )
            if overlay_experiment:
                for j, (load_e, ext_e) in enumerate(load_experimental(direction, ang)):
                    ax.scatter(ext_e, load_e, s=8, color="tab:blue", alpha=0.5,
                               label="experiment" if j == 0 else None)
            ax.axvline(0.0, color="grey", lw=0.6, ls=":")
            ax.set_title(f"{DIRECTIONS[direction]} | alpha = {ang:g} deg")
            ax.set_ylabel("load P (kN)")
        axes[r][0].legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("extension (mm)")
    extra = " with experimental data" if overlay_experiment else ""
    fig.suptitle(
        f"Extension vs load at the experimental angles{extra} "
        "(top: normal; bottom: shear; lines = nominal/Irene/Hexcel E1 slices)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTDIR / ("vs_load_with_experiment.png" if overlay_experiment else "vs_load.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_angle_table(gps):
    """Small CSV mirroring the vs-angle plots: extension at each experimental
    angle, for each fixed load, at nominal uncontrollables. One row per
    (direction, load); columns are the three angles."""
    import csv

    out = OUTDIR / "vs_angle_nominal_extension.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P_kN", "direction"] + [f"ext_{a:g}deg_mm" for a in EXP_ANGLES_DEG])
        # Order rows by load, then direction: (5, normal), (5, shear), (10, ...).
        for load in FIXED_LOADS:
            for direction, label in DIRECTIONS.items():
                gp = gps[direction]
                vals = [
                    gp.predict(make_rows(load, np.radians(a), NOMINAL_THETA))[0]
                    for a in EXP_ANGLES_DEG
                ]
                w.writerow([f"{load:g}", label] + [f"{v:.6f}" for v in vals])
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    P_grid = np.linspace(0.0, 10.0, 120)
    alpha_grid_deg = np.linspace(1.0, 179.0, 120)

    print("FE functional-form exercise")
    print("=" * 60)
    written = []
    gps, ranges_by_dir = {}, {}
    for direction, label in DIRECTIONS.items():
        X, y = load_direction(direction)
        ranges = {n: (X[:, i].min(), X[:, i].max()) for i, n in enumerate(INPUT_NAMES)}
        gp = GPSurrogate(X, y).fit()
        gps[direction] = gp
        ranges_by_dir[direction] = ranges
        r2 = gp.loo_r2()
        ls = np.exp(np.asarray(gp.params[: gp.d]))
        print(f"\n[{direction}] {label}")
        print(f"  fitted on {gp.n} runs, LOO-CV R^2 = {r2:.4f}")
        print("  standardised lengthscales (small = influential):")
        for name, l in sorted(zip(INPUT_NAMES, ls), key=lambda t: t[1]):
            print(f"    {name:5s} {l:8.3f}")
        written.append(plot_vs_angle(gp, direction, ranges, alpha_grid_deg))

    written.append(plot_vs_load(gps, ranges_by_dir, P_grid))
    written.append(plot_vs_load(gps, ranges_by_dir, P_grid, overlay_experiment=True))
    written.append(write_angle_table(gps))

    print("\nFigures written:")
    for p in written:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
