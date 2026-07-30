"""Simplest empirical model — single-parameter Bayesian fit.

A deliberately minimal baseline to build on. Two extension channels share ONE
scalar parameter mu, with fixed (non-random) trigonometric projections and no
per-specimen bias:

    y_v = P * cos(alpha) * mu + eps        (vertical / normal extension)
    y_h = 2 * P * sin(alpha) * mu + eps     (horizontal / shear extension)
    eps ~ Normal(0, sigma^2)                (single shared error variance)

Everything is treated as independent (v independent of h; observations iid).

Proportional (load-dependent) error model: the error variance grows with load,
    eps ~ Normal(0, sigma^2 * P)             (Var proportional to P; sd = sigma*sqrt(P))
so the scatter fans out with load and pinches to zero at the origin. Rows with
P == 0 carry zero variance and are excluded from the likelihood.

Priors:
    mu    ~ Normal(0.0054, 0.001)            (mean = current mu_emulator prior mean)
    sigma ~ HalfNormal(0.02)                 (weakly informative; units mm/sqrt(kN))

Data: averaged experimental data (DIC sensors averaged), angles 45/90/135 deg,
truncated to loads <= 10 kN.

This is a STANDALONE diagnostic script (like scripts/fe_functional_form.py). It
does not touch the config-driven inference pipeline or the existing
`model_empirical`, and it does not create a paper-evidence run bundle.

Run: uv run python scripts/empirical_simple_model.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.io.data_loader import load_experiment_data  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

EXP_DATA = REPO / "data" / "experimental" / "full_experimental_data"
OUTDIR = REPO / "figures" / "empirical_simple_model"
ANGLES_DEG = [45, 90, 135]
MAX_LOAD = 10.0  # kN

MU_PRIOR = (0.0054, 0.001)     # Normal(mean, sd)
SIGMA_PRIOR_SCALE = 0.02       # HalfNormal(scale); units mm/sqrt(kN)

N_WARMUP = 1000
N_SAMPLES = 2000
N_CHAINS = 4
SEED = 0

DIR_LABEL = {"v": "Normal extension", "h": "Shear extension"}


# --------------------------------------------------------------------------- #
# Data: averaged experimental extension, per channel and angle, load <= 10 kN
# --------------------------------------------------------------------------- #


def load_averaged(direction):
    """Return {angle_deg: (load, extension)} with sensors averaged and load<=MAX_LOAD.
    Pools all specimens at each angle into one array."""
    inputs, exts = load_experiment_data(EXP_DATA, ANGLES_DEG, direction=direction)
    by_angle = {a: ([], []) for a in ANGLES_DEG}
    for xy, ext in zip(inputs, exts):
        load = np.asarray(xy[:, 0], float)
        ang = np.asarray(xy[:, 1], float)
        y = np.asarray(ext, float).mean(axis=1)  # average across DIC sensors
        keep = (load <= MAX_LOAD) & np.isfinite(load) & np.isfinite(y)
        a = int(round(np.degrees(ang[0])))
        by_angle[a][0].append(load[keep])
        by_angle[a][1].append(y[keep])
    return {a: (np.concatenate(L), np.concatenate(Y)) for a, (L, Y) in by_angle.items()}


def assemble():
    """Stack v and h into flat arrays for the joint fit."""
    data = {"v": load_averaged("v"), "h": load_averaged("h")}
    P, alpha, chan, y = [], [], [], []
    for ci, d in enumerate(["v", "h"]):  # 0 = v, 1 = h
        for a in ANGLES_DEG:
            load, ext = data[d][a]
            P.append(load)
            alpha.append(np.full(load.shape, np.radians(a)))
            chan.append(np.full(load.shape, ci))
            y.append(ext)
    return (
        data,
        np.concatenate(P),
        np.concatenate(alpha),
        np.concatenate(chan),
        np.concatenate(y),
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def model(P, alpha, chan, y=None):
    mu = numpyro.sample("mu", dist.Normal(MU_PRIOR[0], MU_PRIOR[1]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(SIGMA_PRIOR_SCALE))
    # chan == 0 -> cos(alpha); chan == 1 -> 2 sin(alpha)
    trig = jnp.where(chan == 0, jnp.cos(alpha), 2.0 * jnp.sin(alpha))
    mean = P * trig * mu  # through the origin: no intercept term
    # Proportional error model: Var(eps) = sigma^2 * P, i.e. sd = sigma*sqrt(P).
    sd = sigma * jnp.sqrt(P)
    numpyro.sample("y", dist.Normal(mean, sd), obs=y)


def summarise(name, s):
    return (f"  {name:6s} mean={s.mean():.6f}  sd={s.std():.6f}  "
            f"94% HDI=[{np.percentile(s,3):.6f}, {np.percentile(s,97):.6f}]")


# --------------------------------------------------------------------------- #
# Prediction overlay plot (extension on x, load on y — matching vs_load)
# --------------------------------------------------------------------------- #


def plot_predictions(data, mu_s, sigma_s):
    P_grid = np.linspace(0.0, MAX_LOAD, 100)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex="row", squeeze=False)
    for r, d in enumerate(["v", "h"]):
        for ax, a in zip(axes[r], ANGLES_DEG):
            trig = np.cos(np.radians(a)) if d == "v" else 2.0 * np.sin(np.radians(a))
            # posterior of the mean line: P * trig * mu
            lines = np.outer(mu_s, P_grid * trig)          # (n_samples, n_grid)
            mean_line = lines.mean(0)
            lo_m, hi_m = np.percentile(lines, [3, 97], axis=0)
            # posterior predictive band: proportional noise, sd = sigma*sqrt(P).
            rng = np.random.default_rng(SEED)
            noise_sd = sigma_s[:, None] * np.sqrt(P_grid)[None, :]
            noise = rng.normal(0.0, 1.0, size=lines.shape) * noise_sd
            pred = lines + noise
            lo_p, hi_p = np.percentile(pred, [3, 97], axis=0)

            ax.fill_betweenx(P_grid, lo_p, hi_p, color="tab:orange", alpha=0.15,
                             label="94% predictive")
            ax.fill_betweenx(P_grid, lo_m, hi_m, color="tab:orange", alpha=0.45,
                             label="94% mean")
            ax.plot(mean_line, P_grid, color="black", lw=2.0, label="posterior mean")
            load, ext = data[d][a]
            ax.scatter(ext, load, s=8, color="tab:blue", alpha=0.5, label="experiment (avg)")
            ax.axvline(0.0, color="grey", lw=0.6, ls=":")
            ax.set_title(f"{DIR_LABEL[d]} | alpha = {a} deg")
            ax.set_ylabel("load P (kN)")
        axes[r][0].legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("extension (mm)")
    fig.suptitle(
        "Simplest empirical model: posterior predictions vs averaged experimental "
        "data (single shared mu, proportional error Var=sigma^2*P; load <= 10 kN)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTDIR / "predictions_vs_data.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(verbose=True):
    """Fit the model, write the prediction figure, and return a results dict.

    Returns keys: mu, sigma (posterior sample arrays), data (per-channel/angle),
    counts (obs per channel/angle), n_obs, fig (figure path)."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(N_CHAINS)
    data, P, alpha, chan, y = assemble()
    # Proportional error model has zero variance at P=0; exclude those rows.
    keep = P > 0
    P, alpha, chan, y = P[keep], alpha[keep], chan[keep], y[keep]
    counts = {
        d: {a: int(((chan == ci) & np.isclose(alpha, np.radians(a))).sum())
            for a in ANGLES_DEG}
        for ci, d in enumerate(["v", "h"])
    }

    if verbose:
        print("Simplest empirical model")
        print("=" * 60)
        print("Model:")
        print("  y_v = P*cos(alpha)*mu + eps ;  y_h = 2*P*sin(alpha)*mu + eps  (through origin)")
        print("  eps ~ Normal(0, sigma^2 * P)  [proportional error; single shared sigma]")
        print("Priors:")
        print(f"  mu    ~ Normal({MU_PRIOR[0]}, {MU_PRIOR[1]})")
        print(f"  sigma ~ HalfNormal({SIGMA_PRIOR_SCALE})")
        print(f"Data: averaged (sensor-mean) experimental, angles {ANGLES_DEG} deg, "
              f"load <= {MAX_LOAD} kN")
        for d in ["v", "h"]:
            print(f"  {DIR_LABEL[d]:17s} n per angle = {counts[d]}, "
                  f"total = {sum(counts[d].values())}")
        print(f"  total observations = {y.size}")
        print(f"MCMC: NUTS, {N_CHAINS} chains x {N_SAMPLES} samples "
              f"({N_WARMUP} warmup), seed {SEED}")

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(SEED), P=jnp.asarray(P), alpha=jnp.asarray(alpha),
             chan=jnp.asarray(chan), y=jnp.asarray(y))
    post = mcmc.get_samples()
    mu_s = np.asarray(post["mu"])
    sigma_s = np.asarray(post["sigma"])

    if verbose:
        print("\nPosterior:")
        print(summarise("mu", mu_s))
        print(summarise("sigma", sigma_s))

    out = plot_predictions(data, mu_s, sigma_s)
    if verbose:
        print(f"\nFigure written: {out.relative_to(REPO)}")

    return {"mu": mu_s, "sigma": sigma_s, "data": data, "counts": counts,
            "n_obs": int(y.size), "fig": out}


if __name__ == "__main__":
    main()
