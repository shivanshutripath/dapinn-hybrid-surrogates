"""run_ball.py - Experiment 1: the bouncing ball (paper, Numerical Experiments).

Hybrid system: xi = (h, v), one mode, flow (h' = v, v' = -g), guard gamma = h,
reset (h, v) -> (h, -e v). The restitution e is the unknown parameter theta.
Data: K noisy measurements of the height h at random times in [0, T].
Unknown and found by the DA-PINN: e, the number of impacts J (Algorithm 1),
the impact times, the initial state and the unmeasured velocity.

Baseline (Proposition 1): a single smooth network on [0, T] trained on the same
data with the penalty-regularized contact model
    v' = -g + k [-h]_+ - c v 1[h < 0],   k = 1e4,  c chosen so that e = 0.8.

Normalization: every residual is divided by its tolerance (data: the noise
level sigma; flow / reset / guard: 1% of the natural scale of the state), so
each loss term is a mean squared misfit in units of the tolerance and a total
loss of order one means "everything fits within tolerance" (tolerance eps = 10
in Algorithm 1).

Usage (inside bouncing_ball):  python3 run_ball.py              (about 15 min, 1 CPU thread)
                              python3 run_ball.py --plot-only  (redraw from saved results)
Writes: results/ball.json, results/ball_curves.npz, results/ball_table.tex, figs/ball_dapinn.pdf and .png
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dapinn as dp  # noqa: E402  (the paper's DA-PINN)

RESULTS = HERE / "results"
FIGS = HERE / "figs"

# --- the true system and the data ---
G, E_TRUE, T, H0, V0 = 9.81, 0.8, 2.5, 1.0, 0.0
K, SIGMA = 50, 0.01            # measurements of h, noise std (m)
SEEDS = [0, 1, 2]              # noise realizations (and network initializations)
TOL, J_MAX, RESTARTS = 10.0, 6, 3
STATE_SCALE = np.array([1.0, 5.0])      # natural size of (h, v)
REL_TOL = 0.01                          # physics tolerance: 1% of the natural scale
WEIGHTS = (10.0, 10.0, 10.0)            # lambda_1, lambda_2, lambda_3


class Ball(dp.HybridFamily):
    modes = ["flight"]
    guards = {"flight": ["impact"]}
    nu = {("flight", "impact"): "flight"}
    theta_spec = {"e": (0.5, 0.3, 1.0)}      # initial guess 0.5, known range
    state_scale = STATE_SCALE.tolist()
    rate_scale = (REL_TOL * STATE_SCALE / T).tolist()     # tolerance on xi' - f
    reset_scale = (REL_TOL * STATE_SCALE).tolist()        # tolerance on xi+ - R(xi-)
    guard_scale = {"impact": REL_TOL * STATE_SCALE[0]}    # tolerance on h at the jump
    T = T
    n_state = 2

    def f(self, mode, xi, t, theta):
        return torch.stack([xi[:, 1], -G * torch.ones_like(xi[:, 0])], 1)

    def gamma(self, mode, guard, xi, t, theta):
        return xi[:, 0]

    def reset(self, mode, guard, xi, t, theta):
        return torch.stack([xi[0], -theta["e"] * xi[1]])

    def h(self, mode, xi, theta):
        return xi[:, :1]                      # only the height is measured


def exact(t, e=E_TRUE):
    """Height, velocity and impact times of the true solution (closed form)."""
    t = np.asarray(t, dtype=float)
    h, v, taus = np.zeros_like(t), np.zeros_like(t), []
    tj, hj, vj = 0.0, H0, V0
    while True:
        tf = (vj + math.sqrt(vj ** 2 + 2 * G * hj)) / G      # time of flight
        m = (t >= tj) & (t <= tj + tf)
        s = t[m] - tj
        h[m], v[m] = hj + vj * s - G * s ** 2 / 2, vj - G * s
        if tj + tf > T:
            return h, v, np.array(taus)
        taus.append(tj + tf)
        vj, hj, tj = -e * (vj - G * tf), 0.0, tj + tf


def make_data(seed):
    rng = np.random.default_rng(seed)
    tk = np.sort(rng.uniform(0, T, K))
    hk, _, _ = exact(tk)
    return tk, (hk + rng.normal(0, SIGMA, K)).reshape(-1, 1)


# --- the smooth baselines (Proposition 1) ---
class SmoothNet(torch.nn.Module):
    """One smooth network h(t) on [0, T]; v = h', a = h'' by autograd.
    lam = 0: pure data fit.  lam = 1: smooth PINN with the penalty contact model
    h'' = -g + k [-h]_+ - c h' 1[h < 0] (k = 1e4, c from e), with the same
    normalization (data / sigma, flow / rate_scale) and weight as the DA-PINN."""

    def __init__(self, tk, yk, k_pen=1e4, e=E_TRUE, width=64, depth=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = dp.IntervalNet(1, [STATE_SCALE[0]], width, depth)
        self.register_buffer("tk", torch.as_tensor(tk / T).reshape(-1, 1))
        self.register_buffer("yk", torch.as_tensor(yk))
        zeta = abs(math.log(e)) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
        self.k, self.c = k_pen, 2 * zeta * math.sqrt(k_pen)     # unit mass
        self.register_buffer("s", ((torch.arange(1024) + 0.5) / 1024).reshape(-1, 1))
        self.lam = 1.0

    def forward(self, s):                     # -> (h, v) at s = t / T
        s = s.clone().requires_grad_(True)
        h = self.net(s)[:, 0]
        v = torch.autograd.grad(h.sum(), s, create_graph=True)[0][:, 0] / T
        return h, v, s

    def loss(self):
        r = (self.net(self.tk)[:, 0] - self.yk[:, 0]) / SIGMA
        l_data = (r ** 2).mean()
        h, v, s = self.forward(self.s)
        a = torch.autograd.grad(v.sum(), s, create_graph=True)[0][:, 0] / T
        a_model = -G + self.k * torch.relu(-h) - self.c * v * (h < 0).double()
        l_flow = (((a - a_model) / Ball.rate_scale[1]) ** 2).mean()
        return l_data + self.lam * WEIGHTS[0] * l_flow, {"data": l_data, "flow": l_flow}

    def evaluate(self, t):
        h, v, _ = self.forward(torch.as_tensor(t / T).reshape(-1, 1))
        return np.stack([h.detach().numpy(), v.detach().numpy()], 1)


def train_smooth(model, physics):
    """Data-only fit (Adam + L-BFGS); if `physics`, then the full loss from that
    warm start (Adam + L-BFGS), the same budget as one DA-PINN candidate."""
    model.lam = 0.0
    loss, terms = dp.train(model, n_adam=2000, lr=2e-3, n_lbfgs=1000)
    if physics:
        model.lam = 1.0
        loss, terms = dp.train(model, n_adam=2000, lr=1e-3, n_lbfgs=1000)
    return loss, terms


# --- metrics ---
def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def downward_crossings(h, level=0.0):
    return int(np.sum((h[:-1] > level) & (h[1:] <= level)))


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    t_fine = np.linspace(0, T, 2001)
    h_ex, v_ex, taus = exact(t_fine)
    print(f"true impact times: {taus}")
    out = {"settings": dict(g=G, e=E_TRUE, T=T, h0=H0, v0=V0, K=K, sigma=SIGMA, tol=TOL,
                            J_max=J_MAX, restarts=RESTARTS, seeds=SEEDS),
           "environment": dp.environment(),
           "true_jump_times": taus.tolist(), "runs": []}
    keep = {}
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        tk, yk = make_data(seed)
        t0 = time.perf_counter()
        model, record = dp.algorithm1(Ball(), tk, yk, [SIGMA], J_MAX, TOL, restarts=RESTARTS, seed=seed,
                                      train_kwargs=dict(n_adam=2000, lr=5e-3, n_lbfgs=1000),
                                      model_kwargs=dict(weights=WEIGHTS))
        secs = time.perf_counter() - t0
        xi, _ = model.trajectory(t_fine)
        tau_hat = model.jump_times().detach().numpy()[1:model.J + 1]
        theta = model.theta_values()
        run = dict(seed=seed, J=model.J, jump_times=tau_hat.tolist(),
                   jump_time_error_ms=(1000 * np.abs(tau_hat - taus)).tolist() if model.J == len(taus) else None,
                   e=theta["e"], e_rel_error_pct=100 * abs(theta["e"] - E_TRUE) / E_TRUE,
                   rel_l2_h=rel_l2(xi[:, 0], h_ex), rel_l2_v=rel_l2(xi[:, 1], v_ex),
                   loss_by_J={str(J): min(r["loss"] for r in record if r["J"] == J)
                              for J in sorted({r["J"] for r in record})},
                   final_terms=min(record, key=lambda r: r["loss"])["terms"],
                   seconds=secs, record=record)
        print(f"DA-PINN: J = {model.J}, e = {theta['e']:.4f}, jump-time errors (ms) = "
              f"{run['jump_time_error_ms']}, rel L2 h = {run['rel_l2_h']:.2e}, v = {run['rel_l2_v']:.2e}")

        # Baselines: one smooth network, data only / with the penalty model.
        xs = {}
        for name, physics in [("smooth_fit", False), ("smooth_pinn", True)]:
            t0 = time.perf_counter()
            smooth = SmoothNet(tk, yk, seed=seed)
            s_loss, s_terms = train_smooth(smooth, physics)
            xs[name] = smooth.evaluate(t_fine)
            run[name] = dict(loss=s_loss, terms=s_terms, rel_l2_h=rel_l2(xs[name][:, 0], h_ex),
                             rel_l2_v=rel_l2(xs[name][:, 1], v_ex), impacts=downward_crossings(xs[name][:, 0]),
                             seconds=time.perf_counter() - t0)
            print(f"{name}: loss = {s_loss:.3e}, rel L2 h = {run[name]['rel_l2_h']:.2f}, "
                  f"v = {run[name]['rel_l2_v']:.2f}, downward zero crossings = {run[name]['impacts']}")
        out["runs"].append(run)
        if seed == SEEDS[0]:
            keep = dict(tk=tk, yk=yk, xi=xi, xs=xs, tau_hat=tau_hat, model=model)

    # --- summary over seeds (mean, std) ---
    runs = out["runs"]

    def ms_tuple(vals):
        vals = np.array(vals, dtype=float)
        return float(vals.mean()), float(vals.std())

    def ms(key, sub=None):
        return ms_tuple([(r[sub][key] if sub else r[key]) for r in runs])

    summary = {
        "J_found": [r["J"] for r in runs],
        "max_jump_time_error_ms": ms_tuple([max(r["jump_time_error_ms"]) for r in runs]),
        "e_rel_error_pct": ms("e_rel_error_pct"),
        "e": ms("e"),
        "loss": ms_tuple([min(r["loss_by_J"].values()) for r in runs]),
        "rel_l2_h": ms("rel_l2_h"), "rel_l2_v": ms("rel_l2_v"),
        "seconds_dapinn": ms("seconds"),
    }
    for name in ["smooth_fit", "smooth_pinn"]:
        summary[name] = {"loss": ms("loss", name), "rel_l2_h": ms("rel_l2_h", name),
                         "rel_l2_v": ms("rel_l2_v", name), "impacts": [r[name]["impacts"] for r in runs],
                         "seconds": ms("seconds", name)}
    out["summary"] = summary
    with open(RESULTS / "ball.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nsummary:", json.dumps(summary, indent=1, default=float))
    # curves of seed 0 for the figure (saved, so that the figure can be redrawn)
    np.savez(RESULTS / "ball_curves.npz", t_fine=t_fine, h_ex=h_ex, v_ex=v_ex, xi=keep["xi"],
             xs_fit=keep["xs"]["smooth_fit"], xs_pinn=keep["xs"]["smooth_pinn"], tk=keep["tk"],
             yk=keep["yk"], tau_hat=keep["tau_hat"],
             loss_by_J=np.array([runs[0]["loss_by_J"][str(j)] for j in sorted(int(j) for j in runs[0]["loss_by_J"])]))
    write_table()
    make_figure()
    print("saved", FIGS / "ball_dapinn.pdf", "and", RESULTS / "ball.json")


def write_table():
    """results/ball_table.tex from results/ball.json:
    surrogate & impacts & max tau error (ms) & e & loss & rel. err. h & rel. err. v."""
    summary = json.load(open(RESULTS / "ball.json"))["summary"]

    def pm(m, s, fmt="{:.2f}"):
        return "$" + fmt.format(m) + " \\pm " + fmt.format(s) + "$"

    def pct(m, s):
        d = 1 if 100 * m < 10 else 0
        return f"${100 * m:.{d}f} \\pm {100 * s:.{d}f}$\\%"

    def sci1(x):
        mant, ex = f"{x:.1e}".split("e")
        return f"${mant}{{\\times}}10^{{{int(ex)}}}$"

    rows = ["DA-PINN & " + "/".join(str(j) for j in summary["J_found"]) + " & "
            + pm(*summary["max_jump_time_error_ms"], "{:.1f}") + " & " + pm(*summary["e"], "{:.4f}") + " & "
            + pm(*summary["loss"], "{:.1f}") + " & " + pct(*summary["rel_l2_h"]) + " & "
            + pct(*summary["rel_l2_v"]) + " \\\\"]
    for name, label in [("smooth_fit", "smooth net, data only"),
                        ("smooth_pinn", "smooth PINN, penalty")]:
        s = summary[name]
        rows.append(f"{label} & " + "/".join(str(j) for j in s["impacts"]) + " & -- & -- & "
                    + sci1(s["loss"][0]) + " & " + pct(*s["rel_l2_h"]) + " & " + pct(*s["rel_l2_v"]) + " \\\\")
    header = ("surrogate & impacts & $\\max_j|\\hat\\tau_j-\\tau_j|$ (ms) & $\\hat e$ & loss & "
              "rel.\\ err.\\ $h$ & rel.\\ err.\\ $v$ \\\\")
    with open(RESULTS / "ball_table.tex", "w") as f:
        f.write("% generated by run_ball.py\n\\begin{tabular}{lcccccc}\\hline\n" + header + "\n\\hline\n"
                + "\n".join(rows) + "\n\\hline\\end{tabular}\n")
    print("\n".join(rows))


def make_figure():
    """Figure ball_dapinn.pdf from results/ball_curves.npz (seed 0)."""
    path = RESULTS / "ball_curves.npz"
    if not path.exists():
        raise SystemExit("missing results/ball_curves.npz; run python3 run_ball.py (without --plot-only)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = np.load(path)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, k, ex, ylim, yl, title in [(axes[0], 0, c["h_ex"], (-0.6, 1.6), "$h$ (m)", "(a) height (measured)"),
                                       (axes[1], 1, c["v_ex"], (-8, 8), "$v$ (m/s)", "(b) velocity (not measured)")]:
        ax.plot(c["t_fine"], ex, "k-", lw=1.8, alpha=0.35, label="exact")
        ax.plot(c["t_fine"], c["xi"][:, k], "b-", lw=1, label="DA-PINN")
        ax.plot(c["t_fine"], c["xs_fit"][:, k], "g-.", lw=0.8, label="smooth network, data only")
        ax.plot(c["t_fine"], c["xs_pinn"][:, k], "r--", lw=0.8, label="smooth PINN, penalty model")
        if k == 0:
            ax.plot(c["tk"], c["yk"][:, 0], "ko", ms=2.5, label="data")
        for tau in c["tau_hat"]:
            ax.axvline(tau, color="b", lw=0.5, ls=":")
        ax.set_ylim(*ylim), ax.set_xlabel("$t$ (s)"), ax.set_ylabel(yl), ax.set_title(title, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.2, handlelength=2.2)
    ax = axes[2]
    Js = np.arange(len(c["loss_by_J"]))
    ax.semilogy(Js, c["loss_by_J"], "bo-", ms=4)
    ax.axhline(TOL, color="k", ls="--", lw=0.8)
    ax.text(Js[0] + 0.1, TOL * 1.4, r"tolerance $\varepsilon$", fontsize=7)
    ax.set_xlabel("number of jumps $J$"), ax.set_ylabel("loss $L$ (best of 3 starts)")
    ax.set_xticks(Js), ax.set_title("(c) Algorithm 1", fontsize=8)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "ball_dapinn.pdf")
    fig.savefig(FIGS / "ball_dapinn.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:        # redraw table and figure from saved results
        write_table()
        make_figure()
    else:
        main()
