"""dapinn_pin.py - the DA-PINN of the paper on the MuJoCo pin extraction.

The surrogate, its loss and Algorithm 1 are those of the paper, imported
unchanged from ../dapinn.py. This file only describes the hybrid family (what
is known, up to theta) and the data.

Press-fit pin: modes S (stick), K (slip), R (released)
  state xi = (delta, v): spring deflection of the puller, speed of the pin.
  Input: the hook height U(t) and its speed U'(t) (the robot's z encoder).
  The pin lift is x = U - delta.
    S:  delta' = U'(t) - v,    v' = 0      (v = 0 while stuck)
    K:  delta' = U'(t) - v,    m v' = F - m g - mu_k N
    R:  delta' = U'(t) - v,    m v' = F - m g
  with the spring force F = k delta + c (U'(t) - v).
  guards  S: breakaway  mu_s N - (F - m g)         (-> K)
          K: arrest     max(v, F - m g - mu_k N)   (-> S)
          K: exit       L - x                      (-> R)
  resets: identity. Measured: delta (a series-elastic tool measures its spring
  deflection). theta = (mu_s, mu_k). K has TWO guards: Algorithm 1 must
  choose between arrest and exit at every slip.

Usage (inside pin_extraction):
  python3 dapinn_pin.py                 # ~7 min on 7 processes
  python3 dapinn_pin.py --plot-only     # table and figure from saved results
  python3 dapinn_pin.py --workers 4     # number of processes
Writes: results/pin.json, results/pin_curves.npz, results/pin_table.tex,
        figs/pin_dapinn.pdf and .png
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dapinn as dp  # noqa: E402  (the paper's DA-PINN, unchanged)

DATA = HERE / "data" / "pin.npz"
RESULTS = HERE / "results"
FIGS = HERE / "figs"
SUBSAMPLE = 4                          # 4000 Hz physics -> 1000 Hz measurements
SEED = 0

# Settings. Tolerances follow the paper: every residual is divided by a
# tolerance, so a loss of about one means "met within tolerance", and
# Algorithm 1 accepts the first J whose loss is below TOL.
CFG = dict(T=1.48, noise=[2.5e-5], y_scale=[5e-5], J_max=9, restarts=3, jitter=0.003,
           min_points=4, weights=(1.0, 1.0, 1.0), tol=10.0)
TRAIN = dict(n_adam=2000, lr=5e-3, n_lbfgs=1000)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """The recorded pull of pin_experiment.py (made once if it is missing)."""
    if not DATA.exists():
        subprocess.run([sys.executable, str(HERE / "pin_experiment.py")], check=True, cwd=HERE)
    d = np.load(DATA)
    w = {k: d[k] for k in d.files if k not in ("constants", "results", "settings")}
    return w, json.loads(str(d["constants"])), json.loads(str(d["results"]))


class Signal:
    """A measured input, linearly interpolated, differentiable in t (torch)."""

    def __init__(self, t, y):
        self.t = torch.as_tensor(np.asarray(t, dtype=float))
        self.y = torch.as_tensor(np.asarray(y, dtype=float))
        self.slope = (self.y[1:] - self.y[:-1]) / (self.t[1:] - self.t[:-1])

    def __call__(self, t):
        i = torch.clamp(torch.searchsorted(self.t, t.detach().contiguous(), right=True) - 1, 0, len(self.t) - 2)
        return self.y[i] + self.slope[i] * (t - self.t[i])


def window():
    """Time from the start of the pull, the inputs, the hidden true states and
    events, the known constants, the reference theta."""
    w, c, res = load_data()
    t = w["pin_t"] - w["pin_t"][0]
    U = w["pin_tcp_z"] - c["pin_top0"] - c["puller_rest_length"]     # hook height, as spring stretch
    Ud = w["pin_tcp_vz"]                                             # hook speed (encoder)
    delta = U - w["pin_lift"]                                        # spring deflection (measured)
    truth = dict(delta=delta, v=w["pin_speed"], x=w["pin_lift"], F=w["pin_spring_force"],
                 state=w["pin_friction_state"])
    ev = res["pin"]["events_from_pull_start"]
    ref = dict(jump_times=[e[0] for e in ev], events=[e[1] for e in ev],
               mu_s=c["mu_static"], mu_k=c["mu_kinetic"], itinerary=res["pin"]["itinerary"])
    m = (t >= -1e-9) & (t <= CFG["T"] + 1e-9)
    t, U, Ud = t[m], U[m], Ud[m]
    truth = {k: v[m] for k, v in truth.items()}
    return t, U, Ud, truth, c, ref


def make_data(t, truth, seed=SEED):
    """Measurements of the deflection at 1000 Hz with Gaussian noise."""
    rng = np.random.default_rng(seed)
    tk = t[::SUBSAMPLE]
    yk = (truth["delta"][::SUBSAMPLE] + rng.normal(0, CFG["noise"][0], len(tk))).reshape(-1, 1)
    return tk, yk


# ---------------------------------------------------------------------------
# The hybrid family
# ---------------------------------------------------------------------------
class StickSlip(dp.HybridFamily):
    """A part held by static/kinetic friction over an engagement length,
    pulled through a spring by a robot whose position U(t) is measured.
    State (delta, v): spring deflection and part speed."""
    modes = ["S", "K", "R"]
    initial_modes = ["S"]                 # the part is stuck when the robot starts
    guards = {"S": ["breakaway"], "K": ["arrest", "exit"]}
    nu = {("S", "breakaway"): "K", ("K", "arrest"): "S", ("K", "exit"): "R"}
    n_state = 2

    def __init__(self, U, Ud, mass, k, c, load, normal, exit_at, T, theta_spec, scales):
        self.U, self.Ud = U, Ud
        self.mass, self.k, self.c, self.load, self.normal, self.exit_at = mass, k, c, load, normal, exit_at
        self.T = T
        self.theta_spec = theta_spec
        self.state_scale, self.rate_scale, self.reset_scale = scales["state"], scales["rate"], scales["reset"]
        self.guard_scale = scales["guard"]
        self.v_ref, self.f_ref = scales["arrest_ref"]

    def force(self, xi, t):
        return self.k * xi[:, 0] + self.c * (self.Ud(t) - xi[:, 1])

    def friction(self, theta):
        names = list(self.theta_spec)
        return theta[names[0]] * self.normal, theta[names[1]] * self.normal   # static, kinetic

    def f(self, mode, xi, t, theta):
        Ud = self.Ud(t)
        # delta' = U' - v in every mode (kinematics). In S the part does not
        # accelerate, and v = 0 comes from the arrest (and from the data); with
        # delta' = U' in S instead, v would be invisible there and could take
        # any value, which moves the damping term in the breakaway guard.
        if mode == "S":
            return torch.stack([Ud - xi[:, 1], torch.zeros_like(Ud)], 1)
        F_static, F_kinetic = self.friction(theta)
        acc = self.force(xi, t) - self.load - (F_kinetic if mode == "K" else 0.0)
        return torch.stack([Ud - xi[:, 1], acc / self.mass], 1)

    def gamma(self, mode, guard, xi, t, theta):
        F_static, F_kinetic = self.friction(theta)
        net = self.force(xi, t) - self.load
        if guard == "breakaway":
            return F_static - net
        if guard == "arrest":            # v reaches 0 while decelerating (max-composition)
            return torch.maximum(xi[:, 1] / self.v_ref, (net - F_kinetic) / self.f_ref)
        return self.exit_at - (self.U(t) - xi[:, 0])          # exit: lift x = U - delta reaches L

    def h(self, mode, xi, theta):
        return xi[:, :1]                 # the measured deflection


def make_family():
    t, U, Ud, truth, c, ref = window()
    return StickSlip(Signal(t, U), Signal(t, Ud), c["pin_mass"], c["puller_k"], c["puller_c"],
                     c["pin_mass"] * c["g"], c["fit_normal"], c["fit_length"], CFG["T"],
                     theta_spec={"mu_s": (0.37, 0.2, 0.8), "mu_k": (0.24, 0.1, 0.6)},
                     scales=dict(state=[0.01, 0.3], rate=[5e-3, 2.0], reset=[2e-5, 5e-3],
                                 guard={"breakaway": 0.1, "arrest": 0.01, "exit": 1e-4},
                                 arrest_ref=(0.3, 10.0)))


# ---------------------------------------------------------------------------
# Initial jump times: the paper's least-squares segmentation, made fast
# ---------------------------------------------------------------------------
def segment_costs(t_data, y_data, y_scale, degree=2, min_points=4):
    """cost[a, b] of dapinn.segment_init (polynomial least squares of points
    a..b), computed with running sums instead of one lstsq per segment: the
    same numbers, in O(K^2) vector operations instead of K^2 small lstsq."""
    t = np.asarray(t_data, dtype=float)
    y = np.asarray(y_data, dtype=float).reshape(len(t), -1) / np.asarray(y_scale, dtype=float)
    K, p = len(t), degree + 1
    tn = (t - t[0]) / (t[-1] - t[0])
    V = np.vander(tn, p)
    G = np.concatenate([np.zeros((1, p, p)), np.cumsum(V[:, :, None] * V[:, None, :], 0)])
    B = np.concatenate([np.zeros((1, p, y.shape[1])), np.cumsum(V[:, :, None] * y[:, None, :], 0)])
    S = np.concatenate([[0.0], np.cumsum((y ** 2).sum(1))])
    cost = np.full((K, K), np.inf)
    for a in range(K):
        b = np.arange(a + min_points - 1, K)
        if len(b) == 0:
            break
        Gab = G[b + 1] - G[a]
        Bab = B[b + 1] - B[a]
        coef = np.linalg.solve(Gab + 1e-12 * np.eye(p), Bab)
        cost[a, b] = np.maximum(S[b + 1] - S[a] - np.einsum("kpj,kpj->k", Bab, coef), 0.0)
    return cost


def segment_from_costs(t, cost, J):
    """The dynamic program of dapinn.segment_init on a precomputed cost matrix."""
    K = len(t)
    if J == 0:
        return np.zeros(0)
    best = np.full((J + 2, K), np.inf)
    arg = np.zeros((J + 2, K), dtype=int)
    best[1] = cost[0]
    for n in range(2, J + 2):
        for b in range(K):
            cands = best[n - 1, :b] + cost[1:b + 1, b]
            if len(cands):
                k = int(np.argmin(cands))
                best[n, b], arg[n, b] = cands[k], k
    breaks, b = [], K - 1
    for n in range(J + 1, 1, -1):
        k = arg[n, b]
        breaks.append(0.5 * (t[k] + t[k + 1]))
        b = k
    return np.array(breaks[::-1])


# ---------------------------------------------------------------------------
# Algorithm 1, with the candidates trained in parallel
# ---------------------------------------------------------------------------
_CACHE = {}


def _setup():
    if "pin" not in _CACHE:
        t, U, Ud, truth, c, ref = window()
        tk, yk = make_data(t, truth)
        _CACHE["pin"] = (make_family(), tk, yk)
    return _CACHE["pin"]


def _train_one(task):
    """One (J, candidate, restart) of Algorithm 1: the same seeds and the same
    jump-time initialization as dapinn.algorithm1."""
    J, c, sigma0, guard_seq, r, tau_init = task
    fam, tk, yk = _setup()
    t0 = time.perf_counter()
    seed = SEED + 1000 * J + 100 * c + r
    model = dp.DAPINN(fam, sigma0, guard_seq, tk, yk, CFG["y_scale"], seed=seed, weights=CFG["weights"])
    rng = np.random.default_rng(seed)
    tau = np.asarray(tau_init) + (CFG["jitter"] * fam.T * rng.standard_normal(J) if r else 0.0)
    dp.set_jump_times(model, tau)
    loss, terms = dp.train(model, **TRAIN)
    entry = dict(J=J, sigma0=sigma0, guards=list(guard_seq), restart=r, loss=loss, terms=terms,
                 theta=model.theta_values(), jump_times=model.jump_times().detach().numpy()[1:J + 1].tolist(),
                 seconds=time.perf_counter() - t0)
    state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    return entry, state


def algorithm1(workers):
    """Algorithm 1 of the paper. Candidates with the same J are independent,
    so every (J, candidate, restart) up to J_max is trained in parallel; the
    rule of Algorithm 1 is then applied: the first J whose smallest loss is
    below the tolerance (the losses for larger J are kept for the figure)."""
    fam, tk, yk = _setup()
    cost = segment_costs(tk, yk, CFG["y_scale"], min_points=CFG["min_points"])
    tasks = []
    for J in range(CFG["J_max"] + 1):
        tau_init = segment_from_costs(tk, cost, J)
        for c, (sigma0, guard_seq) in enumerate(fam.candidates(J)):
            for r in range(CFG["restarts"]):
                tasks.append((J, c, sigma0, tuple(guard_seq), r, tau_init.tolist()))
    print(f"[pin] {len(tasks)} trainings ({CFG['J_max'] + 1} values of J) on {workers} processes")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(_train_one, tasks))
    secs = time.perf_counter() - t0
    record = [e for e, _ in out]
    for e in record:
        print(f"  J = {e['J']}  {e['sigma0']}" + "".join(f" -{i}-> " for i in e["guards"]) +
              f"  restart {e['restart']}: loss {e['loss']:.3e} ({e['seconds']:.0f} s)")
    best_by_J = {}
    for e, st in out:
        if e["J"] not in best_by_J or e["loss"] < best_by_J[e["J"]][0]["loss"]:
            best_by_J[e["J"]] = (e, st)
    chosen = None
    for J in sorted(best_by_J):
        if best_by_J[J][0]["loss"] < CFG["tol"]:
            chosen = J
            break
    if chosen is None:
        chosen = min(best_by_J, key=lambda J: best_by_J[J][0]["loss"])
        print(f"[pin] no J below the tolerance; the smallest loss is at J = {chosen}")
    e, st = best_by_J[chosen]
    model = dp.DAPINN(fam, e["sigma0"], e["guards"], tk, yk, CFG["y_scale"], weights=CFG["weights"])
    model.load_state_dict({k: torch.as_tensor(v) for k, v in st.items()})
    print(f"[pin] Algorithm 1 stops at J = {chosen}: {model.modes}, loss {e['loss']:.3g} ({secs:.0f} s wall)")
    return model, record, best_by_J, secs


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(workers):
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    t, U, Ud, truth, c, ref = window()
    tk, yk = make_data(t, truth)
    print(f"\n[pin] window {CFG['T']} s, {len(t)} physics steps, {len(tk)} measurements; "
          f"reference events (s): {np.round(ref['jump_times'], 4).tolist()}")
    model, record, best_by_J, secs = algorithm1(workers)
    theta = model.theta_values()
    tau_hat = model.jump_times().detach().numpy()[1:model.J + 1]
    xi_hat, piece = model.trajectory(t)
    tau_ref = np.array(ref["jump_times"])
    match = len(tau_ref) == model.J
    true_modes = list(ref["itinerary"])
    out = dict(name="pin", settings=dict(CFG, subsample=SUBSAMPLE, seed=SEED, train=TRAIN),
               environment=dp.environment(), reference=ref, J=model.J, modes=list(model.modes),
               jump_times=tau_hat.tolist(),
               jump_time_error_ms=(1e3 * np.abs(tau_hat - tau_ref)).tolist() if match else None,
               theta=theta, loss=best_by_J[model.J][0]["loss"], terms=best_by_J[model.J][0]["terms"],
               loss_by_J={str(J): best_by_J[J][0]["loss"] for J in sorted(best_by_J)},
               seconds=secs, record=record,
               rel_l2={"delta": rel_l2(xi_hat[:, 0], truth["delta"]), "v": rel_l2(xi_hat[:, 1], truth["v"])},
               true_modes=true_modes, modes_correct=list(model.modes) == true_modes)
    with open(RESULTS / "pin.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    np.savez(RESULTS / "pin_curves.npz", t=t, tk=tk, yk=yk, xi_hat=xi_hat, piece=piece,
             tau_hat=tau_hat, tau_ref=tau_ref, modes=np.array(model.modes),
             truth0=truth["delta"], truth1=truth["v"],
             loss_by_J=np.array([out["loss_by_J"][str(J)] for J in sorted(best_by_J)]))
    print(f"[pin] J = {model.J}, modes {''.join(model.modes)} (true {''.join(true_modes)}), "
          f"theta {({k: round(v, 4) for k, v in theta.items()})}, jump-time errors (ms) "
          f"{np.round(out['jump_time_error_ms'], 2).tolist() if match else 'n/a'}")
    write_table()
    make_figure()


# ---------------------------------------------------------------------------
# Table and figure
# ---------------------------------------------------------------------------
def write_table():
    out = json.load(open(RESULTS / "pin.json"))
    ref, theta, tau = out["reference"], out["theta"], out["jump_times"]
    rows = []
    if out["jump_time_error_ms"] is not None:
        for i, (r, e) in enumerate(zip(ref["jump_times"], tau)):
            rows.append((f"$\\tau_{{{i + 1}}}$ ({ref['events'][i]}), s", r, e, "{:.4f}"))
    rows += [("$\\mu_s$", ref["mu_s"], theta["mu_s"], "{:.4f}"), ("$\\mu_k$", ref["mu_k"], theta["mu_k"], "{:.4f}")]
    lines = [f"{lab} & {fmt.format(r)} & {fmt.format(e)} \\\\" for lab, r, e, fmt in rows]
    loss_row = ", ".join(f"$J={J}$: {v:.3g}" for J, v in out["loss_by_J"].items())
    with open(RESULTS / "pin_table.tex", "w") as f:
        f.write(f"% generated by dapinn_pin.py; modes {''.join(out['modes'])}; loss by J: {loss_row}; "
                f"rel L2 {out['rel_l2']}; {out['seconds']:.0f} s\n")
        f.write("\\begin{tabular}{lcc}\\hline\nquantity & MuJoCo reference & DA-PINN \\\\\n\\hline\n"
                + "\n".join(lines) + "\n\\hline\\end{tabular}\n")
    print("\n".join(lines))
    print("loss by J:", loss_row)


def make_figure():
    c = np.load(RESULTS / "pin_curves.npz")
    out = json.load(open(RESULTS / "pin.json"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    colors = {"S": "#dde8ff", "K": "#ffe2d6", "R": "0.9"}
    modes = [str(m) for m in c["modes"]]
    T = CFG["T"]
    bounds = np.concatenate([[0.0], c["tau_hat"], [T]])
    const = window()[4]
    conv = [(const["puller_k"], "spring force $k\\delta$ (N)", "(a) measured: puller deflection"),
            (1.0, "pin speed $v$ (m/s)", "(b) not measured: pin speed")]
    for col, ax in enumerate(axes[:2]):
        scale, ylabel, title = conv[col]
        for j, mode in enumerate(modes):
            ax.axvspan(bounds[j], bounds[j + 1], color=colors[mode], lw=0)
            if bounds[j + 1] - bounds[j] > 0.04 * T:
                ax.text(0.5 * (bounds[j] + bounds[j + 1]), 0.93, mode, transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=6, color="0.3")
        if col == 0:
            ax.plot(c["tk"], c["yk"][:, 0] * scale, ".", color="0.55", ms=1.2, label="data (1 kHz)")
        ax.plot(c["t"], c[f"truth{col}"] * scale, "k-", lw=0.8, label="MuJoCo")
        for j in range(len(modes)):
            m = c["piece"] == j
            ax.plot(c["t"][m], c["xi_hat"][m, col] * scale, "--", color="tab:blue", lw=1,
                    label="DA-PINN" if j == 0 else None)
        for i, tau in enumerate(c["tau_ref"]):
            ax.axvline(tau, color="tab:red", lw=0.5, ls=":", label="MuJoCo events" if i == 0 else None)
        ax.set_xlabel("$t$ (s)"), ax.set_ylabel(ylabel), ax.set_title(title, fontsize=8), ax.set_xlim(0, T)
    axes[0].legend(loc="lower right", fontsize=6)
    ax = axes[2]
    Js = np.arange(len(c["loss_by_J"]))
    ax.semilogy(Js, c["loss_by_J"], "o-", color="tab:blue", ms=3)
    ax.axhline(CFG["tol"], color="k", ls="--", lw=0.8)
    ax.axvline(out["J"], color="tab:red", lw=0.6)
    ax.set_xlabel("number of jumps $J$"), ax.set_ylabel("loss $L$ (best start)")
    ax.set_xticks(Js), ax.set_title(f"(c) Algorithm 1 stops at $J={out['J']}$", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "pin_dapinn.pdf")
    fig.savefig(FIGS / "pin_dapinn.png", dpi=150)
    plt.close(fig)
    print("saved", FIGS / "pin_dapinn.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plot-only", action="store_true", help="table and figure from saved results")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    if args.plot_only:
        write_table()
        make_figure()
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
