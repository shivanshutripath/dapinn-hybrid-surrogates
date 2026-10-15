"""dapinn.py - discontinuity-aware physics-informed neural networks (Section 3).

The surrogate of a hybrid system on [0, T] with J jumps is

    xi_hat_j(t) = N_{w_j}(s),   s = (t - tau_j) / (tau_{j+1} - tau_j),   j = 0..J,

one small network per flow interval (piece). The jump times tau_1..tau_J, the
weights w_0..w_J and the physical parameter theta are trained together on the
loss of Eq. (loss-terms):

    L = L_data + lambda_1 L_flow + lambda_2 L_guard + lambda_3 L_reset.

The number of jumps J, the initial mode and the guard that triggers each jump
are discrete and chosen by Algorithm 1 (search over J, train each admissible
candidate, keep the smallest J whose loss is below a tolerance).

A hybrid family is described by a `HybridFamily` object (see run_ball.py and
run_cover.py for two examples). Everything is in PyTorch, double precision,
on one CPU thread, so that a run is reproducible bit for bit on one machine.
"""
import itertools
import math
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("PYTHONHASHSEED", "0")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# The hybrid family H_theta (what is known, up to theta)
# ---------------------------------------------------------------------------
class HybridFamily:
    """The known structure of the hybrid system (Section 2).

    modes        list of mode names sigma
    guards       dict: mode -> list of guard names i in I(sigma)
    nu           dict: (mode, guard) -> next mode
    theta_spec   dict: name -> (initial value, lower bound, upper bound); theta is
                 kept inside its bounds by a sigmoid (the known set Theta)
    state_scale  typical size of each component of xi (used to make the
                 residuals dimensionless)
    rate_scale   expected size of the flow residual per component (None =
                 state_scale / T); reset_scale: expected size of the reset
                 residual (None = state_scale). Dividing every residual by its
                 expected size makes each loss term a normalized misfit.
    T            the horizon
    guard_scale  typical size of each guard value (dict guard -> float)

    Subclasses implement f, gamma, reset and h (all in torch, batched over the
    first axis of xi, with t a tensor of shape (N,) and theta a dict of tensors).
    """

    modes = []
    initial_modes = None        # modes allowed at t = 0 (None: any mode)
    guards = {}
    nu = {}
    theta_spec = {}
    state_scale = None
    rate_scale = None
    reset_scale = None
    guard_scale = {}
    T = 1.0
    n_state = 1

    def f(self, mode, xi, t, theta):          # flow in `mode`: (N, n) -> (N, n)
        raise NotImplementedError

    def gamma(self, mode, guard, xi, t, theta):   # guard value: (N, n) -> (N,)
        raise NotImplementedError

    def reset(self, mode, guard, xi, t, theta):   # reset map at one point: (n,) -> (n,)
        return xi

    def h(self, mode, xi, theta):             # output map: (N, n) -> (N, m)
        return xi                             # (may depend on the mode and on theta)

    # -- the admissible candidates (sigma_0, i_1, ..., i_J) of Algorithm 1 --
    def candidates(self, J):
        out = []
        for sigma0 in (self.initial_modes or self.modes):
            out += self._extend([sigma0], [], J)
        return out

    def _extend(self, modes, guards, J):
        if len(guards) == J:
            return [(modes[0], tuple(guards))]
        out = []
        for i in self.guards.get(modes[-1], []):
            out += self._extend(modes + [self.nu[(modes[-1], i)]], guards + [i], J)
        return out

    def mode_sequence(self, sigma0, guard_seq):
        modes = [sigma0]
        for i in guard_seq:
            modes.append(self.nu[(modes[-1], i)])
        return modes


# ---------------------------------------------------------------------------
# One interval network N_w(s)
# ---------------------------------------------------------------------------
class IntervalNet(torch.nn.Module):
    def __init__(self, n_out, scale, width=32, depth=2):
        super().__init__()
        layers, n_in = [], 1
        for _ in range(depth):
            layers.append(torch.nn.Linear(n_in, width))
            n_in = width
        self.hidden = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(n_in, n_out)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64))
        for layer in list(self.hidden) + [self.out]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, s):                     # s: (N, 1) in [0, 1]
        x = 2.0 * s - 1.0
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        return self.out(x) * self.scale


# ---------------------------------------------------------------------------
# The DA-PINN for one candidate structure
# ---------------------------------------------------------------------------
class DAPINN(torch.nn.Module):
    def __init__(self, family, sigma0, guard_seq, t_data, y_data, y_scale,
                 n_colloc=64, width=32, depth=2, weights=(1.0, 10.0, 10.0), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.family = family
        self.T = float(family.T)
        self.guard_seq = tuple(guard_seq)
        self.J = len(self.guard_seq)
        self.modes = family.mode_sequence(sigma0, self.guard_seq)
        self.nets = torch.nn.ModuleList(
            [IntervalNet(family.n_state, family.state_scale, width, depth) for _ in range(self.J + 1)])
        # Jump times: Delta_j = T softmax(alpha)_j keeps them ordered inside [0, T].
        self.alpha = torch.nn.Parameter(0.3 * torch.randn(self.J + 1))
        # theta inside its bounds: theta = lo + (hi - lo) sigmoid(raw).
        self.theta_names = list(family.theta_spec)
        raw = []
        for name in self.theta_names:
            init, lo, hi = family.theta_spec[name]
            p = min(max((init - lo) / (hi - lo), 1e-4), 1 - 1e-4)
            raw.append(math.log(p / (1 - p)))
        self.theta_raw = torch.nn.Parameter(torch.tensor(raw))
        self.register_buffer("t_data", torch.as_tensor(t_data, dtype=torch.float64))
        self.register_buffer("y_data", torch.as_tensor(y_data, dtype=torch.float64))
        self.register_buffer("y_scale", torch.as_tensor(y_scale, dtype=torch.float64))
        self.register_buffer("s_colloc", ((torch.arange(n_colloc) + 0.5) / n_colloc).reshape(-1, 1))
        self.register_buffer("state_scale", torch.as_tensor(family.state_scale, dtype=torch.float64))
        rate = family.rate_scale if family.rate_scale is not None else np.asarray(family.state_scale) / self.T
        self.register_buffer("rate_scale", torch.as_tensor(np.asarray(rate, dtype=float)))
        reset = family.reset_scale if family.reset_scale is not None else family.state_scale
        self.register_buffer("reset_scale", torch.as_tensor(np.asarray(reset, dtype=float)))
        self.weights = weights

    # -- trained quantities --
    def deltas(self):
        return self.T * torch.softmax(self.alpha, 0)

    def jump_times(self):
        """tau_0 = 0 < tau_1 < ... < tau_J < tau_{J+1} = T as one tensor."""
        d = self.deltas()
        return torch.cat([torch.zeros(1), torch.cumsum(d, 0)])

    def theta(self):
        out = {}
        for k, name in enumerate(self.theta_names):
            _, lo, hi = self.family.theta_spec[name]
            out[name] = lo + (hi - lo) * torch.sigmoid(self.theta_raw[k])
        return out

    def theta_values(self):
        return {k: float(v.detach()) for k, v in self.theta().items()}

    # -- evaluation --
    def piece(self, j, s, derivative=False):
        """xi_hat_j at rescaled times s (N, 1); optionally also d xi / ds."""
        if derivative:
            s = s.clone().requires_grad_(True)
        xi = self.nets[j](s)
        if not derivative:
            return xi
        dxi = torch.stack([torch.autograd.grad(xi[:, k].sum(), s, create_graph=True)[0][:, 0]
                           for k in range(xi.shape[1])], 1)
        return xi, dxi

    def trajectory(self, t):
        """xi_hat(t) and the piece index for plotting (numpy in, numpy out)."""
        t = np.asarray(t, dtype=float)
        tau = self.jump_times().detach().numpy()
        j = np.clip(np.searchsorted(tau[1:self.J + 1], t, side="right"), 0, self.J)
        xi = np.zeros((len(t), self.family.n_state))
        with torch.no_grad():
            for k in range(self.J + 1):
                m = j == k
                if m.any():
                    s = (t[m] - tau[k]) / (tau[k + 1] - tau[k])
                    xi[m] = self.piece(k, torch.as_tensor(s).reshape(-1, 1)).numpy()
        return xi, j

    def outputs(self, t):
        """y_hat(t) = h(mode, xi_hat(t), theta) for plotting (numpy)."""
        xi, j = self.trajectory(t)
        with torch.no_grad():
            theta = self.theta()
            y = None
            for k in range(self.J + 1):
                m = j == k
                if m.any():
                    yk = self.family.h(self.modes[k], torch.as_tensor(xi[m]), theta).numpy()
                    if y is None:
                        y = np.zeros((len(t), yk.shape[1]))
                    y[m] = yk
        return y, j

    # -- the loss of Eq. (loss-terms) --
    def loss_terms(self):
        fam, theta = self.family, self.theta()
        tau, delta = self.jump_times(), self.deltas()
        T = self.T

        # L_data: each measurement belongs to the piece j_k = number of jumps before t_k.
        jk = torch.clamp(torch.searchsorted(tau[1:self.J + 1].detach(), self.t_data, right=True), 0, self.J)
        l_data = torch.zeros(())
        for j in range(self.J + 1):
            m = jk == j
            if m.any():
                s = ((self.t_data[m] - tau[j]) / delta[j]).reshape(-1, 1)
                r = (fam.h(self.modes[j], self.piece(j, s), theta) - self.y_data[m]) / self.y_scale
                l_data = l_data + (r ** 2).sum()
        l_data = l_data / len(self.t_data)

        # L_flow and the flow condition of L_guard, on fixed collocation points
        # of the rescaled time s (Eq. flow-rescaled): the integral over the
        # piece is Delta_j times the mean over s.
        l_flow, l_guard = torch.zeros(()), torch.zeros(())
        for j in range(self.J + 1):
            xi, dxi_ds = self.piece(j, self.s_colloc, derivative=True)
            t = tau[j] + self.s_colloc[:, 0] * delta[j]
            xi_dot = dxi_ds / delta[j]
            r = (xi_dot - fam.f(self.modes[j], xi, t, theta)) / self.rate_scale
            l_flow = l_flow + (delta[j] / T) * (r ** 2).sum(1).mean()
            for i in fam.guards.get(self.modes[j], []):
                g = fam.gamma(self.modes[j], i, xi, t, theta) / fam.guard_scale[i]
                l_guard = l_guard + (delta[j] / T) * (torch.relu(-g) ** 2).mean()

        # Jump condition and reset at each jump.
        l_reset = torch.zeros(())
        one, zero = torch.ones(1, 1), torch.zeros(1, 1)
        for j in range(1, self.J + 1):
            i = self.guard_seq[j - 1]
            xi_minus = self.piece(j - 1, one)[0]
            xi_plus = self.piece(j, zero)[0]
            tj = tau[j].reshape(1)
            g = fam.gamma(self.modes[j - 1], i, xi_minus.reshape(1, -1), tj, theta)[0] / fam.guard_scale[i]
            l_guard = l_guard + torch.relu(g) ** 2
            r = (xi_plus - fam.reset(self.modes[j - 1], i, xi_minus, tj, theta)) / self.reset_scale
            l_reset = l_reset + (r ** 2).sum()
        return {"data": l_data, "flow": l_flow, "guard": l_guard, "reset": l_reset}

    def loss(self):
        terms = self.loss_terms()
        w = self.weights
        return terms["data"] + w[0] * terms["flow"] + w[1] * terms["guard"] + w[2] * terms["reset"], terms


# ---------------------------------------------------------------------------
# Initial jump times: least-squares segmentation of the measured outputs
# ---------------------------------------------------------------------------
def segment_init(t_data, y_data, y_scale, J, degree=2, min_points=4):
    """Split the data into J+1 consecutive segments, each fitted by a polynomial
    of the given degree in t (one per output), so that the total squared misfit
    (outputs divided by y_scale) is smallest. Dynamic programming, exact.

    Returns the J break times (midway between the last point of a segment and
    the first point of the next). This is a physics-free initial guess for the
    jump times; the DA-PINN then trains them jointly with everything else.
    """
    t = np.asarray(t_data, dtype=float)
    y = np.asarray(y_data, dtype=float).reshape(len(t), -1) / np.asarray(y_scale, dtype=float)
    K = len(t)
    if J == 0:
        return np.zeros(0)
    tn = (t - t[0]) / (t[-1] - t[0])
    V = np.vander(tn, degree + 1)
    cost = np.full((K, K), np.inf)              # cost[a, b]: segment of points a..b inclusive
    for a in range(K):
        for b in range(a + min_points - 1, K):
            coef, *_ = np.linalg.lstsq(V[a:b + 1], y[a:b + 1], rcond=None)
            cost[a, b] = np.sum((V[a:b + 1] @ coef - y[a:b + 1]) ** 2)
    best = np.full((J + 2, K), np.inf)          # best[n, b]: n segments covering points 0..b
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


def set_jump_times(model, taus):
    """Write the jump times tau_1..tau_J into the model (through alpha)."""
    taus = np.clip(np.sort(np.asarray(taus, dtype=float)), 1e-3 * model.T, (1 - 1e-3) * model.T)
    delta = np.diff(np.concatenate([[0.0], taus, [model.T]]))
    delta = np.maximum(delta, 1e-3 * model.T)
    with torch.no_grad():
        model.alpha.copy_(torch.as_tensor(np.log(delta / delta.sum())))


# ---------------------------------------------------------------------------
# Training: Adam, then L-BFGS
# ---------------------------------------------------------------------------
def train(model, n_adam=2000, lr=5e-3, n_lbfgs=1000, verbose=False):
    """Three stages: Adam on everything (weights, jump times, theta); L-BFGS on
    everything; L-BFGS on the weights and theta with the jump times fixed.
    The loss is discontinuous in the jump times where a measurement changes
    piece. With few measurements (ball) L-BFGS still refines the jump times;
    with dense data (cover) its line search stalls, and the last stage, whose
    loss is smooth, finishes the job."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_adam, eta_min=lr / 20)
    for it in range(n_adam):
        opt.zero_grad()
        total, terms = model.loss()
        total.backward()
        opt.step()
        sched.step()
        if verbose and (it % 500 == 0 or it == n_adam - 1):
            print(f"    adam {it:5d}  loss {float(total):.3e}  " +
                  "  ".join(f"{k} {float(v):.2e}" for k, v in terms.items()))
    for stage, params in enumerate([list(model.parameters()),
                                    [p for name, p in model.named_parameters() if name != "alpha"]]):
        if n_lbfgs <= 0:
            break
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=n_lbfgs, history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total, _ = model.loss()
            total.backward()
            return total

        opt.step(closure)
        if verbose:
            state = opt.state[opt._params[0]]
            print(f"    lbfgs stage {stage + 1}: {state['n_iter']} iterations, "
                  f"{state['func_evals']} evaluations, loss {float(closure()):.3e}")
    # (no torch.no_grad here: the flow residual needs d xi / ds from autograd)
    total, terms = model.loss()
    total = total.detach()
    if verbose:
        print(f"    final loss {float(total):.3e}  " +
              "  ".join(f"{k} {float(v):.2e}" for k, v in terms.items()))
    return float(total), {k: float(v.detach()) for k, v in terms.items()}


# ---------------------------------------------------------------------------
# Algorithm 1: choosing the number of jumps and the guards
# ---------------------------------------------------------------------------
def algorithm1(family, t_data, y_data, y_scale, J_max, tol, restarts=3, seed=0,
               train_kwargs=None, model_kwargs=None, jitter=0.02, min_points=4, verbose=True):
    """Try J = 0, 1, ..., J_max; for each admissible (sigma_0, i_1..i_J) train the
    surrogate (`restarts` initializations, keep the best); stop at the first J
    whose smallest loss is below `tol`.

    The jump times start from the segmentation of the data (segment_init, with
    segments of at least `min_points` measurements); restart r >= 1 perturbs
    them by N(0, (jitter T)^2) and uses another network seed.
    Returns (best model, record), where record lists every candidate trained.
    """
    train_kwargs = train_kwargs or {}
    model_kwargs = model_kwargs or {}
    record, best_overall = [], None
    for J in range(J_max + 1):
        best_J = None
        tau_init = segment_init(t_data, y_data, y_scale, J, min_points=min_points)
        for c, (sigma0, guard_seq) in enumerate(family.candidates(J)):
            for r in range(restarts):
                t0 = time.perf_counter()
                model = DAPINN(family, sigma0, guard_seq, t_data, y_data, y_scale,
                               seed=seed + 1000 * J + 100 * c + r, **model_kwargs)
                rng = np.random.default_rng(seed + 1000 * J + 100 * c + r)
                set_jump_times(model, tau_init + (jitter * family.T * rng.standard_normal(J) if r else 0.0))
                loss, terms = train(model, **train_kwargs)
                entry = dict(J=J, sigma0=sigma0, guards=list(guard_seq), restart=r, loss=loss,
                             terms=terms, theta=model.theta_values(),
                             jump_times=model.jump_times().detach().numpy()[1:J + 1].tolist(),
                             seconds=time.perf_counter() - t0)
                record.append(entry)
                if verbose:
                    print(f"J = {J}  {sigma0}" + "".join(f" -{i}-> " for i in guard_seq) +
                          f"  restart {r}: loss {loss:.3e} ({entry['seconds']:.0f} s)")
                if best_J is None or loss < best_J[0]:
                    best_J = (loss, model, entry)
        if best_overall is None or best_J[0] < best_overall[0]:
            best_overall = best_J
        if best_J[0] < tol:
            if verbose:
                print(f"Algorithm 1 stops at J = {J}: loss {best_J[0]:.3e} < tol {tol:.1e}")
            return best_J[1], record
    if verbose:
        print(f"no candidate reached tol {tol:.1e}; returning the smallest loss {best_overall[0]:.3e}")
    return best_overall[1], record


def environment():
    """Interpreter and library versions, written into the result JSON files."""
    out = dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__)
    try:
        import mujoco
        out["mujoco"] = mujoco.__version__
    except ImportError:
        pass
    try:
        import matplotlib
        out["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    return out
