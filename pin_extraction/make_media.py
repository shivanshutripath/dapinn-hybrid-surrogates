"""make_media.py - video of the pin extraction with live force and displacement graphs.

Runs the MuJoCo simulation of pin_experiment.py twice. The simulation is
deterministic, so the first run gives the event times, and the second run
renders frames on a schedule that slows down around every slip:

  approach of the robot     2x speed       (0.08 s of simulation per frame)
  the pull, pin stuck       real time      (0.04 s per frame)
  each slip, and the exit   20x slow motion (2 ms per frame), from 5 ms before
                            the breakaway to 25 ms after the arrest (60 ms
                            after the exit, to show the ringing)

Each frame shows the MuJoCo render next to the spring force and the
displacements (hook and pin) recorded so far, colored by the state of the pin.

Saves (in ../media)
  pin_extraction.gif              the video, 25 frames per second (for the README)
  pin_extraction.mp4              the same video in H.264 (needs ffmpeg on the PATH)
  pin_force_displacement.png/.pdf static graphs of the whole pull: force and
                                  displacement over time, and force over pin lift

Usage (inside pin_extraction)
  python3 make_media.py              # ~30 s
  python3 make_media.py --graphs-only
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

import pin_experiment as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "media"

FPS = 25
APPROACH_DT, NORMAL_DT, SLOW_DT = 0.08, 0.04, 0.002   # s of simulation per frame
SLOW_BEFORE, SLOW_AFTER, SLOW_AFTER_EXIT = 0.005, 0.025, 0.060
END_HOLD = 2.0                                         # s the last frame stays on screen
RENDER_W, RENDER_H = 600, 450
CAMERA = dict(lookat=[0.0, 0.0, 0.488], distance=0.205, azimuth=60, elevation=-15)

# Colors: three validated categorical slots for the three states of the pin,
# ink for text, a neutral for the robot's hook and the reference lines.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
STATE_NAME = {0: "stuck", 1: "slipping", 2: "released"}
STATE_COLOR = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
HOOK_COLOR = "#6b6a66"


def schedule(events, pull_t0, t_end):
    """Simulation times at which a frame is taken."""
    windows = []
    for (t, e), nxt in zip(events, events[1:] + [(None, None)]):
        if e == "breakaway":
            t_stop, after = nxt if nxt[0] is not None else (t, None)
            windows.append((pull_t0 + t - SLOW_BEFORE,
                            pull_t0 + t_stop + (SLOW_AFTER_EXIT if after == "exit" else SLOW_AFTER)))
    times, t = [], 0.0
    while t <= t_end:
        times.append(t)
        slow = any(a <= t < b for a, b in windows)
        t += SLOW_DT if slow else (APPROACH_DT if t < pull_t0 else NORMAL_DT)
        nxt_slow = [a for a, _ in windows if t < a < t + NORMAL_DT]
        if not slow and nxt_slow:           # land exactly on the start of a slow window
            t = nxt_slow[0]
    return np.array(times), windows


class Grabber:
    """on_step callback of pin_experiment.run: renders a frame at each scheduled time."""

    def __init__(self, times, windows):
        self.times, self.windows, self.k = times, windows, 0
        self.renderer, self.frames = None, []
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = CAMERA["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = (CAMERA["distance"], CAMERA["azimuth"],
                                                                   CAMERA["elevation"])

    def __call__(self, model, data, law, text, rows):
        if self.k >= len(self.times) or data.time < self.times[self.k] - 1e-9:
            return
        while self.k < len(self.times) and self.times[self.k] <= data.time + 1e-9:
            self.k += 1
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
        self.renderer.update_scene(data, camera=self.cam)
        self.frames.append(dict(image=self.renderer.render(), time=data.time, step=text, n=len(rows),
                                state=law.code(), slips=sum(e == "breakaway" for _, e in law.events),
                                slow=any(a <= data.time < b for a, b in self.windows)))


def curves(w, const):
    """Time since the pull started, spring force, hook and pin displacement (mm)."""
    c = pe.COL
    t = w[:, c["t"]] - w[0, c["t"]]
    hook = 1e3 * (w[:, c["tcp_z"]] - w[0, c["tcp_z"]])
    lift = 1e3 * w[:, c["lift"]]
    return t, w[:, c["spring_force"]], hook, lift, w[:, c["friction_state"]].astype(int)


def colored(ax, lw=2.0):
    lc = LineCollection([], linewidths=lw, capstyle="round", joinstyle="round")
    ax.add_collection(lc)
    return lc


def set_colored(lc, x, y, state):
    if len(x) < 2:
        lc.set_segments([])
        return
    pts = np.column_stack([x, y])
    lc.set_segments(np.stack([pts[:-1], pts[1:]], 1))
    lc.set_colors([STATE_COLOR[s] for s in state[:-1]])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def friction_levels(const):
    """Spring force at which the pin breaks away and at which a slip settles."""
    m_g = const["pin_mass"] * const["g"]
    return const["mu_static"] * const["fit_normal"] + m_g, const["mu_kinetic"] * const["fit_normal"] + m_g


def make_video(w, const, res, frames):
    t, F, hook, lift, state = curves(w, const)
    sub = np.r_[np.arange(0, len(t), 4), len(t) - 1]           # 1 kHz is plenty for drawing
    F_s, F_k = friction_levels(const)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11.0, 5.0), dpi=100, facecolor=SURFACE)
    ax_img = fig.add_axes([0.0, 0.0, RENDER_W / 1100, RENDER_H / 500])
    ax_img.set_axis_off()
    im = ax_img.imshow(frames[0]["image"])
    ax_f = fig.add_axes([0.605, 0.555, 0.375, 0.33])
    ax_x = fig.add_axes([0.605, 0.115, 0.375, 0.33], sharex=ax_f)
    for ax in (ax_f, ax_x):
        style(ax)
    t_max = t[-1] + 0.02
    ax_f.set_xlim(-0.02, t_max), ax_f.set_ylim(-12, 23)
    ax_x.set_ylim(-3, 50)
    ax_f.set_ylabel("spring force (N)", color=INK, fontsize=9)
    ax_x.set_ylabel("displacement (mm)", color=INK, fontsize=9)
    ax_x.set_xlabel("time since the pull started (s)", color=INK, fontsize=9)
    plt.setp(ax_f.get_xticklabels(), visible=False)
    ax_f.set_title("Pulling force", loc="left", color=INK, fontsize=10, fontweight="bold")
    ax_x.set_title("Displacement", loc="left", color=INK, fontsize=10, fontweight="bold")
    for level, label in [(F_s, f"μs·N + mg = {F_s:.1f} N"), (F_k, f"μk·N + mg = {F_k:.1f} N")]:
        ax_f.axhline(level, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax_f.text(t_max, level + 0.6, label, color=INK2, fontsize=7.5, ha="right", va="bottom")
    L = 1e3 * const["fit_length"]
    ax_x.axhline(L, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax_x.text(0.0, L + 0.8, f"end of the fit ({L:.1f} mm)", color=INK2, fontsize=7.5, ha="left", va="bottom")
    lc_f, lc_x = colored(ax_f), colored(ax_x)
    hook_line, = ax_x.plot([], [], color=HOOK_COLOR, lw=1.2, ls=(0, (5, 2)))
    dot_f, = ax_f.plot([], [], "o", ms=7, mec=SURFACE, mew=1.5)
    dot_x, = ax_x.plot([], [], "o", ms=7, mec=SURFACE, mew=1.5)
    hook_label = ax_x.text(0, 0, "hook (robot)", color=INK2, fontsize=8, ha="left", va="bottom")
    pin_label = ax_x.text(0, 0, "pin", color=INK2, fontsize=8, ha="left", va="top")
    handles = [plt.Line2D([], [], color=STATE_COLOR[s], lw=2.5, label=STATE_NAME[s]) for s in STATE_NAME]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK, handlelength=1.6, columnspacing=1.2, title="pin state:",
               title_fontsize=9, alignment="left").get_title().set_color(INK2)
    head = fig.text(0.012, 0.955, "Robot pulls a press-fitted pin (MuJoCo)", color=INK, fontsize=12, fontweight="bold", va="center")
    sub_head = fig.text(0.012, 0.915, "", color=INK2, fontsize=9.5, va="center")
    badge = fig.text(RENDER_W / 1100 - 0.012, RENDER_H / 500 - 0.035, "", color=SURFACE, fontsize=10,
                     fontweight="bold", ha="right", va="top",
                     bbox=dict(boxstyle="round,pad=0.35", fc=INK, ec="none"))
    chip = fig.text(0.012, 0.035, "", color=INK, fontsize=10.5, va="center",
                    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=MUTED, lw=0.8))
    pull_t0 = w[0, pe.COL["t"]]
    images = []
    for fr in frames:
        im.set_data(fr["image"])
        n = fr["n"]
        idx = sub[sub < n]
        if n > 0:
            idx = np.r_[idx, n - 1]
        set_colored(lc_f, t[idx], F[idx], state[idx])
        set_colored(lc_x, t[idx], lift[idx], state[idx])
        hook_line.set_data(t[idx], hook[idx])
        if n > 0:
            k, col = n - 1, STATE_COLOR[fr["state"]]
            dot_f.set_data([t[k]], [F[k]]), dot_f.set_color(col), dot_f.set_markeredgecolor(SURFACE)
            dot_x.set_data([t[k]], [lift[k]]), dot_x.set_color(col), dot_x.set_markeredgecolor(SURFACE)
            right = t[k] < 0.8 * t_max                     # labels follow the cursor, inside the axes
            for lab, y, dy in [(hook_label, hook[k], 1.5), (pin_label, lift[k], -2.2)]:
                lab.set_position((t[k] + (0.03 if right else -0.03), y + dy))
                lab.set_horizontalalignment("left" if right else "right")
                lab.set_visible(True)
            chip.set_text(f"pin {STATE_NAME[fr['state']]}   ·   force {F[k]:5.1f} N   ·   lift {lift[k]:4.1f} mm"
                          f"   ·   slips {fr['slips']}")
            chip.get_bbox_patch().set_edgecolor(col)
            chip.get_bbox_patch().set_linewidth(2.0)
        else:
            dot_f.set_data([], []), dot_x.set_data([], [])
            hook_label.set_visible(False), pin_label.set_visible(False)
            chip.set_text("pin stuck in the housing   ·   force 0.0 N   ·   lift 0.0 mm   ·   slips 0")
            chip.get_bbox_patch().set_edgecolor(STATE_COLOR[0])
            chip.get_bbox_patch().set_linewidth(2.0)
        sub_head.set_text(f"robot: {fr['step']}   ·   simulation time {fr['time']:.3f} s")
        badge.set_text("SLOW MOTION ×20" if fr["slow"] else ("2× SPEED" if fr["time"] < pull_t0 else ""))
        badge.set_visible(bool(badge.get_text()))
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return images


def save_gif(images, path):
    from PIL import Image
    some = [images[i] for i in np.linspace(0, len(images) - 1, 24).astype(int)]
    palette = Image.fromarray(np.concatenate(some)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                             dither=Image.Dither.NONE)
    frames = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in images]
    durations = [int(1000 / FPS)] * (len(frames) - 1) + [int(1000 * END_HOLD)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def save_mp4(images, path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found: skipping", path)
        return False
    h, w = images[0].shape[:2]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for img in images + [images[-1]] * int(END_HOLD * FPS):
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        proc.stdin.close()
        proc.wait()
    return proc.returncode == 0


def make_graphs(w, const, res):
    """Static figure of the whole pull: (a) force, (b) displacements, (c) force over pin lift."""
    t, F, hook, lift, state = curves(w, const)
    F_s, F_k = friction_levels(const)
    L = 1e3 * const["fit_length"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    for ax in axes:
        style(ax)
    ax_f, ax_x, ax_fx = axes
    for ax, x, y in [(ax_f, t, F), (ax_x, t, lift), (ax_fx, lift, F)]:
        set_colored(colored(ax, lw=1.6), x, y, state)
    ax_f.set(xlim=(0, t[-1]), ylim=(-12, 22))
    ax_x.set(xlim=(0, t[-1]), ylim=(-3, 48))
    ax_fx.set(xlim=(-1, 46), ylim=(-12, 22))
    ax_x.plot(t, hook, color=HOOK_COLOR, lw=1.1, ls=(0, (5, 2)))
    ax_x.text(0.42, 12.5, "hook (robot)", color=INK2, ha="right", va="bottom")
    ax_x.text(1.03, 9.8, "pin", color=INK2, ha="center", va="top")
    for ax in (ax_f, ax_fx):
        for level, label in [(F_s, f"μs·N + mg = {F_s:.1f} N"), (F_k, f"μk·N + mg = {F_k:.1f} N")]:
            ax.axhline(level, color=MUTED, lw=0.7, ls=(0, (4, 3)))
            ax.text(ax.get_xlim()[1], level + 0.5, label, color=INK2, fontsize=7, ha="right", va="bottom")
    ax_fx.text(L + 0.6, -9, "end of\nthe fit", color=INK2, fontsize=7)
    ax_x.axhline(L, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax_x.text(0.02, L + 0.8, f"end of the fit ({L:.1f} mm)", color=INK2, fontsize=7)
    ax_fx.axvline(L, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax_f.set(xlabel="time since the pull started (s)", ylabel="spring force (N)")
    ax_x.set(xlabel="time since the pull started (s)", ylabel="displacement (mm)")
    ax_fx.set(xlabel="pin lift (mm)", ylabel="spring force (N)")
    for ax, title in zip(axes, ["(a) pulling force", "(b) hook and pin displacement", "(c) force over displacement"]):
        ax.set_title(title, loc="left", color=INK, fontsize=9)
    ev = res["pin"]["events_from_pull_start"]
    for k, (te, e) in enumerate(ev):
        if e == "breakaway":
            j = np.searchsorted(t, te)
            ax_f.annotate(f"slip {k // 2 + 1}", (te, F[j]), xytext=(te - 0.05, 20.6), color=INK2, fontsize=7,
                          ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    handles = [plt.Line2D([], [], color=STATE_COLOR[s], lw=2.2, label=f"pin {STATE_NAME[s]}") for s in STATE_NAME]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(MEDIA / f"pin_force_displacement.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("saved", MEDIA / "pin_force_displacement.png", "and .pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--graphs-only", action="store_true", help="only the static graphs (no video)")
    args = parser.parse_args()
    MEDIA.mkdir(exist_ok=True)
    last = {}
    w, const, res, _ = pe.run(verbose=False, on_step=lambda m, d, *a: last.update(t=d.time))   # pass 1: events
    make_graphs(w, const, res)
    if args.graphs_only:
        return
    pull_t0 = float(w[0, pe.COL["t"]])
    events = [(float(t), e) for t, e in res["pin"]["events_from_pull_start"]]
    times, windows = schedule(events, pull_t0, last["t"])
    grab = Grabber(times, windows)
    w2, *_ = pe.run(verbose=False, on_step=grab)               # pass 2: the frames
    assert np.array_equal(w, w2), "the simulation is not deterministic"
    print(f"{len(grab.frames)} frames ({len(grab.frames) / FPS:.1f} s at {FPS} fps)")
    images = make_video(w, const, res, grab.frames)
    save_gif(images, MEDIA / "pin_extraction.gif")
    print("saved", MEDIA / "pin_extraction.gif", f"({(MEDIA / 'pin_extraction.gif').stat().st_size / 1e6:.1f} MB)")
    if save_mp4(images, MEDIA / "pin_extraction.mp4"):
        print("saved", MEDIA / "pin_extraction.mp4",
              f"({(MEDIA / 'pin_extraction.mp4').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
