"""make_submission_video.py - the supplementary video of the pin extraction (MP4).

A one-minute video for a multimedia submission: 1920x1080, 30 frames per
second, H.264 (High profile, yuv420p) in an MP4 container, no audio. All
explanations are on-screen captions. The authors are not named (double-blind
review).

  0  title card
  1  the cell: a slow orbit around the MuJoCo scene, with the parts labelled
  2  approach: the robot puts the hook on the pin (real time)
  3  the pull in real time, with the spring force and the displacements
  4  replay in slow motion, with a caption for each stick, slip and release
  5  summary: what happened, and what the DA-PINN recovers
  6  credits

The simulation is that of pin_experiment.py. It is deterministic, so it is run
once to find the events and then again for each section that shows it.

Usage (inside pin_extraction)
  python3 make_submission_video.py --out ../media/pin_extraction_video.mp4
  python3 make_submission_video.py --preview DIR     # a few frames as PNG, no video
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

import make_media as mm
import pin_experiment as pe
import sim_utils as su

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
W, H, FPS = 1920, 1080, 30
RW, RH = 1080, 810                         # the MuJoCo render
TOP, BOTTOM = 90, 180                      # header and caption bars (pixels)
SURFACE, INK, INK2, MUTED = mm.SURFACE, mm.INK, mm.INK2, mm.MUTED
CODE = {0: "S", 1: "K", 2: "R"}
NAME = {0: "STUCK", 1: "SLIPPING", 2: "RELEASED"}
WIDE = dict(lookat=[0.0, 0.0, 0.58], distance=1.05, azimuth=25.0, elevation=-22.0)
WIDE_END = dict(WIDE, azimuth=60.0)
CLOSE = dict(lookat=list(mm.CAMERA["lookat"]), distance=mm.CAMERA["distance"],
             azimuth=float(mm.CAMERA["azimuth"]), elevation=float(mm.CAMERA["elevation"]))

TITLE = ("Hybrid Surrogates for Hybrid Dynamical Systems:", "Physics-Informed Learning Across Discontinuities")

CAP = {
    "cell": "A gantry robot pulls a press-fitted pin out of its housing through a series-elastic puller.",
    "approach": "The robot moves the puller over the pin, lowers the hook and engages the spring.",
    "realtime": "Real time: the hook rises at 20 mm/s. The pin sticks and slips four times, then leaves the fit.",
    "stick": "Stick (S): static friction holds the pin while the spring stretches, so the force rises.",
    "slip": "Breakaway at μs·N + mg = 18.5 N: the pin slips (K) and jumps 5.7 mm in 17 ms.",
    "arrest": "Arrest: the pin stops and sticks again (S). The spring stretches again.",
    "repeat": "The stick-slip cycle repeats every 0.29 s.",
    "exit": "Fourth slip: the pin passes the end of the fit (19.5 mm) after 7 ms and is released (R).",
    "released": "Released (R): without friction, the pin rings on the spring (32 Hz), then follows the hook.",
}
SETUP_TEXT = [
    ("The cell", "bold"),
    ("Gantry robot: three prismatic axes,", None),
    ("each driven by a position servo.", "indent"),
    ("Puller: a hook and a spring (k = 2000 N/m,", None),
    ("c = 2 N s/m) that measures its deflection δ.", "indent"),
    ("Pin: 50 g, held by an interference fit", None),
    ("over L = 19.5 mm with a normal force N = 40 N.", "indent"),
    ("Friction: static μs = 0.45 (18 N),", None),
    ("kinetic μk = 0.30 (12 N).", "indent"),
    ("Task: raise the hook at a constant 20 mm/s.", None),
    ("", None),
    ("Simulated in MuJoCo, time step 0.25 ms. MuJoCo has", "muted"),
    ("one friction value per joint, so it is set to μs·N", "muted"),
    ("while the pin is at rest and to μk·N while it slides.", "muted"),
]
LABELS = [("gantry robot", "bridge", (-300, -70)), ("puller and spring", "puller_body", (170, -60)),
          ("press-fitted pin", "pin_top", (190, 40)), ("housing", "housing_block", (-230, 70))]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class Writer:
    """Frames to ffmpeg (H.264 in MP4), or to PNG files for a preview."""

    def __init__(self, out=None, preview=None, frames=None):
        self.count, self.preview = 0, preview
        self.frames = set(frames) if frames else None
        self.proc = None
        if preview is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise SystemExit("ffmpeg is needed for the MP4 (https://ffmpeg.org)")
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                   "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                   "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p", "-tag:v", "avc1",
                   "-map_metadata", "-1", "-metadata", f"title={TITLE[0]} {TITLE[1]} (supplementary video)",
                   "-movflags", "+faststart", str(out)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            Path(preview).mkdir(parents=True, exist_ok=True)

    def write(self, image, repeat=1):
        for _ in range(repeat):
            if self.proc is not None:
                self.proc.stdin.write(np.ascontiguousarray(image).tobytes())
            elif (self.count in self.frames) if self.frames else self.count % 60 == 0:
                from PIL import Image
                Image.fromarray(image).save(Path(self.preview) / f"frame_{self.count:05d}.png")
            self.count += 1

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait():
                raise SystemExit("ffmpeg failed")


def fade(writer, image, seconds, fade_s=0.5, background=SURFACE):
    """Write a still for `seconds`, fading in from and out to the background."""
    bg = np.array(matplotlib.colors.to_rgb(background)) * 255
    n, nf = int(round(seconds * FPS)), int(round(fade_s * FPS))
    for i in range(n):
        a = min(1.0, (i + 1) / nf, (n - i) / nf)
        writer.write((image * a + bg * (1 - a)).astype(np.uint8) if a < 1 else image)


def figure_image(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def card(lines):
    """A still with centered lines: (text, y, size, weight, color)."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    for text, y, size, weight, color in lines:
        fig.text(0.5, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color)
    img = figure_image(fig)
    plt.close(fig)
    return img


def title_card():
    return card([(TITLE[0], 0.60, 38, "bold", INK), (TITLE[1], 0.525, 38, "bold", INK),
                 ("Supplementary video: robotic extraction of a press-fitted pin, simulated in MuJoCo",
                  0.40, 24, "normal", INK2)])


def credits_card():
    return card([(TITLE[0], 0.60, 30, "bold", INK), (TITLE[1], 0.54, 30, "bold", INK),
                 ("Supplementary video", 0.45, 22, "normal", INK2),
                 ("Simulation: MuJoCo 3.14.0   ·   Plots: Matplotlib 3.10.8", 0.34, 16, "normal", MUTED)])


def summary_card(w, const, res, ref):
    """The whole pull (force, displacement, force over lift) and the DA-PINN result."""
    t, F, hook, lift, state = mm.curves(w, const)
    F_s, F_k = mm.friction_levels(const)
    L = 1e3 * const["fit_length"]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    fig.text(0.04, 0.93, "Summary of the pull", fontsize=26, fontweight="bold", color=INK, va="center")
    axes = [fig.add_axes([0.05 + 0.32 * k, 0.43, 0.26, 0.40]) for k in range(3)]
    for ax in axes:
        mm.style(ax)
        ax.tick_params(labelsize=12)
    for ax, x, y in [(axes[0], t, F), (axes[1], t, lift), (axes[2], lift, F)]:
        mm.set_colored(mm.colored(ax, lw=2.2), x, y, state)
    axes[1].plot(t, hook, color=mm.HOOK_COLOR, lw=1.6, ls=(0, (5, 2)))
    axes[1].text(0.45, 12.5, "hook (robot)", color=INK2, fontsize=13, ha="right", va="bottom")
    axes[1].text(1.03, 9.5, "pin", color=INK2, fontsize=13, ha="center", va="top")
    axes[0].set(xlim=(0, t[-1]), ylim=(-12, 22))
    axes[1].set(xlim=(0, t[-1]), ylim=(-3, 48))
    axes[2].set(xlim=(-1, 46), ylim=(-12, 22))
    for ax in (axes[0], axes[2]):
        for level, label in [(F_s, f"μs·N + mg = {F_s:.1f} N"), (F_k, f"μk·N + mg = {F_k:.1f} N")]:
            ax.axhline(level, color=MUTED, lw=1.0, ls=(0, (4, 3)))
            ax.text(ax.get_xlim()[1], level + 0.5, label, color=INK2, fontsize=11, ha="right", va="bottom")
    axes[1].axhline(L, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    axes[1].text(0.02, L + 0.8, f"end of the fit ({L:.1f} mm)", color=INK2, fontsize=11)
    axes[2].axvline(L, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    for ax, xl, yl, title in [(axes[0], "time since the pull started (s)", "spring force (N)", "Pulling force"),
                              (axes[1], "time since the pull started (s)", "displacement (mm)", "Displacement"),
                              (axes[2], "pin lift (mm)", "spring force (N)", "Force over displacement")]:
        ax.set_xlabel(xl, fontsize=14, color=INK), ax.set_ylabel(yl, fontsize=14, color=INK)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color=INK)
    handles = [plt.Line2D([], [], color=mm.STATE_COLOR[s], lw=3, label=f"pin {mm.STATE_NAME[s]}")
               for s in mm.STATE_NAME]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.97, 0.93), ncol=3, frameon=False,
               fontsize=15, labelcolor=INK)
    ev = res["pin"]["events_from_pull_start"]
    n_slips = sum(e == "breakaway" for _, e in ev)
    seq = " ".join(res["pin"]["itinerary"])
    left = [("In the MuJoCo simulation", 0.29, 19, "bold", INK),
            (f"{n_slips} breakaways at μs·N + mg = {F_s:.1f} N, every 0.29 s; each slip lasts 17 ms", 0.23,
             16, "normal", INK),
            (f"mode sequence   {seq}", 0.17, 16, "normal", INK)]
    right = [("Recovered by the DA-PINN, from the measured spring deflection only", 0.29, 17, "bold", INK),
             (f"J = {ref['J']} events, mode sequence   {' '.join(ref['modes'])}", 0.23, 16, "normal", INK),
             (f"μs = {ref['theta']['mu_s']:.4f} (true {const['mu_static']:.2f}),   "
              f"μk = {ref['theta']['mu_k']:.4f} (true {const['mu_kinetic']:.2f})", 0.17, 16, "normal", INK),
             (f"all {ref['J']} event times within {max(ref['jump_time_error_ms']):.1f} ms", 0.11, 16, "normal", INK)]
    for x, block in [(0.05, left), (0.47, right)]:
        for text, y, size, weight, color in block:
            fig.text(x, y, text, fontsize=size, fontweight=weight, color=color, va="center")
    img = figure_image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# The simulation frames
# ---------------------------------------------------------------------------
def camera(params):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = params["lookat"]
    cam.distance, cam.azimuth, cam.elevation = params["distance"], params["azimuth"], params["elevation"]
    return cam


def blend(a, b, u):
    """Camera between a (u = 0) and b (u = 1), eased, zooming in log distance."""
    u = su.smooth(u)
    out = {k: (1 - u) * np.asarray(a[k], float) + u * np.asarray(b[k], float) for k in ("lookat", "azimuth",
                                                                                           "elevation")}
    out["distance"] = float(np.exp((1 - u) * np.log(a["distance"]) + u * np.log(b["distance"])))
    out["lookat"] = out["lookat"].tolist()
    return out


def project(scene, point):
    """Pixel position of a world point in the last render (MuJoCo free camera)."""
    c = scene.camera[0]
    pos, fwd, up = np.array(c.pos), np.array(c.forward), np.array(c.up)
    right = np.cross(fwd, up)
    d = np.asarray(point) - pos
    z = d @ fwd
    xn, yn = (d @ right) * c.frustum_near / z, (d @ up) * c.frustum_near / z
    half_h = 0.5 * (c.frustum_top - c.frustum_bottom)
    half_w = half_h * RW / RH
    px = 0.5 * ((xn - c.frustum_center) / half_w + 1.0) * RW
    py = 0.5 * (1.0 - (yn - 0.5 * (c.frustum_top + c.frustum_bottom)) / half_h) * RH
    return px, py


def label_points(model, data):
    bridge = model.body_geomadr[model.body("gantry_bridge").id]
    return {"bridge": data.geom_xpos[bridge], "puller_body": data.geom("puller_body").xpos,
            "pin_top": data.site("pin_top").xpos, "housing_block": data.geom("housing_block").xpos}


class Screen:
    """The layout of a simulation frame: header, render, plots or text, captions."""

    def __init__(self, w, const):
        self.t, self.F, self.hook, self.lift, self.state = mm.curves(w, const)
        self.sub = np.r_[np.arange(0, len(self.t), 4), len(self.t) - 1]
        F_s, F_k = mm.friction_levels(const)
        L = 1e3 * const["fit_length"]
        plt.rcParams.update({"font.family": "DejaVu Sans"})
        self.fig = fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
        self.ax_img = fig.add_axes([0, BOTTOM / H, RW / W, RH / H])
        self.ax_img.set_axis_off()
        self.im = self.ax_img.imshow(np.zeros((RH, RW, 3), np.uint8), interpolation="nearest")
        self.ax_img.set_xlim(-0.5, RW - 0.5), self.ax_img.set_ylim(RH - 0.5, -0.5)
        self.ax_f = fig.add_axes([0.605, 0.585, 0.375, 0.29])
        self.ax_x = fig.add_axes([0.605, 0.225, 0.375, 0.29], sharex=self.ax_f)
        for ax in (self.ax_f, self.ax_x):
            mm.style(ax)
            ax.tick_params(labelsize=13)
        t_max = self.t[-1] + 0.02
        self.t_max = t_max
        self.ax_f.set_xlim(-0.02, t_max), self.ax_f.set_ylim(-12, 23)
        self.ax_x.set_ylim(-3, 50)
        self.ax_f.set_ylabel("spring force (N)", fontsize=15, color=INK)
        self.ax_x.set_ylabel("displacement (mm)", fontsize=15, color=INK)
        self.ax_x.set_xlabel("time since the pull started (s)", fontsize=15, color=INK)
        plt.setp(self.ax_f.get_xticklabels(), visible=False)
        self.ax_f.set_title("Pulling force", loc="left", fontsize=18, fontweight="bold", color=INK)
        self.ax_x.set_title("Displacement", loc="left", fontsize=18, fontweight="bold", color=INK)
        for level, label in [(F_s, f"μs·N + mg = {F_s:.1f} N"), (F_k, f"μk·N + mg = {F_k:.1f} N")]:
            self.ax_f.axhline(level, color=MUTED, lw=1.0, ls=(0, (4, 3)))
            self.ax_f.text(t_max, level + 0.6, label, color=INK2, fontsize=12, ha="right", va="bottom")
        self.ax_x.axhline(L, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        self.ax_x.text(0.0, L + 0.8, f"end of the fit ({L:.1f} mm)", color=INK2, fontsize=12, va="bottom")
        self.lc_f, self.lc_x = mm.colored(self.ax_f, lw=2.6), mm.colored(self.ax_x, lw=2.6)
        self.hook_line, = self.ax_x.plot([], [], color=mm.HOOK_COLOR, lw=1.8, ls=(0, (5, 2)))
        self.dots = [ax.plot([], [], "o", ms=11, mec=SURFACE, mew=2)[0] for ax in (self.ax_f, self.ax_x)]
        self.hook_label = self.ax_x.text(0, 0, "hook (robot)", color=INK2, fontsize=13, va="bottom")
        self.pin_label = self.ax_x.text(0, 0, "pin", color=INK2, fontsize=13, va="top")
        handles = [plt.Line2D([], [], color=mm.STATE_COLOR[s], lw=4, label=f"{mm.STATE_NAME[s]} ({CODE[s]})")
                   for s in mm.STATE_NAME]
        self.legend = fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.985, 0.947), ncol=3,
                                 frameon=False, fontsize=15, labelcolor=INK, handlelength=1.6,
                                 columnspacing=1.4, title="pin:", title_fontsize=15, alignment="left")
        self.legend.get_title().set_color(INK2)
        self.header = fig.text(0.012, 1 - 0.5 * TOP / H, "", fontsize=24, fontweight="bold", color=INK,
                               va="center")
        self.badge = fig.text(RW / W - 0.012, (BOTTOM + RH) / H - 0.04, "", fontsize=17, fontweight="bold",
                              color=SURFACE, ha="right", va="top",
                              bbox=dict(boxstyle="round,pad=0.4", fc=INK, ec="none"))
        self.clock = fig.text(0.012, (BOTTOM + RH) / H - 0.04, "", fontsize=15, color=SURFACE, va="top",
                              bbox=dict(boxstyle="round,pad=0.4", fc="#000000", ec="none", alpha=0.55))
        self.caption = fig.text(0.5, 0.112, "", fontsize=25, color=INK, ha="center", va="center")
        self.chip = fig.text(0.5, 0.042, "", fontsize=17, color=INK, ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=MUTED, lw=2.5))
        self.setup = []
        y = 0.86
        for text, style in SETUP_TEXT:
            kw = dict(fontsize=24 if style == "bold" else 17, fontweight="bold" if style == "bold" else "normal",
                      color=MUTED if style == "muted" else INK)
            self.setup.append(fig.text(0.60 + (0.02 if style == "indent" else 0.0), y, text, va="top", **kw))
            y -= 0.065 if style == "bold" else 0.046
        self.callouts = [self.ax_img.annotate(
            text, xy=(0, 0), xytext=(0, 0), fontsize=16, color=INK, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=INK, lw=1.0),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0, shrinkA=2, shrinkB=4, mutation_scale=18))
            for text, _, _ in LABELS]

    def panel(self, kind):
        plots = kind == "plots"
        for a in [self.ax_f, self.ax_x, self.legend]:
            a.set_visible(plots)
        for a in self.setup:
            a.set_visible(not plots)

    def plots(self, n, state_now):
        idx = self.sub[self.sub < n]
        if n > 0:
            idx = np.r_[idx, n - 1]
        t, F, lift, hook, st = self.t, self.F, self.lift, self.hook, self.state
        mm.set_colored(self.lc_f, t[idx], F[idx], st[idx])
        mm.set_colored(self.lc_x, t[idx], lift[idx], st[idx])
        self.hook_line.set_data(t[idx], hook[idx])
        if n == 0:
            for d in self.dots:
                d.set_data([], [])
            self.hook_label.set_visible(False), self.pin_label.set_visible(False)
            return
        k = n - 1
        col = mm.STATE_COLOR[state_now]
        for d, y in zip(self.dots, (F[k], lift[k])):
            d.set_data([t[k]], [y]), d.set_color(col), d.set_markeredgecolor(SURFACE)
        right = t[k] < 0.8 * self.t_max
        for lab, y, dy in [(self.hook_label, hook[k], 1.6), (self.pin_label, lift[k], -2.4)]:
            lab.set_position((t[k] + (0.03 if right else -0.03), y + dy))
            lab.set_horizontalalignment("left" if right else "right")
            lab.set_visible(True)

    def draw(self, image, header, caption, badge="", clock="", chip=None, chip_color=None, callouts=None,
             n=0, state_now=0, panel="plots"):
        self.im.set_data(image)
        self.header.set_text(header)
        self.caption.set_text(caption)
        self.badge.set_text(badge), self.badge.set_visible(bool(badge))
        self.clock.set_text(clock), self.clock.set_visible(bool(clock))
        self.chip.set_text(chip or ""), self.chip.set_visible(bool(chip))
        if chip:
            self.chip.get_bbox_patch().set_edgecolor(chip_color or MUTED)
        self.panel(panel)
        if panel == "plots":
            self.plots(n, state_now)
        for ann, (text, key, off) in zip(self.callouts, LABELS):
            if callouts is None:
                ann.set_visible(False)
                continue
            x, y = callouts[key]
            ann.xy = (x, y)
            ann.set_position((x + off[0], y + off[1]))
            ann.set_visible(True)
        return figure_image(self.fig)


def status(law, data, rows, F, lift):
    """The status line under the caption, and its color."""
    if not rows:
        return "pin seated in the housing (S)   ·   spring force 0.0 N   ·   lift 0.0 mm", mm.STATE_COLOR[0]
    seq = "S" + "".join({"breakaway": "K", "arrest": "S", "exit": "R"}[e] for _, e in law.events)
    k = len(rows) - 1
    code = law.code()
    f_k, x_k = round(F[k], 1) + 0.0, round(lift[k], 1) + 0.0         # + 0.0: no "-0.0"
    return (f"pin {NAME[code]} ({CODE[code]})   ·   spring force {f_k:5.1f} N   ·   lift {x_k:4.1f} mm"
            f"   ·   modes so far: {' '.join(seq)}", mm.STATE_COLOR[code])


class Pass:
    """on_step callback of pin_experiment.run: draws a frame at each requested time.
    requests: list of (sim time, dict(header, caption, badge, camera, hold)) in time order."""

    def __init__(self, screen, writer, requests, pull_t0):
        self.screen, self.writer, self.req, self.k, self.pull_t0 = screen, writer, requests, 0, pull_t0
        self.renderer = None

    def __call__(self, model, data, law, text, rows):
        while self.k < len(self.req) and self.req[self.k][0] <= data.time + 1e-9:
            t_req, spec = self.req[self.k]
            self.k += 1
            if self.renderer is None:
                model.vis.global_.offwidth, model.vis.global_.offheight = RW, RH
                self.renderer = mujoco.Renderer(model, RH, RW)
            self.renderer.update_scene(data, camera=camera(spec["camera"]))
            s = self.screen
            chip, color = status(law, data, rows, s.F, s.lift)
            clock = (f"t = {data.time - self.pull_t0:.3f} s since the pull started" if rows
                     else f"t = {data.time:.2f} s")
            img = s.draw(self.renderer.render(), spec["header"], spec["caption"], badge=spec["badge"],
                         clock=clock, chip=chip, chip_color=color, n=len(rows), state_now=law.code())
            self.writer.write(img, repeat=1 + int(round(spec.get("hold", 0.0) * FPS)))


def cell_section(screen, writer, seconds=8.0):
    """A slow orbit around the scene at rest, with the parts labelled."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene_pin.xml"))
    model.vis.global_.offwidth, model.vis.global_.offheight = RW, RH
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, RH, RW)
    pts = label_points(model, data)
    n = int(round(seconds * FPS))
    for i in range(n):
        renderer.update_scene(data, camera=camera(blend(WIDE, WIDE_END, i / (n - 1))))
        image = renderer.render()
        callouts = {k: project(renderer.scene, p) for k, p in pts.items()}
        writer.write(screen.draw(image, "1 · The cell", CAP["cell"], callouts=callouts, panel="setup"))
    renderer.close()


def realtime_requests(pull_t0, t_end):
    req = []
    for t in np.arange(0.0, pull_t0, 1.0 / FPS):
        u = min(t / (pull_t0 - 0.3), 1.0)               # zoom in while the robot approaches
        req.append((t, dict(header="2 · Approach", caption=CAP["approach"], badge="REAL TIME",
                            camera=blend(WIDE_END, CLOSE, u))))
    ts = np.arange(pull_t0, t_end, 1.0 / FPS)
    for k, t in enumerate(ts):
        req.append((t, dict(header="3 · The pull, in real time", caption=CAP["realtime"], badge="REAL TIME",
                            camera=CLOSE, hold=3.0 if k == len(ts) - 1 else 0.0)))
    return req


def replay_requests(events, pull_t0, t_end):
    """Slow motion: (start, end, speed, caption) pieces, in time since the pull started."""
    tb = [t for t, e in events if e == "breakaway"]
    ta = [t for t, e in events if e == "arrest"]
    te = [t for t, e in events if e == "exit"][0]
    end = t_end - pull_t0
    pieces = [(0.0, tb[0] - 0.01, 4, "stick"), (tb[0] - 0.01, ta[0] + 0.01, 100, "slip"),
              (ta[0] + 0.01, tb[1] - 0.01, 12, "arrest"),
              (tb[1] - 0.01, ta[1] + 0.01, 40, "repeat"), (ta[1] + 0.01, tb[2] - 0.01, 4, "repeat"),
              (tb[2] - 0.01, ta[2] + 0.01, 40, "repeat"), (ta[2] + 0.01, tb[3] - 0.01, 4, "repeat"),
              (tb[3] - 0.01, te + 0.02, 100, "exit"), (te + 0.02, te + 0.10, 40, "released"),
              (te + 0.10, end, 2, "released")]
    req = []
    for a, b, slow, cap in pieces:
        for t in np.arange(a, b, 1.0 / (FPS * slow)):
            req.append((pull_t0 + t, dict(header="4 · Replay in slow motion", caption=CAP[cap],
                                          badge=f"1/{slow} SPEED", camera=CLOSE,
                                          hold=1.0 if not req else 0.0)))
    req[-1][1]["hold"] = 2.0
    return req


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(HERE.parent / "media" / "pin_extraction_video.mp4"))
    parser.add_argument("--preview", help="save frames as PNG in this folder instead of a video")
    parser.add_argument("--frames", help="with --preview: comma-separated frame numbers (default: every 60th)")
    args = parser.parse_args()
    import json
    last = {}
    w, const, res, _ = pe.run(verbose=False, on_step=lambda m, d, *a: last.update(t=d.time))
    assert res["pin"]["itinerary"] == "SKSKSKSKR", "the captions describe the default pull"
    events = [(float(t), e) for t, e in res["pin"]["events_from_pull_start"]]
    pull_t0, t_end = float(w[0, pe.COL["t"]]), last["t"]
    ref = json.load(open(HERE / "results" / "pin.json"))
    writer = Writer(args.out, args.preview, [int(f) for f in args.frames.split(",")] if args.frames else None)
    screen = Screen(w, const)
    fade(writer, title_card(), 6.0)
    cell_section(screen, writer)
    for requests in (realtime_requests(pull_t0, t_end - 1e-6), replay_requests(events, pull_t0, t_end - 1e-6)):
        w2, *_ = pe.run(verbose=False, on_step=Pass(screen, writer, requests, pull_t0))
        assert np.array_equal(w, w2), "the simulation is not deterministic"
    fade(writer, summary_card(w, const, res, ref), 12.0)
    fade(writer, credits_card(), 5.0)
    writer.close()
    print(f"{writer.count} frames, {writer.count / FPS:.1f} s" + ("" if args.preview else f" -> {args.out}"))


if __name__ == "__main__":
    main()
