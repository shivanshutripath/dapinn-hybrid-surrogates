"""pin_experiment.py - the MuJoCo experiment of the paper: press-fit extraction.

A gantry robot (scene_pin.xml) pulls a press-fitted pin (device_pin.xml) out
of its housing through a series-elastic puller: a hook connected to the pin by
a spring (k = 2000 N/m, c = 2 N s/m) whose deflection the puller measures.
The robot pulls the hook up at a constant 20 mm/s. Static friction (18 N) is
larger than kinetic friction (12 N), so the pin sticks and slips until it
leaves the fit after 19.5 mm, then rings on the spring.

The physics is MuJoCo's (robot axes, gravity, joint friction constraint, the
tendon spring), with the two rules of sim_utils.py: FrictionLaw (static or
kinetic friction value, 0 after the fit) and engage() (the hook engages).

Saves
  data/pin.npz               every physics step (4000 per s) of the pull, with
                             the hidden states and events: the data of the
                             DA-PINN (python3 dapinn_pin.py)
  results/pin_summary.json   the events and the constants
  ../media/pin_cell.png      a picture of the cell during the pull (--picture)

The video with the force and displacement graphs is made by make_video.py,
which runs this same simulation.

Usage (inside pin_extraction)
  python3 pin_experiment.py [--picture]
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

import sim_utils as su

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "pin.npz"
MEDIA = HERE.parent / "media"

FIT_NORMAL, MU_STATIC, MU_KINETIC, FIT_LENGTH = 40.0, 0.45, 0.30, 0.0195
PULLER_K, PULLER_C = 2000.0, 2.0
PULL_SPEED, PULL_HEIGHT, HOOK_ABOVE_PIN = 0.02, 0.045, 0.03
SAFE_Z, TRAVEL_SPEED, APPROACH_SPEED = 0.52, 0.2, 0.05
PULL_STEP = "pull the pin out"
COLUMNS = ["t", "tcp_z", "tcp_vz", "spring_length", "spring_rate", "spring_force", "lift", "speed",
           "friction_state"]
COL = {c: k for k, c in enumerate(COLUMNS)}


def plan_of(data):
    pin = data.site("pin_top").xpos.copy()
    hook = pin + [0, 0, HOOK_ABOVE_PIN]
    return [("move", "move above the pin", [hook[0], hook[1], SAFE_Z], TRAVEL_SPEED, "smooth"),
            ("move", "lower the hook to the pin", hook, APPROACH_SPEED, "smooth"),
            ("engage", "hook the pin (spring)", None, 0.3, None),
            ("move", PULL_STEP, hook + [0, 0, PULL_HEIGHT], PULL_SPEED, "linear"),
            ("wait", "hold the pulled pin", None, 0.3, None)]


def run(picture=False, on_step=None, verbose=True):
    """Run the robot program. `on_step(model, data, law, text, rows)` is called
    after every physics step (make_video.py uses it to grab frames); `rows`
    holds the recorded steps of the pull so far."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene_pin.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    law = su.FrictionLaw(model, "pin", MU_STATIC * FIT_NORMAL, MU_KINETIC * FIT_NORMAL, FIT_LENGTH,
                         v_slip=1e-3, v_stick=3e-4)
    model.dof_frictionloss[law.dof] = law.static
    act = [model.actuator(n).id for n in ["motor_x", "motor_y", "motor_z"]]
    ten = model.tendon("puller").id
    plan = plan_of(data)
    target = data.site("tcp").xpos.copy()
    rows, rest_length, shot = [], None, None
    renderer = mujoco.Renderer(model, 720, 960) if picture else None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.47]
    cam.distance, cam.azimuth, cam.elevation = 0.25, 62, -18
    for kind, text, xyz, value, profile in plan:
        if verbose:
            print(f"t = {data.time:6.3f} s  {text}")
        t0, start = data.time, target.copy()
        if kind == "move":
            xyz = np.array(xyz, float)
            dist = float(np.linalg.norm(xyz - start))
            duration = dist / value + 0.5 * su.RAMP if profile == "linear" else max(1.5 * dist / value, 0.3)
        else:
            duration = value
        if kind == "engage":
            rest_length = su.engage(model, data, "puller", PULLER_K, PULLER_C)
        measuring = text == PULL_STEP
        while True:
            el = data.time - t0
            if kind == "move":
                u = (su.linear_profile(el, dist, value) / max(dist, 1e-12) if profile == "linear"
                     else su.smooth(el / duration))
                target = start + u * (xyz - start)
                if el >= duration and np.linalg.norm(data.site("tcp").xpos - xyz) < 0.002:
                    break
            elif el >= duration:
                break
            if el > duration + 3.0:
                raise RuntimeError(f"step '{text}' did not finish")
            data.ctrl[act] = target
            law.update(data)
            mujoco.mj_step(model, data)
            if measuring:
                rows.append([data.time, data.site("tcp").xpos[2], data.joint("slide_z").qvel[0],
                             data.ten_length[ten], data.ten_velocity[ten], su.tendon_force(model, data, "puller"),
                             law.position(data), data.joint("pin").qvel[0], law.code()])
                if picture and shot is None and el > 1.2:
                    renderer.update_scene(data, camera=cam)
                    shot = renderer.render()
            if on_step is not None:
                on_step(model, data, law, text, rows)
    w = np.array(rows)
    ev = [(round(float(t - w[0, 0]), 4), e) for t, e in law.events]
    itinerary = "S" + "".join({"breakaway": "K", "arrest": "S", "exit": "R"}[e] for _, e in ev)
    pin_body = model.body("pin").id
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    const = {"g": float(-model.opt.gravity[2]), "pin_mass": float(M[law.dof, law.dof]),
             "puller_k": PULLER_K, "puller_c": PULLER_C, "puller_rest_length": rest_length,
             "pin_top0": float(model.body_pos[pin_body, 2] + model.site("pin_top").pos[2]),
             "fit_normal": FIT_NORMAL, "mu_static": MU_STATIC, "mu_kinetic": MU_KINETIC,
             "fit_length": FIT_LENGTH, "pull_speed": PULL_SPEED, "timestep": model.opt.timestep}
    slips = [round(float(1e3 * (b[0] - a[0])), 2) for a, b in zip(ev[:-1], ev[1:]) if a[1] == "breakaway"]
    res = {"pin": {"events_from_pull_start": ev, "itinerary": itinerary,
                   "peak_force": round(float(w[:, COL["spring_force"]].max()), 3), "slip_durations_ms": slips,
                   "out_of_fit": law.state == "free"}}
    if verbose:
        print(f"itinerary {itinerary}, events {ev}, slips (ms) {slips}, peak {res['pin']['peak_force']} N")
    return w, const, res, shot


def save(w, const, res):
    DATA.parent.mkdir(exist_ok=True)
    out = {f"pin_{c}": w[:, k] for k, c in enumerate(COLUMNS)}
    np.savez_compressed(DATA, constants=json.dumps(const), results=json.dumps(res),
                        settings=json.dumps({"setup": "scene_pin.xml"}), **out)
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "pin_summary.json", "w") as f:
        json.dump({"constants": const, "results": res}, f, indent=1)
    print("saved", DATA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--picture", action="store_true", help="also save ../media/pin_cell.png")
    args = parser.parse_args()
    w, const, res, shot = run(picture=args.picture)
    save(w, const, res)
    if shot is not None:
        from PIL import Image
        MEDIA.mkdir(exist_ok=True)
        Image.fromarray(shot).save(MEDIA / "pin_cell.png")
        print("saved", MEDIA / "pin_cell.png")
    if not res["pin"]["out_of_fit"]:
        sys.exit("the pin did not leave the fit")


if __name__ == "__main__":
    main()
