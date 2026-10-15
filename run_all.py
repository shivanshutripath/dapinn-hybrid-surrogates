"""Run everything, in the order that reproduces every number, table, figure
and video of the repository (about 25 minutes on 8 cores).

    python3 run_all.py

This is equivalent to

    cd pin_extraction
    python3 pin_experiment.py --picture   # MuJoCo pull -> data/pin.npz        (~1 s)
    python3 make_media.py                 # GIF, MP4, force/displacement graphs (~15 s)
    python3 dapinn_pin.py                 # DA-PINN on the pin                  (~7 min, 7 processes)
    cd ../bouncing_ball
    python3 run_ball.py                   # DA-PINN on the bouncing ball        (~15 min, 1 thread)

All random seeds are fixed in the scripts; PYTHONHASHSEED is set here so
that hash randomization does not affect the run.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [("pin_extraction", ["pin_experiment.py", "--picture"]),
         ("pin_extraction", ["make_media.py"]),
         ("pin_extraction", ["dapinn_pin.py"]),
         ("bouncing_ball", ["run_ball.py"])]


def main():
    env = dict(os.environ, PYTHONHASHSEED="0")
    for folder, cmd in STEPS:
        print(f"\n==== {folder}/{' '.join(cmd)} ====\n", flush=True)
        r = subprocess.run([sys.executable, "-u", *cmd], cwd=HERE / folder, env=env)
        if r.returncode:
            sys.exit(r.returncode)
    print("\nall done: media/, pin_extraction/results, pin_extraction/figs, bouncing_ball/results, bouncing_ball/figs")


if __name__ == "__main__":
    main()
