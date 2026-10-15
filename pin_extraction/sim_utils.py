"""sim_utils.py - the two rules written in Python, and the robot's motion profiles.

MuJoCo simulates the robot axes, gravity, the joint friction constraint and the
puller spring (a tendon). Two things are written here:

  FrictionLaw   MuJoCo has one friction value per joint, so a static value larger
                than the kinetic one does not exist. FrictionLaw switches the
                joint's `frictionloss` between mu_s N (stuck) and mu_k N (sliding),
                with a small velocity hysteresis, and sets 0 once the part has left
                its fit. Every switch is logged as a true event of the hybrid system.
  engage()      the hook switches its tendon spring on when it is on the pin,
                like closing a gripper.

The breakaway itself is MuJoCo physics: the friction constraint holds the pin
until the load exceeds mu_s N, then the pin moves, and only then does
FrictionLaw lower the friction to mu_k N. Likewise, the arrest happens because
the friction constraint stops the pin; FrictionLaw only notices it.
"""
import mujoco


class FrictionLaw:
    """Coulomb friction with a static value larger than the kinetic one, on
    one joint, over an engagement length.

    state "stick": frictionloss = static. MuJoCo holds the joint while the load
                   is below it; when the load exceeds it, the joint starts to
                   move. Above v_slip, the state becomes "slip".
    state "slip":  frictionloss = kinetic. Below v_stick, back to "stick".
    state "free":  the part has left its fit (position >= exit): no friction.

    The thresholds are a small hysteresis: a stuck joint creeps slowly (MuJoCo
    friction is a soft constraint, here below 0.1 mm/s), and a sliding joint
    must not be stopped by its own noise. Every change of state is a true event
    of the hybrid system, kept in `events`.
    """
    STATES = {"stick": 0, "slip": 1, "free": 2}

    def __init__(self, model, joint, static, kinetic, exit_position, v_slip, v_stick):
        self.model = model
        self.dof = model.joint(joint).dofadr[0]
        self.qadr = model.joint(joint).qposadr[0]
        self.static, self.kinetic, self.exit = static, kinetic, exit_position
        self.v_slip, self.v_stick = v_slip, v_stick
        self.state = "stick"
        self.events = []             # (time, "breakaway" / "arrest" / "exit")
        self.origin = 0.0            # position counted from here

    def position(self, data):
        return data.qpos[self.qadr] - self.origin

    def update(self, data):
        x, v = self.position(data), data.qvel[self.dof]
        if self.state != "free" and x >= self.exit:
            self.state = "free"
            self.events.append((data.time, "exit"))
        elif self.state == "stick" and v > self.v_slip:
            self.state = "slip"
            self.events.append((data.time, "breakaway"))
        elif self.state == "slip" and v < self.v_stick:
            self.state = "stick"
            self.events.append((data.time, "arrest"))
        self.model.dof_frictionloss[self.dof] = {"stick": self.static, "slip": self.kinetic,
                                                 "free": 0.0}[self.state]

    def code(self):
        return self.STATES[self.state]


def engage(model, data, tendon, stiffness, damping):
    """Switch a coupling spring on, unstretched at the present tendon length."""
    i = model.tendon(tendon).id
    mujoco.mj_forward(model, data)
    model.tendon_stiffness[i] = stiffness
    model.tendon_damping[i] = damping
    model.tendon_lengthspring[i] = [data.ten_length[i]] * 2
    model.tendon_rgba[i, 3] = 1.0          # draw the spring from now on
    return float(data.ten_length[i])


def tendon_force(model, data, tendon):
    """Tension of a coupling (N): what a load cell in series with the spring
    measures, k (L - L0) + c dL/dt."""
    i = model.tendon(tendon).id
    return float(model.tendon_stiffness[i] * (data.ten_length[i] - model.tendon_lengthspring[i, 0])
                 + model.tendon_damping[i] * data.ten_velocity[i])


def smooth(u):
    """0 -> 1 with zero speed at both ends (smoothstep)."""
    u = min(max(u, 0.0), 1.0)
    return 3 * u**2 - 2 * u**3


RAMP = 0.05   # s, a "linear" move reaches its constant speed in this time


def linear_profile(t, distance, speed):
    """Distance travelled after t for a constant-speed move with a short
    linear speed ramp at the start (so the robot is not kicked)."""
    if t < RAMP:
        s = 0.5 * speed * t * t / RAMP
    else:
        s = speed * (t - 0.5 * RAMP)
    return min(s, distance)
