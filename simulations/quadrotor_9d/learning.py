"""Generate observations, learn Theta, calibrate OLACP and set adaptation gains."""


import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from olacp import OLACP
from systems.quadrotor_9d.quadrotor_9d import QUADROTOR_9D, THETA_INITIAL
from systems.quadrotor_9d.nonlinear_uncertainty import WindEnvironment, nonlinear_uncertainty
from .results import dump_json, core_hashes, repository_state


A_LOWER, A_UPPER = -np.ones(4), np.ones(4)
THETA_DELTA = np.broadcast_to(np.array([.08, .08, .06, .03, .03])[:, None], (5, 4))
THETA_LOWER, THETA_UPPER = THETA_INITIAL - THETA_DELTA, THETA_INITIAL + THETA_DELTA


def rk4_step(system, x, u, dt, t=0.):
    def dynamics(state, tau):
        return system.dynamics(state, u, tau)
    k1 = dynamics(x, t)
    k2 = dynamics(x+dt*k1/2, t+dt/2)
    k3 = dynamics(x+dt*k2/2, t+dt/2)
    k4 = dynamics(x+dt*k3, t+dt)
    return x+dt*(k1+2*k2+2*k3+k4)/6


def pd_input(system, x, x_d, u_ref):
    """Uploaded nominal pretraining feedback; no true-uncertainty input."""
    acc = system.dynamics_nominal(x_d, u_ref)[3:6]
    acc += 3*(x_d[:3]-x[:3])+2.5*(x_d[3:6]-x[3:6])
    force = np.clip(acc, -5, 5)+np.array([0, 0, system.grav])
    thrust = np.linalg.norm(force)
    b3 = force/max(thrust, 1e-9)
    yaw = x_d[8]
    bx = np.cos(yaw)*b3[0]+np.sin(yaw)*b3[1]
    by = -np.sin(yaw)*b3[0]+np.cos(yaw)*b3[1]
    eta = np.array([np.arcsin(np.clip(-by, -1, 1)), np.arctan2(bx, b3[2]), yaw])
    err = eta-x[6:9]
    err[2] = (err[2]+np.pi) % (2*np.pi)-np.pi
    eta_dot = system.g(x_d)[6:9, 1:] @ np.asarray(u_ref)[1:]+4*err
    return np.r_[thrust, np.linalg.solve(system.g(x)[6:9, 1:], eta_dot)]


def make_system(metric, theta):
    return QUADROTOR_9D(metric, {
        "Theta_init": theta, "a_lb": A_LOWER, "a_ub": A_UPPER,
    })


def make_learner(system, args, learning_rate, online_theta=True):
    representation = {}
    if online_theta:
        representation = dict(
            Theta_init=system.Theta_hat.copy(), Theta_lb=THETA_LOWER,
            Theta_ub=THETA_UPPER, representation_period=args.representation_period,
            representation_lr=learning_rate, Y_Theta=system.Y_Theta,
            representation_loss_gradient=system.representation_loss_gradient,
        )
    return OLACP(
        [], N_cal=args.calibration_intervals, acp_lr=args.acp_lr,
        delta_target=args.delta, delta_init=args.delta,
        buffer_maxlen=args.interval_steps, **representation,
    )


def put_interval(learner, system, data, index, dt):
    learner.clear_buffers()
    for x, nominal, measured in zip(
        data["x"][index], data["nominal"][index], data["xdot"][index]
    ):
        learner.add_data_to_buffers(x, nominal, system.Y(x), xdot=measured)
    learner.estimate_uncertainty(dt)


def calibrate(metric, args, theta, data, learning_rate, online_theta=True):
    system = make_system(metric, theta)
    learner = make_learner(system, args, learning_rate, online_theta)
    for index in range(len(data["x"])):
        put_interval(learner, system, data, index, args.dt)
        learner.append_score(learner.compute_score(A_UPPER, A_LOWER))
    learner.compute_quantile()
    learner.clear_buffers()
    return learner


def collect_calibration_data(metric, args, intervals, seed, residual_scale):
    """Generate observations using the uploaded physical plant and nominal PD.

    The controller sees only position/velocity and nominal dynamics. The measured
    derivative is recorded separately, as in planar_quad's pretraining stage.
    """
    rng = np.random.default_rng(seed)
    system = make_system(metric, THETA_INITIAL)
    phases = rng.uniform(0, 2*np.pi, 3)
    amplitude, frequency = np.array([1.2, 1., .25]), np.array([.8, .7, .31])
    x = np.zeros(9)
    data = {key: [] for key in ("x", "u", "nominal", "xdot", "t", "environment")}
    for interval in range(intervals):
        wind = rng.uniform([-.15, -.15, -.05], [.15, .15, .05])
        env = WindEnvironment(*wind, payload=rng.uniform(.07, .15),
                              phase=rng.uniform(0, 2*np.pi),
                              residual_scale=residual_scale)
        system.true_uncertainty_fcn = lambda state, t, e=env: nonlinear_uncertainty(state, t, e)
        block = {key: [] for key in ("x", "u", "nominal", "xdot", "t")}
        for step in range(args.interval_steps):
            t = (interval*args.interval_steps+step)*args.dt
            desired = np.zeros(9)
            desired[:3] = amplitude*np.sin(frequency*t+phases)
            desired[3:6] = amplitude*frequency*np.cos(frequency*t+phases)
            u = np.clip(pd_input(system, x, desired, [9.81, 0, 0, 0]),
                        args.u_min, args.u_max)
            nominal = system.dynamics_nominal(x, u)
            measured = system.dynamics(x, u, t)
            for key, value in {"x": x.copy(), "u": u, "nominal": nominal,
                               "xdot": measured, "t": t}.items():
                block[key].append(value)
            x = rk4_step(system, x, u, args.dt, t)
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 100:
                raise RuntimeError(f"Data collection diverged at t={t+args.dt}")
        for key in block:
            data[key].append(block[key])
        data["environment"].append(list(asdict(env).values()))
    return {key: np.asarray(value) for key, value in data.items()}


def pretrain_representation(metric, args, data):
    """Fit a_k each interval and update Theta every B intervals, like planar_quad."""
    system = make_system(metric, THETA_INITIAL)
    learner = make_learner(system, args, args.learning_rate, online_theta=True)
    theta_history, scores, fitted = [learner.Theta.copy()], [], []
    for index in range(len(data["x"])):
        put_interval(learner, system, data, index, args.dt)
        score = float(learner.compute_score(A_UPPER, A_LOWER))
        scores.append(score)
        fitted.append(learner.a_k.copy())
        learner.append_score(score)
        update = learner.update_representation()
        if update is not None:
            system.set_representation(update["Theta"])
        theta_history.append(learner.Theta.copy())
        learner.clear_buffers()
    if learner._representation_intervals:
        raise RuntimeError("Pretraining must end on a complete representation block")
    return learner.Theta.copy(), dict(theta=np.asarray(theta_history),
                                    score=np.asarray(scores), a_fit=np.asarray(fitted))


def prepare_from_scratch(metric, args, output):
    """Produce reusable results rather than requiring a pre-existing results folder."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    training = collect_calibration_data(metric, args, args.pretrain_intervals,
                                        args.seed, args.residual_scale)
    theta, history = pretrain_representation(metric, args, training)
    # A separately seeded observation run calibrates the final, frozen Theta.
    # Online Theta updates are enabled only after this independent calibration.
    calibration = collect_calibration_data(metric, args, args.calibration_intervals,
                                           args.seed+1, args.residual_scale)
    np.savez_compressed(output/"data_pretraining.npz", **training)
    np.savez_compressed(output/"pretraining_history.npz", **history)
    np.savez_compressed(output/"data_calibration.npz", **calibration)
    metadata = dict(
        schema_version=1, generation="fresh_physical_plant_observations",
        dt=args.dt, interval_steps=args.interval_steps,
        intervals={"pretraining":args.pretrain_intervals,
                   "calibration":args.calibration_intervals},
        representation_period=args.representation_period,
        learning_rate=args.learning_rate, theta_initial=THETA_INITIAL,
        theta_learned=theta, controller_rate=metric.rate,
        seeds={"pretraining":args.seed,"calibration":args.seed+1},
        residual_scale=args.residual_scale, core_hashes=core_hashes(),
        repository=repository_state(),
        metric_source_sha256=metric.metadata["source_sha256"],
        representation_updates=args.pretrain_intervals//args.representation_period,
        theta_change_norm=float(np.linalg.norm(theta-THETA_INITIAL)),
    )
    dump_json(output/"summary.json", metadata)
    return metadata, calibration


def normalized_gamma(theta, physical_gain, regularization):
    gram = np.asarray(theta).T @ np.asarray(theta) + regularization*np.eye(4)
    gamma = physical_gain*np.linalg.inv(gram)
    gamma = 0.5*(gamma+gamma.T)
    np.linalg.cholesky(gamma)
    return gamma


def payload_boosted_gamma(theta, physical_gain, regularization, boost):
    """Increase ``theta[2] Gamma theta[2].T`` by exactly ``boost``."""
    if boost < 0:
        raise ValueError("payload boost must be nonnegative")
    gamma = normalized_gamma(theta, physical_gain, regularization)
    direction = np.asarray(theta)[2]
    norm_squared = float(direction @ direction)
    if norm_squared <= np.finfo(float).eps:
        raise ValueError("Theta has no payload direction")
    gamma += boost*np.outer(direction, direction)/norm_squared**2
    gamma = 0.5*(gamma+gamma.T)
    np.linalg.cholesky(gamma)
    return gamma


def finish_interval(learner, system, args):
    learner.estimate_uncertainty(args.dt)
    score = float(learner.compute_score(system.a_ub, system.a_lb))
    previous_quantile = learner.Q_k
    missed = learner.update_delta(score)
    learner.append_score(score)
    update = learner.update_representation()
    if update is not None:
        system.set_representation(update["Theta"])
    system.cp_quantile = learner.compute_quantile()
    event = dict(score=score, previous_quantile=previous_quantile,
                 next_quantile=system.cp_quantile, miscoverage=missed,
                 delta=float(learner.delta), a_fit=learner.a_k.copy(),
                 representation_updated=update is not None)
    learner.clear_buffers()
    return event


def clone_learner(pretrained, system):
    """Copy interval state without copying the old model/metric through callbacks."""
    learner = copy.copy(pretrained)
    learner.__dict__ = copy.deepcopy({
        key: value for key, value in vars(pretrained).items()
        if key not in ("Y_Theta", "representation_loss_gradient")
    })
    learner.clear_buffers()
    if learner._representation_intervals:
        raise ValueError("Pretraining must end at a complete representation block")
    learner.Y_Theta = system.Y_Theta
    learner.representation_loss_gradient = system.representation_loss_gradient
    if learner.Theta is not None:
        system.set_representation(learner.Theta)
    system.cp_quantile = learner.compute_quantile()
    return learner
