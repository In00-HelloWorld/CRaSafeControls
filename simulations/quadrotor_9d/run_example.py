"""9D comparison using the cloud controller/planner and the uploaded reference model.

By default this generates fresh representation-learning and calibration data
using the uploaded physical plant. --source optionally reuses saved observations.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ccm.neural_metric import load_metric
from systems.quadrotor_9d.nonlinear_uncertainty import WindEnvironment, nonlinear_uncertainty
from simulations.quadrotor_9d import learning, results
from simulations.quadrotor_9d.simulation import run_simulation
from simulations.quadrotor_9d.reference import WaypointReference, validate_timing


ARMS = (("proposed", "Proposed", True, True),
        ("no_cp", "No CP", True, False),
        ("cp_no_adaptation", "Yes CP + no adaptation", False, True),
        ("no_adaptation", "No CP + no adaptation", False, False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        help="Optional saved summary.json/data_calibration.npz directory")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent/"assets/tracking_zonly_metric.npz",
                        help="CCM .npz: original neural, invariant LMI, or retrained neural")
    parser.add_argument("--pretrain-intervals", type=int, default=64)
    parser.add_argument("--calibration-intervals", type=int, default=32)
    parser.add_argument("--interval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--sim-time", type=float, default=12.)
    parser.add_argument("--dt", type=float, default=.02)
    parser.add_argument("--trajectory", choices=("spatial", "point"), default="spatial")
    parser.add_argument("--waypoints", type=Path, help="JSON array of K+1 position waypoints [x,y,z]")
    parser.add_argument("--initial-reference", type=float, nargs=9, default=np.zeros(9),
                        help="Complete initial reference state; position must match the first waypoint")
    parser.add_argument("--mpc-horizon-steps",type=int,default=40)
    parser.add_argument("--mpc-input-steps", type=int, default=1,
                        help="Simulation steps per held MPC reference input")
    parser.add_argument("--mpc-replan-steps",type=int,default=20)
    parser.add_argument("--goal", type=float, nargs=3, default=[.5,.3,.3])
    parser.add_argument("--online-theta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physical-gamma", type=float, default=3e-5)
    parser.add_argument("--payload-physical-boost", type=float, default=9.7e-4)
    parser.add_argument("--gamma-regularization", type=float, default=1e-5)
    parser.add_argument("--representation-period", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.)
    parser.add_argument("--rho-divergence-threshold",type=float,default=1e4)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(
        acp_lr=.01, delta=.1, eta=500., slack_weight=10000., geo_degree=2, geo_nodes=4,
        initial_offset=np.zeros(9), u_min=np.array([0, -2, -2, -2]),
        u_max=np.array([20, 2, 2, 2]),
    )
    args = parser.parse_args(argv)
    numeric = [args.sim_time, args.dt, args.physical_gamma, args.payload_physical_boost,
               args.gamma_regularization, args.residual_scale, args.learning_rate,
               args.rho_divergence_threshold, *args.goal, *args.initial_reference]
    if (not np.all(np.isfinite(numeric)) or min(args.sim_time, args.dt,
            args.physical_gamma, args.gamma_regularization, args.residual_scale,
            args.learning_rate, args.rho_divergence_threshold) <= 0
            or args.payload_physical_boost < 0 or args.representation_period < 1
            or args.mpc_horizon_steps < 2 or args.mpc_replan_steps < 1
            or not np.isclose(args.sim_time/args.dt, round(args.sim_time/args.dt))):
        parser.error("Times, gains, periods, scale and goal must be finite and valid")
    paths = [args.checkpoint]
    if args.source is not None:
        paths += [args.source/"summary.json",args.source/"data_calibration.npz"]
    if args.waypoints is not None:
        paths.append(args.waypoints)
    for path in paths:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.source is None and (args.calibration_intervals < 30 or args.interval_steps < 10
            or args.pretrain_intervals < 1 or args.seed < 0
            or args.pretrain_intervals % args.representation_period
            or args.calibration_intervals % args.representation_period
            or round(args.sim_time/args.dt) % args.interval_steps):
        parser.error("Pretraining/calibration need complete blocks, at least 30 calibration intervals and 10 steps")
    if args.output.is_file() or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Choose an empty output directory")
    try:
        validate_timing(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def prepare_learning(config, metric):
    """Generate or reuse observations, then calibrate the same initial learner for all arms."""
    source = config.source or config.output/"pretraining"
    if config.source is None:
        print("Generating pretraining and calibration observations...", flush=True)
        metadata, calibration = learning.prepare_from_scratch(metric, config, source)
    else:
        metadata = json.loads((source/"summary.json").read_text(encoding="utf-8"))
        if config.interval_steps != int(metadata["interval_steps"]):
            raise ValueError("Saved observations use a different --interval-steps value")
        config.calibration_intervals = int(metadata["intervals"]["calibration"])
        if "dt" in metadata and not np.isclose(metadata["dt"], config.dt):
            raise ValueError("Saved observations use a different dt")
        if metadata.get("metric_source_sha256", metric.metadata["source_sha256"]) != metric.metadata["source_sha256"]:
            raise ValueError("Saved pretraining used a different metric checkpoint")
        if np.isclose(config.residual_scale, metadata.get("residual_scale", 1.)):
            with np.load(source/"data_calibration.npz", allow_pickle=False) as data:
                calibration = {key: data[key].copy() for key in data.files}
        else:
            calibration = learning.collect_calibration_data(
                metric, config, config.calibration_intervals,
                int(metadata["seeds"]["calibration"]), config.residual_scale,
            )
    config.rate = float(metadata["controller_rate"])
    config.learning_rate = float(metadata["learning_rate"])
    if not np.isfinite(config.rate) or config.rate <= 0:
        raise ValueError("Controller rate must be finite and positive")
    if (config.interval_steps < 10 or config.calibration_intervals < 30
            or config.calibration_intervals % config.representation_period
            or round(config.sim_time/config.dt) % config.interval_steps):
        raise ValueError("Simulation and calibration must contain complete intervals/blocks")
    expected = (config.calibration_intervals, config.interval_steps, 9)
    for key in ("x", "nominal", "xdot"):
        if calibration[key].shape != expected or not np.isfinite(calibration[key]).all():
            raise ValueError(f"Calibration {key} must be finite with shape {expected}")
    theta = np.asarray(metadata["theta_learned"], dtype=float)
    learner = learning.calibrate(metric, config, theta, calibration, config.learning_rate, config.online_theta)
    if learner.Theta is not None:
        theta = learner.Theta.copy()
    gamma = learning.payload_boosted_gamma(theta, config.physical_gamma,
                                          config.gamma_regularization, config.payload_physical_boost)
    np.savez_compressed(config.output/"data_calibration.npz", **calibration)
    results.dump_json(config.output/"calibration.json", dict(
        scores=list(learner.S_cal), quantile=learner.Q_k, delta=learner.delta, theta=theta,
        source=str(source.resolve()), source_mode="reused" if config.source else "generated",
    ))
    print(f"Calibration complete: intervals={len(learner.S_cal)}, Q_0={learner.Q_k:.6g}", flush=True)
    return source, metadata, theta, learner, gamma


def main(argv=None):
    config = parse_args(argv)
    reference = WaypointReference(config)
    config.output.mkdir(parents=True, exist_ok=True)
    metric = load_metric(config.checkpoint)
    source, metadata, theta, learner, gamma = prepare_learning(config, metric)
    environment = WindEnvironment(residual_scale=config.residual_scale)
    plant = lambda x, t: nonlinear_uncertainty(x, t, environment)
    histories, intervals, summaries = {}, {}, {}
    for key, label, adaptive, cp in ARMS:
        print(f"Running {label}...", flush=True)
        data, events, outcome = run_simulation(metric, config, reference, learner, theta, gamma, adaptive, cp, plant)
        data["target"] = reference.guide(data["t"]).T  # Cost guide, not x_d.
        data["true_w"] = np.asarray([plant(x, t) for x, t in zip(data["x"], data["t"])])
        summary = results.summarize(data, events, learning.make_system(metric, theta), config.interval_steps)
        summaries[key] = dict(summary, **outcome, label=label, cp=cp, adaptive=adaptive)
        histories[label], intervals[key] = data, events
        np.savez_compressed(config.output/f"{key}.npz", **data)
    hashes = results.core_hashes()
    report = dict(
        schema_version=2,
        method="Position-waypoint natural spline guide; native CE-MPC; coupled live reference",
        status="completed" if all(arm["status"] == "completed" for arm in summaries.values()) else "controller_failures",
        source=str(source.resolve()), source_mode="reused" if config.source else "generated",
        pretraining_summary=metadata, metric=metric.metadata, trajectory=config.trajectory,
        environment=asdict(environment), configuration=vars(config), arms=summaries,
        reference=reference.metadata(),
        common_prefix_comparison=results.common_prefix_comparison(histories),
        repository=results.repository_state(), core_hashes=hashes,
        changed_core_files_from_local_source=[name for name in hashes if metadata.get("core_hashes", {}).get(name) != hashes[name]],
    )
    results.dump_json(config.output/"metric.json", metric.metadata)
    results.dump_json(config.output/"intervals.json", intervals)
    results.dump_json(config.output/"summary.json", report)
    results.save_plot(histories, config.output/"comparison.png")
    results.save_input_plot(histories, config.output/"inputs.png")
    results.save_reference_plot(histories, reference, config.output/"reference.png")
    print(json.dumps(summaries, default=lambda value: value.tolist(), indent=2))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "completed" else 1)

