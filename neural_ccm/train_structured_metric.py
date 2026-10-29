"""Fit an exact-C2 metric by sampled semidefinite optimization, without Torch.

Q>0 ensures positivity; right invariance ensures full C2=0 identically.
The finite C1 constraints and counterexample search are not an SOS certificate.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_ccm.ccm_conditions import model_derivatives, projection
from neural_ccm.neural_metric import NeuralCCMMetric
from neural_ccm.structured_metric import invariant_frame


def coefficients(x, b):
    f, A, B, *_ = model_derivatives(x, b)
    J, dJ = invariant_frame(x)
    P = projection(B)
    return P.T@J, P.T@(A@J - np.einsum("i,ijk->jk", f.ravel(), dJ))


def domain_pool(domain, count, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(domain["x_min"], domain["x_max"], (count, 9))
    b = rng.uniform(domain["a_min"], domain["a_max"], (count, 5))
    corners = np.array(list(product((-1., 1.), repeat=6)))
    xc = np.zeros((len(corners), 9))
    xc[:, 3:] = corners*np.asarray(domain["x_max"])[3:]
    bc = np.array(list(product(*zip(domain["a_min"], domain["a_max"]))))
    return (np.vstack((np.zeros((1, 9)), x, np.repeat(xc, len(bc), axis=0))),
            np.vstack((np.zeros((1, 5)), b, np.tile(bc, (len(xc), 1)))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--pool-size", type=int, default=8192)
    parser.add_argument("--velocity-limit", type=float, default=.75)
    parser.add_argument("--tilt-limit", type=float, default=.25)
    parser.add_argument("--yaw-limit", type=float, default=.2)
    parser.add_argument("--rate", type=float, default=.5)
    parser.add_argument("--margin", type=float, default=.005)
    parser.add_argument("--q-floor", type=float, default=.001)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args(argv)
    if (args.output.exists() or args.pool_size < 256 or args.seed < 0
            or not np.isfinite([args.velocity_limit, args.tilt_limit, args.yaw_limit,
                               args.rate, args.margin, args.q_floor]).all()
            or min(args.velocity_limit, args.tilt_limit, args.yaw_limit, args.rate,
                   args.margin, args.q_floor, args.rounds) <= 0 or args.tilt_limit >= 1.5):
        parser.error("Choose a new output folder and finite positive settings")
    import cvxpy as cp
    baseline_path = ROOT/"simulations/quadrotor_9d/assets/tracking_zonly_metric.npz"
    baseline = NeuralCCMMetric(baseline_path)
    domain = {key: value.tolist() for key, value in baseline.domain.items()}
    domain["x_min"][3:6], domain["x_max"][3:6] = [-args.velocity_limit]*3, [args.velocity_limit]*3
    domain["x_max"][6:9] = [args.tilt_limit, args.tilt_limit, args.yaw_limit]
    domain["x_min"][6:9] = [-args.tilt_limit, -args.tilt_limit, -args.yaw_limit]
    x, b = domain_pool(domain, args.pool_size, args.seed)
    U, V = (np.asarray(values) for values in zip(*(coefficients(xx, bb) for xx, bb in zip(x, b))))
    target = baseline.W(np.zeros(9), np.zeros(5))
    active, history = list(range(257)), []
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    for iteration in range(args.rounds):
        Q = cp.Variable((9, 9), symmetric=True)
        constraints = [Q >> args.q_floor*np.eye(9), cp.trace(Q) == np.trace(target)]
        for i in active:
            H = V[i]@Q@U[i].T + U[i]@Q@V[i].T + 2*args.rate*U[i]@Q@U[i].T
            constraints.append(H << -args.margin*np.eye(5))
        problem = cp.Problem(cp.Minimize(cp.sum_squares(Q-target)), constraints)
        problem.solve(solver="CLARABEL", max_iter=150, tol_feas=1e-8, tol_gap_abs=1e-8)
        if Q.value is None:
            raise RuntimeError(f"No feasible matrix: {problem.status}; output contains no deployable metric")
        value = (Q.value+Q.value.T)/2
        H = V@value@U.transpose(0, 2, 1) + U@value@V.transpose(0, 2, 1)
        H += 2*args.rate*U@value@U.transpose(0, 2, 1)
        eigenvalues = np.linalg.eigvalsh((H+H.transpose(0, 2, 1))/2)[:, -1]
        record = dict(round=iteration, constraints=len(active), solver_status=problem.status,
                      worst_pool_eigenvalue=float(eigenvalues.max()),
                      minimum_Q_eigenvalue=float(np.linalg.eigvalsh(value)[0]))
        history.append(record)
        (args.output/"progress.json").write_text(json.dumps(history, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        if eigenvalues.max() < -.9*args.margin and np.linalg.eigvalsh(value)[0] >= args.q_floor-1e-7:
            break
        active_set = set(active)
        new = [int(i) for i in np.argsort(eigenvalues)[::-1] if i not in active_set][:128]
        active.extend(new)
    else:
        raise RuntimeError("C1 search did not converge; output contains no deployable metric")
    fingerprint = hashlib.sha256(value.tobytes()).hexdigest()
    vmax2 = 3*args.velocity_limit**2
    metadata = dict(
        architecture="right_invariant_constant", parameterization="physical",
        source_checkpoint="sampled_LMI/right_invariant_Q", source_sha256=fingerprint,
        task="QUADROTOR_9D_EULER_PAYLOAD_PLANAR_STYLE", rate=args.rate, domain=domain,
        w_lb=float(np.linalg.eigvalsh(value)[0]/(9+3*vmax2)),
        baseline_metric_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        training=dict(seed=args.seed, pool_size=len(x), margin=args.margin,
                      q_floor=args.q_floor, history=history,
                      wall_seconds=time.perf_counter()-started),
        guarantee=dict(positivity="Q positive definite and J invertible on regular Euler coordinates",
                       C2="Full matrix identity from right invariance; floating point evaluation has rounding error",
                       C1="Finite sampled LMI constraints and search only; no whole-domain certificate"),
    )
    np.savez_compressed(args.output/"metric.npz", Q=value, metadata=json.dumps(metadata))
    np.savez_compressed(args.output/"training_samples.npz", x=x[active], b=b[active])
    (args.output/"summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


if __name__ == "__main__":
    main()
