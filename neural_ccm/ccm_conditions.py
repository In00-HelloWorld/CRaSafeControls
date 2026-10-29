"""CCM matrices for the unchanged 9D estimated model F=f+Psi b."""

from functools import lru_cache
from types import SimpleNamespace

import casadi as ca
import numpy as np

from systems.quadrotor_9d.quadrotor_9d import QUADROTOR_9D


@lru_cache(maxsize=1)
def dynamics_function():
    system = QUADROTOR_9D(SimpleNamespace(rate=.5))
    x, b = ca.SX.sym("x", 9), ca.SX.sym("b", 5)
    state = np.array([x[i] for i in range(9)], dtype=object)
    matrix = lambda rows: ca.vertcat(*[ca.horzcat(*row) for row in rows])
    f = ca.vertcat(*system.f(state)) + matrix(system.psi(state)) @ b
    B = matrix(system.g(state))
    return ca.Function("metric_model_derivatives", [x, b], [
        f, ca.jacobian(f, x), B, *[ca.jacobian(B[:, j], x) for j in range(4)],
    ])


def model_derivatives(x, b):
    return tuple(np.asarray(value) for value in dynamics_function()(x, b))


def projection(B):
    return np.linalg.svd(B, full_matrices=True)[0][:, 4:]


def ccm_matrices(metric, x, b):
    f, A, B, *DB = model_derivatives(x, b)
    W = metric.W(x, b)
    dW = metric.derivatives(x, b)[0]
    H = -np.einsum("ijk,k->ij", dW, f.ravel()) + A@W + W@A.T + 2*metric.rate*W
    C2 = np.array([np.einsum("ijk,k->ij", dW, B[:, j]) - DB[j]@W - W@DB[j].T
                   for j in range(4)])
    P = projection(B)
    return W, P.T@H@P, C2, np.array([P.T@C@P for C in C2])


def evaluate(metric, states, parameters):
    rows = []
    for x, b in zip(states, parameters):
        W, C1, C2, projected = ccm_matrices(metric, x, b)
        rows.append((np.linalg.eigvalsh(W)[0], np.linalg.eigvalsh((C1+C1.T)/2)[-1],
                     np.max(np.linalg.norm(C2, axis=(1, 2))),
                     np.max(np.linalg.norm(projected, axis=(1, 2))),
                     metric.in_training_domain(x, b)))
    data = np.asarray(rows)
    return dict(samples=len(data), minimum_W_eigenvalue=float(data[:, 0].min()),
                C1_pass_fraction=float(np.mean(data[:, 1] < 0)),
                worst_C1_eigenvalue=float(data[:, 1].max()),
                max_full_C2_frobenius=float(data[:, 2].max()),
                max_projected_C2_frobenius=float(data[:, 3].max()),
                in_domain_fraction=float(data[:, 4].mean())), data
