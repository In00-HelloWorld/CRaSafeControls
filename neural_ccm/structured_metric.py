"""Right-invariant dual metric with exact input Killing equations (C2=0).

Treat (v,R) as SE(3), with v its translation coordinate; p is unchanged.
Thrust and body-rate input flows are right translations on this group.
The frame J below is preserved by these flows, so W=J Q J.T, Q>0, has
full-matrix C2=0. Q is fitted offline; no auxiliary feedback is imported.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np


def skew(v):
    a, b, c = v
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def invariant_frame(x):
    """Map invariant tangent coordinates to [delta p, delta v, delta Euler]."""
    pitch, yaw = x[7:9]
    if abs(np.cos(pitch)) < 1e-5:
        raise ValueError("Euler pitch singularity")
    s, c = np.sin(yaw), np.cos(yaw)
    t, sec = np.tan(pitch), 1/np.cos(pitch)
    J = np.eye(9)
    J[3:6, 6:9] = -skew(x[3:6])
    J[6:9, 6:9] = [[c*sec, s*sec, 0], [-s, c, 0], [c*t, s*t, 1]]
    dJ = np.zeros((9, 9, 9))  # derivative index first
    for i in range(3):
        dJ[3+i, 3:6, 6:9] = -skew(np.eye(3)[i])
    dJ[7, 6:9, 6:9] = [[c*sec*t, s*sec*t, 0], [0, 0, 0],
                         [c*sec**2, s*sec**2, 0]]
    dJ[8, 6:9, 6:9] = [[-s*sec, c*sec, 0], [-c, -s, 0], [-s*t, c*t, 0]]
    return J, dJ


class InvariantCCMMetric:
    """State-dependent W=J(x) Q J(x).T with a constant learned SPD Q."""

    parameterization, adim = "physical", 5

    def __init__(self, checkpoint):
        self.checkpoint_path = Path(checkpoint).resolve()
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"].item()))
            self.Q = np.asarray(data["Q"], dtype=float)
        if (self.Q.shape != (9, 9) or not np.isfinite(self.Q).all()
                or not np.allclose(self.Q, self.Q.T, atol=1e-12)):
            raise ValueError("Q must be a finite symmetric 9x9 matrix")
        np.linalg.cholesky(self.Q)
        self.Q.setflags(write=False)
        self.rate = float(self.metadata["rate"])
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("Contraction rate must be finite and positive")
        self.domain = {key: np.asarray(value, dtype=float)
                       for key, value in self.metadata["domain"].items()}
        for low, high, size in (("x_min", "x_max", 9), ("a_min", "a_max", 5)):
            if (self.domain[low].shape != (size,) or self.domain[high].shape != (size,)
                    or not np.isfinite([self.domain[low], self.domain[high]]).all()
                    or np.any(self.domain[low] > self.domain[high])):
                raise ValueError("Metric domain must contain finite ordered bounds")
        if max(abs(self.domain["x_min"][7]), abs(self.domain["x_max"][7])) >= np.pi/2:
            raise ValueError("Metric domain crosses an Euler singularity")
        self.w_lb = float(self.metadata["w_lb"])
        vmax = np.maximum(abs(self.domain["x_min"][3:6]), abs(self.domain["x_max"][3:6]))
        bound = np.linalg.eigvalsh(self.Q)[0]/(9+3*(vmax@vmax))
        if not np.isfinite(self.w_lb) or not 0 < self.w_lb <= bound*(1+1e-6):
            raise ValueError("Invalid certified lower bound for W on the declared domain")

    @lru_cache(maxsize=128)
    def _evaluate(self, state):
        J, dJ = invariant_frame(np.asarray(state))
        W = J @ self.Q @ J.T
        derivative = dJ @ self.Q @ J.T
        derivative += derivative.transpose(0, 2, 1)
        W.setflags(write=False)
        derivative.setflags(write=False)
        return W, np.moveaxis(derivative, 0, -1)

    def _state(self, x, parameter):
        x = np.asarray(x, dtype=float).reshape(9)
        parameter = np.asarray(parameter, dtype=float).reshape(5)
        if not np.isfinite(x).all() or not np.isfinite(parameter).all():
            raise ValueError("Metric state and parameters must be finite")
        return tuple(x)

    def W(self, x, parameter):
        return self._evaluate(self._state(x, parameter))[0]

    def derivatives(self, x, parameter):
        return self._evaluate(self._state(x, parameter))[1], np.zeros((9, 9, 5))

    def in_training_domain(self, x, parameter, atol=1e-10):
        x, parameter = np.asarray(x), np.asarray(parameter)
        return bool(np.isfinite(x).all() and np.isfinite(parameter).all()
                    and np.all(x >= self.domain["x_min"]-atol)
                    and np.all(x <= self.domain["x_max"]+atol)
                    and np.all(parameter >= self.domain["a_min"]-atol)
                    and np.all(parameter <= self.domain["a_max"]+atol))
