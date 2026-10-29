"""NumPy evaluation and exact derivatives of the exported Neural CCM."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

WEIGHT_NAMES = ("weight0", "bias0", "weight2", "bias2", "weight4")


def load_metric(checkpoint):
    """Select an explicit exported metric format; legacy NPZs remain unchanged."""
    with np.load(checkpoint, allow_pickle=False) as data:
        kind = json.loads(str(data["metadata"].item())).get("architecture", "neural")
    if kind == "right_invariant_constant":
        from .structured_metric import InvariantCCMMetric
        return InvariantCCMMetric(checkpoint)
    if kind != "neural":
        raise ValueError(f"Unknown metric architecture: {kind}")
    return NeuralCCMMetric(checkpoint)


class NeuralCCMMetric:
    """W=L.T@L+w_lb I; network inputs are x[3:9] and five physical coefficients."""

    def __init__(self, checkpoint):
        self.checkpoint_path = Path(checkpoint).resolve()
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"].item()))
            self.weights = tuple(np.array(data[key], dtype=float) for key in WEIGHT_NAMES)
        self.parameterization = self.metadata["parameterization"]
        if self.parameterization != "physical":
            raise ValueError("The 9D metrics use five physical coefficients Theta@a")
        self.adim = 5
        w0, b0, w2, b2, w4 = self.weights
        hidden = len(b0)
        expected = ((hidden, 6+self.adim), (hidden,), (hidden, hidden),
                    (hidden,), (81, hidden))
        if tuple(w.shape for w in self.weights) != expected:
            raise ValueError("Unexpected NeuralCCM metric architecture")
        if not all(np.all(np.isfinite(w)) for w in self.weights):
            raise ValueError("Metric weights must be finite")
        for weight in self.weights:
            weight.setflags(write=False)
        self.w_lb = float(self.metadata["w_lb"])
        self.rate = float(self.metadata["rate"])
        if not np.isfinite(self.w_lb) or self.w_lb <= 0:
            raise ValueError("w_lb must be finite and positive")
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("The contraction rate must be finite and positive")
        domain = self.metadata.get("domain")
        self.domain = None if domain is None else {
            name: np.asarray(domain[name], dtype=float).reshape(size)
            for name, size in (("x_min", 9), ("x_max", 9),
                               ("a_min", self.adim), ("a_max", self.adim))
        }
        if self.domain is not None:
            if (not all(np.all(np.isfinite(value)) for value in self.domain.values())
                    or np.any(self.domain["x_min"] > self.domain["x_max"])
                    or np.any(self.domain["a_min"] > self.domain["a_max"])):
                raise ValueError("Training bounds must be finite and ordered")

    def _features(self, x, parameter):
        x = np.asarray(x, dtype=float).reshape(9)
        parameter = np.asarray(parameter, dtype=float).reshape(self.adim)
        features = np.r_[x[3:9], parameter]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(features)):
            raise ValueError("Metric state and parameters must be finite")
        return tuple(features)

    @lru_cache(maxsize=128)
    def _evaluate(self, features):
        # Forward-mode differentiation of the two tanh layers. Keeping all
        # derivatives together avoids repeating backpropagation at each node.
        w0, b0, w2, b2, w4 = self.weights
        h1 = np.tanh(w0 @ features + b0)
        h2 = np.tanh(w2 @ h1 + b2)
        L = (w4 @ h2).reshape(9, 9)
        dL = (w4 @ ((1-h2*h2)[:, None] *
                     (w2 @ ((1-h1*h1)[:, None] * w0)))).reshape(9, 9, -1)
        W = L.T @ L + self.w_lb*np.eye(9)
        derivative = np.einsum("jip,jk->ikp", dL, L)
        derivative += np.einsum("ji,jkp->ikp", L, dL)
        W.setflags(write=False)
        derivative.setflags(write=False)
        return W, derivative

    def W(self, x, parameter):
        return self._evaluate(self._features(x, parameter))[0]

    def derivatives(self, x, parameter):
        derivative = self._evaluate(self._features(x, parameter))[1]
        dx = np.zeros((9, 9, 9))
        dx[:, :, 3:9] = derivative[:, :, :6]
        return dx, derivative[:, :, 6:]

    def in_training_domain(self, x, parameter, atol=1e-10):
        """Return None when domain metadata is unavailable, never assume it."""
        if self.domain is None:
            return None
        x = np.asarray(x, dtype=float).reshape(9)
        parameter = np.asarray(parameter, dtype=float).reshape(self.adim)
        d = self.domain
        return bool(np.all(np.isfinite(x)) and np.all(np.isfinite(parameter))
                    and np.all(x >= d["x_min"]-atol)
                    and np.all(x <= d["x_max"]+atol)
                    and np.all(parameter >= d["a_min"]-atol)
                    and np.all(parameter <= d["a_max"]+atol))
