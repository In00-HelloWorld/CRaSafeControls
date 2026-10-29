"""Training gradients and exported Neural CCM runtime agree independently."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from neural_ccm.neural_metric import NeuralCCMMetric


@unittest.skipIf(torch is None, "Torch is an optional neural-training dependency")
class NeuralTrainingTests(unittest.TestCase):
    def setUp(self):
        from neural_ccm.train_neural_metric import MetricNetwork
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1]/"simulations/quadrotor_9d/assets/tracking_zonly_metric.npz"
        self.initial = NeuralCCMMetric(path)
        self.model = MetricNetwork(self.initial)

    def test_forward_derivatives_match_autograd_and_numpy(self):
        x = torch.linspace(-.2, .2, 9, dtype=torch.float64)
        b = torch.linspace(-.1, .1, 5, dtype=torch.float64)
        W, dW = self.model(x[None], b[None])
        jacobian = torch.autograd.functional.jacobian(
            lambda state: self.model(state[None], b[None])[0][0], x, vectorize=True)
        np.testing.assert_allclose(dW[0].detach(), jacobian.detach(), atol=1e-13)
        np.testing.assert_allclose(W[0].detach(), self.initial.W(x.numpy(), b.numpy()), atol=1e-14)
        np.testing.assert_allclose(dW[0].detach(), self.initial.derivatives(x.numpy(), b.numpy())[0], atol=1e-13)

    def test_C2_weight_gradient_matches_finite_difference(self):
        from neural_ccm.train_neural_metric import matrices, observations
        batch = observations(self.initial.domain, 3, 84)
        objective = matrices(self.model, batch, .5)[2].square().sum()
        objective.backward()
        weight = next(self.model.parameters())
        index = np.unravel_index(weight.grad.abs().argmax().item(), weight.shape)
        analytic = weight.grad[index].item()
        value, eps = weight[index].item(), 1e-6
        with torch.no_grad():
            weight[index] = value+eps
            plus = matrices(self.model, batch, .5)[2].square().sum().item()
            weight[index] = value-eps
            minus = matrices(self.model, batch, .5)[2].square().sum().item()
            weight[index] = value
        self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), delta=max(1e-9, abs(analytic)*1e-6))

    def test_trained_weight_export_matches_runtime_and_weight_hash(self):
        from neural_ccm.train_neural_metric import export
        with torch.no_grad():
            next(self.model.parameters())[0, 0] += .001
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = export(self.model, self.initial, root, dict(method="test neural weight update"))
            metric = NeuralCCMMetric(root/"metric.npz")
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(
                b"".join(p.detach().numpy().tobytes() for p in self.model.layers.parameters())).hexdigest())
            self.assertEqual([p.name for p in root.iterdir()], ["metric.npz"])
            x, b = torch.ones((1, 9), dtype=torch.float64)*.1, torch.zeros((1, 5), dtype=torch.float64)
            W, derivative = self.model(x, b)
            np.testing.assert_allclose(metric.W(x[0], b[0]), W[0].detach(), atol=1e-14)
            np.testing.assert_allclose(metric.derivatives(x[0], b[0])[0], derivative[0].detach(), atol=1e-13)


if __name__ == "__main__":
    unittest.main()
