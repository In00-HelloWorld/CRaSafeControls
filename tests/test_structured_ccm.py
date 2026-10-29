"""Independent derivative and input-flow checks for exact-C2 metrics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from neural_ccm.ccm_conditions import ccm_matrices, model_derivatives
from neural_ccm.neural_metric import load_metric


class StructuredCCMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(702)
        L = rng.normal(size=(9, 9))
        self.Q = L@L.T+.1*np.eye(9)
        self.path = Path(self.temp.name)/"metric.npz"
        self.metadata = dict(architecture="right_invariant_constant", rate=.5, w_lb=1e-4,
                             domain=dict(x_min=[-5]*3+[-1]*3+[-.35]*3,
                                         x_max=[5]*3+[1]*3+[.35]*3,
                                         a_min=[-1]*5, a_max=[1]*5))
        self.save(self.Q)
        self.metric = load_metric(self.path)

    def save(self, Q):
        np.savez(self.path, Q=Q, metadata=json.dumps(self.metadata))

    def test_full_C2_identity_and_metric_derivatives(self):
        rng = np.random.default_rng(73)
        for _ in range(12):
            x, b = rng.uniform(-.3, .3, 9), rng.uniform(-.4, .4, 5)
            W, _, C2, _ = ccm_matrices(self.metric, x, b)
            self.assertGreater(np.linalg.eigvalsh(W)[0], 0)
            self.assertLess(np.max(np.abs(C2)), 1e-12)
            dx, db = self.metric.derivatives(x, b)
            for i in range(9):
                offset = 1e-5*np.eye(9)[i]
                numerical = (self.metric.W(x+offset, b)-self.metric.W(x-offset, b))/2e-5
                np.testing.assert_allclose(dx[:, :, i], numerical, atol=2e-8)
            np.testing.assert_array_equal(db, 0.)
            np.testing.assert_array_equal(W, self.metric.W(x, -b))

    def test_invariance_under_actual_input_flows(self):
        x0 = np.array([.1, -.1, .3, .24, -.15, .12, .13, -.12, .16])
        W0 = self.metric.W(x0, np.zeros(5))
        for j in range(4):
            def flow(t, state):
                x, Phi = state[:9], state[9:].reshape(9, 9)
                _, _, B, *DB = model_derivatives(x, np.zeros(5))
                return np.r_[B[:, j], (DB[j]@Phi).ravel()]
            solution = solve_ivp(flow, (0, .17), np.r_[x0, np.eye(9).ravel()],
                                 rtol=1e-11, atol=1e-12)
            self.assertTrue(solution.success)
            final = solution.y[:, -1]
            Phi = final[9:].reshape(9, 9)
            np.testing.assert_allclose(self.metric.W(final[:9], np.zeros(5)),
                                       Phi@W0@Phi.T, atol=2e-9)

    def test_invalid_matrix_and_domain_checks(self):
        self.save(np.diag([-1]+[1]*8))
        with self.assertRaises(np.linalg.LinAlgError):
            load_metric(self.path)
        self.assertTrue(self.metric.in_training_domain(np.zeros(9), np.zeros(5)))
        self.assertFalse(self.metric.in_training_domain(np.ones(9)*10, np.zeros(5)))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.metric.W(np.full(9, np.nan), np.zeros(5))

    def test_model_jacobians_against_finite_differences(self):
        x, b = np.arange(9)/50, np.arange(5)/20
        _, A, _, *DB = model_derivatives(x, b)
        for i in range(9):
            offset = 1e-6*np.eye(9)[i]
            plus, minus = model_derivatives(x+offset, b), model_derivatives(x-offset, b)
            np.testing.assert_allclose(A[:, i], ((plus[0]-minus[0])/2e-6).ravel(), atol=1e-8)
            for j in range(4):
                np.testing.assert_allclose(DB[j][:, i], ((plus[2]-minus[2])/2e-6)[:, j], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
