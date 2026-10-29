"""The paper's guide/MPC/reference distinction and shared-boundary timing."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from motion_planner import MotionPlanner
from neural_ccm.neural_metric import load_metric
from simulations.quadrotor_9d.learning import make_learner, make_system
from simulations.quadrotor_9d.reference import WaypointReference, make_planner
from simulations.quadrotor_9d.simulation import run_simulation
from systems.quadrotor_9d.quadrotor_9d import QUADROTOR_9D, THETA_INITIAL


class QuadrotorReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            dt=.05, sim_time=1.5, interval_steps=10, representation_period=2,
            mpc_input_steps=2, mpc_replan_steps=4, mpc_horizon_steps=12,
            initial_reference=np.zeros(9), initial_offset=np.zeros(9), waypoints=None,
            trajectory="point", goal=[.02, .01, .03],
            u_min=np.array([0, -2, -2, -2]), u_max=np.array([20, 2, 2, 2]),
            calibration_intervals=30, acp_lr=.01, delta=.1, learning_rate=1e-4,
            slack_weight=1e4, eta=500., rate=.5, geo_degree=2, geo_nodes=4,
            rho_divergence_threshold=1e4, quiet=True,
        )
        path = Path(__file__).resolve().parents[1]/"simulations/quadrotor_9d/assets/tracking_zonly_metric.npz"
        self.metric = load_metric(path)
        self.system = make_system(self.metric, THETA_INITIAL)

    def test_natural_position_spline_and_separate_initial_state(self):
        c = self.config
        positions = np.array([[.1, .2, .3], [.2, .25, .4], [.3, .2, .3], [.4, .1, .2]])
        c.initial_reference = np.r_[positions[0], .04, -.02, .01, .03, -.01, .02]
        with tempfile.TemporaryDirectory() as temp:
            c.waypoints = Path(temp)/"waypoints.json"
            c.waypoints.write_text(json.dumps(positions.tolist()))
            ref = WaypointReference(c)
            np.testing.assert_allclose(ref.guide(ref.times).T, positions, atol=1e-14)
            np.testing.assert_allclose(ref.spline([0., 1.5], 2), 0., atol=1e-13)
            np.testing.assert_allclose(ref.guide([-1., 2., 4.]).T, positions[[0, -1, -1]], atol=1e-14)
            np.testing.assert_array_equal(ref.initial_state, c.initial_reference)
            self.assertEqual(ref.guide([.12, .43]).shape, (3, 2))
            c.initial_reference[0] += .1
            with self.assertRaisesRegex(ValueError, "start"):
                WaypointReference(c)

    def test_cost_is_position_only_discrete_time_and_native_MPC_has_no_waypoint_equalities(self):
        ref = WaypointReference(self.config)
        planner = make_planner(self.system, self.config)
        np.testing.assert_array_equal(planner.output_matrix, np.eye(3, 9))
        np.testing.assert_array_equal(planner.S, 0.)
        X = np.arange(27).reshape(9, 3)/100
        U = np.tile([9.81, 0., 0., 0.], (2, 1)).T
        U[0] += [.2, -.1]
        U[1] = [.05, -.03]
        guide = np.zeros((3, 3)); hover = np.array([9.81, 0, 0, 0])
        cost = lambda states: float(planner._tracking_cost(ca.DM(states), ca.DM(U),
                                                            ca.DM(guide), ca.DM(hover), ca.DM(hover)))
        expected = 8*np.sum(X[:3, :-1]**2)+.03*np.sum((U-hover[:, None])**2)+60*np.sum(X[:3, -1]**2)
        self.assertAlmostEqual(cost(X), expected)
        X[3:] += 20
        self.assertAlmostEqual(cost(X), expected)
        a = np.array([.1, .2, -.1, .05])
        predicted, inputs = planner.plan_certainty_equivalent_through_waypoints(
            ref.initial_state, ref.positions, self.config.interval_steps, 0,
            THETA_INITIAL, a, hover, mpc_dt=ref.mpc_dt, mpc_horizon=ref.horizon,
            output_reference=ref.guide,
        )
        np.testing.assert_allclose(predicted[:, 0], ref.initial_state, atol=1e-10)
        N = self.config.mpc_horizon_steps
        self.assertEqual(int(planner._mpc_problem["opti"].g.numel()), 9*(N+1)+8*N)
        for k in (0, N//2, N-1):
            solution = solve_ivp(lambda t, x: self.system.dynamics_nominal(x, inputs[:, k])
                                + self.system.Y_Theta(x, THETA_INITIAL)@a,
                                (0, ref.mpc_dt), predicted[:, k], rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(predicted[:, k+1], solution.y[:, -1], atol=2e-6)
        problem = planner._mpc_problem
        planner.plan_certainty_equivalent_through_waypoints(
            predicted[:, -1], ref.positions, self.config.interval_steps, 26,
            THETA_INITIAL, a, hover, mpc_dt=ref.mpc_dt, mpc_horizon=ref.horizon,
            output_reference=ref.guide,
        )
        self.assertIs(planner._mpc_problem, problem)
        padded = problem["opti"].value(problem["parameters"]["output_reference"])
        np.testing.assert_allclose(padded[:, -1], ref.positions[-1], atol=1e-14)

    def test_held_inputs_live_estimate_no_reset_and_update_before_replan(self):
        c, calls, boundaries = self.config, [], []
        ref = WaypointReference(c)
        learner = make_learner(self.system, c, c.learning_rate)
        for _ in range(c.calibration_intervals):
            learner.append_score(0.)
        da = np.array([.01, .02, .03, .04])

        def plan(planner, xd, positions, interval, k, theta, a, hover, **kwargs):
            calls.append(dict(k=k, theta=theta.copy(), a=a.copy(), xd=xd.copy()))
            inputs = np.tile(hover[:, None], (1, c.mpc_horizon_steps))
            inputs[0] += .05*np.arange(c.mpc_horizon_steps)
            return np.full((9, c.mpc_horizon_steps+1), 999.), inputs

        def control(system, x, a, xd, ud, solver):
            system.gamma = np.column_stack((xd, (x+xd)/2, x))
            return np.array([9.81, 0, 0, 0]), 0., {}

        def finish(online, system, config):
            boundaries.append(True)
            updated = len(boundaries) % c.representation_period == 0
            if updated:
                system.set_representation(system.Theta_hat+.001)
            online.clear_buffers()
            return dict(representation_updated=updated)

        with patch.object(MotionPlanner, "plan_certainty_equivalent_through_waypoints", new=plan), \
             patch("simulations.quadrotor_9d.simulation.compute_control", side_effect=control), \
             patch("simulations.quadrotor_9d.simulation.finish_interval", side_effect=finish), \
             patch.object(QUADROTOR_9D, "adaptation_craccm", return_value=(da, 0.)):
            data, events, outcome = run_simulation(self.metric, c, ref, learner, THETA_INITIAL,
                                                   3e-5*np.eye(4), True, False,
                                                   lambda x, t: np.zeros(9))
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual([row["k"] for row in calls], list(range(0, 30, 4)))
        self.assertLess(np.abs(data["x_d"]).max(), 1.)  # No reset to fake predicted states.
        for row in calls:
            k = row["k"]
            np.testing.assert_array_equal(row["xd"], data["x_d"][k])
            np.testing.assert_array_equal(row["a"], data["a"][k])
            np.testing.assert_allclose(row["theta"], THETA_INITIAL+(.001 if k >= 20 else 0))
        largest_frozen_error = 0.
        for k, t in enumerate(data["t"][:-1]):
            if k % 2:
                np.testing.assert_array_equal(data["u_d"][k], data["u_d"][k-1])
            theta, a, ud = data["theta"][k], data["a"][k], data["u_d"][k]
            def rhs(tau, x, live=True):
                parameter = a+da*(tau-t) if live else a
                return self.system.dynamics_nominal(x, ud)+self.system.Y_Theta(x, theta)@parameter
            live = solve_ivp(rhs, (t, t+c.dt), data["x_d"][k], rtol=1e-11, atol=1e-13)
            frozen = solve_ivp(lambda tau, x: rhs(tau, x, False), (t, t+c.dt),
                               data["x_d"][k], rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(data["x_d"][k+1], live.y[:, -1], atol=1e-8)
            largest_frozen_error = max(largest_frozen_error, np.max(np.abs(live.y[:, -1]-frozen.y[:, -1])))
        self.assertGreater(largest_frozen_error, 1e-6)

    def test_misaligned_update_or_input_hold_is_rejected(self):
        for key, value in (("representation_period", 3), ("mpc_input_steps", 3),
                           ("mpc_horizon_steps", 1)):
            config = SimpleNamespace(**vars(self.config))
            setattr(config, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                WaypointReference(config)


if __name__ == "__main__":
    unittest.main()
