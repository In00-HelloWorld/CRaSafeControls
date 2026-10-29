"""Physics, derivatives, live reference integration and reproducible four-case runs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from subprocess import CalledProcessError

import numpy as np
from scipy.integrate import solve_ivp

from motion_planner import MotionPlanner
from neural_ccm.neural_metric import NeuralCCMMetric
from simulations.quadrotor_9d import learning, results
from simulations.quadrotor_9d.run_example import main, parse_args
from simulations.quadrotor_9d.simulation import run_simulation
from simulations.quadrotor_9d.reference import make_planner, WaypointReference
from systems.control_affine_system import ControlAffineSystem
from systems.quadrotor_9d.quadrotor_9d import QUADROTOR_9D, THETA_INITIAL
from systems.quadrotor_9d.nonlinear_uncertainty import WindEnvironment, nonlinear_uncertainty

ASSETS = Path(__file__).resolve().parents[1]/"simulations/quadrotor_9d/assets"


class Quadrotor9DTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metric = NeuralCCMMetric(ASSETS/"tracking_zonly_metric.npz")
        cls.system = QUADROTOR_9D(cls.metric)

    def test_hover_and_symbolic_dynamics(self):
        system = self.system
        np.testing.assert_allclose(system.dynamics_nominal(np.zeros(9), [9.81, 0, 0, 0]), 0)
        planner = MotionPlanner(system, .05, np.eye(9), np.eye(4), np.eye(9))
        rng = np.random.default_rng(123)
        for _ in range(8):
            x = rng.uniform(-.25, .25, 9)
            u = np.r_[9.81+rng.uniform(-.3, .3), rng.uniform(-.1, .1, 3)]
            symbolic = np.asarray(planner._nominal_dynamics(x, u)).reshape(9)
            np.testing.assert_allclose(symbolic, system.dynamics_nominal(x, u), atol=1e-13)
        # Body rates -> Euler rates retain their coupling at nonzero attitudes.
        x = np.r_[np.zeros(6), .2, -.15, .1]
        self.assertGreater(abs(system.g(x)[6, 3]), .1)
        self.assertIs(QUADROTOR_9D.ctrl_craccm, ControlAffineSystem.ctrl_craccm)
        self.assertIs(QUADROTOR_9D.adaptation_craccm, ControlAffineSystem.adaptation_craccm)

    def test_thrust_and_uncertainty_axes(self):
        system = self.system
        x = np.array([0, 0, 0, .3, -.2, .1, .12, -.09, .2])
        phi, pitch, yaw = x[6:]
        Rx = np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
        Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        thrust = (Rz @ Ry @ Rx)[:, 2]
        np.testing.assert_allclose(system.g(x)[3:6, 0], thrust)
        np.testing.assert_allclose(system.psi(x)[3:6, 2], -9.81*thrust)
        self.assertEqual(system.psi(x)[3, 0], -.3)
        self.assertEqual(system.psi(x)[4, 1], .2)
        np.testing.assert_array_equal(system.psi(x)[6:8, 3:5], np.eye(2))

    def test_metric_derivatives_and_theta_chain_rule(self):
        rng = np.random.default_rng(17)
        x = rng.uniform(-.1, .1, 9)
        a = rng.uniform(-.1, .1, 4)
        for name in ("tracking_zonly_metric.npz", "retrained_neural_metric.npz"):
            system = QUADROTOR_9D(NeuralCCMMetric(ASSETS/name))
            self.assertGreaterEqual(np.linalg.eigvalsh(system.W_fcn(x, a)).min(), system.metric.w_lb-1e-12)
            for i in range(9):
                offset = np.eye(9)[i]*1e-5
                numerical = (system.W_fcn(x+offset, a)-system.W_fcn(x-offset, a))/2e-5
                np.testing.assert_allclose(system.dW_dxi_fcn(i, x, a), numerical, atol=1e-7)
            for i in range(4):
                offset = np.eye(4)[i]*1e-5
                numerical = (system.W_fcn(x, a+offset)-system.W_fcn(x, a-offset))/2e-5
                np.testing.assert_allclose(system.dW_dai_fcn(i, x, a), numerical, atol=1e-7)
            before = system.W_fcn(x, a).copy()
            system.set_representation(THETA_INITIAL+.01)
            self.assertGreater(np.max(np.abs(system.W_fcn(x, a)-before)), 1e-10)

    def test_representation_gradient(self):
        system = self.system
        x, a, w = np.arange(9)/20, np.arange(4)/10, np.arange(9)/30
        theta = THETA_INITIAL.copy()
        gradient = system.representation_loss_gradient(x, theta, a, w)
        for index in np.ndindex(theta.shape):
            offset = np.zeros_like(theta)
            offset[index] = 1e-6
            plus = np.linalg.norm(system.Y_Theta(x, theta+offset) @ a-w)**2
            minus = np.linalg.norm(system.Y_Theta(x, theta-offset) @ a-w)**2
            self.assertAlmostEqual(gradient[index], (plus-minus)/2e-6, places=7)

    def test_native_prediction_matches_the_9d_estimated_dynamics(self):
        config = SimpleNamespace(dt=.02, u_min=np.array([0, -2, -2, -2]), u_max=np.array([20, 2, 2, 2]))
        planner = make_planner(self.system, config)
        dynamics = planner._build_certainty_equivalent_dynamics_function()
        rk4 = planner._build_rk4_function("test_9d_prediction", dynamics)
        theta, a = THETA_INITIAL.copy(), np.array([.1, .3, .02, -.01])
        rng = np.random.default_rng(31)
        for _ in range(8):
            x, u = rng.uniform(-.2, .2, 9), np.r_[10., rng.uniform(-.1, .1, 3)]
            expected = self.system.dynamics_nominal(x, u)+self.system.Y_Theta(x, theta)@a
            np.testing.assert_allclose(np.asarray(dynamics(x, u, theta, a)).ravel(), expected, atol=1e-13)
            solution = solve_ivp(lambda t, state: self.system.dynamics_nominal(state, u)
                                +self.system.Y_Theta(state, theta)@a,
                                (0, config.dt), x, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(np.asarray(rk4(x, u, theta, a)).ravel(), solution.y[:, -1], atol=1e-8)

    def test_four_arms_clone_the_learner_and_satisfy_the_qp(self):
        config = SimpleNamespace(
            dt=.05, sim_time=1., interval_steps=10, representation_period=2,
            calibration_intervals=30, learning_rate=1e-5, acp_lr=.01, delta=.1,
            slack_weight=1e4, eta=500., rate=.5, geo_degree=2, geo_nodes=4,
            rho_divergence_threshold=1e4, initial_offset=np.r_[0, 0, .002, np.zeros(6)],
            trajectory="point", goal=[0, 0, .01], mpc_horizon_steps=20, mpc_replan_steps=10,
            mpc_input_steps=1, initial_reference=np.zeros(9), waypoints=None,
            u_min=np.array([0, -2, -2, -2]), u_max=np.array([20, 2, 2, 2]), quiet=True,
        )
        physical = np.array([.02, .02, .004, .0002, -.0001])
        plant = lambda x, t: self.system.psi(x)@physical+np.r_[np.zeros(5), .001*np.sin(t), np.zeros(3)]
        x = np.random.default_rng(82).uniform(-.1, .1, (30, 10, 9))
        nominal = np.array([[self.system.dynamics_nominal(row, [9.81, 0, 0, 0]) for row in block] for block in x])
        measured = nominal+np.array([[plant(row, (i*10+k)*config.dt) for k, row in enumerate(block)] for i, block in enumerate(x)])
        calibration = dict(x=x, nominal=nominal, xdot=measured)
        learner = learning.calibrate(self.metric, config, THETA_INITIAL, calibration, config.learning_rate)
        theta_before, scores_before = learner.Theta.copy(), list(learner.S_cal)
        histories = {}
        for cp, adaptive in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(cp=cp, adaptive=adaptive):
                data, events, outcome = run_simulation(self.metric, config, WaypointReference(config), learner, THETA_INITIAL,
                                                       3e-5*np.eye(4), adaptive, cp, plant)
                self.assertEqual(outcome["status"], "completed")
                self.assertEqual(len(events), 2)
                self.assertEqual(sum(event["representation_updated"] for event in events), 1)
                self.assertLess(data["constraint_residual"].max(), 1e-8)
                self.assertTrue(np.isfinite(data["x"]).all())
                np.testing.assert_allclose(data["qp_correction"], data["u"]-data["u_d"], atol=1e-13)
                if adaptive:
                    self.assertGreater(np.linalg.norm(data["a"][-1]-data["a"][0]), 1e-12)
                else:
                    np.testing.assert_array_equal(data["a"], 0)
                    np.testing.assert_array_equal(data["rho"], 0)
                histories[cp, adaptive] = data
        np.testing.assert_allclose(histories[True, False]["x_d"], histories[False, False]["x_d"], atol=1e-12)
        np.testing.assert_array_equal(learner.Theta, theta_before)
        self.assertEqual(list(learner.S_cal), scores_before)

    def test_invalid_configuration_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            for option in (["--dt", "0"], ["--sim-time", "1.1"], ["--mpc-horizon-steps", "0"],
                           ["--representation-period", "0"], ["--calibration-intervals", "12"]):
                with self.subTest(option=option), self.assertRaises(SystemExit):
                    parse_args(["--output", tmp, *option])
        with patch("subprocess.check_output", side_effect=CalledProcessError(128, "git")):
            self.assertEqual(results.repository_state(), dict(branch=None, commit=None))


    def test_fresh_pretraining_physical_observations_and_saved_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/"fresh"
            common = ["--sim-time","1","--dt",".05","--interval-steps","10",
                      "--trajectory","point","--goal","0","0",".01",
                      "--representation-period","2","--quiet"]
            with redirect_stdout(io.StringIO()):
                fresh = main(["--output",str(output),"--pretrain-intervals","4",
                              "--calibration-intervals","30",*common])
            source = output/"pretraining"
            metadata = json.loads((source/"summary.json").read_text())
            self.assertEqual(fresh["source_mode"],"generated")
            self.assertEqual({(arm["cp"],arm["adaptive"]) for arm in fresh["arms"].values()},
                             {(True,True),(False,True),(True,False),(False,False)})
            self.assertTrue((output/"inputs.png").is_file())
            self.assertEqual(metadata["representation_updates"],2)
            self.assertGreater(metadata["theta_change_norm"],0.)
            self.assertNotEqual(metadata["seeds"]["pretraining"],metadata["seeds"]["calibration"])
            with np.load(source/"data_calibration.npz",allow_pickle=False) as data:
                self.assertEqual(data["x"].shape,(30,10,9))
                for interval in (0,12,29):
                    environment = WindEnvironment(*data["environment"][interval])
                    for step in (0,9):
                        expected = nonlinear_uncertainty(data["x"][interval,step],
                                                         data["t"][interval,step],environment)
                        actual = data["xdot"][interval,step]-data["nominal"][interval,step]
                        np.testing.assert_allclose(actual,expected,atol=1e-14)
            with redirect_stdout(io.StringIO()):
                replay = main(["--output",str(root/"replay"),"--source",str(source),*common])
            self.assertEqual(replay["source_mode"],"reused")
            self.assertEqual(replay["changed_core_files_from_local_source"],[])
            for arm in ("proposed","no_cp","cp_no_adaptation","no_adaptation"):
                self.assertEqual(fresh["arms"][arm]["status"],"completed")
                with np.load(output/f"{arm}.npz",allow_pickle=False) as first, \
                        np.load(root/"replay"/f"{arm}.npz",allow_pickle=False) as second:
                    np.testing.assert_allclose(first["x"],second["x"],atol=1e-12)
            self.assertTrue((source/"data_pretraining.npz").is_file())
            self.assertTrue((source/"pretraining_history.npz").is_file())
            # A diverging arm must retain samples and allow other arms to run.
            with redirect_stdout(io.StringIO()):
                guarded = main(["--output",str(root/"guarded"),"--source",str(source),
                                "--rho-divergence-threshold","1e-12",*common])
            self.assertEqual(guarded["status"],"controller_failures")
            self.assertEqual(guarded["arms"]["proposed"]["failure_reason"],"rho_divergence")
            self.assertEqual(guarded["arms"]["no_adaptation"]["status"],"completed")
            self.assertEqual(guarded["common_prefix_comparison"]["sample_count"],
                             min(arm["sample_count"] for arm in guarded["arms"].values()))
            self.assertTrue((root/"guarded"/"summary.json").is_file())
            self.assertTrue((root/"guarded"/"comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
