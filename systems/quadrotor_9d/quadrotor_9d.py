"""9D Euler quadrotor adapter for the repository's existing CRaCCM controller.

Dynamics and features follow the supplied NeuralCCM payload-adaptive system:
x=[position(3), inertial velocity(3), roll, pitch, yaw];
u=[thrust acceleration, body rates(3)]. This is an attitude-rate model, not
a 12-state rigid-body model. Physical plant uncertainty is supplied separately.
"""

import numpy as np

from systems.control_affine_system import ControlAffineSystem


THETA_INITIAL = np.array([
    [.22, 0, 0, 0], [.22, 0, 0, 0], [0, .22, 0, 0],
    [0, 0, .04, 0], [0, 0, 0, .04],
])


class QUADROTOR_9D(ControlAffineSystem):
    xdim, udim, adim = 9, 4, 4
    theta_shape = (5, 4)

    def __init__(self, metric, params=None):
        params = dict(params or {})
        self.metric = metric
        self.grav = float(params.get("g", 9.81))
        if self.grav != 9.81:
            raise ValueError("The CCM metrics use g=9.81")
        self.set_representation(params.get("Theta_init", THETA_INITIAL))
        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is not None and not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x,t)")
        params.setdefault("a_lb", -np.ones(self.adim))
        params.setdefault("a_ub", np.ones(self.adim))
        radius = 0.5*np.linalg.norm(np.asarray(params["a_ub"])-params["a_lb"])
        params.setdefault("epsilon", .01)
        params.setdefault("a_hat_norm_max", radius+params["epsilon"])
        params.setdefault("ccm_rate", metric.rate)
        super().__init__(params)

    def f(self, x):
        return np.array([x[3], x[4], x[5], 0., 0., -self.grav, 0., 0., 0.])

    @staticmethod
    def _thrust_direction(x):
        phi, pitch, yaw = x[6:9]
        if type(x[0]).__module__.startswith("casadi"):
            import casadi as math
        else:
            math = np
        sp, cp = math.sin(phi), math.cos(phi)
        st, ct = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        return cy*st*cp+sy*sp, sy*st*cp-cy*sp, ct*cp

    def g(self, x):
        phi, pitch = x[6:8]
        symbolic = type(x[0]).__module__.startswith("casadi")
        if not symbolic and abs(np.cos(pitch)) < 1e-5:
            raise ValueError("Euler pitch singularity")
        if symbolic:
            import casadi as math
        else:
            math = np
        sp, cp = math.sin(phi), math.cos(phi)
        tangent, secant = math.tan(pitch), 1/math.cos(pitch)
        bx, by, bz = self._thrust_direction(x)
        rows = (
            [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
            [bx, 0., 0., 0.], [by, 0., 0., 0.], [bz, 0., 0., 0.],
            [0., 1., sp*tangent, cp*tangent],
            [0., 0., cp, -sp], [0., 0., sp*secant, cp*secant],
        )
        # Scalar matrix entries match the branch's symbolic MotionPlanner interface.
        if symbolic:
            output = np.empty((self.xdim,self.udim), dtype=object)
            for i, row in enumerate(rows):
                for j,value in enumerate(row):
                    output[i,j] = value
            return output
        return np.vstack(rows)

    def psi(self, x):
        symbolic = type(x[0]).__module__.startswith("casadi")
        x = np.asarray(x, dtype=object if symbolic else float).reshape(self.xdim)
        bx, by, bz = self._thrust_direction(x)
        result = np.zeros((self.xdim, 5),dtype=object if symbolic else float)
        result[3, 0], result[4, 1] = -x[3], -x[4]
        for i,value in enumerate((bx,by,bz)):
            result[3+i,2] = -self.grav*value
        result[6, 3], result[7, 4] = 1., 1.
        return result

    def Y(self, x):
        return self.Y_Theta(x, self.Theta_hat)

    def Y_Theta(self, x, theta):
        theta = np.asarray(theta)
        if theta.shape != self.theta_shape:
            raise ValueError("Theta must be finite with shape (5,4)")
        if theta.dtype != object or all(np.isscalar(value) for value in theta.flat):
            theta = np.asarray(theta,dtype=float)
            if not np.all(np.isfinite(theta)):
                raise ValueError("Theta must be finite with shape (5,4)")
        result = self.psi(x) @ theta
        return result if result.dtype == object else self._validate_Y_shape(result)

    def dynamics_certainty_equivalent(self,x,u,a_hat):
        return self.dynamics_nominal(x,u)+self.Y(x) @ np.asarray(a_hat).reshape(self.adim)

    def representation_loss_gradient(self, x, theta, a, w):
        a = np.asarray(a, dtype=float).reshape(self.adim)
        w = np.asarray(w, dtype=float).reshape(self.xdim)
        error = self.Y_Theta(x, theta) @ a - w
        return 2*np.outer(self.psi(x).T @ error, a)

    def set_representation(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta_shape or not np.all(np.isfinite(theta)):
            raise ValueError("Theta must be finite with shape (5,4)")
        self.Theta_hat = theta.copy()

    def true_uncertainty(self, x, t=0.):
        if self.true_uncertainty_fcn is None:
            return np.zeros(self.xdim)
        value = np.asarray(self.true_uncertainty_fcn(x, t), dtype=float)
        if value.shape != (self.xdim,) or not np.all(np.isfinite(value)):
            raise ValueError("true_uncertainty must return a finite vector of shape (9,)")
        return value

    def dynamics(self, x, u, t=0.):
        return self.dynamics_nominal(x, u)+self.true_uncertainty(x, t)

    def ctrl_nominal(self, x):
        return np.array([self.grav, 0., 0., 0.])

    def dynamics_extended(self, state, x_d, u, geodesic_solver, t=0.):
        state = np.asarray(state, dtype=float).reshape(self.xdim+self.adim+1)
        x, a, rho = state[:9], state[9:13], state[-1]
        if self.use_adaptive:
            da, drho = self.adaptation_craccm(x, x_d, a, rho, geodesic_solver)
        else:
            da, drho = np.zeros(self.adim), 0.
        return np.r_[self.dynamics(x, u, t), np.asarray(da).reshape(-1), drho]

    def _metric_parameter(self, a):
        a = np.asarray(a, dtype=float).reshape(self.adim)
        return self.Theta_hat @ a

    def W_fcn(self, x, a):
        return self.metric.W(x, self._metric_parameter(a))

    def dW_dxi_fcn(self, i, x, a):
        if not 0 <= i < self.xdim:
            raise IndexError(i)
        return self.metric.derivatives(x, self._metric_parameter(a))[0][:, :, i]

    def dW_dai_fcn(self, i, x, a):
        if not 0 <= i < self.adim:
            raise IndexError(i)
        derivative = self.metric.derivatives(x, self._metric_parameter(a))[1]
        return derivative @ self.Theta_hat[:, i]

    def metric_in_training_domain(self, x, a):
        return self.metric.in_training_domain(x, self._metric_parameter(a))
