"""Independent physical disturbance for representation-learning experiments.

The plant never evaluates Psi, Theta or an estimated a. Velocities are inertial;
quadratic drag is evaluated in body coordinates relative to an ambient wind.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WindEnvironment:
    wind_x: float = .08
    wind_y: float = -.06
    wind_z: float = .02
    payload: float = .11
    phase: float = .4
    residual_scale: float = 1.0


def rotation_matrix(x):
    phi, pitch, yaw = np.asarray(x, dtype=float)[6:9]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
                     [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
                     [-st, ct*sp, ct*cp]])


def nonlinear_uncertainty(x, t, environment=WindEnvironment()):
    """Return a 9-vector of acceleration/rate disturbances, never a sensor error.

    beta contains drag coefficients divided by nominal mass, in 1/m.
    Payload retains the existing additive hover approximation. A small smooth
    gust and state/time-varying rate biases provide unmodelled residuals.
    """
    x = np.asarray(x, dtype=float).reshape(9)
    if environment.residual_scale < 0:
        raise ValueError("residual_scale must be nonnegative")
    R = rotation_matrix(x)
    wind = np.array([environment.wind_x, environment.wind_y, environment.wind_z])
    relative_velocity = R.T @ (x[3:6]-wind)
    beta = np.array([.09, .23, .07])
    drag_body = -beta*relative_velocity*np.abs(relative_velocity)
    phase = environment.phase
    gust = environment.residual_scale * np.array([
        .008*np.sin(.71*t+phase), .008*np.cos(.53*t+.3+phase),
        .006*np.sin(.37*t+.2+phase),
    ])
    result = np.zeros(9)
    result[3:6] = R @ drag_body - 9.81*environment.payload*R[:,2] + gust
    result[6] = (.04*environment.wind_x + environment.residual_scale *
                 (.003*np.sin(.43*t+phase) + .002*np.sin(x[7])))
    result[7] = (-.04*environment.wind_y + environment.residual_scale *
                 (.003*np.cos(.39*t+phase) + .002*np.sin(x[6])))
    return result
