"""Native certainty-equivalent MPC, inherited CRaCCM and coupled plant/reference ODE."""

import time
import numpy as np
from scipy.integrate import solve_ivp

from geodesic_solver import GeodesicSolver
from systems.quadrotor_9d.quadrotor_9d import QUADROTOR_9D
from .learning import clone_learner, finish_interval
from .reference import make_planner


def compute_control(system, x, a, xd, ud, solver):
    """Invoke the inherited controller and measure, without altering, its output."""
    ud = np.asarray(ud).reshape(system.udim)
    u, slack = system.ctrl_craccm(x, a, xd, ud, solver,
                                use_qpsolvers=True, use_slack=True)
    u = np.asarray(u).reshape(system.udim)
    end = np.linalg.solve(system.W_fcn(x, a), system.gamma_s[:, -1])
    start = np.linalg.solve(system.W_fcn(xd, a), system.gamma_s[:, 0])
    Yx = system.Y(x) @ a if system.use_adaptive else np.zeros(system.xdim)
    Yd = system.Y(xd) @ a if system.use_adaptive else np.zeros(system.xdim)
    authority = end @ system.g(x)
    authority_norm = float(np.linalg.norm(authority))
    ratio = min(authority_norm/system.ccm_authority_threshold,1.)
    scale = ratio**3*(10.-15.*ratio+6.*ratio**2)
    if system.use_cp:
        tightening = float(scale*np.linalg.norm(end)*system.cp_quantile)
    else:
        tightening = 0.
    residual = (end @ (system.dynamics_nominal(x, u)+Yx)
                - start @ (system.dynamics_nominal(xd, ud)+Yd)
                + system.ccm_rate*system.Erem+tightening-slack)
    return u, float(slack), dict(
        energy=system.Erem, tightening=tightening,
        constraint_residual=float(residual), correction_norm=float(np.linalg.norm(u-ud)),
        qp_correction=(u-ud).copy(),
    )


def run_simulation(metric, config, reference, pretrained, theta, gamma, adaptive, cp, plant):
    """One independent arm, using planar_quad's live-estimate reference integration."""
    system = QUADROTOR_9D(metric, dict(
        Theta_init=theta, use_cp=cp, use_adaptive=adaptive, Gamma_ccm=gamma,
        weight_slack=config.slack_weight, eta_ccm=config.eta,
        ccm_rate=config.rate, dt=config.dt, true_uncertainty=plant,
    ))
    learner = clone_learner(pretrained, system)
    planner = make_planner(system, config)
    geodesic = GeodesicSolver(9, config.geo_degree, config.geo_nodes,
                              system.W_fcn, system.dW_dxi_fcn, system.dW_dai_fcn)
    times = np.arange(round(config.sim_time/config.dt)+1)*config.dt

    def rho_event(t, state):
        return config.rho_divergence_threshold-abs(float(state[13]))
    rho_event.terminal, rho_event.direction = True, -1.

    # State order: plant x(9), adaptive a(4), rho(1), desired x_d(9).
    state = np.r_[reference.initial_state+config.initial_offset, system.a_center, 0., reference.initial_state]
    ud = np.array([system.grav, 0, 0, 0])
    controls, plan_start_step, solve_seconds = None, 0, []
    rows, events, outcome = [], [], dict(status="completed")
    begin = time.perf_counter()
    for k, t in enumerate(times):
        x, a, rho, xd = state[:9], state[9:13], float(state[13]), state[14:]
        reference_a = a.copy() if adaptive else np.zeros(4)
        replanned = False
        if k < len(times)-1:
            if k % config.mpc_replan_steps == 0:
                start = time.perf_counter()
                # Predicted states are deliberately discarded; only inputs are applied.
                _, controls = planner.plan_certainty_equivalent_through_waypoints(
                    xd, reference.positions, config.interval_steps, k, system.Theta_hat, reference_a,
                    np.r_[system.grav, np.zeros(3)],
                    mpc_horizon=reference.horizon, mpc_dt=reference.mpc_dt,
                    output_reference=reference.guide, u_previous=ud,
                )
                solve_seconds.append(time.perf_counter()-start)
                plan_start_step, replanned = k, True
            index = (k-plan_start_step)//config.mpc_input_steps
            ud = controls[:, index].copy()
        u, slack, diagnostics = compute_control(system, x, a, xd, ud, geodesic)
        domain = [system.metric_in_training_domain(node, a) for node in system.gamma.T]
        rows.append(dict(
            t=t, x=x.copy(), a=a.copy(), rho=rho, x_d=xd.copy(), u=u.copy(), u_d=ud.copy(),
            b=system.Theta_hat@a, theta=system.Theta_hat.copy(), reference_a=reference_a,
            quantile=system.cp_quantile, slack=slack, mpc_replanned=replanned,
            in_domain=np.nan if any(value is None for value in domain) else float(all(domain)),
            **diagnostics,
        ))
        if k == len(times)-1:
            break
        learner.add_data_to_buffers(x, system.dynamics_nominal(x, u), system.Y(x),
                                    xdot=system.dynamics(x, u, t))

        def dynamics(tau, current):
            adaptive_state, desired = current[:14], current[14:]
            model_a = adaptive_state[9:13] if adaptive else np.zeros(4)
            return np.r_[system.dynamics_extended(adaptive_state, desired, u, geodesic, tau),
                         system.dynamics_certainty_equivalent(desired, ud, model_a)]

        solution = solve_ivp(dynamics, (t, times[k+1]), state, method="RK45",
                             rtol=1e-7, atol=1e-9, t_eval=[times[k+1]],
                             events=rho_event if adaptive else None)
        if adaptive and len(solution.t_events[0]):
            outcome = dict(status="diverged", failure_reason="rho_divergence",
                           failure_time=float(solution.t_events[0][-1]),
                           failure_state=solution.y_events[0][-1].copy())
            break
        if not solution.success:
            outcome = dict(status="failed", failure_reason="integration_failure",
                           failure_time=float(t), failure_message=solution.message)
            break
        state = solution.y[:, -1]
        if (not np.all(np.isfinite(state)) or abs(state[13]) > 1e6
                or np.linalg.norm(state[:9]) > 100):
            outcome = dict(status="diverged", failure_reason="state_divergence",
                           failure_time=float(times[k+1]), failure_state=state.copy())
            break
        if (k+1) % config.interval_steps == 0:
            # Install Theta before the next loop's MPC call at a shared boundary.
            event = finish_interval(learner, system, config)
            events.append(dict(event, time=float(times[k+1])))
        if not config.quiet and (k+1) % max(1, round(.5/config.dt)) == 0:
            print(f"t={times[k+1]:.2f}, reference error={np.linalg.norm(state[:3]-state[14:17]):.4f}",
                  flush=True)

    result = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    outcome.update(mpc_replans=len(solve_seconds), wall_seconds=time.perf_counter()-begin)
    return result, events, outcome
