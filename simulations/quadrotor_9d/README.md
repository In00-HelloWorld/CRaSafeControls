# 9D quadrotor: 코드 읽기와 실행

기준 branch는 `adaptive_reference_trajectory`입니다. 공통 controller, MPC,
GeodesicSolver, OLACP는 branch의 구현을 그대로 사용합니다.

## 읽는 순서

| 순서 | 파일 | 확인할 내용 |
| --- | --- | --- |
| 1 | [run_example.py](run_example.py) | 설정, metric 선택, Theta 사전학습/보정, 네 조건 실행 |
| 2 | [reference.py](reference.py) | 위치 waypoint, 별도 초기 기준 상태, natural spline, 위치 비용 |
| 3 | [simulation.py](simulation.py) | MPC 재계획, 입력 유지, plant·적응·기준 상태 결합 적분 |
| 4 | [learning.py](learning.py) | 관측 수집, Theta 학습, CP 보정, 온라인 Θ 설치 |
| 5 | [quadrotor_9d.py](../../systems/quadrotor_9d/quadrotor_9d.py) | f, g, Psi, Y_Theta, W와 미분 연결 |
| 별도 | [results.py](results.py) | 평가 수치, 파일 저장, 그래프 |

실제 제어식은 [control_affine_system.py](../../systems/control_affine_system.py)의
`ctrl_craccm`/`adaptation_craccm`, 예측·비용·제약은
[motion_planner.py](../../motion_planner.py)의 `_tracking_cost`,
`_build_certainty_equivalent_mpc_problem`, `plan_certainty_equivalent_through_waypoints`에 있습니다.
실제 외란은 [nonlinear_uncertainty.py](../../systems/quadrotor_9d/nonlinear_uncertainty.py)에서
plant에만 더하며, MPC/controller가 실제 외란 함수를 직접 사용하지 않습니다.

## 기준 생성: 문단과 코드의 대응

경로는 `position waypoints -> natural spline cost guide -> CE-MPC inputs -> CE reference ODE`입니다.
Spline은 비용의 위치 guide이며, 실제 기준 상태 x_d는 ODE의 적분 결과입니다.

| 문단의 조건 | 9D 구현 |
| --- | --- |
| p_i^wp, T_i=i|I| | `reference.DEFAULT_WAYPOINTS` 또는 `--waypoints` JSON, `WaypointReference.times` |
| natural cubic spline | `WaypointReference.spline`, `bc_type="natural"` |
| 위치 선택 C_p | `make_planner`: `[I_3,0]`, 3×9 |
| 별도 완전 초기 상태 x_d,0 | `--initial-reference`의 9개 값; 첫 waypoint 위치와 일치 검사 |
| 중간 속도/자세를 지정하지 않음 | waypoint와 guide는 위치 3개만 포함 |
| running/terminal 위치 비용, hover 입력 비용 | Q=8I_3, P=60I_3, R=.03I_4, 이산시간 가중치 |
| S=0 | `make_planner`에서 명시적으로 4×4 영행렬 |
| z_0=x_d(t_l) | `simulation`에서 현재 적분된 x_d를 planner에 전달 |
| frozen Θ/a_hat RK4 예측 | 공통 `MotionPlanner`의 parameterized RK4 |
| 별도 waypoint penalty/equality 없음 | 공통 MPC의 비용과 제약을 그대로 사용 |
| 최적화된 입력 순차 유지 | `index=(k-plan_start_step)//mpc_input_steps` |
| live a_hat, 현재 Θ로 x_d 적분 | `simulation.dynamics`의 결합 RK45 |
| 예측 상태로 x_d를 reset하지 않음 | planner의 첫 반환값을 버리고 입력만 사용 |
| Θ 갱신 후 공유 경계의 재계획 | `finish_interval` 실행 후 다음 루프에서 MPC |
| B|I|/Delta t_r 정수 | `validate_timing`으로 강제 |
| 종료 뒤 마지막 waypoint 유지 | `WaypointReference.guide`에서 시간 clip |

문단의 planar는 위치2·상태6·입력2이고, 여기서는 **위치3·상태9·입력4**입니다.
상태는 `[p_x,p_y,p_z,v_x,v_y,v_z,roll,pitch,yaw]`, 입력은
`[thrust_acceleration,omega_x,omega_y,omega_z]`입니다.
Hover 입력은 `[9.81,0,0,0]`, MPC 입력 범위는 `[0,-2,-2,-2]`~`[20,2,2,2]`입니다.
이 범위는 MPC 기준 입력 u_d의 제약이며, 최종 CRaCCM 입력 u의 포화 제약은 아닙니다.

## 시간 설정과 바꿀 위치

| 설정 | 기본값 | 수정 위치 |
| --- | --- | --- |
| 제어/적분 간격 dt | .02 s | `--dt` |
| 공통 waypoint/학습 구간 | 2 s | `--interval-steps` × dt; 기본100 |
| MPC 예측 간격 Delta t_p | .02 s | `--mpc-input-steps` × dt; 기본1 |
| 예측 스텝 N_p / horizon | 40 / .8 s | `--mpc-horizon-steps` |
| 재계획 간격 Delta t_r | .4 s | `--mpc-replan-steps` × dt; 기본20 |
| 재계획 사이 사용하는 입력 수 | 20개 | Delta t_r / Delta t_p |
| Θ 갱신 주기 | B=4 구간=8 s | `--representation-period` |
| 초기 기준 상태 | 9개 모두0 | `--initial-reference` |
| 기본 위치 경로 | 7개의3D 위치, 12초 | `reference.DEFAULT_WAYPOINTS` |
| 비용 Q/R/P/S | 위치/hover 비용 | `reference.make_planner` |
| Γ·slack·적응 관련 설정 | CLI 및 `set_defaults` | `run_example.parse_args` |

주신 문단과 동일한 시간 수치를 쓰려면 아래 옵션을 추가하면 됩니다.
9D 모델과 입력 범위는 위 설명을 따릅니다.

```bash
--dt .01 --interval-steps 200 --mpc-input-steps 5 \
--mpc-horizon-steps 80 --mpc-replan-steps 20
```

## 실행과 확인

저장소 root에서 실행합니다. 출력 폴더는 새 폴더 또는 빈 폴더여야 합니다.

```bash
source /workspace/.venvs/crasafecontrols/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/crasafecontrols/matplotlib
python simulations/quadrotor_9d/run_example.py --output /workspace/outputs/quad9d_new
```

처음 실행하면 Θ 사전학습·독립 CP 보정부터 만듭니다. 같은 관측을 재사용하려면
`--source /workspace/outputs/quad9d_new/pretraining`을 추가합니다.
재사용 관측과 `--dt`, `--interval-steps`가 일치해야 합니다.

직접 위치를 지정하려면 K+1개의 `[x,y,z]`를 담은 JSON 파일을 `--waypoints`로 전달합니다.
K=sim_time/(interval_steps*dt)이고, 기본값은7개 waypoint입니다.
`--initial-reference`에는 위치뿐 아니라 초기 속도·자세까지9개 값을 지정할 수 있습니다.
첫 위치는 첫 waypoint와 같아야 합니다. 실제 plant 초기 상태는 이 값에 `initial_offset`을 더합니다.

| 확인 파일 | 볼 내용 |
| --- | --- |
| `reference.png` | 공통 spline guide·waypoint와 조건별 적분된 x_d가 서로 다른지 |
| `comparison.png` | 실제 위치와 guide, 실제 x와 자체 x_d 사이의 추적 오차 |
| `inputs.png` | 네 조건의 원시 실제 입력 |
| 각 조건 `.npz` | t, x, x_d, u, u_d, a, rho, theta, quantile, slack, mpc_replanned |
| `summary.json`의 `reference` | 위치 waypoint, 시간, 초기 상태, MPC 시간 설정 |
| `intervals.json` | Θ 갱신 시점과 적용 여부 |

NPZ의 `target`은 3D 위치 비용 guide입니다. `x_d`가 실제 기준 상태입니다.
네 조건은 `(CP, continuous adaptation)=(Yes,Yes),(No,Yes),(Yes,No),(No,No)`입니다.
No adaptation은 a_hat/rho 및 기준 모델의 해당 항을 끄며, 온라인 Θ/OLACP 설정은 공통입니다.

## Metric과 검증

세 metric은 [metric 안내](../../neural_ccm/README.md)에 정리했습니다.
**CCM metric의 offline 학습**, `learning.py`의 **Θ 학습**, **CP 보정**은 별개입니다.

```bash
python -m unittest discover -s tests -p 'test_quadrotor_9d.py' -v
python -m unittest discover -s tests -p 'test_quadrotor_reference.py' -v
```

두 번째 검증은 natural spline·위치 비용·고정 파라미터 RK4·입력 유지·live a_hat 적분·
예측 상태 reset 방지·Θ 갱신 순서·종료 후 guide 유지 조건을 확인합니다.

최근 결과는 `/workspace/outputs/quadrotor_9d_waypoints/`에 있습니다.
과거 실험 결과와 이전 소스 ZIP은 삭제했습니다. 코드 변경 전 백업은 `/workspace/backups/`에 있습니다.
