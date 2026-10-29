# 9D CCM metric 안내

Runtime은 NumPy로 metric과 정확한 미분을 평가하고, controller는 저장소 공통 구현을 사용합니다.
`load_metric()`이 명시적인 NPZ 형식을 선택합니다. 세 metric 파일은 변경하지 않았습니다.

| 파일 (`simulations/quadrotor_9d/assets/`) | 방법 | 구현 |
| --- | --- | --- |
| `tracking_zonly_metric.npz` | 제공된 학습 완료 Neural CCM; 기본값 | `neural_metric.py` |
| `invariant_metric.npz` | W=JQJ.T, SPD Q를 LMI로 최적화 | `structured_metric.py` |
| `retrained_neural_metric.npz` | 기존 NN을 초기값으로 Adam 재학습 | `neural_metric.py` |

실행에서 `--checkpoint simulations/quadrotor_9d/assets/파일명.npz`로 선택합니다.
세 metric 모두 5개 물리 계수 b=Theta@a_hat를 입력으로 받습니다.
LMI metric은 b에 의존하지 않으므로 그 파라미터 미분은0입니다.

## 계산과 보장 범위

- Neural: `11 -> 128(tanh) -> 128(tanh) -> 81`, W=L.T@L+.03I. 양정성은 구조로 보장합니다.
- LMI: `(v,R)`의 right-invariant frame J와 Q>0를 사용합니다. 정규 Euler 영역에서
  양정성과 **전체 C2=0**이 구조적으로 성립합니다.
- Neural 학습의 C2는 원래 NeuralCCM의 **투영 C2** 잔차입니다. 정확한 등식이 아닙니다.
- C1은 두 방식 모두 유한 표본과 수치 탐색으로 검증했습니다. 영역 전체의 SOS 인증이나
  실제 외란·온라인 Θ 갱신을 포함한 전체 적응 폐루프 수축 증명은 없습니다.
- 영역·수축률·원본 SHA256·학습 설정은 각 NPZ의 metadata에 있습니다.
  Neural 영역은 속도±2, roll/pitch±.35, yaw±.4이며 LMI 영역은±.75,±.25,±.2로 좁습니다.

## 읽을 파일

| 파일 | 역할 |
| --- | --- |
| `neural_metric.py` | NN forward, 상태/물리 계수 미분, 형식 선택 |
| `structured_metric.py` | 불변 frame J, W=JQJ.T 및 미분 |
| `ccm_conditions.py` | 공통9D 모델로 C1/C2 계산 |
| `train_neural_metric.py` | NN 가중치 Adam 학습, C1/C2/보존 손실, export |
| `train_structured_metric.py` | SPD Q의 LMI 최적화 |
| `verify_metric.py` | 독립 seed 표본 및 C1 위반점 탐색 |

NN 재학습은96 epochs 중95를 선택했고 가중치 L2 변화는.358521입니다.
8,192개 무작위 점,2,048개 모서리,이전 위반점 주변128개를 학습에 사용했습니다.
checkpoint는 경계/validation C1과 C2/보존 손실로 선택했고 시뮬레이션 성능을 선택 기준으로 쓰지 않았습니다.
학습에 사용한 경계점을 독립 검증으로 부르지 않습니다.

## 재현

```bash
python -m pip install -r neural_ccm/requirements-neural-training.txt
python neural_ccm/train_neural_metric.py --output /workspace/training/neural_new
```

LMI 재학습에는 `requirements-training.txt`와 `train_structured_metric.py`를 사용합니다.
Runtime에는 Torch/CVXPY가 필요하지 않습니다.

```bash
python neural_ccm/verify_metric.py \
  --checkpoint simulations/quadrotor_9d/assets/retrained_neural_metric.npz \
  --output /workspace/training/metric_check_new
python -m unittest discover -s tests -p 'test_neural_metric_training.py' -v
python -m unittest discover -s tests -p 'test_structured_ccm.py' -v
```

기본 학습 설정과 위반점 입력은 최신 재학습 버전에 맞춰 두었습니다.
학습 결과는 `metric.npz`, `summary.json`, `history.json`, 재현용 소스/위반점으로 저장합니다.
NPZ에 신경망 가중치가 모두 들어 있으므로 별도의 Torch checkpoint와 표본 복사본은 만들지 않습니다.
`--initial`에 다른 neural NPZ를 지정하면 그 가중치에서 이어서 학습할 수 있습니다.

최근 실행 결과는 `/workspace/outputs/quadrotor_9d_waypoints/`에만 보존합니다.
