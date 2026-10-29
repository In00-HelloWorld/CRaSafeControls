"""Neural CCM fine-tuning; runtime uses the repository's CRaCCM controller."""

import argparse
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_ccm.ccm_conditions import model_derivatives, projection
from neural_ccm.neural_metric import NeuralCCMMetric, WEIGHT_NAMES


class MetricNetwork(nn.Module):
    def __init__(self, initial):
        super().__init__()
        hidden = len(initial.weights[1])
        self.layers = nn.Sequential(nn.Linear(11, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 81, bias=False)).double()
        with torch.no_grad():
            for target, source in zip(self.layers.parameters(), initial.weights):
                target.copy_(torch.from_numpy(source.copy()))
        self.w_lb = initial.w_lb
        self.rate = initial.rate

    def forward(self, x, b):
        w0, b0, w2, b2, w4 = tuple(self.layers.parameters())
        h1 = torch.tanh(torch.nn.functional.linear(torch.cat((x[:, 3:], b), 1), w0, b0))
        h2 = torch.tanh(torch.nn.functional.linear(h1, w2, b2))
        L = torch.nn.functional.linear(h2, w4).reshape(-1, 9, 9)
        W = L.transpose(1, 2)@L + self.w_lb*torch.eye(9, dtype=x.dtype, device=x.device)
        dh1 = (1-h1.square())[:, :, None]*w0[None, :, :6]
        dh2 = (1-h2.square())[:, :, None]*torch.einsum("ij,bjk->bik", w2, dh1)
        dL = torch.einsum("ij,bjk->bik", w4, dh2).reshape(-1, 9, 9, 6)
        derivative = torch.einsum("bjik,bjl->bilk", dL, L)
        derivative = derivative + derivative.transpose(1, 2)
        return W, torch.cat((torch.zeros_like(derivative[..., :3]), derivative), -1)


def observations(domain, count, seed, boundary_samples=0, counterexamples=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(domain["x_min"], domain["x_max"], (count, 9))
    b = rng.uniform(domain["a_min"], domain["a_max"], (count, 5))
    if boundary_samples:
        corners = np.array(list(product((False, True), repeat=11)))
        corners = corners[rng.permutation(len(corners))[:boundary_samples]]
        xc = np.zeros((len(corners), 9))
        xc[:, 3:] = np.where(corners[:, :6], domain["x_max"][3:], domain["x_min"][3:])
        bc = np.where(corners[:, 6:], domain["a_max"], domain["a_min"])
        x, b = np.vstack((x, xc)), np.vstack((b, bc))
    if counterexamples:
        point = np.asarray(json.loads(Path(counterexamples).read_text())["adversarial_search"]["point"])
        low, high = np.r_[domain["x_min"][3:], domain["a_min"]], np.r_[domain["x_max"][3:], domain["a_max"]]
        hard = np.clip(point+rng.uniform(-.04, .04, (128, 11))*(high-low), low, high)
        hard[0] = point
        x, b = np.vstack((x, np.c_[np.zeros((128, 3)), hard[:, :6]])), np.vstack((b, hard[:, 6:]))
    rows = [model_derivatives(xx, bb) for xx, bb in zip(x, b)]
    F, A, B = (np.stack([row[i] for row in rows]) for i in range(3))
    DB = np.stack([np.stack(row[3:]) for row in rows])
    P = np.stack([projection(matrix) for matrix in B])
    return tuple(torch.from_numpy(value).double() for value in (x, b, F[..., 0], A, B, DB, P))


def matrices(model, batch, rate):
    x, b, F, A, B, DB, P = batch
    W, dW = model(x, b)
    H = -torch.einsum("bijk,bk->bij", dW, F) + A@W + W@A.transpose(1, 2) + 2*rate*W
    C1 = P.transpose(1, 2)@H@P
    C2 = torch.einsum("bijk,bkl->blij", dW, B) - DB@W[:, None] - W[:, None]@DB.transpose(2, 3)
    projected = P.transpose(1, 2)[:, None]@C2@P[:, None]
    return W, (C1+C1.transpose(1, 2))/2, projected


def losses(model, anchor, batch, rate, margin):
    W, C1, C2 = matrices(model, batch, rate)
    with torch.no_grad():
        target = anchor(batch[0], batch[1])[0]
    maximum = torch.linalg.eigvalsh(C1)[:, -1]
    violation = torch.relu(maximum+margin).square()
    return (violation.max(),
            C2.square().sum((1, 2, 3)).mean(), (W-target).square().mean()), maximum


def export(model, initial, output, training):
    arrays = {name: parameter.detach().numpy().copy()
              for name, parameter in zip(WEIGHT_NAMES, model.layers.parameters())}
    metadata = dict(initial.metadata, architecture="neural", training=training,
                    initial_metric_sha256=hashlib.sha256(initial.checkpoint_path.read_bytes()).hexdigest(),
                    source_checkpoint="neural_fine_tuning/weights",
                    source_sha256=hashlib.sha256(b"".join(a.tobytes() for a in arrays.values())).hexdigest())
    np.savez_compressed(output/"metric.npz", metadata=json.dumps(metadata), **arrays)
    return metadata


def batches(data, size, shuffle=False):
    indices = torch.randperm(len(data[0])) if shuffle else torch.arange(len(data[0]))
    for index in indices.split(size):
        if len(index):
            yield tuple(value[index] for value in data)


@torch.no_grad()
def validate(model, anchor, data, hard, args):
    totals, eigenvalues, boundary = [], [], []
    for batch in batches(data, args.batch_size):
        terms, maximum = losses(model, anchor, batch, model.rate, args.margin)
        totals.append(np.array([v.item() for v in terms])*len(batch[0]))
        eigenvalues.extend(maximum.tolist())
    for batch in batches(hard, args.batch_size):
        boundary.extend(torch.linalg.eigvalsh(matrices(model, batch, model.rate)[1])[:, -1].tolist())
    c1, c2, preservation = np.sum(totals, axis=0)/len(data[0])
    fraction = float(np.mean(np.array(eigenvalues)<0))
    boundary_fraction = float(np.mean(np.array(boundary)<0)) if boundary else None
    worst, boundary_worst = float(max(eigenvalues)), float(max(boundary)) if boundary else None
    record = dict(C1_pass_fraction=fraction, worst_C1_eigenvalue=worst, C1_hinge=float(c1),
                  C2_mean_squared=float(c2), anchor_mean_squared=float(preservation),
                  worst_boundary_eigenvalue=boundary_worst, boundary_pass_fraction=boundary_fraction)
    score = (1-min(fraction, boundary_fraction if boundary else 1.),
             max(0., max(worst, boundary_worst if boundary else -np.inf)+args.margin),
             float(c2*args.c2_weight+preservation*args.anchor_weight))
    return record, score


DEFAULTS = dict(epochs=96, train_samples=8192, validation_samples=2048, boundary_samples=2048,
                batch_size=128, seed=20261031, learning_rate=3e-5, c1_weight=1000.,
                c2_weight=100., anchor_weight=10., margin=.02)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path,
                        default=ROOT/"simulations/quadrotor_9d/assets/tracking_zonly_metric.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterexamples", type=Path,
                        default=Path(__file__).with_name("quadrotor_counterexample.json"))
    for name, default in DEFAULTS.items():
        parser.add_argument("--"+name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)
    gains = [args.learning_rate, args.c1_weight, args.c2_weight, args.anchor_weight, args.margin]
    if (args.output.exists() or args.seed < 0 or not 0 <= args.boundary_samples <= 2048
            or (args.counterexamples and not args.counterexamples.is_file()) or min(args.epochs, args.train_samples,
            args.validation_samples, args.batch_size) < 1 or not np.isfinite(gains).all()
            or min(gains) <= 0):
        parser.error("Choose a new output folder and valid positive training settings")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    initial = NeuralCCMMetric(args.initial)
    args.output.mkdir(parents=True)
    (args.output/"train_source.py").write_bytes(Path(__file__).read_bytes())
    model = MetricNetwork(initial)
    anchor = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train = observations(initial.domain, args.train_samples, args.seed,
                         args.boundary_samples, args.counterexamples)
    validation = observations(initial.domain, args.validation_samples, args.seed+1)
    if args.counterexamples:
        (args.output/"counterexample_source.json").write_bytes(args.counterexamples.read_bytes())
    hard = tuple(value[args.train_samples:] for value in train)
    weights = torch.tensor([args.c1_weight, args.c2_weight, args.anchor_weight], dtype=torch.float64)
    history, best, best_state = [], None, None
    for epoch in range(args.epochs+1):
        if epoch:
            for batch in batches(train, args.batch_size, shuffle=True):
                terms, _ = losses(model, anchor, batch, initial.rate, args.margin)
                objective = (torch.stack(terms)*weights).sum()
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
        record, score = validate(model, anchor, validation, hard, args)
        record["epoch"] = epoch
        if best is None or score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
            record["selected"] = True
        history.append(record)
        (args.output/"history.json").write_text(json.dumps(history, indent=2)+"\n")
        print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    delta = np.sqrt(sum(float((a-b).square().sum().detach()) for a, b in zip(model.parameters(), anchor.parameters())))
    if delta <= 1e-12:
        raise RuntimeError("No changed neural weights met selection criteria; no new metric exported")
    training = dict(method="Neural CCM: Adam fine-tuning of tanh network weights",
                    configuration={key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
                    selected_epoch=next(row["epoch"] for row in reversed(history) if row.get("selected")),
                    parameter_change_l2=delta,
                    training_script_sha256=hashlib.sha256((args.output/"train_source.py").read_bytes()).hexdigest(),
                    actual_train_samples=len(train[0]), trainable_parameters=sum(p.numel() for p in model.parameters()),
                    C2_definition="Projected C2 as in the supplied NeuralCCM training code; approximate penalty",
                    selection="Boundary and validation C1 feasibility, then weighted C2/anchor loss; no simulation test selection")
    metadata = export(model, initial, args.output, training)
    (args.output/"summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(training), flush=True)
    return metadata


if __name__ == "__main__":
    main()
