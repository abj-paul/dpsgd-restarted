# WRN-16-1 DP-SGD gradient-sketch data collection

## Status and canonical location

This document describes the completed ten-run CIFAR-10 experiment for studying
adaptive clipping predictors. All ten runs completed successfully, with 510
DP-SGD updates and a full 50,000-example gradient sweep after every update.

The canonical dataset is stored in the private Chameleon/Ceph object store at:

```text
rclone_s3:dpsgd-as-a-optimization-problem/current-work/adaptive-clipping-wrn/collection_10/restart_20260922
```

It contains `run_00` through `run_09`. This is the only prefix that should be
treated as the completed ten-run dataset.

An older, interrupted attempt remains at:

```text
rclone_s3:dpsgd-as-a-optimization-problem/current-work/adaptive-clipping-wrn/collection_10/run_00
```

That older attempt ended after step 406. It is retained for provenance but is
not part of the canonical dataset and should not be mixed with it.

At completion, the canonical prefix contained:

- 10 complete runs;
- 5,100 post-update checkpoints;
- 255,000,000 per-example gradient sketches;
- 2.008 TiB (2,207,353,188,469 bytes) before the small `results/` bundles were
  added;
- 20,440 original objects, plus four compact result files per run.

The data-generation code was at Git commit
`cc30f6e7f13b447fffb057e02054d2db18a3da74`. The main implementation files are
[`model.py`](model.py), [`dp_train.py`](dp_train.py),
[`gradient_collection.py`](gradient_collection.py), and
[`collect_ten_runs.py`](collect_ten_runs.py).

## Purpose

The dataset records how all CIFAR-10 per-example gradients evolve along ten
independent DP-SGD trajectories. A full WRN-16-1 gradient has 176,602 scalar
parameters, so storing every gradient in FP32 would be prohibitively large.
Instead, each gradient is represented by:

1. a fixed 4,275-dimensional block-wise Gaussian random projection in BF16;
2. six exact, unprojected block L2 norms in FP32; and
3. the exact model checkpoint at which those gradients were evaluated.

The intended downstream use is training and evaluating predictors for adaptive
gradient clipping while retaining useful information about gradient direction,
relative geometry, and block scale.

## Experiment definition

### Dataset and preprocessing

The training and test data are CIFAR-10. The archive used for collection had
SHA-256:

```text
6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce
```

Training updates used:

- random 32 x 32 crop after four-pixel reflect padding;
- random horizontal flip;
- conversion to a float tensor; and
- channel normalization with mean `(0.4914, 0.4822, 0.4465)` and standard
  deviation `(0.2470, 0.2435, 0.2616)`.

The gradient-collection sweep deliberately did **not** use random augmentation.
Every saved gradient was evaluated on the canonical CIFAR-10 image after only
tensor conversion and the normalization above. This makes row `i` comparable
across all steps and runs.

The data files do not duplicate CIFAR-10 images or labels. Row `i` always means
original CIFAR-10 training-set index `i`, for `0 <= i < 50000`.

### Model

The target is a DP-friendly WideResNet-16-1 with 176,602 trainable parameters.
It uses scaled weight-standardized convolutions and GroupNorm rather than
BatchNorm, avoiding batch-dependent normalization statistics. The six parameter
groups used by the collector are:

| Block | Parameters | Projection dimensions | Feature slice |
|---|---:|---:|---:|
| `stem` | 464 | 131 | `[0:131]` |
| `block1` | 9,792 | 601 | `[131:732]` |
| `block2` | 33,344 | 1,110 | `[732:1842]` |
| `block3` | 132,224 | 2,209 | `[1842:4051]` |
| `final_norm` | 128 | 69 | `[4051:4120]` |
| `classifier` | 650 | 155 | `[4120:4275]` |
| **Total** | **176,602** | **4,275** | `[0:4275]` |

### DP-SGD hyperparameters

| Setting | Value |
|---|---:|
| Updates | 510 |
| Sampling | Poisson / uniform with replacement |
| Sample rate `q` | exactly `1 / 3` |
| Expected logical batch | 16,666.67 examples |
| Configured logical batch | 16,667 examples |
| Maximum physical batch | 2,048 examples |
| Optimizer | SGD |
| Learning rate | 4.0, constant |
| Momentum | 0.0 |
| Weight decay | 0.0 |
| Flat clipping norm `C` | 1.0 |
| Gaussian noise multiplier `sigma` | 4.888246618 |
| Privacy delta | `1e-5` |
| Per-example clipping implementation | Opacus ghost clipping |
| EMA decay | 0.9999, with warm-start decay |
| Test evaluation | every 50 steps and at step 510 |
| Model dropout | 0.0 |
| GroupNorm groups | 16 |

The actual Poisson batch size varies. Across all 5,100 updates, its mean was
16,667.003 examples with a sample standard deviation of 105.744 and a range of
16,236 to 17,022.

The final per-run privacy estimates are:

- PRV accountant: epsilon = 7.433064708180238;
- RDP accountant: epsilon = 7.9999999328736715;
- delta = `1e-5`.

The noise multiplier was selected to make the RDP estimate approximately 8.0.
The two accountants use different bounds, hence the smaller PRV value.

Opacus was run with `secure_mode=False`. This is appropriate for this research
collection but means the noise PRNG was not the cryptographically secure PRNG
recommended for a production privacy release.

### Ten trajectories

The training seeds are exactly 0 through 9:

| Object-store run | Training seed |
|---|---:|
| `run_00` | 0 |
| `run_01` | 1 |
| `run_02` | 2 |
| `run_03` | 3 |
| `run_04` | 4 |
| `run_05` | 5 |
| `run_06` | 6 |
| `run_07` | 7 |
| `run_08` | 8 |
| `run_09` | 9 |

Changing the training seed changes model initialization, Poisson sampling,
augmentation, DP noise, and the other Python, NumPy, CPU Torch, and CUDA Torch
RNG streams. Architecture, dataset, optimizer, privacy settings, preprocessing,
projection matrices, and evaluation schedule are identical.

## Gradient definition and checkpoint alignment

At training update `t`:

1. DP-SGD consumes a Poisson-sampled, randomly augmented logical batch.
2. Per-example gradients are clipped, aggregated, noised, and applied, producing
   post-update parameters `theta_t`.
3. The EMA is updated and, when scheduled, evaluated on the test set.
4. `step_T/checkpoint.pt` saves `theta_t` and the associated optimizer,
   accountant, RNG, EMA, and metric state.
5. The collector evaluates all 50,000 canonical, unaugmented CIFAR-10 training
   examples at `theta_t` and writes their sketches and block norms.

Therefore, row `i` in `step_T/sketch.bf16` is a projection of

```text
gradient_theta cross_entropy(model(theta_T, normalized_image_i), label_i)
```

It is an **unclipped** gradient. It is not the clipped/noised contribution used
by the DP-SGD update that produced `theta_T`.

The `initial/initial.pt` object contains `theta_0` and initial RNG/optimizer
state. No gradient sketch was collected for `theta_0`.

The collector evaluates the shadow model in evaluation mode. There is no model
dropout and GroupNorm has no running batch statistics, so the collected gradient
definition is stable across collection batch sizes.

## Projection construction

For each parameter block `l`, let the flattened per-example gradient be a row
vector `g_l` of dimension `d_l`. The serialized projection tensor has shape
`[d_l, k_l]` and entries sampled independently as:

```text
R_l[a, b] ~ Normal(0, 1 / k_l)
```

The saved block feature is `z_l = g_l @ R_l`, and the final feature is:

```text
z = concat(z_stem, z_block1, z_block2, z_block3,
           z_final_norm, z_classifier)
```

Projection seed 4275 was used once. Exactly the same matrices and therefore the
same projected-coordinate semantics are used across every step and every run.
All ten copies have SHA-256:

```text
158ffc0b292a6a8ce8106eed4b23ab96e25fe2a9ce4b2b329fe1de3909a2b9a5
```

Projection multiplication is performed in FP32, then the concatenated result is
cast to BF16 for storage. The exact block norms are calculated from the original
FP32 gradients before projection and remain FP32 on disk.

The 4,275 total dimensions were motivated by a 15% Johnson-Lindenstrauss target
for 50,000 points, then heuristically divided among model blocks. The ordinary
global JL bound applies to one global random map. It does **not** prove 15%
distortion for every block of this split map or uniformly over all 255 million
gradients. Treat 15% as a design target to validate empirically, not a guarantee
of this stored representation.

## Object-store layout

The canonical prefix has this structure:

```text
restart_20260922/
  run_00/
    initial/
      initial.pt
      manifest.json
    projection_spec/
      projection.pt
      manifest.json
    step_000001/
      checkpoint.pt
      sketch.bf16
      norms.f32
      manifest.json
    ...
    step_000510/
      checkpoint.pt
      sketch.bf16
      norms.f32
      manifest.json
    results/
      checkpoint.pt
      config.json
      metrics.json
      summary.json
  ...
  run_09/
    ...
```

There are 510 `step_XXXXXX` directories per run. Step numbers are zero-padded to
six digits and begin at 1.

### Per-step files

| File | Contents | Typical size |
|---|---|---:|
| `sketch.bf16` | Raw row-major `[50000, 4275]` BF16 tensor | 427,500,000 bytes |
| `norms.f32` | Raw row-major `[50000, 6]` little-endian FP32 tensor | 1,200,000 bytes |
| `checkpoint.pt` | Post-update model and reproducibility state | about 1.48 MB |
| `manifest.json` | Shapes, semantics, hashes, and sweep time | about 1.1 KB |

The raw files have no NumPy or Torch container header. Their shape and dtype
come from `manifest.json` and the fixed schema above.

Each step checkpoint is a Python dictionary with keys:

```text
step, model, ema_model, optimizer, accountant, python_rng, numpy_rng,
torch_rng, cuda_rng, metrics
```

`model` and `ema_model` are unwrapped `WideResNet.state_dict()` mappings, so
their parameter names do not have an Opacus `_module.` prefix.

### Initialization files

`initial/initial.pt` contains the pre-update model, EMA model, optimizer, all
recorded RNG states, and the full argument mapping. Its manifest records the
dataset checksum, package versions, CUDA version, GPU name, and source commit.

### Projection files

`projection_spec/projection.pt` contains:

```python
{
    "seed": 4275,
    "dimensions": (131, 601, 1110, 2209, 69, 155),
    "matrices": {
        "stem": Tensor[464, 131],
        "block1": Tensor[9792, 601],
        "block2": Tensor[33344, 1110],
        "block3": Tensor[132224, 2209],
        "final_norm": Tensor[128, 69],
        "classifier": Tensor[650, 155],
    },
}
```

All projection tensors are FP32.

### Compact results

The `results/` directory was added after collection so results can be inspected
without downloading 510 checkpoints:

- `config.json`: complete run arguments;
- `metrics.json`: one metrics record per update, including scheduled test
  evaluations and collection durations;
- `summary.json`: final privacy, runtime, and accuracy summary;
- `checkpoint.pt`: final model, EMA model, optimizer, and summary convenience
  checkpoint.

This final convenience checkpoint has a different dictionary layout from the
per-step checkpoint. Use `step_000510/checkpoint.pt` when RNG and accountant
state are needed.

## Retrieving data

The examples below assume the configured private rclone remote is named
`rclone_s3`. Credentials are intentionally not stored in this repository.

```bash
DATA_REMOTE='rclone_s3:dpsgd-as-a-optimization-problem/current-work/adaptive-clipping-wrn/collection_10/restart_20260922'
```

List runs or the contents of one run:

```bash
rclone lsf "$DATA_REMOTE" --dirs-only
rclone lsf "$DATA_REMOTE/run_00" --dirs-only
```

Download one complete step:

```bash
mkdir -p downloaded/run_00/step_000100
rclone copy \
  "$DATA_REMOTE/run_00/step_000100" \
  downloaded/run_00/step_000100
```

Download only compact metrics and the projection specification:

```bash
rclone copy "$DATA_REMOTE/run_00/results" downloaded/run_00/results
rclone copy \
  "$DATA_REMOTE/run_00/projection_spec" \
  downloaded/run_00/projection_spec
```

Avoid copying an entire run accidentally: one run is approximately 220.7 GB
including its projection and checkpoints. Direct `rclone` access is preferred
to the FUSE mount because a stale or disconnected mount can surface as
`Transport endpoint is not connected`.

## Loading a shard

### PyTorch and NumPy

This loader maps the files without initially reading the whole sketch into RAM:

```python
from pathlib import Path

import json
import numpy as np
import torch


def load_step(step_dir: str | Path):
    step_dir = Path(step_dir)
    manifest = json.loads((step_dir / "manifest.json").read_text())
    n = manifest["examples"]
    k = manifest["feature_dim"]
    blocks = tuple(manifest["block_order"])

    sketch = torch.from_file(
        str(step_dir / "sketch.bf16"),
        shared=False,
        size=n * k,
        dtype=torch.bfloat16,
    ).reshape(n, k)

    norms = np.memmap(
        step_dir / "norms.f32",
        dtype="<f4",
        mode="r",
        shape=(n, len(blocks)),
        order="C",
    )

    checkpoint = torch.load(
        step_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    return manifest, sketch, norms, checkpoint


manifest, sketch, norms, checkpoint = load_step(
    "downloaded/run_00/step_000100"
)
print(sketch.shape, sketch.dtype)  # torch.Size([50000, 4275]), bfloat16
print(norms.shape, norms.dtype)    # (50000, 6), float32
print(checkpoint["step"])         # 100

# Materialize a manageable slice for model input.
x = sketch[0:1024].float()
layer_norms = torch.from_numpy(np.array(norms[0:1024], copy=True))
```

Do not call `sketch.float()` on the full mapping unless enough RAM is available:
the full FP32 expansion is 855 MB for one step, before additional copies.

### Extracting block features and clipping targets

```python
BLOCK_SLICES = {
    "stem": slice(0, 131),
    "block1": slice(131, 732),
    "block2": slice(732, 1842),
    "block3": slice(1842, 4051),
    "final_norm": slice(4051, 4120),
    "classifier": slice(4120, 4275),
}
BLOCKS = (
    "stem", "block1", "block2", "block3", "final_norm", "classifier"
)

example_index = 123
block3_feature = sketch[example_index, BLOCK_SLICES["block3"]].float()

# Blocks are disjoint, so this recovers the exact full-gradient norm.
exact_total_norm = torch.linalg.vector_norm(layer_norms, dim=1)
clip_factor = torch.clamp(1.0 / exact_total_norm, max=1.0)  # C = 1.0
was_clipped = exact_total_norm > 1.0
```

The exact norm, clipping factor, or clipping indicator can serve as predictor
targets without reconstructing the original 176,602-dimensional gradient.

### Associating rows with CIFAR-10

```python
from torchvision import datasets, transforms

canonical = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])
train = datasets.CIFAR10(
    "data", train=True, download=False, transform=canonical
)

image, label = train[example_index]
feature = sketch[example_index]
norm = norms[example_index]
```

The row mapping is invariant across every step and run.

### Loading a checkpoint into the model

```python
import torch
from model import WideResNet

payload = torch.load(
    "downloaded/run_00/step_000100/checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)
model = WideResNet(
    num_classes=10, depth=16, width=1, groups=16, dropout_rate=0.0
)
model.load_state_dict(payload["model"])
model.eval()

ema_model = WideResNet(
    num_classes=10, depth=16, width=1, groups=16, dropout_rate=0.0
)
ema_model.load_state_dict(payload["ema_model"])
ema_model.eval()
```

Reconstructing a resumable private optimizer requires first wrapping an
identically constructed model and optimizer with Opacus, then loading the saved
optimizer and accountant state. Loading model weights alone is safest for
analysis. Bit-for-bit continuation is not guaranteed because deterministic CUDA
algorithms were not enforced and the run used multi-worker data loading.

### Loading projection matrices

```python
projection = torch.load(
    "downloaded/run_00/projection_spec/projection.pt",
    map_location="cpu",
    weights_only=True,
)
assert projection["seed"] == 4275
assert tuple(projection["dimensions"]) == (131, 601, 1110, 2209, 69, 155)
assert projection["matrices"]["block3"].shape == (132224, 2209)
```

Projection matrices are not needed to train directly from saved sketches.

## Integrity verification

Each step manifest stores the byte size and SHA-256 of all binary artifacts.
After downloading a step, verify it with:

```bash
cd downloaded/run_00/step_000100
jq -r '.files | to_entries[] | "\(.value.sha256)  \(.key)"' manifest.json \
  | sha256sum --check -
```

Also check that each manifest's `projection_sha256` equals the projection hash
given above. During collection, local directories were deleted only after
`rclone copy` succeeded and a size-based `rclone check` passed. Manifest hashes
provide stronger validation after download.

## Results

Accuracy below is the EMA model's accuracy on all 10,000 CIFAR-10 test examples
at step 510. Training accuracy covers only the final stochastic logical batch.

| Run | Seed | EMA test accuracy | EMA test loss | Final-batch train accuracy | Final-batch train loss | Runtime (min) |
|---|---:|---:|---:|---:|---:|---:|
| `run_00` | 0 | 61.30% | 1.2760 | 57.09% | 1.4620 | 161.6 |
| `run_01` | 1 | 60.64% | 1.2872 | 55.85% | 1.5771 | 161.3 |
| `run_02` | 2 | 62.19% | 1.2454 | 58.04% | 1.4735 | 191.1 |
| `run_03` | 3 | 60.31% | 1.2913 | 57.90% | 1.4232 | 194.4 |
| `run_04` | 4 | 61.64% | 1.2232 | 58.07% | 1.4052 | 163.5 |
| `run_05` | 5 | 61.93% | 1.2463 | 59.93% | 1.3508 | 161.3 |
| `run_06` | 6 | 60.85% | 1.2668 | 55.92% | 1.6390 | 161.6 |
| `run_07` | 7 | 61.40% | 1.2507 | 60.60% | 1.3284 | 160.6 |
| `run_08` | 8 | 61.91% | 1.2506 | 60.87% | 1.3154 | 161.9 |
| `run_09` | 9 | **62.60%** | **1.2195** | 58.90% | 1.3899 | 161.6 |

Across ten runs:

- mean final EMA test accuracy: **61.477%**;
- sample standard deviation: **0.721 percentage points**;
- range: **60.31% to 62.60%**;
- mean final EMA test loss: **1.2557**;
- mean runtime: **167.87 minutes per run**;
- mean collection sweep: **14.419 seconds per step**;
- total summed per-run runtime: approximately **27.98 hours**.

Runtime includes training, evaluation, collection, hashing, staging, and waits
for asynchronous upload. Runs 2 and 3 took longer; their outputs are complete.

### Aggregate learning curve

| Step | Mean EMA test accuracy | Sample SD | Min | Max |
|---:|---:|---:|---:|---:|
| 50 | 34.803% | 1.164% | 32.88% | 36.26% |
| 100 | 43.540% | 0.945% | 42.50% | 45.27% |
| 150 | 47.766% | 1.043% | 46.80% | 49.87% |
| 200 | 50.759% | 0.906% | 49.84% | 52.15% |
| 250 | 53.192% | 0.805% | 52.12% | 54.45% |
| 300 | 55.201% | 0.673% | 54.02% | 56.14% |
| 350 | 56.939% | 0.714% | 55.84% | 58.08% |
| 400 | 58.559% | 0.683% | 57.58% | 59.49% |
| 450 | 59.948% | 0.714% | 59.01% | 61.02% |
| 500 | 61.198% | 0.676% | 60.22% | 62.31% |
| 510 | **61.477%** | **0.721%** | **60.31%** | **62.60%** |

## Suggested predictor dataset construction

A supervised example can be indexed by `(run, step, example_id)`:

- input: the 4,275-dimensional sketch;
- scale features: the six block norms, optionally transformed with `log1p`;
- optional context: step, privacy epsilon, class, or checkpoint statistics;
- target: exact total norm, clipping indicator, clipping factor, or an adaptive
  clipping-threshold target.

Split by complete runs or contiguous time intervals rather than randomly mixing
all rows. Random row splitting leaks nearly identical model states and repeated
CIFAR-10 identities between partitions. A useful first split is runs 0-7 for
predictor training, run 8 for validation, and run 9 as a held-out test. Fit all
normalization statistics only on predictor-training runs.

The layout is step-major. Sequential access over selected runs and steps gives
far better object-store and disk locality than random access across all 255
million records.

## Privacy and handling requirements

Projected per-example gradients and exact norms are sensitive derivatives of
individual training records. They were collected before clipping and without
noise and are **not differentially private outputs**. Random projection is
compression, not anonymization or privacy.

Keep the object-store prefix private; do not publish sketches, norms, or derived
row-level datasets. Apply access control, encryption, and retention appropriate
for raw training-data derivatives.

Within one run, the privacy estimate accounts for 510 DP update mechanisms. Ten
independently trained models do not jointly retain the per-run epsilon if all
ten are released; their privacy losses must be composed. The raw sketch dataset
itself cannot be described as a DP release, regardless of model accountant
values.

## Reproducibility environment

| Component | Version/value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| Opacus | 1.5.4 |
| NumPy | 2.5.2 |
| CUDA reported by PyTorch | 12.4 |
| GPU | NVIDIA Quadro RTX 6000, 24 GB |
| Source commit | `cc30f6e7f13b447fffb057e02054d2db18a3da74` |
| Projection seed | 4275 |

RNG state is saved to maximize reproducibility. Exact replay remains subject to
CUDA kernel determinism, library behavior, worker scheduling, and hardware.

The sequence ran under task-spooler (`tsp`). Each step was staged to local SSD,
uploaded asynchronously, size-verified, and then removed locally. This bounded
local storage use while overlapping object-store transfer with computation.

## Known limitations

1. The block-wise dimension split is heuristic and has no uniform 15% JL
   guarantee over the full collection.
2. BF16 storage quantizes projected coordinates; exact block norms remain FP32.
3. Collected gradients use canonical images, not the particular augmentation
   used by an update.
4. Gradients exist only after updates 1 through 510, not at initialization.
5. `secure_mode=False` was used for performance.
6. Full checkpoints at every step are highly redundant.
7. The layout favors sequential step access rather than random access over all
   255 million records.
8. Exact continuation can depend on software and CUDA nondeterminism despite
   saved RNG and optimizer state.
9. The old 406-step prefix is a separate attempt and must not be combined with
   the canonical restart without an explicit research reason.

## Minimal validation checklist

Before using a downloaded subset:

1. Confirm the remote path contains `restart_20260922`.
2. Confirm the run is `run_00` through `run_09`.
3. Confirm the step is in `[1, 510]`.
4. Verify hashes from `manifest.json`.
5. Verify `examples == 50000` and `feature_dim == 4275`.
6. Verify the projection SHA-256 shown above.
7. Interpret rows as original CIFAR-10 training indices.
8. Preserve the documented block order and dtypes.
9. Keep the data private.
