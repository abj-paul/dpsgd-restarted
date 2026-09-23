# Binary clipping-advantage dataset (`C = 0.25` versus `C = 1.0`)

## Status and canonical location

This document specifies the completed 20,400-row counterfactual dataset for
learning a binary per-example clipping policy. Its only actions are

\[
  C_{\mathrm{low}}=0.25,
  \qquad C_{\mathrm{high}}=1.0.
\]

No other candidate clipping action is present in this artifact. The canonical
object-store prefix is:

```text
rclone_s3:dpsgd-as-a-optimization-problem/current-work/adaptive-clipping-wrn/counterfactual_advantage_binary_20400/v1_20260923
```

The local copy is
[`runs/counterfactual_advantage_binary_20400_v1`](runs/counterfactual_advantage_binary_20400_v1).
It was materialized by [`refine_binary_advantage.py`](refine_binary_advantage.py)
at Git commit `715722583b55ca94bbf3980bfe5144c1a2884e74`.

## Purpose and row semantics

Let \(\theta_{s,\ell}\) be the post-update checkpoint at step
\(s\in\{1,\ldots,510\}\) of training run \(\ell\in\{0,…,9\}\). A row asks:

> Starting from \(\theta_{s,\ell}\), is the next one-example clipping decision
> better with \(C=0.25\) or with \(C=1.0\), as measured by a paired one-step
> change in loss on a fixed independent public probe set?

In the logical predictor notation \(t=s+1\), the row represents

\[
  \left(\theta_{t-1,\ell},h_i^{1:t-1},
  C_{\mathrm{low}},C_{\mathrm{high}},A_{s\ell ri}\right).
\]

The checkpoint and gradient history are referenced rather than copied into
each row. For anchor \(i\), \(h_i^{1:s}\) is the sequence of its existing
4,275-dimensional gradient sketches and six exact block norms through step
\(s\).

## Inputs

### Checkpoint trajectories

The states come from ten WRN-16-1 CIFAR-10 DP-SGD trajectories. Each model has
176,602 trainable parameters. Relevant original training settings are:

| Setting | Value |
|---|---:|
| Runs / training seeds | 10 / `0` through `9` |
| Checkpoints per run | 510 post-update states |
| Original sample rate | \(q=1/3\) |
| Original clipping norm | \(C=1.0\) |
| Noise multiplier | \(\sigma=4.888246618\) |
| Learning rate | \(\eta=4.0\) |
| Momentum / weight decay | `0.0` / `0.0` |
| Privacy target | RDP \(\varepsilon\approx8\), \(\delta=10^{-5}\) |

The checkpoint and history prefix is documented in
[`DATA-COLLECTION.md`](DATA-COLLECTION.md). The binary dataset does not embed
model tensors or gradient sketches.

### Declared-public data

Let \(D\) be the 50,000-example CIFAR-10 training set. A deterministic,
class-stratified public subset was selected with seed `20260923`:

\[
  D_{\mathrm{pub}}=D_{\mathrm{upd}}\mathbin{\dot\cup}D_{\mathrm{probe}},
  \qquad |D_{\mathrm{upd}}|=4{,}000,
  \quad |D_{\mathrm{probe}}|=1{,}000.
\]

Every class contributes 400 update examples and 100 probe examples. The sets
are disjoint. Their exact CIFAR-10 indices are in `public_split.json`; the
logical split identity is:

```text
7ab3da293eb55fb24b9ca3757e25c49b29a40d5970aa24b6b1b4d42765e1cbc3
```

Only \(D_{\mathrm{upd}}\) supplies synthetic update batches and anchors.
Only \(D_{\mathrm{probe}}\) supplies the evaluation loss.

## Counterfactual construction

Define flat clipping

\[
  \operatorname{clip}(g,C)
  =g\min\!\left(1,\frac{C}{\lVert g\rVert_2}\right).
\]

For every \((s,\ell)\), two independent Poisson batches are drawn from
\(D_{\mathrm{upd}}\):

\[
  B_{s\ell r}=\{j\in D_{\mathrm{upd}}:Z_j=1\},
  \qquad Z_j\sim\operatorname{Bernoulli}(1/3),
  \quad r\in\{1,2\}.
\]

The expected batch size is \(4{,}000/3\). The observed mean was 1,333.170,
with range 1,216--1,451. Two distinct anchors are selected uniformly without
replacement from each batch.

Update images use the recorded random four-pixel reflect-padded crop and
horizontal flip. Probe images use canonical, unaugmented CIFAR-10 images.
Both use channel mean `(0.4914, 0.4822, 0.4465)` and standard deviation
`(0.2470, 0.2435, 0.2616)`.

Let \(N=50{,}000\), \(n_{\mathrm{upd}}=4{,}000\), \(q=1/3\),
\(C_{\mathrm{cap}}=1\), and
\(\xi_{s\ell r}\sim\mathcal N(0,\sigma^2C_{\mathrm{cap}}^2I)\). The common
high-clipping next state is

\[
  \theta^{H}_{s\ell r}
  =\theta_{s,\ell}-\eta\left[
    \frac{1}{q n_{\mathrm{upd}}}
    \sum_{j\in B_{s\ell r}}\operatorname{clip}(g_j,1)
    +\frac{\xi_{s\ell r}}{qN}
  \right].
\]

For selected anchor \(i\), only its clipping decision is changed:

\[
  \theta^{L}_{s\ell ri}
  =\theta^{H}_{s\ell r}
   -\frac{\eta}{qN}
    \left[\operatorname{clip}(g_i,0.25)
          -\operatorname{clip}(g_i,1)\right].
\]

The public batch estimates the background direction at public-set scale,
whereas the one-record intervention and Gaussian noise retain the deployment
scale \(N=50{,}000\). The paired states use identical checkpoint, batch,
augmentation, and noise; hence their difference is only anchor \(i\)'s clip.

The probe objective is

\[
  L_{\mathrm{probe}}(\theta)
  =\frac{1}{|D_{\mathrm{probe}}|}
    \sum_{(x,y)\in D_{\mathrm{probe}}}
      \operatorname{CE}(f_\theta(x),y).
\]

The authoritative target is

\[
  A_{s\ell ri}
  =L_{\mathrm{probe}}(\theta^H_{s\ell r})
   -L_{\mathrm{probe}}(\theta^L_{s\ell ri}).
\]

Therefore:

| Condition | Label | Meaning |
|---|---|---|
| \(A>10^{-12}\) | `low` | \(C=0.25\) gives lower probe loss |
| \(A<-10^{-12}\) | `high` | \(C=1.0\) gives lower probe loss |
| \(|A|\le10^{-12}\) | `neutral` | the two updates are equal at stored precision |

The reward gauge is `reward_high = 0` and `reward_low = A`. Adding a common
constant to both action rewards would not change the optimal action.
Temporary next states are discarded after evaluating the probe loss.

### Dataset size

There is one binary decision row per selected anchor:

\[
  10\ \text{runs}\times510\ \text{states}\times
  2\ \text{batches}\times2\ \text{anchors}
  =\boxed{20{,}400\ \text{rows}}.
\]

The materialization selects the exact paired \(C=0.25\)-versus-baseline
measurement and makes \(C=1.0\) explicit. It neither interpolates nor
recomputes the advantage. Row-level validation found zero differences from
the measured source values.

## Predictor splits

Rows are split by complete training trajectory, not randomly:

| Split | Runs | Rows |
|---|---:|---:|
| Train | 0--7 | 16,320 |
| Validation | 8 | 2,040 |
| Test | 9 | 2,040 |

This prevents checkpoints from one trajectory appearing in multiple splits.
The same declared-public example identity can occur in different runs, so this
split measures generalization to unseen training trajectories, not to unseen
CIFAR-10 identities.

## Schema

The SQLite table is named `binary_advantages`. Its primary key is
`(run_id, checkpoint_step, batch_replicate, anchor_slot)`.

| Fields | Meaning |
|---|---|
| `run_id`, `checkpoint_step` | Source \((\ell,s)\) |
| `batch_replicate`, `anchor_slot` | Batch \(r\in\{1,2\}\) and anchor \(1\) or \(2\) |
| `example_id` | Original CIFAR-10 training index in \(D_{\mathrm{upd}}\) |
| `dataset_split` | `train`, `validation`, or `test` |
| `c_low`, `c_high` | Constants `0.25`, `1.0` |
| `baseline_high_probe_ce` | \(L_{\mathrm{probe}}(\theta^H)\) |
| `candidate_low_probe_ce` | \(L_{\mathrm{probe}}(\theta^L)\) |
| `reward_high`, `reward_low` | \(0\), \(A\) |
| `advantage_low_vs_high` | Raw authoritative target \(A\) |
| `advantage_scaled` | \(A/(s_A+10^{-12})\) |
| `absolute_advantage` | \(|A|\) |
| `preferred_action` | `low`, `high`, or `neutral` |
| `gradient_norm` | Unclipped \(\lVert g_i(\theta_{s,\ell})\rVert_2\) |
| `high_clipping_factor` | \(\min(1,1/\lVert g_i\rVert_2)\) |
| `low_clipping_factor` | \(\min(1,0.25/\lVert g_i\rVert_2)\) |
| `parameter_delta_norm` | \(\lVert\theta^L-\theta^H\rVert_2\) |
| `logical_batch_size` | Realized Poisson batch size |
| `batch_seed`, `augmentation_seed`, `noise_seed` | Deterministic paired RNG seeds |
| `checkpoint_uri` | Exact `checkpoint.pt` used for \(\theta_{s,\ell}\) |
| `history_remote` | Object-store run prefix containing \(h_i^{1:s}\) |
| `history_end_step` | Last included history step, equal to \(s\) |

The global scale is

\[
  s_A=\operatorname{median}|A|=4.868945805358571\times10^{-5},
  \qquad (s_A+10^{-12})^{-1}=20538.3263531.
\]

Use raw `advantage_low_vs_high` for scientific interpretation. The scaled
field is a numerically convenient regression target.

## Files and integrity

| File | Bytes | SHA-256 |
|---|---:|---|
| `binary_advantages.sqlite3` | 10,366,976 | `e258b3972160288a7c0ab9aca5c1f9960d2d247f56e40257ee87b0a4ab197c19` |
| `binary_advantages.jsonl.gz` | 2,422,305 | `713ed260452adb437a09e08c407694375bcfda46958821b33087c9e107571b31` |
| `manifest.json` | 988 | `a461147b52ef8b70c009f0cd94e838156b00441298e3623a1296bbe3537ec0fc` |
| `public_split.json` | 54,089 | `edf16825ee70e46777b9d46d286ed696e4d573c1172576197273a13c1b60c154` |
| `summary.json` | 8,579 | `2b0f14f068063b12996679dac22a223783b5fe5f0cb7fb91bd441c66cfc4f670` |

`manifest.json` records construction identity and provenance. `summary.json`
contains full per-run, per-split, and 50-step-stage statistics. The SQLite and
compressed JSONL files contain the same 20,400 logical rows.

## Retrieving and loading

Download the complete compact artifact:

```bash
BINARY_REMOTE='rclone_s3:dpsgd-as-a-optimization-problem/current-work/adaptive-clipping-wrn/counterfactual_advantage_binary_20400/v1_20260923'
mkdir -p downloaded/binary_20400
rclone copy "$BINARY_REMOTE" downloaded/binary_20400
rclone check downloaded/binary_20400 "$BINARY_REMOTE" --size-only
```

Load SQLite without reading the entire table into memory:

```python
import sqlite3

db = sqlite3.connect("downloaded/binary_20400/binary_advantages.sqlite3")
db.row_factory = sqlite3.Row

rows = db.execute(
    """
    SELECT run_id, checkpoint_step, example_id,
           advantage_scaled, preferred_action,
           checkpoint_uri, history_remote, history_end_step
    FROM binary_advantages
    WHERE dataset_split = 'train'
    ORDER BY run_id, checkpoint_step, batch_replicate, anchor_slot
    """
)
for row in rows:
    target = float(row["advantage_scaled"])
    # Resolve theta and h lazily from the recorded object-store references.
```

Load the streaming JSONL representation:

```python
import gzip
import json

with gzip.open("downloaded/binary_20400/binary_advantages.jsonl.gz", "rt") as f:
    for line in f:
        row = json.loads(line)
        advantage = row["advantage_low_vs_high"]
```

For each row, `checkpoint_uri` points directly to the required model state.
For history step \(u\le s\), the corresponding sketch and norms are under:

```text
{history_remote}/step_UUUUUU/sketch.bf16
{history_remote}/step_UUUUUU/norms.f32
```

Row `example_id` selects the correct row in each `[50000, 4275]` sketch and
`[50000, 6]` norm array. See `DATA-COLLECTION.md` for memory-mapped loading and
the six feature slices. Access only declared-public `example_id` rows when
building predictor inputs.

### Recommended learning targets

For regression, predict \(\widehat A=f(\theta_{s,\ell},h_i^{1:s})\) and choose

\[
  \widehat C=
  \begin{cases}
    0.25,&\widehat A>0,\\
    1.0,&\widehat A\le0.
  \end{cases}
\]

For classification, use `preferred_action`; either retain `neutral` as a third
class or mask it for a strictly binary loss. Regression retains effect size and
is preferable when policy regret matters. Fit all feature and target
normalization statistics on runs 0--7 only.

An action-conditioned representation may expand each row into
`(state, C=1.0, reward=0)` and `(state, C=0.25, reward=A)`. Keep both expanded
records in the same split and training group.

## Reproduction

### Re-materialize the canonical binary rows

Use commit `715722583b55ca94bbf3980bfe5144c1a2884e74` and the completed source
database recorded by `manifest.json`. Its required database SHA-256 is:

```text
359f26e47908f512b42744a14af5de97864bf354576443f038a97985ad101766
```

Choose a new output directory because the script refuses to overwrite a
non-empty artifact:

```bash
.venv/bin/python refine_binary_advantage.py \
  --source-dir runs/counterfactual_advantage_40800_v1 \
  --output-dir runs/reproduced_binary_20400 \
  --no-upload
```

This operation is CPU-only and should complete in seconds. It validates the
source row count, selects exactly one measured low-versus-high decision per
anchor, assigns run-level splits, computes global scaling and statistics,
exports SQLite and JSONL, and checks SQLite integrity. Timestamps and Git
metadata may differ; row values must agree exactly.

To publish a reproduction, supply an explicit private prefix with
`--output-remote` and omit `--no-upload`. The script runs `rclone check
--size-only` after upload.

### Regenerate measurements from checkpoints

Full regeneration requires CUDA, local CIFAR-10, access to the ten checkpoint
trajectories, and the registered seeds. The measurement implementation is
[`counterfactual_advantage.py`](counterfactual_advantage.py). Use
`public_seed=20260923`, `experiment_seed=314159`, \(q=1/3\), two batches, two
anchors, \(\eta=4\), \(\sigma=4.888246618\), cap `1.0`, and candidate actions
`0.25 1.0`; then run `refine_binary_advantage.py` over its completed output.
Exact bitwise replay can still depend on PyTorch, CUDA, and kernel
determinism.

## Completed statistics

### Labels and advantage

| Preferred action | Rows | Fraction |
|---|---:|---:|
| Low, \(C=0.25\) | 12,178 | 59.696% |
| High, \(C=1.0\) | 7,293 | 35.750% |
| Neutral | 929 | 4.554% |

Among non-neutral decisions, low wins 62.544%. All 929 neutral rows have
\(\lVert g_i\rVert_2\le0.25\), so neither action clips the anchor and the two
updates coincide.

| Advantage statistic | Value |
|---|---:|
| Mean | \(2.66743\times10^{-5}\) |
| Sample standard deviation | \(8.10482\times10^{-5}\) |
| Minimum / maximum | \(-3.97591\times10^{-4}\) / \(4.32615\times10^{-4}\) |
| 5th / 50th / 95th percentile | \(-1.02394\times10^{-4}\) / \(1.53759\times10^{-5}\) / \(1.66438\times10^{-4}\) |

| Split | Low | High | Neutral | Mean \(A\) |
|---|---:|---:|---:|---:|
| Train | 9,736 | 5,849 | 735 | \(2.65409\times10^{-5}\) |
| Validation | 1,220 | 713 | 107 | \(2.87891\times10^{-5}\) |
| Test | 1,222 | 731 | 87 | \(2.56270\times10^{-5}\) |

The low-preference fraction ranges from 58.14% to 60.83% across runs. The
mean advantage across run means is \(2.66743\times10^{-5}\), with standard
error \(4.84235\times10^{-7}\).

The in-sample one-step oracle value is

\[
  \frac1n\sum_j\max(A_j,0)=4.48544\times10^{-5}.
\]

This is a post-hoc upper bound, not predictor performance. Always choosing low
has mean reward \(2.66743\times10^{-5}\); always choosing high has reward zero
under the chosen gauge.

### Coverage and norms

The anchors cover 3,979 of 4,000 public update identities; 21 are unobserved.
Among observed identities, the median occurrence count is 5, the mean is
5.127, and the maximum is 14.

The anchor-gradient norm has mean 25.183, median 19.003, sample standard
deviation 22.600, and range 0.000735--192.742. Of all rows, 18,707 (91.70%)
have \(\lVert g_i\rVert_2>1\), 764 (3.75%) lie in \((0.25,1]\), and 929
(4.55%) lie at or below 0.25.

## Privacy and limitations

The construction uses a declared-public subset and DP checkpoint releases.
Under that premise it is post-processing and adds no privacy loss for the
remaining non-public records. This statement does not make the linked full
gradient-history collection DP: its unnoised per-example sketches and norms
remain sensitive. Do not access non-public history rows through
`history_remote`, and keep the object-store prefixes private.

The principal scientific limitations are:

1. \(A\) is a local, one-step effect around a synthetic public-data update;
   it is not the long-horizon return of an adaptive clipping policy.
2. The fixed 1,000-example probe set makes labels comparable but permits
   policy selection to overfit that objective. Final policies require fresh
   end-to-end evaluation.
3. Rows sharing checkpoints, batches, examples, and runs are correlated and
   must not be treated as 20,400 independent observations.
4. The registered split holds out trajectories, not example identities.
5. Two actions do not identify an optimal continuous clipping threshold.
6. Advantages are small one-step loss differences; use paired labels and the
   global scaling rather than separately normalizing each run or step.
7. A perfect-action oracle computed on these labels is an in-sample upper
   bound and cannot be reported as learned-policy performance.

## Minimal validation checklist

Before training a predictor:

1. Confirm exactly 20,400 rows and 5,100 distinct `(run_id, checkpoint_step)`
   pairs.
2. Confirm the only action pair is `(c_low, c_high) = (0.25, 1.0)`.
3. Run `PRAGMA integrity_check;` and require `ok`.
4. Verify the file hashes above or run `rclone check`.
5. Use the registered run-level split and fit normalization on train only.
6. Interpret positive advantage as evidence for low clipping.
7. Treat neutral rows explicitly rather than assigning them arbitrarily.
8. Resolve checkpoint/history references lazily and access public rows only.
