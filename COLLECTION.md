# All-example gradient collection

`dp_train.py --collect-gradients` saves a checkpoint and a 4,275-dimensional
block-wise Gaussian sketch for every one of the 50,000 CIFAR-10 training images
after every DP-SGD update. The sketch is the concatenation of projections for
stem, block1, block2, block3, final_norm, and classifier, with dimensions
`(131, 601, 1110, 2209, 69, 155)`. Each matrix has independent
`N(0, 1/k_l)` entries, generated from `--projection-seed` and fixed across
steps and runs. These dimensions are a heuristic allocation, **not** a
15%-distortion JL guarantee for the block-wise map.

Gradients are unclipped, computed at the post-update checkpoint against a
canonical, unaugmented, normalized CIFAR-10 training view. Rows follow the
original CIFAR-10 training index order. This is distinct from the stochastic
augmentation used for the private training update. Norms are computed in
float32 before projection; sketches are computed in float32 and stored in
bfloat16. The projection is serialized once per run with a SHA-256 digest.

Each `step_XXXXXX` shard contains `checkpoint.pt`, `sketch.bf16` (raw
little-endian uint16 representation of a row-major `[50000, 4275]` BF16
array), `norms.f32` (row-major `[50000, 6]` float32), and `manifest.json`
with SHA-256 digests and the exact shape. `initial/initial.pt` contains
pre-update weights and RNG state. `projection_spec/projection.pt` contains
the matrices. The local stage is uploaded asynchronously by `rclone`; a stage
directory is deleted only after a successful copy and size verification.
The root `projection.pt` remains local for later runs. A failed upload keeps
its stage directory and eventually stops training instead of filling the disk.

The sketches and checkpoints are sensitive training-data derivatives and are
**not themselves DP-protected**. Do not publish them or put them in a public
bucket. Ten independently released DP models also require joint privacy
accounting; epsilon for each run does not apply to their combined release.

The measured full-dataset sweep was 81.4 s with collection batch 32 and
14.7 s with batch 512 on the Quadro RTX 6000. At 510 steps, collection alone
is roughly 2.1 hours per run with batch 512, before training and upload.

For a single run, pass `--steps 510 --logical-batch-size 16667
--physical-batch-size 2048 --noise-multiplier 4.888246618 --delta 1e-5
--collect-gradients --collection-batch-size 512 --collection-remote REMOTE
--ntfy-topic TOPIC --output-dir RUN_DIR`. The remote should be a private,
explicitly confirmed destination. A full run stores 196.2 GB of sketches;
the ten-run total is 1.962 TB before metadata and checkpoints.
