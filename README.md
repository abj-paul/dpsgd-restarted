# Adaptive clipping WideResNet sandbox

A small PyTorch baseline for CIFAR-10 using the DP-friendly WideResNet from
JAX Privacy's historical image-classification experiments. The implementation
keeps the reference model's scaled weight-standardized convolutions,
GroupNorm, activation/norm/convolution ordering, global average pooling, and
configurable skip initialization.

The default WRN-16-4 is deliberately smaller than the common WRN-28-10. It is
a better starting point for DP-SGD experiments because per-example gradients
use substantially more memory than ordinary minibatch training. The model has
no BatchNorm, whose batch-dependent statistics are unsuitable for standard
DP-SGD accounting.

## Setup and training

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

CIFAR-10 downloads to `data/`, and the best checkpoint plus metrics are saved
under `runs/wrn16-4/`. For a quick end-to-end check:

```bash
python main.py --epochs 1 --max-train-samples 1024 --workers 2
```

Scale the architecture with `--depth` (which must be `6n + 4`) and `--width`.
For example, `--depth 28 --width 10` selects WRN-28-10, though that is likely
too expensive once per-example gradients are introduced.

`main.py` is intentionally a non-private SGD baseline: it makes no privacy
claim and does not spend a privacy budget. It provides a known-good target
training loop on which an adaptive clipping predictor and a properly
accounted DP optimizer can be developed next.

## Architecture source

Ported from the historical JAX Privacy CIFAR model (Apache-2.0), corresponding
to the source link supplied for this project:

https://github.com/google-deepmind/jax_privacy/blob/a0d0600bf151222a6eedb44b4bc09a4188ea363e/jax_privacy/experiments/image_classification/models/cifar.py
