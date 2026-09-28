# Model card — DEMIST (EEG → Source → EEG)

A self-supervised transformer that estimates cortical **source activity** from
scalp **EEG**, using the subject's leadfield as the physical bridge. Trained
without any ground-truth source: the model reconstructs the input EEG through
the forward model, regularised towards physiologically plausible sources.

| | |
| :---- | :---- |
| Architecture | Dual-stream transformer: montage-invariant EEG encoder + leadfield encoder, autoregressive source decoder |
| Parameters | **15.25 M** (encoder 4.74 M, decoder 6.32 M, EEG projector 1.52 M, position embedding 0.20 M, leadfield encoder 0.013 M, generator 0.15 M) |
| Hidden size / FF / heads / depth | 256 / 1024 / 8 / 6 (encoder and decoder) |
| Source space | Colin27, **5003** cortical vertices, 3 orientations |
| Input window | 200 samples per channel |
| Decoder chunk `K` | 150 voxels per autoregressive step (runtime-adjustable) |
| Objective | `‖Φ − L·Ŝ‖² + α·Tr(ŜᵀWŜ)`, eLORETA weights, α = 0.1 |
| Precision | bf16 |

## Which checkpoint this is

The released weights are **step 485,904 (epoch 53)** of the pre-training run
`naive_autoregressive_leadfieldprojection_memory_continue_20251104_160634`.

The paper's own figures were produced from **step 467,568 (epoch 51)** of that
same run. That exact step was not preserved; epoch 53 is the nearest surviving
save from the identical training, and it reproduces the published simulation
result:

| | LocErr, 676 trials @ SNR 16 dB |
| :---- | ----: |
| released checkpoint (epoch 53) | **11.79 mm** mean, 9.36 mm median |
| paper, Table II row 1 | 12.27 mm |

Metric definitions match the paper's evaluation script: peak of `sum_t |j|^2`,
distance to the true source; SD weighted by `(sum_t |j|^2)^2`; AUC over a 20 mm
positive radius.

## Inputs and outputs

**Inputs** — windowed EEG `(C, T)`; the subject's leadfield `(C, V, 3)`;
electrode and source 3-D positions; optionally eLORETA weights `(3, 3, V)`.

**Output** — estimated source activity `Ŝ` of shape `(3, V, T)`, returned as the
model's `hidden_states`. Forward-projecting `Ŝ` through the leadfield
reconstructs the EEG (`logits`).

`C` is not fixed: the EEG encoder embeds electrodes by their 3-D coordinates,
so a montage the model has never seen is handled without retraining — subject
to the limits below.

## Training data

| Source | Channels | Notes |
| :---- | ----: | :---- |
| Tsinghua SSVEP | 62 | real EEG |
| EEGLAB/FieldTrip simulation (Colin27 5003) | 62 | single / double / multiple source |
| Resting-state, epilepsy sets | 30 / varies | present in the config, disabled for the released model |

Not redistributed with this repository — see `DATA.md` for where to obtain each
one and the expected file layout.

**Pre-processing is deliberately minimal**: a single global z-score per trial.
No filtering, no re-referencing, no baseline correction. Bad channels are
recorded in `info['bads']` but **not** dropped, so the montage stays aligned
with the leadfield.

## Intended use

Research on EEG source imaging: estimating source distributions, comparison
against MNE / dSPM / eLORETA / LCMV, and as a pre-trained starting point for
self-supervised fine-tuning on a new dataset.

**Not** for clinical decision-making. Outputs are distributed source estimates
with tens of millimetres of localisation error on real data; they are not a
substitute for intracranial recording.

## Limitations

**1. `pos_embedding_type: "sinusoid"` is NOT montage-invariant.**
The released model uses `"Learned"`, which embeds each electrode from its 3-D
coordinates and therefore transfers across montages. The `"sinusoid"` option
indexes a fixed table by **channel ordinal**, so changing the electrode set
silently reassigns every position. Measured zero-shot on 256-channel data after
62-channel training:

| `pos_embedding_type` | AUC |
| :---- | ----: |
| **`Learned`** (released model) | **0.805** |
| `sinusoid` (+ leadfield) | 0.509 |
| `sinusoid` (EEG only) | 0.552 |

0.5 is chance. Do not select `"sinusoid"` unless the evaluation montage is the
training montage.

**2. Large channel counts are extrapolation.**
Pre-training used ≤ 62 channels. Applying the model to a 256-channel montage
works because positions are coordinate-based, but that regime was never seen in
training and accuracy is correspondingly lower. Fine-tuning on the target
montage recovers a large part of it. Training on high-density montages is
future work.

**3. Fixed source space and template anatomy.**
The released weights assume the Colin27 5003-vertex cortical surface. Using
individual anatomy requires a leadfield on that same source space, and the
template-to-individual mismatch is part of the error on real subjects.

**4. Simulation sources are single-frequency.**
The simulation used for training and for the reported simulation results
produced **10 Hz** sources, not the randomised 6–14 Hz the original script
intended (`cfg.dip.frequency` was set where FieldTrip reads `cfg.frequency`;
measured peak frequency is 10.00 Hz on 200/200 trials). The training mixture is
therefore narrower in spectrum than intended.

**5. Non-stationary signals, artifacts and correlated sources.**
Training sources are stationary sinusoids with sensor-level white noise. Real
EEG artifacts are not modelled.

## Loading a checkpoint

`from_pretrained` does not round-trip these configs on `transformers` 5.x. Use
the pinned `transformers==4.46.1`, or load the weights manually:

```python
from safetensors.torch import load_file
from eeg_sl.models import SLTConfig, EEG2Source2EEG_autoregressive

cfg = SLTConfig.from_pretrained(ckpt)
model = EEG2Source2EEG_autoregressive(cfg)
model.load_state_dict(load_file(f'{ckpt}/model.safetensors'), strict=False)
model.eval()
```

`strict=False` is required: the sinusoidal positional-encoding tables are
deterministic buffers, rebuilt identically at `__init__`.

On Windows, `import torch` **before** `import pandas` (c10.dll conflict).

## Citation

See `CITATION.cff`.
