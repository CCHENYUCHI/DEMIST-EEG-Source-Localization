# EEG → Source → EEG Source Localization

A self-supervised deep-learning model that estimates brain **source activity**
from scalp **EEG**, using the physical forward model (leadfield) as the bridge.
This folder is a cleaned, runnable repackaging of the deep-learning half of the
multi-modal source-localization project (the ESI / leadfield-building /
simulation / plotting work lives elsewhere).

> **Data layout:** no dataset ships with this repository.
> See **`..\DATA_MAP.md`** for exactly where every dataset is on `D:` and which
> config field points at it. The configs are already filled in for that layout.

---

## 1. The idea in one picture

```
   EEG (scalp, C channels)                                  reconstructed EEG
          │                                                        ▲
          ▼                                                        │
   ┌─────────────┐    sensor positions      ┌──────────────┐  leadfield forward
   │  Encoder    │◀─────(pos embedding)      │   Generator  │──(L · Ĵ)──────────┘
   └─────────────┘                           └──────────────┘
          │ memory                                  ▲
          ▼                                          │  source activity Ĵ
   ┌───────────────────────────────────────────────────────┐   (3 × V × T)
   │  Autoregressive Decoder over source voxels (K / step)  │   ← this is the
   │  source tokens built from the leadfield (per voxel)    │     localization output
   └───────────────────────────────────────────────────────┘
```

* **Input:** windowed EEG `(C, T)` + the subject's leadfield `(C, V, 3)` +
  electrode/source 3-D positions + (optionally) eLORETA weights.
* **Output:** estimated source activity `Ĵ` of shape `(3, V, T)` — the
  `hidden_states` of the model. Forward-projecting `Ĵ` through the leadfield
  reconstructs the EEG.
* **Training signal:** the model reconstructs the *input EEG itself*
  (`labels = src`). It is **fully self-supervised** — no ground-truth source is
  needed. A source regulariser (eLORETA / L1 / L2) keeps `Ĵ` physiologically
  plausible.

> Why this works: the leadfield `L` is fixed physics. The network is free to
> choose any source `Ĵ` that both (a) reproduces the measured EEG via `L·Ĵ` and
> (b) is "simple" under the regulariser — exactly the inverse-problem setup of
> classical ESI, but learned and amortised across subjects/montages.

---

## 2. Repository layout

```
eeg_source_localization/
├── README.md
├── requirements.txt
├── configs/                # ← edit these (paths + hyper-parameters)
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   └── inference.yaml
├── eeg_sl/                 # the package
│   ├── config_io.py        # tiny YAML loader (+ --set overrides)
│   ├── metrics.py          # eval metrics for the Trainer
│   ├── data/
│   │   ├── constants.py    # channel montages + MNE coordinate alignment
│   │   ├── bids_io.py      # load_bids (localize-mi reader)
│   │   ├── datasets.py     # the 3 datasets + MergeDataset + random_subset
│   │   └── collator.py     # EEG_2_Source_DataCollector (padding + labels=src)
│   └── models/
│       ├── config.py       # SLTConfig (HuggingFace PretrainedConfig)
│       ├── transformer.py  # encoder/decoder/attention building blocks
│       └── eeg2source2eeg.py  # EEG2Source2EEG_autoregressive (the model)
└── scripts/                # ← run these
    ├── train.py            # pre-training
    ├── finetune.py         # fine-tuning on intracerebral (HIS) data
    └── inference.py        # extract + save source activity
```

---

## 3. The three-stage pipeline

| Stage | Script | Data | What it does |
|-------|--------|------|--------------|
| **1. Pre-train** | `scripts/train.py` | SSVEP / resting / epilepsy / EEGLAB-simulated EEG | Learn the EEG→source→EEG mapping across montages, self-supervised. |
| **2. Fine-tune** | `scripts/finetune.py` | localize-mi intracerebral stimulation (HIS BIDS, `sub-01…07`) | Adapt the pre-trained model to real subjects with ground-truth-adjacent data. |
| **3. Inference** | `scripts/inference.py` | held-out HIS subjects | Run the model, save estimated source activity (`.mat` per run) for downstream ESI evaluation/plotting. |

Each stage is driven by its YAML config:

```bash
pip install -r requirements.txt

# 1) pre-train
python scripts/train.py     --config configs/pretrain.yaml

# 2) fine-tune  (set pretrained.checkpoint_path first)
python scripts/finetune.py  --config configs/finetune.yaml

# 3) inference (set model.checkpoint_path to the fine-tuned model)
python scripts/inference.py --config configs/inference.yaml
```

Override any field on the command line without editing the file:

```bash
python scripts/train.py --config configs/pretrain.yaml \
    --set training.batch_size=8 model.K=200 datasets.use_resting=true
```

---

## 4. What you must edit before running

Everything machine-specific lives in the YAML configs — **no paths are
hard-coded in the Python**. Check these:

1. **`head_model_path`** — the EEGLAB Colin27 head model `.mat`
   (`head_modelColin27_5003_Standard-10-5-Cap339.mat`). Source/sensor positions
   are read from it and aligned into the MNE `fsaverage` frame.
2. **Per-dataset paths** — `train_path` / `val_path` (`.npy` EEG),
   `leadfield_path`, `eloreta_weight_path`, simulation `.mat` lists, etc.
3. **`training.output_base_dir`** — where checkpoints are written.
4. **`pretrained.checkpoint_path`** (finetune) and **`model.checkpoint_path`**
   (inference) — point at the right run/checkpoint folder.
5. **wandb** — set `wandb.enabled: false` to run without logging, or
   `wandb login` beforehand (don't commit API keys).

### Data contract (if you add a new dataset)
A dataset's `__getitem__` must return this dict; the collator handles padding:

```python
{
  'src':            torch.FloatTensor (C, T),     # EEG, z-scored
  'leadfield':      torch.FloatTensor (C, V, 3),  # forward model
  'leadfield_id':   torch.long  scalar,           # 0=tsinghua,1=resting,2=other
  'sensor_pos':     torch.FloatTensor (C, 3),     # MNE frame
  'source_pos':     torch.FloatTensor (V, 3),     # MNE frame
  'eloreta_weight': torch.FloatTensor (3, 3, V) or None,   # note axis order
  # 'metadata': dict   # HIS dataset only, used by inference grouping
}
```

---

## 5. Key hyper-parameters (`configs/*.yaml → model:`)

| Field | Meaning | Trained value |
|-------|---------|---------------|
| `d_model`, `d_ff`, `h` | transformer width / FFN / heads | 256 / 1024 / 8 |
| `N` | **encoder *and* decoder depth** | 6 |
| `K` | source voxels decoded per autoregressive step | 150 |
| `source_voxel_time` | generator width = `3 × T` | 600 |
| `tgt_len` | number of source voxels `V` | 5003 |
| `embedding_type` | `leadfield_projection` (trained) or `encoder` | leadfield_projection |
| `pos_embedding_type` | `Learned` (MLP over xyz) or `sinusoid` | Learned |
| `regularization_loss` | `eloreta` / `L1` / `L2` / `None` | eloreta |
| `reg_alpha` | regulariser weight | 0.1 (pretrain), 0.5 (finetune) |

> ⚠️ **Depth gotcha (kept honest):** in the original notebook `encoder_n=4` /
> `decoder_n=4` were passed but **never used** — the model has always built its
> backbone from `config.N` (default **6**). So the released checkpoints have 6
> layers. `N` is the real knob; `encoder_n`/`decoder_n` are accepted only so old
> `config.json` files still load. See `eeg_sl/models/config.py`.

---

## 6. Notes / provenance

* The model, datasets and collator are extracted verbatim (logic-preserving)
  from the original `Trainer.ipynb` + `FirstMultiModel/EEGART/tf_model.py` +
  `DataLoader/SLT_dataloader.py`. Dead experiments (FFT-only variants, VAE
  variants, ROI/Power/DeepSIF datasets, the old `SLTModel*` classes) were
  dropped — they remain in the original files / git history if ever needed.
* The model is time-domain only (the FFT / phasor-loss experiments were
  removed). The SDPA/flash-attention path (`sdp_attention: true`) is wired up
  but was **not** part of the trained pipeline; treat it as experimental.
* Windows note: `dataloader_num_workers` is fixed to 0 (avoids multiprocessing
  issues on Windows + the head-model objects held in the datasets).
```
