"""Smallest end-to-end run: checkpoint in, source estimate out.

    python scripts/demo.py --ckpt <checkpoint> --assets <asset_dir>

`--assets` is a directory holding the three released files that make the model
usable at all:

    head_modelColin27_5003_Standard-10-5-Cap339.mat   electrode + cortex geometry
    _62_leadfield.mat                                  (62, 5003, 3) leadfield
    _62_eloreta_weight.mat                             source regulariser weights

Without a leadfield there is nothing to localise into, so there is no
assets-free mode: a made-up spherical leadfield sits far outside the training
distribution and produces numbers that look like failure but only measure the
mismatch.

By default the demo excites one cortical vertex with a 10 Hz oscillation,
forward-projects it to EEG, and asks the model to recover it -- so the repo is
runnable with no EEG recording. Pass --eeg to use your own instead.

Note that geometry comes from `EEG2Source2EEG_Dataset`, not from the .mat files
directly: electrode coordinates are aligned into the MNE head frame (with an
x-flip) before they reach the model, and feeding raw file coordinates instead
silently degrades the estimate.
"""
import torch            # import first: avoids a c10.dll conflict with pandas
import argparse
import os
import os.path as op
import sys
import tempfile

import numpy as np

sys.path.insert(0, op.dirname(op.dirname(op.abspath(__file__))))
from eeg_sl.models import SLTConfig, EEG2Source2EEG_autoregressive  # noqa: E402

T_WIN = 200
FS = 100.0
RAW_CH = 64                 # the loader takes 64 channels and drops CB1/CB2
DROP = (59, 63)


def load_model(ckpt, device):
    """`from_pretrained` does not round-trip these configs on transformers 5.x,
    so load the weights directly. strict=False is expected: the sinusoidal
    positional-encoding tables are deterministic buffers rebuilt at __init__."""
    from safetensors.torch import load_file
    cfg = SLTConfig.from_pretrained(ckpt)
    if getattr(cfg, 'pos_embedding_type', None) is None:
        cfg.pos_embedding_type = 'Learned'
    model = EEG2Source2EEG_autoregressive(cfg)
    missing, unexpected = model.load_state_dict(
        load_file(op.join(ckpt, 'model.safetensors')), strict=False)
    real = [k for k in missing if not k.endswith('.pe')]
    if real or unexpected:
        raise SystemExit(f'weights do not match the config: missing={real} '
                         f'unexpected={list(unexpected)}')
    return model.to(device).eval(), cfg


def build_dataset(assets):
    """The real dataset object, so electrode/source coordinates reach the model
    in exactly the frame it was trained on."""
    from eeg_sl.data import EEG2Source2EEG_Dataset
    need = {'head': 'head_modelColin27_5003_Standard-10-5-Cap339.mat',
            'lf': '_62_leadfield.mat', 'w': '_62_eloreta_weight.mat'}
    paths = {k: op.join(assets, v) for k, v in need.items()}
    for k, p in paths.items():
        if not op.exists(p):
            raise SystemExit(f'missing asset: {p}')
    fd, tmp = tempfile.mkstemp(suffix='.npy')
    os.close(fd)
    try:
        np.save(tmp, np.zeros((1, RAW_CH, T_WIN), dtype=np.float32))
        ds = EEG2Source2EEG_Dataset(
            subject_folder=tmp, leadfield_path=paths['lf'], ch_list='tshinghwa',
            is_shuffle=False, is_npy=True, leadfield_reduction=False,
            head_model=paths['head'], eloreta_weight_path=paths['w'])
    finally:
        os.unlink(tmp)
    return ds


def set_eeg(ds, eeg62):
    """Put a (62, T) trial back into the loader's 64-channel layout."""
    full = np.zeros((1, RAW_CH, T_WIN), dtype=np.float32)
    keep = [i for i in range(RAW_CH) if i not in DROP]
    full[0, keep] = eeg62
    ds.eeg_data = full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint directory')
    ap.add_argument('--assets', required=True, help='directory with the released .mat assets')
    ap.add_argument('--eeg', default=None, help='your own EEG, .npy shaped (62, T)')
    ap.add_argument('--vertex', type=int, default=2500, help='vertex to excite')
    ap.add_argument('--save', default=None, help='write the estimate to this .npy')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(args.ckpt, device)
    print(f'model: d_model={cfg.d_model} N={cfg.N} K={cfg.K} sources={cfg.tgt_len} '
          f'pos_embedding={cfg.pos_embedding_type}  [{device}]')

    ds = build_dataset(args.assets)
    item = ds[0]
    leadfield = item['leadfield'].numpy().astype(np.float64)      # (C, V, 3)
    source_pos = item['source_pos'].numpy()                       # (V, 3), metres
    print(f'geometry: {leadfield.shape[0]} channels, {leadfield.shape[1]} sources')

    gt = None
    if args.eeg:
        eeg = np.load(args.eeg)[:, :T_WIN].astype(np.float64)
    else:
        gt = args.vertex
        t = np.arange(T_WIN) / FS
        src = np.zeros((3, leadfield.shape[1], T_WIN))
        ori = source_pos[gt] / np.linalg.norm(source_pos[gt])     # radial
        src[:, gt, :] = ori[:, None] * np.cos(2 * np.pi * 10.0 * t)
        eeg = np.einsum('cvk,kvt->ct', leadfield, src)
        print(f'synthetic: one 10 Hz radial source at vertex {gt}')
    eeg = (eeg - eeg.mean()) / eeg.std()      # the loader's only normalisation
    set_eeg(ds, eeg)

    from eeg_sl.data import EEG_2_Source_DataCollector
    batch = EEG_2_Source_DataCollector()([ds[0]])
    batch = {k: v.to(device) for k, v in batch.items()
             if torch.is_tensor(v) and k != 'labels'}
    with torch.no_grad():
        out = model(**batch, labels=batch['src'], return_dict=True)

    est = out['hidden_states'][0].cpu().numpy()                   # (3, V, T)
    rec = out['logits'][0].cpu().numpy()                          # (C, T)
    x = batch['src'][0].cpu().numpy()

    power = (est ** 2).sum(axis=(0, 2))                           # sum_t |j|^2
    peak = int(np.argmax(power))
    corr = float(np.corrcoef(rec.ravel(), x.ravel())[0, 1])
    print(f'\nsource estimate {est.shape}   peak vertex {peak}')
    print(f'EEG reconstruction corr(L*S, X) = {corr:+.4f}')
    if gt is not None:
        mm = np.linalg.norm(source_pos[peak] - source_pos[gt]) * 1000
        print(f'peak is {mm:.1f} mm from the excited vertex')

    if args.save:
        np.save(args.save, est)
        print(f'saved -> {args.save}')


if __name__ == '__main__':
    main()
