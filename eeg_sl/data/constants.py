"""Channel montages and small shared helpers for the datasets.

The two montages below come from the EEGLAB Colin27 head model channel labels.
Note the resting-state ↔ EEGLAB naming: T5=P7, T6=P8, T3=T7, T4=T8.
"""

from mne.transforms import _get_trans, apply_trans


# 62-channel SSVEP (Tsinghua) montage (CB1/CB2 dropped at load time)
TSINGHUA_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5",
    "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2",
]

# 30-channel resting-state montage
RESTING_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8",
    "T7", "C3", "Cz", "C4", "T8",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

CHANNEL_LISTS = {
    "tshinghwa": TSINGHUA_CHANNELS,
    "resting": RESTING_CHANNELS,
}


def align_positions(pos, trans, coord_frame="mri", fro="head"):
    """Align EEGLAB head-model positions into the MNE source coordinate frame.

    Swaps the x/y axes (EEGLAB convention) then applies the head->mri transform.
    Mutates ``pos`` in place (matching the original behaviour).
    """
    trans = _get_trans(trans, fro=fro, to=coord_frame)[0]
    pos[:, [0, 1]] = pos[:, [1, 0]]
    return apply_trans(trans, pos)
