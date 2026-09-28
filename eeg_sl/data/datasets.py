"""Datasets for the EEG -> Source -> EEG model.

All datasets return the same per-sample dict (consumed by
``EEG_2_Source_DataCollector``)::

    {
        'src':           (n_channels, T)      normalised EEG
        'leadfield':     (n_channels, V, 3)   forward model
        'leadfield_id':  scalar long          group id (0=tsinghua, 1=resting, 2=other)
        'sensor_pos':    (n_channels, 3)       electrode positions (MNE frame)
        'source_pos':    (V, 3)                source voxel positions
        'eloreta_weight':(3, 3, V) or None     eLORETA regulariser weights
        'metadata':      dict (HIS dataset only)
    }

Three datasets are used in the pipeline:
  * ``EEG2Source2EEG_Dataset``         - npy EEG or simulated .mat (pre-training)
  * ``EEG2Source2EEG_epilepsy_Dataset``- BUH epilepsy folder (pre-training)
  * ``HISDataset_BIDS``                - localize-mi intracerebral data (fine-tune/eval)

The original code base loaded a Colin27 EEGLAB head model and aligned both
source and sensor positions into the MNE ``fsaverage`` coordinate frame; that
shared logic lives in ``_HeadModelMixin``.
"""

import os

import mne
import numpy as np
import scipy.io
import torch
from mne.datasets import fetch_fsaverage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, Subset
import bisect

from .constants import CHANNEL_LISTS, align_positions

# Default EEGLAB Colin27 head model (5003 sources, Standard 10-5 cap).
DEFAULT_HEAD_MODEL = (
    "C:\\Program Files\\MATLAB\\R2022b\\toolbox\\eeglab\\functions\\"
    "supportfiles\\head_modelColin27_5003_Standard-10-5-Cap339.mat"
)



def _load_bids_fn():
    """Return the Localize-MI BIDS reader, which this package does not bundle.

    `load_bids` belongs to the dataset authors' own release
    (https://github.com/iTCf/mikulan_et_al_2020, `fx_bids.py`). That repository
    carries no licence, so it is not redistributed here. Fetch `fx_bids.py`
    yourself and make it importable, e.g. drop it next to this file or anywhere
    on PYTHONPATH.

    Only `HISDataset_BIDS` needs it; the model, the collator and the simulation
    datasets work without it.
    """
    try:
        from fx_bids import load_bids
    except ImportError:
        try:
            from .fx_bids import load_bids
        except ImportError as exc:
            raise ImportError(
                "HISDataset_BIDS needs load_bids from the Localize-MI "
                "dataset authors' own scripts, which are not bundled "
                "with this package because that repository carries no "
                "licence. Get fx_bids.py from "
                "https://github.com/iTCf/mikulan_et_al_2020 and put it "
                "on PYTHONPATH (or in eeg_sl/data/). Please also cite "
                "Mikulan et al., Scientific Data 7, 127 (2020)."
            ) from exc
    return load_bids


class _HeadModelMixin:
    """Shared head-model loading, MNE alignment and source reduction."""

    def _load_head_model(self, head_model_path):
        self.head_model = scipy.io.loadmat(head_model_path)
        self.fs_dir = fetch_fsaverage(verbose=True)
        self.mne_trans = mne.read_trans(self.fs_dir + '\\bem\\fsaverage-trans.fif')
        self.labels = [label.item() for label in self.head_model['labels'][0]]

    def _aligned_cortex(self, flip_x=True):
        cortex = align_positions(self.head_model['cortex'][0][0][0], self.mne_trans)
        if flip_x:
            cortex[:, 0] *= -1  # fix left/right flip
        return cortex

    def _aligned_sensors(self, flip_x=True):
        sensors = align_positions(self.head_model['channelSpace'], self.mne_trans)
        if flip_x:
            sensors[:, 0] *= -1
        return sensors

    def do_alignment(self, pos, trans, coord_frame='mri', fro='head'):
        return align_positions(pos, trans, coord_frame=coord_frame, fro=fro)

    def reduce_sources_proportionally(self, target_total_points=1000):
        """Down-sample the 5003 cortex sources per atlas region (KMeans centroids)."""
        atlas_labels = self.head_model['atlas']['colorTable'][0][0]
        reduced_pos, indices = [], []
        unique_labels = np.unique(atlas_labels)
        region_counts = np.array([np.sum(atlas_labels == l) for l in unique_labels])
        region_ratios = region_counts / region_counts.sum()
        region_targets = np.clip(np.round(region_ratios * target_total_points).astype(int), 1, None)

        for label, n_points in zip(unique_labels, region_targets):
            if label == 0:
                continue
            region_mask, _ = np.where(atlas_labels == label)
            region_pos = self.cortex_pos[region_mask]
            region_idx = np.where(region_mask)[0]
            if len(region_pos) <= n_points:
                reduced_pos.append(region_pos)
                indices.append(region_idx)
            else:
                kmeans = KMeans(n_clusters=n_points, random_state=0).fit(region_pos)
                centers = kmeans.cluster_centers_
                reduced_pos.append(centers)
                dist_matrix = cdist(centers, region_pos)
                closest_idx = region_idx[np.argmin(dist_matrix, axis=1)]
                indices.append(region_mask[closest_idx])

        return np.vstack(reduced_pos), np.concatenate(indices)


class EEG2Source2EEG_Dataset(_HeadModelMixin, Dataset):
    """EEG from .npy (real recordings) or .mat (EEGLAB simulation).

    Args:
        subject_folder: path to a .npy file (is_npy=True) or list of .mat files.
        leadfield_path: path to the leadfield .mat (key 'leadfield').
        ch_list:        'tshinghwa' (62ch) or 'resting' (30ch).
        is_npy:         True for .npy EEG, False for a list of simulation .mat files.
        is_shuffle:     shuffle the EEG trials at load time.
        leadfield_reduction: reduce 5003 sources to ~1000 (KMeans per region).
        eloreta_weight_path: optional .mat with key 'W' (V,3,3) eLORETA weights.
        head_model:     EEGLAB Colin27 head-model .mat path.
    """

    def __init__(self, subject_folder, leadfield_path, ch_list='tshinghwa',
                 is_shuffle=True, is_npy=True, leadfield_reduction=True,
                 head_model=DEFAULT_HEAD_MODEL, eloreta_weight_path=None):
        self._load_head_model(head_model)
        self.leadfield_path = leadfield_path
        self.leadfield_reduction = leadfield_reduction

        self.cortex_pos = self._aligned_cortex()
        self.source_pos = self.cortex_pos
        self.sensor_pos = self._aligned_sensors()

        if self.leadfield_reduction:
            self.source_pos, self.indices = self.reduce_sources_proportionally(1000)

        self.ch_list = ch_list
        self.use_ch = CHANNEL_LISTS[ch_list]

        if eloreta_weight_path is not None:
            self.eloreta_weight = scipy.io.loadmat(eloreta_weight_path)['W']

        # leadfield group id: 0 = tsinghua montage, 1 = resting montage
        self.leadfield_id_fixed = 0 if self.ch_list == 'tshinghwa' else 1

        self.use_sensor_pos = np.stack(
            [self.sensor_pos[self.labels.index(label)] for label in self.use_ch])

        # EEG trials
        if is_npy:
            self.eeg_data = np.load(subject_folder)
        else:
            self.eeg_data = np.concatenate(
                [scipy.io.loadmat(f)['data'][:, :, :200] for f in subject_folder], axis=0)
        if is_shuffle:
            np.random.shuffle(self.eeg_data)

        leadfield = scipy.io.loadmat(self.leadfield_path)['leadfield'][0]
        self.leadfield = np.stack(leadfield).transpose(1, 0, 2)  # (ch, V, 3)
        if self.leadfield_reduction:
            self.leadfield = self.leadfield[:, self.indices, :]

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        if self.ch_list == 'tshinghwa':
            eeg = np.delete(eeg, [59, 63], axis=0)  # drop CB1/CB2

        eeg = torch.tensor(eeg, dtype=torch.float32)
        mean = torch.mean(eeg, axis=(0, 1), keepdims=True)
        std = torch.std(eeg, axis=(0, 1), keepdims=True).clamp(min=1e-6)
        eeg = (eeg - mean) / std

        return {
            'src': eeg,
            'leadfield': torch.tensor(self.leadfield, dtype=torch.float32),
            'leadfield_id': torch.tensor(self.leadfield_id_fixed, dtype=torch.long),
            'sensor_pos': torch.tensor(self.use_sensor_pos, dtype=torch.float32),
            'source_pos': torch.tensor(self.source_pos, dtype=torch.float32),
            'eloreta_weight': torch.tensor(self.eloreta_weight, dtype=torch.float32)
                              if hasattr(self, 'eloreta_weight') else None,
        }


class EEG2Source2EEG_epilepsy_Dataset(_HeadModelMixin, Dataset):
    """BUH epilepsy dataset: one subfolder per subject, multiple .mat recordings.

    Each subfolder holds channame.mat, leadfield.mat, eloreta_weight.mat and
    eegdata_S{i}.mat files (100 Hz), windowed into 2 s (200 sample) segments.
    """

    def __init__(self, folder_path=None, leadfield_reduction=True,
                 use_eloreta_weight=False, head_model=DEFAULT_HEAD_MODEL):
        self.leadfield_id_fixed = 2  # third leadfield type
        self.leadfield_reduction = leadfield_reduction
        self.use_eloreta_weight = use_eloreta_weight
        self._load_head_model(head_model)
        self.labels = [ch.lower() for ch in self.labels]

        self.cortex_pos = self._aligned_cortex()
        self.source_pos = self.cortex_pos
        self.sensor_pos = self._aligned_sensors()
        if self.leadfield_reduction:
            self.source_pos, self.indices = self.reduce_sources_proportionally(1000)

        self.folder_path = folder_path
        replace_map = {'t5': 'p7', 't6': 'p8', 't3': 't7', 't4': 't8'}
        folders = [e for e in os.listdir(folder_path)
                   if os.path.isdir(os.path.join(folder_path, e))]

        self.eeg_data_list, self.sensor_pos_list = [], []
        self.source_pos_list, self.leadfield_list = [], []
        self.eloreta_weight_list, self.use_ch = [], []

        for folder in folders:
            subfolder_path = f'{folder_path}\\{folder}'
            ch_name = scipy.io.loadmat(f'{subfolder_path}\\channame.mat')['chan_name'][0]
            ch_name = [name.item().lower() for name in ch_name]
            print(subfolder_path)
            print(f"Use channel: {ch_name}")

            leadfield = scipy.io.loadmat(f'{subfolder_path}\\leadfield.mat')['leadfield'][0]
            leadfield = np.stack(leadfield).transpose(1, 0, 2)
            if self.leadfield_reduction:
                leadfield = leadfield[:, self.indices, :]

            if use_eloreta_weight:
                eloreta_weight = scipy.io.loadmat(
                    f'{subfolder_path}\\eloreta_weight.mat')['eloreta_w']

            use_sensor_pos = []
            for label in ch_name:
                label_std = replace_map.get(label, label)
                idx = self.labels.index(label_std)
                self.use_ch.append(label)
                use_sensor_pos.append(self.sensor_pos[idx, :])
            use_sensor_pos = np.stack(use_sensor_pos)

            srate, window_size = 100, 2
            for i in range(1, 10):
                eeg_path = f'{subfolder_path}\\eegdata_S{i}.mat'
                if not os.path.exists(eeg_path):
                    print(f'Not found: {eeg_path}')
                    break
                eeg_data = scipy.io.loadmat(eeg_path)['eegdata']
                eeg_len = int(eeg_data.shape[1] / srate)
                for j in range(0, eeg_len, 2):
                    start_ind = j * srate
                    temp_eeg = eeg_data[:, start_ind:start_ind + window_size * srate]
                    if temp_eeg.shape[-1] != 200:
                        break
                    self.eeg_data_list.append(temp_eeg)
                    self.sensor_pos_list.append(use_sensor_pos)
                    self.source_pos_list.append(self.source_pos)
                    self.leadfield_list.append(leadfield)
                    if self.use_eloreta_weight:
                        self.eloreta_weight_list.append(eloreta_weight)

    def __len__(self):
        return len(self.eeg_data_list)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data_list[idx], dtype=torch.float32)
        mean = torch.mean(eeg, axis=(0, 1), keepdims=True)
        std = torch.std(eeg, axis=(0, 1), keepdims=True).clamp(min=1e-6)
        eeg = (eeg - mean) / std
        return {
            'src': eeg,
            'leadfield': torch.tensor(self.leadfield_list[idx], dtype=torch.float32),
            'leadfield_id': torch.tensor(self.leadfield_id_fixed, dtype=torch.long),
            'sensor_pos': torch.tensor(self.sensor_pos_list[idx], dtype=torch.float32),
            'source_pos': torch.tensor(self.source_pos_list[idx], dtype=torch.float32),
            'eloreta_weight': torch.tensor(self.eloreta_weight_list[idx], dtype=torch.float32)
                              if self.use_eloreta_weight else None,
        }


class HISDataset_BIDS(Dataset):
    """localize-mi intracerebral-stimulation BIDS dataset (fine-tune / eval).

    Reads per-subject leadfield / electrode / eLORETA files from
    ``derivatives/epochs`` and the epoched EEG runs via ``load_bids``.  Filenames
    follow ``{subj}_task-{task}_{source_model}_{align_method}_*.mat``.
    """

    def __init__(self,
                 dir_bids=None,
                 task='seegstim', subjects=None, use_eloreta_weight=False,
                 use_source_model='surface', use_elect_align_method='traditional',
                 window_size=2, use_personalized_leadfield=False):
        self.dir_bids = dir_bids
        self.task = task
        self.leadfield_id_fixed = 2
        self.use_eloreta_weight = use_eloreta_weight
        self.window_size = window_size
        self.use_personalized_leadfield = use_personalized_leadfield
        self.use_source_model = use_source_model
        self.use_elect_align_method = use_elect_align_method

        if use_source_model == 'surface':
            self.source_pos_path = os.path.join(dir_bids, 'HIS_dataset_task-seegstim_source_pos.mat')
        elif use_source_model == 'volume':
            self.source_pos_path = os.path.join(dir_bids, 'HIS_dataset_task-seegstim_source_model_loretakey_pos.mat')
        self.source_pos = scipy.io.loadmat(self.source_pos_path)['source_transform']

        if subjects is None:
            subjects = [f'sub-{i:02d}' for i in range(1, 8)]
        self.subjects = subjects

        self.eeg_data_list, self.sensor_pos_list = [], []
        self.source_pos_list, self.leadfield_list = [], []
        self.eloreta_weight_list, self.metadata_list = [], []

        for subj in self.subjects:
            print(f"Loading {subj}... source_model={use_source_model}, align={use_elect_align_method}")
            base = os.path.join(dir_bids, 'derivatives', 'epochs', subj, 'eeg')

            if self.use_personalized_leadfield:
                fwd_path = os.path.join(dir_bids, 'derivatives', 'sourcemodelling',
                                        subj, 'fwd', f'{subj}_fwd.fif')
                if not os.path.exists(fwd_path):
                    print(f"Warning: Forward model not found for {subj}, skipping...")
                    continue
                fwd = mne.read_forward_solution(fwd_path)
                leadfield = fwd['sol']['data']
                n_channels = leadfield.shape[0]
                n_sources = leadfield.shape[1] // 3
                leadfield = leadfield.reshape(n_channels, n_sources, 3)
            else:
                leadfield_filename = f'{subj}_task-{task}_{use_source_model}_{use_elect_align_method}_leadfield.mat'
                fwd_path = os.path.join(base, leadfield_filename)
                if not os.path.exists(fwd_path):
                    print(f"Warning: Leadfield not found: {fwd_path}, skipping...")
                    continue
                leadfield = scipy.io.loadmat(fwd_path)['leadfield_to_save'][0]
                leadfield = np.stack(leadfield).transpose(1, 0, 2)
                print(f"  Leadfield shape: {leadfield.shape}")

            if use_eloreta_weight:
                eloreta_weight_filename = f'{subj}_task-{task}_{use_source_model}_{use_elect_align_method}_eloreta_weight.mat'
                eloreta_weight_path = os.path.join(base, eloreta_weight_filename)
                if not os.path.exists(eloreta_weight_path):
                    print(f"Warning: eLoreta weight not found: {eloreta_weight_path}, skipping...")
                    continue
                eloreta_weight = scipy.io.loadmat(eloreta_weight_path)['eloreta_weight']

            elec_filename = f'{subj}_task-{task}_{use_elect_align_method}_elec_pos.mat'
            electrodes_path = os.path.join(base, elec_filename)
            if not os.path.exists(electrodes_path):
                print(f"Warning: Electrode positions not found: {electrodes_path}, skipping...")
                continue
            sensor_pos = scipy.io.loadmat(electrodes_path)['elec_transform_to_save'] / 1000.0  # mm->m

            epoch_files = [f for f in os.listdir(base) if f.endswith('_epochs.npy')]
            runs = sorted({f.split('_')[2] for f in epoch_files})  # 'run-XX'

            load_bids = _load_bids_fn()
            for run in runs:
                try:
                    epo = load_bids(dir_bids, subj, task, run).resample(sfreq=1000)
                    eeg_data = epo.get_data()  # (n_epochs, n_channels, n_times)
                    window_samples = int(self.window_size)
                    for epoch_idx in range(eeg_data.shape[0]):
                        epoch_data = eeg_data[epoch_idx]
                        if epoch_data.shape[1] >= window_samples:
                            self.eeg_data_list.append(epoch_data[:, -window_samples:])
                            self.sensor_pos_list.append(sensor_pos)
                            self.source_pos_list.append(self.source_pos)
                            self.leadfield_list.append(leadfield)
                            if self.use_eloreta_weight:
                                self.eloreta_weight_list.append(eloreta_weight)
                            self.metadata_list.append({
                                'subject': subj, 'run': run, 'epoch': epoch_idx,
                                'source_model': use_source_model,
                                'align_method': use_elect_align_method,
                            })
                    print(f"  Loaded {run}: {eeg_data.shape[0]} epochs")
                except Exception as e:  # noqa: BLE001 - keep loading remaining runs
                    print(f"Error loading {subj} {run}: {e}")
                    continue

        print(f"\nTotal loaded: {len(self.eeg_data_list)} samples "
              f"(source_model={use_source_model}, align={use_elect_align_method})")

    def reduce_sources_proportionally(self, target_n_sources):
        n_sources = self.source_pos.shape[0]
        step = n_sources / target_n_sources
        indices = np.round(np.arange(0, n_sources, step)).astype(int)[:target_n_sources]
        return self.source_pos[indices], indices

    def __len__(self):
        return len(self.eeg_data_list)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data_list[idx], dtype=torch.float32)
        mean = torch.mean(eeg, dim=(0, 1), keepdim=True)
        std = torch.std(eeg, dim=(0, 1), keepdim=True).clamp(min=1e-6)
        eeg = (eeg - mean) / std
        return {
            'src': eeg,
            'leadfield': torch.tensor(self.leadfield_list[idx], dtype=torch.float32),
            'leadfield_id': torch.tensor(self.leadfield_id_fixed, dtype=torch.long),
            'sensor_pos': torch.tensor(self.sensor_pos_list[idx], dtype=torch.float32),
            'source_pos': torch.tensor(self.source_pos_list[idx], dtype=torch.float32),
            'eloreta_weight': torch.tensor(self.eloreta_weight_list[idx], dtype=torch.float32)
                              if self.use_eloreta_weight else None,
            'metadata': self.metadata_list[idx],
        }


class MergeDataset(Dataset):
    """Concatenate several datasets behind a single index space."""

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.cumulative_lengths = [0]
        for dataset in dataset_list:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
        self.len_ = self.cumulative_lengths[-1]

    def map_idx(self, idx):
        list_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        return list_idx, idx - self.cumulative_lengths[list_idx]

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        list_idx, item_idx = self.map_idx(idx)
        return self.dataset_list[list_idx][item_idx]


def random_subset(ds, n, seed=42):
    """Return a torch Subset of ``ds`` with up to ``n`` randomly chosen items."""
    import random
    length = len(ds)
    if length == 0:
        return Subset(ds, [])
    k = min(n, length)
    idx = list(range(length))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:k])
