"""EEG → Source → EEG self-supervised source-localization model.

Public API re-exports the pieces a typical pipeline needs:

    from eeg_sl import SLTConfig, EEG2Source2EEG_autoregressive
    from eeg_sl import EEG_2_Source_DataCollector, EEG2Source2EEG_Dataset

See the top-level README for the architecture overview and how to run
pre-training / fine-tuning / inference.
"""

from .models.config import SLTConfig
from .models.eeg2source2eeg import EEG2Source2EEG_autoregressive
from .data.collator import EEG_2_Source_DataCollector
from .data.datasets import (
    EEG2Source2EEG_Dataset,
    EEG2Source2EEG_epilepsy_Dataset,
    HISDataset_BIDS,
    MergeDataset,
    random_subset,
)

__all__ = [
    "SLTConfig",
    "EEG2Source2EEG_autoregressive",
    "EEG_2_Source_DataCollector",
    "EEG2Source2EEG_Dataset",
    "EEG2Source2EEG_epilepsy_Dataset",
    "HISDataset_BIDS",
    "MergeDataset",
    "random_subset",
]
