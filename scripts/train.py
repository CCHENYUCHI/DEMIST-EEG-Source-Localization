"""Pre-train the EEG -> Source -> EEG model (self-supervised).

    python scripts/train.py --config configs/pretrain.yaml

Mixes the enabled data sources (SSVEP / resting / epilepsy / simulated), caps
each to ``n_per_dataset_*`` random samples, then trains with the HuggingFace
Trainer.  See configs/pretrain.yaml for all knobs.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from datetime import datetime

import _bootstrap  # noqa: F401  (sets sys.path)
import torch
from transformers import Trainer, TrainingArguments

from eeg_sl.config_io import load_config
from eeg_sl.data import (
    EEG_2_Source_DataCollector,
    EEG2Source2EEG_Dataset,
    EEG2Source2EEG_epilepsy_Dataset,
    MergeDataset,
    random_subset,
)
from eeg_sl.metrics import make_compute_metrics
from eeg_sl.models import SLTConfig, EEG2Source2EEG_autoregressive


def build_datasets(dc, head_model, train: bool):
    """Build the list of (capped) subsets for either the train or val split."""
    subsets = []
    cap = dc["n_per_dataset_train"] if train else dc["n_per_dataset_val"]
    split = "train" if train else "val"
    lf_reduction = dc["leadfield_reduction"]
    seed_base = 2025_0901 + (0 if train else 10)

    if dc["use_ssvep"]:
        s = dc["ssvep"]
        ds = EEG2Source2EEG_Dataset(
            s[f"{split}_path"], s["leadfield_path"], ch_list="tshinghwa",
            leadfield_reduction=lf_reduction, eloreta_weight_path=s["eloreta_weight_path"],
            head_model=head_model)
        subsets.append(random_subset(ds, cap, seed=seed_base + 0))

    if dc["use_resting"]:
        s = dc["resting"]
        ds = EEG2Source2EEG_Dataset(
            s[f"{split}_path"], s["leadfield_path"], ch_list="resting",
            leadfield_reduction=lf_reduction, eloreta_weight_path=s["eloreta_weight_path"],
            head_model=head_model)
        subsets.append(random_subset(ds, cap, seed=seed_base + 1))

    if dc["use_epilepsy"]:
        s = dc["epilepsy"]
        ds = EEG2Source2EEG_epilepsy_Dataset(
            folder_path=s[f"{split}_folder"], use_eloreta_weight=True,
            leadfield_reduction=lf_reduction, head_model=head_model)
        subsets.append(random_subset(ds, cap, seed=seed_base + 2))

    if dc["use_simulated"]:
        s = dc["simulated"]
        ds = EEG2Source2EEG_Dataset(
            s[f"{split}_paths"], s["leadfield_path"], is_npy=False, is_shuffle=True,
            ch_list="tshinghwa", leadfield_reduction=lf_reduction,
            eloreta_weight_path=dc["ssvep"]["eloreta_weight_path"], head_model=head_model)
        subsets.append(random_subset(ds, cap, seed=seed_base + 3))

    if not subsets:
        raise ValueError("No dataset enabled in config under `datasets.use_*`.")
    return MergeDataset(subsets)


def model_init_factory(mc):
    """Return a fresh-model initialiser for Trainer (Xavier init like the original)."""
    def model_init():
        config = SLTConfig(
            d_model=mc["d_model"], d_ff=mc["d_ff"], h=mc["h"], dropout=mc["dropout"],
            N=mc["N"], N_feature_encoder=mc["N_feature_encoder"],
            sensor_time=mc["sensor_time"], source_voxel_time=mc["source_voxel_time"],
            tgt_len=mc["tgt_len"], K=mc["K"], use_embedding=mc["use_embedding"],
            embedding_type=mc["embedding_type"], pos_embedding_type=mc["pos_embedding_type"],
            use_generator=mc["use_generator"],
            sdp_attention=mc["sdp_attention"], regularization_loss=mc["regularization_loss"],
            reg_alpha=mc["reg_alpha"])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = EEG2Source2EEG_autoregressive(config).to(device)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        return model
    return model_init


def main():
    cfg = load_config("Pre-train EEG2Source2EEG")
    dc, mc, tc = cfg["datasets"], cfg["model"], cfg["training"]

    head_model = dc["head_model_path"]
    print("Building training set ...")
    train_dataset = build_datasets(dc, head_model, train=True)
    print(f"Total train samples (capped): {len(train_dataset)}")
    print("Building validation set ...")
    val_dataset = build_datasets(dc, head_model, train=False)
    print(f"Total val samples (capped): {len(val_dataset)}")

    run_name = cfg["run_name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(tc["output_base_dir"], f"{run_name}_{stamp}")

    wc = cfg.get("wandb", {})
    if wc.get("enabled", False):
        import wandb
        if wc.get("api_key"):
            wandb.login(key=wc["api_key"])
        wandb.init(project=wc.get("project", "eeg_source"), name=run_name, reinit=True)
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    steps_per_epoch = max(1, len(train_dataset) // tc["batch_size"])
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        eval_accumulation_steps=4,
        batch_eval_metrics=True,
        num_train_epochs=tc["num_epochs"],
        weight_decay=tc["weight_decay"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        logging_steps=max(1, steps_per_epoch // 10),
        report_to=report_to,
        bf16=tc["bf16"],
        fp16=tc["fp16"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model_init=model_init_factory(mc),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=EEG_2_Source_DataCollector(),
        compute_metrics=make_compute_metrics(reg_alpha=mc["reg_alpha"]),
    )

    trainer.train()
    if wc.get("enabled", False):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
