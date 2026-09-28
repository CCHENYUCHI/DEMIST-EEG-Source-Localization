"""Fine-tune a pre-trained checkpoint on the localize-mi (HIS) subjects.

    python scripts/finetune.py --config configs/finetune.yaml

Loads weights from a pre-training checkpoint, then continues training on real
intracerebral-stimulation EEG with the same self-supervised objective.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from datetime import datetime

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainingArguments

from eeg_sl.config_io import load_config
from eeg_sl.data import EEG_2_Source_DataCollector, HISDataset_BIDS
from eeg_sl.metrics import make_compute_metrics
from eeg_sl.models import SLTConfig, EEG2Source2EEG_autoregressive


def main():
    cfg = load_config("Fine-tune EEG2Source2EEG on HIS data")
    pc, dc, tc = cfg["pretrained"], cfg["data"], cfg["training"]

    # --- load pre-trained model ---
    checkpoint_path = pc["checkpoint_path"]
    print(f"Loading pre-trained model from: {checkpoint_path}")
    config = SLTConfig.from_pretrained(checkpoint_path)
    if pc.get("reg_alpha_override") is not None:
        config.reg_alpha = pc["reg_alpha_override"]
    model = EEG2Source2EEG_autoregressive.from_pretrained(checkpoint_path, config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- datasets (one per subject, concatenated) ---
    print(f"Loading HIS subjects: {dc['subjects']}")
    datasets = [
        HISDataset_BIDS(
            dir_bids=dc["dir_bids"], task=dc["task"], subjects=[subj],
            use_source_model=dc["use_source_model"],
            use_elect_align_method=dc["use_elect_align_method"],
            use_eloreta_weight=dc["use_eloreta_weight"], window_size=dc["window_size"])
        for subj in dc["subjects"]
    ]
    train_dataset = ConcatDataset(datasets)
    print(f"Total training samples: {len(train_dataset)}")

    # --- output / logging ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['run_name_prefix']}_{stamp}"
    output_dir = os.path.join(tc["output_base_dir"], run_name)

    wc = cfg.get("wandb", {})
    if wc.get("enabled", False):
        import wandb
        wandb.init(project=wc.get("project", "finetune_source"), name=run_name, reinit=True)
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        num_train_epochs=tc["num_epochs"],
        learning_rate=tc["learning_rate"],
        save_strategy="epoch",
        eval_strategy="epoch",
        eval_accumulation_steps=4,
        batch_eval_metrics=True,
        logging_steps=50,
        report_to=report_to,
        bf16=tc["bf16"],
        fp16=tc["fp16"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,   # swap in a held-out split if you have one
        data_collator=EEG_2_Source_DataCollector(),
        compute_metrics=make_compute_metrics(reg_alpha=config.reg_alpha),
    )

    print("Starting fine-tuning ...")
    trainer.train()

    final_save_path = os.path.join(output_dir, "final_finetuned_model")
    trainer.save_model(final_save_path)
    print(f"Fine-tuned model saved to: {final_save_path}")

    if wc.get("enabled", False):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
