"""Run a trained model over HIS subjects and save estimated source activity.

    python scripts/inference.py --config configs/inference.yaml

For each subject the estimated source activity (the model's ``hidden_states``,
shape (B, 3, V, T)) is extracted, grouped by BIDS run, and saved as one .mat per
run under ``inference.output_dir``.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from collections import defaultdict

import _bootstrap  # noqa: F401
import torch
import tqdm
from scipy.io import savemat
from torch.utils.data import DataLoader

from eeg_sl.config_io import load_config
from eeg_sl.data import EEG_2_Source_DataCollector, HISDataset_BIDS
from eeg_sl.models import SLTConfig, EEG2Source2EEG_autoregressive


def load_model(model_path):
    print("=" * 90)
    print(f"Loading model from: {model_path}")
    config = SLTConfig.from_pretrained(model_path)
    model = EEG2Source2EEG_autoregressive.from_pretrained(model_path, config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model loaded. Device: {device}")
    return model, device


def extract_hidden_states(dataset, name, model, device, collator, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=num_workers, pin_memory=False)
    hidden_states_list, metadata_list = [], []
    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=f"Processing {name}"):
            outputs = model(
                src=batch['src'].to(device), sensor_pos=batch['sensor_pos'].to(device),
                source_pos=batch['source_pos'].to(device), src_mask=batch['src_mask'].to(device),
                tgt_mask=batch['tgt_mask'].to(device), leadfield=batch['leadfield'].to(device),
                leadfield_id=batch['leadfield_id'].to(device), labels=batch['labels'].to(device),
                eloreta_weight=batch['eloreta_weight'].to(device), return_dict=True)
            hidden_states_list.append(outputs['hidden_states'].detach().cpu())
            for _ in range(batch['src'].shape[0]):
                if sample_idx < len(dataset):
                    metadata_list.append(dataset[sample_idx].get('metadata', {}))
                    sample_idx += 1
                else:
                    metadata_list.append({})
    return torch.cat(hidden_states_list, dim=0), metadata_list


def group_by_run(all_hidden, metadata_list):
    run_indices, run_metadata = defaultdict(list), defaultdict(list)
    for idx, meta in enumerate(metadata_list):
        run = meta.get('run', 'unknown_run')
        run_indices[run].append(idx)
        run_metadata[run].append(meta)
    return {
        run: {'hidden_states': all_hidden[idxs], 'metadata': run_metadata[run]}
        for run, idxs in sorted(run_indices.items())
    }


def save_by_run(run_data, name, out_dir, source_model, align_method, save_format):
    dataset_dir = os.path.join(out_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)
    for run, data in run_data.items():
        filename = os.path.join(
            dataset_dir, f"{name}_{run}_finetuned_{source_model}_{align_method}.{save_format}")
        if save_format == 'mat':
            savemat(filename, {
                'source_voxel_data': data['hidden_states'].numpy(),
                'run': run, 'num_trials': len(data['metadata']),
            }, do_compression=True)
        print(f"Saved: {filename}")


def main():
    cfg = load_config("Inference: extract source activity")
    mc, dc, ic = cfg["model"], cfg["data"], cfg["inference"]

    model, device = load_model(mc["checkpoint_path"])
    collator = EEG_2_Source_DataCollector()
    os.makedirs(ic["output_dir"], exist_ok=True)

    for subj in dc["subjects"]:
        print(f"\nLoading dataset: {subj} ...")
        try:
            dataset = HISDataset_BIDS(
                dir_bids=dc["dir_bids"], task=dc["task"], subjects=[subj],
                use_source_model=dc["use_source_model"],
                use_elect_align_method=dc["use_elect_align_method"],
                use_eloreta_weight=dc["use_eloreta_weight"], window_size=dc["window_size"])
            all_hidden, all_meta = extract_hidden_states(
                dataset, subj, model, device, collator, ic["batch_size"], ic["num_workers"])
            grouped = group_by_run(all_hidden, all_meta)
            save_by_run(grouped, subj, ic["output_dir"],
                        dc["use_source_model"], dc["use_elect_align_method"], ic["save_format"])
            del dataset, all_hidden, all_meta, grouped
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {subj}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
