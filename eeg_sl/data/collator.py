"""Batch collator for the EEG -> Source -> EEG model.

Pads the (variable) channel dimension across the batch, builds a channel mask,
and sets ``labels = src`` (self-supervised EEG reconstruction).  The returned
dict keys match the model's ``forward`` signature exactly so the HuggingFace
``Trainer`` can call ``model(**batch)``.
"""

import torch


class EEG_2_Source_DataCollector:
    def __call__(self, features):
        src, src_mask_list, tgt_mask_list = [], [], []
        sensor_pos_list, source_pos_list = [], []
        leadfield_list, leadfield_id_list, eloreta_weight_list = [], [], []

        pre_len = max(f["src"].shape[0] for f in features)  # max channel count
        feature_dim = features[0]["src"].shape[1]

        for f in features:
            eeg = f["src"]                  # (n_channels, T)
            eeg_pos = f['sensor_pos']
            source_pos = f['source_pos']
            channel_num = eeg.shape[0]
            leadfield = f['leadfield']      # (n_channels, V, 3)
            lf_id = f['leadfield_id']
            eloreta_weight = f['eloreta_weight']

            pad_len = pre_len - channel_num
            src_mask = torch.ones(eeg_pos.shape[0])
            tgt_mask = torch.ones(source_pos.shape[0])

            if pad_len > 0:
                eeg = torch.cat([eeg, torch.zeros((pad_len, feature_dim))], dim=0)
                eeg_pos = torch.cat([eeg_pos, torch.zeros((pad_len, 3))], dim=0)
                leadfield = torch.cat(
                    [leadfield, torch.zeros((pad_len, source_pos.shape[0], 3))], dim=0)
                src_mask = torch.cat([src_mask, torch.zeros(pad_len)], dim=0)

            src.append(eeg)
            sensor_pos_list.append(eeg_pos)
            source_pos_list.append(source_pos)
            leadfield_list.append(leadfield)
            leadfield_id_list.append(lf_id)
            src_mask_list.append(src_mask)
            tgt_mask_list.append(tgt_mask)
            if eloreta_weight is not None:
                eloreta_weight_list.append(eloreta_weight)

        src = torch.stack(src)
        sensor_pos = torch.stack(sensor_pos_list)
        source_pos = torch.stack(source_pos_list)
        leadfield = torch.stack(leadfield_list)
        leadfield_id = torch.stack(leadfield_id_list).long()
        src_mask_list = torch.stack(src_mask_list)
        tgt_mask_list = torch.stack(tgt_mask_list)
        if eloreta_weight_list:
            eloreta_weight_list = torch.stack(eloreta_weight_list)

        return {
            "src": src,
            "tgt": None,
            "labels": src,                      # self-supervised target
            "src_mask": src_mask_list,          # (B, n_channels)
            "tgt_mask": tgt_mask_list,          # (B, V)
            "sensor_pos": sensor_pos,           # (B, n_channels, 3)
            "source_pos": source_pos,           # (B, V, 3)
            "leadfield": leadfield,             # (B, n_channels, V, 3)
            "leadfield_id": leadfield_id,       # (B,)
            "eloreta_weight": eloreta_weight_list if len(eloreta_weight_list) > 0 else None,
            "return_dict": True,
        }
