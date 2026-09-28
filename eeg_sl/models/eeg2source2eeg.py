"""EEG -> Source -> EEG autoregressive model.

This is the only model actually trained / fine-tuned / used for inference in the
project.  It operates entirely in the time domain.  High-level flow::

    src (EEG)  --src_projector + sensor pos--> encoder  ----------------+
                                                                        | (memory)
    leadfield --per-source token + pos--> leadfield_encoder --> source tokens
                                                                        |
    autoregressive decoder over source voxels (K per step) <------------+
                       |
                  generator  -> source activity `hidden` (B, 3, V, T)
                       |
              leadfield forward projection -> reconstructed EEG `logits` (B, C, T)

Loss = EEG reconstruction MSE (channel-masked) + source regulariser
(eLORETA / L1 / L2).  This is fully self-supervised: the target is the input EEG
itself (``labels = src``), no ground-truth source is required.  The estimated
source is returned as ``hidden_states`` and is what inference extracts.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import SLTConfig
from .transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    HybridProjector,
    MLP_projector,
    MLPProjectorPlus,
    MultiHeadedAttention,
    MultiHeaded_FlashAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


class EEG2Source2EEG_autoregressive(PreTrainedModel):
    config_class = SLTConfig

    def __init__(self, config):
        super().__init__(config)
        self.c = copy.deepcopy
        self.config = config

        if self.config.sdp_attention is False:
            self.attn = MultiHeadedAttention(config.h, config.d_model)
        else:
            self.attn = MultiHeaded_FlashAttention(config.h, config.d_model)

        self.ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)

        if config.pos_embedding_type == 'Learned':
            # 3D position -> d_model
            self.pos_embedding = nn.Sequential(
                nn.Linear(3, 128),
                nn.Linear(128, 512),
                nn.Linear(512, config.d_model),
            )
        elif config.pos_embedding_type == 'sinusoid':
            self.pos_embedding_encoder = PositionalEncoding(config.d_model, config.dropout)
            self.pos_embedding_decoder = PositionalEncoding(config.d_model, config.dropout)

        if self.config.use_embedding is True:
            if getattr(self.config, "embedding_type", None) == "leadfield_projection":
                # project each token's 3-D leadfield feature to a small dim d=32
                self.leadfield_projector_1 = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.SiLU(),
                    nn.Linear(32, 32),
                )
                self.pos_embedding_leadfield_encoder = PositionalEncoding(32, config.dropout)
                self.leadfield_pos_embedding = nn.Linear(3, 32)

                # encoder over the (1 + ch_n) sequence, one per source
                self.leadfield_encoder = Encoder(
                    EncoderLayer(32, MultiHeadedAttention(1, 32),
                                 PositionwiseFeedForward(32, 128, config.dropout), config.dropout),
                    config.N_feature_encoder
                )
                self.leadfield_projector_2 = nn.Linear(32, config.d_model)

            if self.config.embedding_type == "encoder":
                self.feature_encoder = Encoder(
                    EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff), config.dropout),
                    config.N_feature_encoder)

        self.encoder = Encoder(
            EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff), config.dropout), config.N)
        self.decoder = Decoder(
            DecoderLayer(config.d_model, self.c(self.attn), self.c(self.attn), self.c(self.ff),
                         config.dropout), config.N)

        self.z_prev_token = nn.Parameter(torch.randn(1, 1, self.config.d_model))

        if self.config.use_embedding is False:
            self.tgt_projector = MLP_projector(config.source_voxel_time, config.d_ff, config.d_model)

        self.src_projector = MLP_projector(config.sensor_time, config.d_ff, config.d_model)
        self.src_projector_feature = MLP_projector(config.sensor_time, config.d_ff, config.d_model)

        self.generator = nn.Linear(config.d_model, config.source_voxel_time)
        if self.config.use_generator == "hybrid":
            self.generator = HybridProjector(config.d_model, 256 * 3, config.source_voxel_time, K=128)
        if self.config.use_generator == "MLP_Plus":
            self.generator = MLPProjectorPlus(config.d_model, 256, config.sensor_time)
        if self.config.use_generator == "MLP":
            self.generator = MLP_projector(config.d_model, 256, config.sensor_time)
        if self.config.use_generator == "linear":
            self.generator = nn.Linear(config.d_model, config.source_voxel_time)

        self.loss_fct = nn.MSELoss()

    def forward(self,
                src=torch.FloatTensor,
                tgt=torch.FloatTensor,
                src_mask: Optional[torch.FloatTensor] = None,
                tgt_mask: Optional[torch.FloatTensor] = None,
                sensor_pos: Optional[torch.FloatTensor] = None,
                source_pos: Optional[torch.FloatTensor] = None,
                leadfield: Optional[torch.FloatTensor] = None,
                leadfield_id: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                eloreta_weight: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                ):
        # ---- EEG encoder ----
        src_embedding = self.src_projector(src)  # (B, ch_n, T) -> (B, ch_n, d_model)
        if self.config.pos_embedding_type == 'Learned':
            src_embedding = src_embedding + self.pos_embedding(sensor_pos)
        elif self.config.pos_embedding_type == 'sinusoid':
            src_embedding = self.pos_embedding_encoder(src_embedding)

        encoder_output, encoder_atten = self.encoder(src_embedding, src_mask.unsqueeze(1))

        # ---- source token construction ----
        if self.config.embedding_type == "leadfield_projection":
            assert leadfield_id is not None, "need leadfield_id to group per-batch"

            B, C_pad, S, _ = leadfield.shape         # leadfield: (B, C_pad, S, 3)
            d = 32
            device = leadfield.device
            dtype = leadfield.dtype

            def build_tokens_for_sample(lf_b, sensor_pos_b, source_pos_b, src_mask_b):
                # lf_b: (C_pad, S, 3); anchor token is all-zeros, placed first
                anchor_feat = torch.zeros(S, 1, 3, device=device, dtype=dtype)
                lf_per_source = lf_b.permute(1, 0, 2).contiguous()             # (S, C_pad, 3)
                lf_tokens_3d = torch.cat([anchor_feat, lf_per_source], dim=1)  # (S, 1+C_pad, 3)

                src_pos_tok = source_pos_b.unsqueeze(1)                        # (S, 1, 3)
                sen_pos_tok = sensor_pos_b.unsqueeze(0).expand(S, C_pad, 3)    # (S, C_pad, 3)
                pos_tokens_3d = torch.cat([src_pos_tok, sen_pos_tok], dim=1)   # (S, 1+C_pad, 3)

                ch_mask = torch.ones(1 + C_pad, device=device, dtype=torch.bool)
                ch_mask[0] = True            # anchor always kept
                ch_mask[1:] = src_mask_b     # real channels True, padding False
                return lf_tokens_3d, pos_tokens_3d, ch_mask

            # up to three unique leadfield types per batch; pick a representative sample for each
            uniques = torch.unique(leadfield_id).tolist()[:3]
            group_to_indices = {t: (leadfield_id == t).nonzero(as_tuple=True)[0] for t in uniques}
            reps = {t: group_to_indices[t][0].item() for t in uniques}

            lf_batch, pos_batch, kpm_batch = [], [], []
            for t in uniques:
                b0 = reps[t]
                # NOTE: this used to pass the literal 32 in place of the mask, so
                # `ch_mask[1:] = 32` filled the slice with True and padded
                # channels were never masked inside the leadfield encoder.  It
                # is a no-op whenever a batch holds a single montage (every
                # inference batch), and only bites when one batch mixes channel
                # counts, i.e. during multi-montage pre-training.
                lf_tokens_3d, pos_tokens_3d, ch_mask = build_tokens_for_sample(
                    leadfield[b0], sensor_pos[b0], source_pos[b0],
                    src_mask[b0].bool())
                lf_batch.append(lf_tokens_3d)
                pos_batch.append(pos_tokens_3d)
                kpm = (~ch_mask).unsqueeze(0).expand(S, -1)     # (S, 1+C_pad) True=mask
                kpm_batch.append(kpm)

            lf_batch = torch.stack(lf_batch, dim=0)             # (G, S, 1+C_pad, 3)
            pos_batch = torch.stack(pos_batch, dim=0)           # (G, S, 1+C_pad, 3)
            kpm_batch = torch.stack(kpm_batch, dim=0)           # (G, S, 1+C_pad)

            G = lf_batch.shape[0]
            lf_tokens_embed = self.leadfield_projector_1(lf_batch)   # (G, S, 1+C_pad, d)

            if self.config.pos_embedding_type == 'Learned':
                pos_tokens_embed = self.leadfield_pos_embedding(pos_batch)
                leadfield_embedding = lf_tokens_embed + pos_tokens_embed
            elif self.config.pos_embedding_type == 'sinusoid':
                leadfield_embedding = self.pos_embedding_leadfield_encoder(lf_tokens_embed, dim=4)

            GS = G * S
            leadfield_seq = leadfield_embedding.reshape(GS, 1 + C_pad, d)
            key_padding_mask = kpm_batch.reshape(GS, 1 + C_pad)

            enc_out, _ = self.leadfield_encoder(leadfield_seq, key_padding_mask.unsqueeze(1))
            anchor = enc_out[:, 0, :].reshape(G, S, d)                            # (G, S, d)

            leadfield_feature = torch.zeros(B, S, d, device=device, dtype=enc_out.dtype)
            for gi, t in enumerate(uniques):
                idx = group_to_indices[t]
                leadfield_feature[idx] = anchor[gi].unsqueeze(0).expand(len(idx), S, d)
            leadfield_feature = self.leadfield_projector_2(leadfield_feature)     # (B, S, d_model)

        if self.config.embedding_type == "encoder":
            eeg_feature = self.src_projector_feature(src)
            eeg_feature, encoder_atten = self.feature_encoder(eeg_feature, src_mask.unsqueeze(1))
            eeg_feature = eeg_feature[:, 0, :].view(eeg_feature.shape[0], -1, self.config.d_model)

        if self.config.use_embedding is False:
            tgt_embedding = self.tgt_projector(tgt)

        # ---- autoregressive decoder over source voxels (K per step) ----
        if self.config.use_embedding is True:
            B, N_src, D = source_pos.shape
            step_num = N_src // self.config.K
            remainder = N_src % self.config.K
            split_sizes = [self.config.K] * (step_num - 1) + [self.config.K + remainder]
            source_pos_steps = torch.split(source_pos, split_sizes, dim=1)

            z_prev = self.z_prev_token.expand(B, 1, -1)

            if self.config.embedding_type == "leadfield_projection":
                outputs = []
                memory_bank = leadfield_feature  # (B, S, d_model)
                for k in range(step_num):
                    pos_step = source_pos_steps[k]
                    start_idx = sum(split_sizes[:k])
                    end_idx = sum(split_sizes[:k + 1])
                    relevant_memory = memory_bank[:, start_idx:end_idx, :]

                    if self.config.pos_embedding_type == 'Learned':
                        pos_embed = self.pos_embedding(pos_step)
                        tgt_step = relevant_memory + pos_embed
                    elif self.config.pos_embedding_type == 'sinusoid':
                        tgt_step = self.pos_embedding_decoder(relevant_memory)

                    tgt_input = torch.cat([z_prev, tgt_step], dim=1)
                    decoder_output, _ = self.decoder(
                        tgt_input, encoder_output, src_mask.unsqueeze(1), None)
                    z_prev = decoder_output[:, -1, :].unsqueeze(1)
                    outputs.append(decoder_output[:, 1:, :])
                decoder_output = torch.cat(outputs, dim=1)  # (B, N, d_model)

            if self.config.embedding_type == "encoder":
                outputs = []
                for k in range(step_num):
                    tgt_embedding = eeg_feature.repeat(1, source_pos_steps[k].shape[1], 1)
                    tgt_step = tgt_embedding
                    pos_step = source_pos_steps[k]
                    if self.config.pos_embedding_type == 'Learned':
                        pos_embed = self.pos_embedding(pos_step)
                        tgt_with_pos = tgt_step[:, :pos_step.shape[1], :] + pos_embed
                    elif self.config.pos_embedding_type == 'sinusoid':
                        tgt_with_pos = self.pos_embedding_decoder(tgt_step[:, :pos_step.shape[1], :])
                    tgt_input = torch.cat([z_prev, tgt_with_pos], dim=1)
                    decoder_output, _ = self.decoder(
                        tgt_input, encoder_output, src_mask.unsqueeze(1), None)
                    z_prev = decoder_output[:, -1, :].unsqueeze(1)
                    outputs.append(decoder_output[:, 1:, :])
                decoder_output = torch.cat(outputs, dim=1)

        # ---- generate source activity and forward-project back to EEG ----
        hidden = self.generator(decoder_output)
        # (B, N, 3*T) -> (B, 3, N, T)
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1], 3, 200).permute(0, 2, 1, 3)
        logits = torch.einsum('bjik,bkit->bjt', leadfield, hidden)  # -> EEG (B, C, T)

        if not return_dict:
            return logits

        loss = None
        if labels is not None:
            weights = src_mask.unsqueeze(-1)
            reconstruction_loss = torch.mean(((logits - src) ** 2) * weights)
            if self.config.regularization_loss == "L2":
                regularization_loss = torch.mean(torch.square(hidden))
                loss = reconstruction_loss + self.config.reg_alpha * regularization_loss
            elif self.config.regularization_loss == "L1":
                regularization_loss = torch.mean(torch.abs(hidden))
                loss = reconstruction_loss + self.config.reg_alpha * regularization_loss
            elif self.config.regularization_loss == "eloreta":
                # Source J, regulariser = J^T W J  (W = eLORETA weight per voxel)
                X = hidden.permute(0, 2, 3, 1)              # (B, V, T, 3)
                Wb = eloreta_weight.permute(0, 3, 1, 2)     # (B, V, 3, 3)
                reg_map = torch.einsum('bvti,bvij,bvtj->bvt', X, Wb, X)
                regularization_loss = reg_map.mean()
                loss = reconstruction_loss + self.config.reg_alpha * regularization_loss
            elif self.config.regularization_loss == "None":
                loss = reconstruction_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=(hidden),
            attentions=(encoder_atten),
        )
