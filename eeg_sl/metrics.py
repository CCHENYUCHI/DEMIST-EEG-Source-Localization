"""Evaluation metrics for the HuggingFace Trainer.

The model already returns the training loss internally; these metrics are only
used during ``trainer.evaluate`` (with ``batch_eval_metrics=True``, so they run
once per eval batch).  They re-compute the reconstruction + regulariser terms
for logging.
"""

import torch
import torch.nn as nn


def make_compute_metrics(reg_alpha=0.1):
    """Return a ``compute_metrics(pred_eval, compute_result)`` callable.

    Expects the model output tuple ``(logits, hidden)`` plus the batch inputs,
    matching ``batch_eval_metrics=True``.  Uses the eLORETA regulariser.
    """
    loss_fct = nn.MSELoss()

    def compute_metrics(pred_eval, compute_result):
        pred_tuple, labels, batch = pred_eval
        pred = pred_tuple[0]      # reconstructed EEG (logits)
        hidden = pred_tuple[1]    # source activity (B, 3, V, T)

        eloreta_weight = batch['eloreta_weight']
        src_mask = batch['src_mask']

        # eLORETA regulariser: J^T W J
        X = hidden.permute(0, 2, 3, 1)            # (B, V, T, 3)
        Wb = eloreta_weight.permute(0, 3, 1, 2)   # (B, V, 3, 3)
        reg_map = torch.einsum('bvti,bvij,bvtj->bvt', X, Wb, X)
        regularization_loss = reg_map.mean()

        weights = src_mask.unsqueeze(-1)
        reconstruction_loss = torch.mean(((pred - labels) ** 2) * weights)
        loss = reconstruction_loss + reg_alpha * regularization_loss

        return {
            'EEG_MSE': loss,
            'Reconstruction_Loss': reconstruction_loss,
            'Regularization_loss': regularization_loss,
        }

    return compute_metrics
