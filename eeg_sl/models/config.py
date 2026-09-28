"""Model configuration.

`SLTConfig` is a HuggingFace `PretrainedConfig`, so it serialises to
`config.json` and is loaded automatically by `from_pretrained`.

Only the fields actually consumed by `EEG2Source2EEG_autoregressive` are kept
here; unused fields from earlier iterations are not carried over.
"""

import warnings

from transformers import PretrainedConfig


class SLTConfig(PretrainedConfig):
    """Configuration for the EEG2Source2EEG autoregressive model.

    Args:
        d_model:            transformer hidden size.
        d_ff:               feed-forward inner size.
        h:                  number of attention heads.
        dropout:            dropout probability.
        N:                  number of encoder AND decoder layers. This is the
                            field that actually controls backbone depth.
        encoder_n, decoder_n:
                            LEGACY / UNUSED. Released checkpoints carry 4 here,
                            but the model has always built its main encoder and
                            decoder from ``config.N`` (default 6), so these have
                            never taken effect. They are kept only so existing
                            ``config.json`` files still load. To change depth,
                            set ``N``.
        N_feature_encoder:  depth of the small leadfield/feature encoder.
        sensor_time:        EEG time length fed into the encoder projector
                            (e.g. 200 samples).
        source_voxel_time:  generator output width per source token.  For a
                            (x,y,z) × T layout this is ``3 * T`` (e.g. 600 = 3*200).
        tgt_len:            number of source voxels (e.g. 5003 or reduced count).
        K:                  number of source voxels decoded per autoregressive step.
        use_embedding:      build source tokens via an embedding path (True for
                            the leadfield-projection model).
        embedding_type:     "leadfield_projection" (default, what is trained) or
                            "encoder".
        pos_embedding_type: "Learned" (MLP over 3-D positions) or "sinusoid".
        use_generator:      "linear" (default), "MLP", "MLP_Plus" or "hybrid".
        sdp_attention:      use the scaled-dot-product / flash-attention block
                            instead of the vanilla multi-head attention.
        regularization_loss: source regulariser — "eloreta", "L1", "L2" or "None".
        reg_alpha:          weight of the regularisation term in the total loss.
    """

    model_type = "SLT"
    _depth_warned = False

    def __init__(
        self,
        # ---- transformer backbone ----
        d_model=256,
        d_ff=1024,
        h=8,
        dropout=0.1,
        N=6,
        encoder_n=4,
        decoder_n=4,
        N_feature_encoder=1,
        # ---- sequence dimensions ----
        sensor_time=200,
        source_voxel_time=600,
        tgt_len=5003,
        K=150,
        # ---- architecture switches ----
        use_embedding=True,
        embedding_type="leadfield_projection",
        pos_embedding_type="Learned",
        use_generator="linear",
        sdp_attention=False,
        # ---- loss ----
        regularization_loss="eloreta",
        reg_alpha=0.1,
        loss_coe=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        # The model builds its main encoder/decoder with `config.N`.
        # `encoder_n`/`decoder_n` are stored only for config compatibility and
        # have never had any effect.  Released checkpoints carry 4 here while
        # the backbone is 6 layers deep, so say it out loud once per process
        # rather than let a reader take the stored value at face value.
        self.N = N
        self.encoder_n = encoder_n
        self.decoder_n = decoder_n
        if (encoder_n != N or decoder_n != N) and not SLTConfig._depth_warned:
            SLTConfig._depth_warned = True
            warnings.warn(
                f'SLTConfig: encoder_n={encoder_n} / decoder_n={decoder_n} are '
                f'legacy fields and are IGNORED; the encoder and decoder are '
                f'both N={N} layers deep. Set N to change depth.',
                stacklevel=2)
        self.N_feature_encoder = N_feature_encoder

        self.sensor_time = sensor_time
        self.source_voxel_time = source_voxel_time
        self.tgt_len = tgt_len
        self.K = K

        self.use_embedding = use_embedding
        self.embedding_type = embedding_type
        self.pos_embedding_type = pos_embedding_type
        self.use_generator = use_generator
        self.sdp_attention = sdp_attention

        self.regularization_loss = regularization_loss
        self.reg_alpha = reg_alpha
        self.loss_coe = loss_coe
