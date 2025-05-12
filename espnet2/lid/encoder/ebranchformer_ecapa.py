import logging
from typing import Optional, Tuple, List

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    ConvolutionalPositionalEmbedding,
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv1dSubsampling1,
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoderLayer


class EBranchformerECAPAEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:

        inter_lang2vec_layers: layer indices for compute intermediate lang2vec predictions
        use_lang2vec_condition: whether to use lang2vec condition on the intermediate layer output
        lang2vec_merge_type: how to use the lang2vec conditions, types include:
            - concat_dim: concatenate the lang2vec condition on the feature dimension
            - concat_time_start: concatenate the lang2vec condition before the first frame
            - add: add the lang2vec condition to the feature

        NOTE(qingzheng): 2 mode of using intermediate lang2vec predictions, first is to only output intermediate
        lang2vec predictions, second is condition the intermediate lang2vec predictions on the intermediate
        layer outputs.
    """

    @typechecked
    def __init__(
        self,
        # === EBranchformer Related Arguments ===
        input_size: int,
        ebrachformer_dim: int = 256,
        attention_heads: int = 4,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        rel_pos_type: str = "latest",
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        zero_triu: bool = False,
        padding_idx: int = -1,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        use_ffn: bool = False,
        macaron_ffn: bool = False,
        ffn_activation_type: str = "swish",
        linear_units: int = 2048,
        positionwise_layer_type: str = "linear",
        merge_conv_kernel: int = 3,
        use_flash_attn=False,
        activation_ckpt=False,
        # === ECAPA Related Arguments ===
        ecapa_scale: int = 8,
        ecapa_dim: int = 512,
        ecapa_output_size: int = 1536,
        # === Intermediate Lang2Vec Related Arguments ===
        inter_lang2vec_layers: List[int] = [],
        lang2vec_merge_type: str = "concat_dim", # concat_dim, concat_time_start, add
    ):
        super().__init__()
        self._ebrachformer_dim = ebrachformer_dim

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert attention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "conv":
            pos_enc_class = ConvolutionalPositionalEmbedding
        elif pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, ebrachformer_dim),
                torch.nn.LayerNorm(ebrachformer_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "wav2vec":
            self.embed = torch.nn.Sequential(
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
                torch.nn.Dropout(dropout_rate),
            )
        elif input_layer == "conv1d1":
            self.embed = Conv1dSubsampling1(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d3":
            self.embed = Conv1dSubsampling3(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                ebrachformer_dim,
                dropout_rate,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, ebrachformer_dim, padding_idx=padding_idx),
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None or input_layer == "none":
            if input_size == ebrachformer_dim:
                self.embed = torch.nn.Sequential(
                    pos_enc_class(ebrachformer_dim, positional_dropout_rate, max_pos_emb_len)
                )
            else:
                self.embed = torch.nn.Linear(input_size, ebrachformer_dim)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        activation = get_activation(ffn_activation_type)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                ebrachformer_dim,
                linear_units,
                dropout_rate,
                activation,
                activation_ckpt,
            )
        elif positionwise_layer_type is None:
            logging.warning("no macaron ffn")
        else:
            raise ValueError("Support only linear.")

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                ebrachformer_dim,
                attention_dropout_rate,
                False,
                use_flash_attn,
                False,
                False,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                ebrachformer_dim,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                ebrachformer_dim,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                ebrachformer_dim,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            ebrachformer_dim,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
            activation_ckpt,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderLayer(
                ebrachformer_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args) if use_ffn else None,
                (
                    positionwise_layer(*positionwise_layer_args)
                    if use_ffn and macaron_ffn
                    else None
                ),
                dropout_rate,
                merge_conv_kernel,
                activation_ckpt,
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(ebrachformer_dim)

        self.inter_lang2vec_layers = inter_lang2vec_layers
        if len(inter_lang2vec_layers) > 0:
            assert 0 <= min(inter_lang2vec_layers) and max(inter_lang2vec_layers) < num_blocks
        
        # ECAPA
        self.ecapa_encoder = EcapaTdnnEncoder(
            input_size=ebrachformer_dim,
            block="EcapaBlock",
            model_scale=ecapa_scale,
            ndim=ecapa_dim,
            output_size=ecapa_output_size,
        )

        # These will be set in the espnet_model and 
        # shared with the final pooling, projector, and loss
        self.pooling = None
        self.projector = None
        self.lang2vec_head = None
        self.lang2vec_type = None
        self.conditioning_layer = None
        self.ecapa_output_size = ecapa_output_size
        self.use_lang2vec_condition = None # assigned in espnet_model
        self.lang2vec_merge_type = lang2vec_merge_type
        self.lang2vec_merge_layer = None
        if lang2vec_merge_type == "concat_dim":
            self.lang2vec_merge_layer = torch.nn.Linear(
                ebrachformer_dim * 2,
                ebrachformer_dim,
            )

    def output_size(self) -> int:
        return self.ecapa_output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            return_all_hs (bool): whether to return all hidden states

        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device) # (B, 1, T), pad place is 0

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv1dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_lang2vec_preds = None
        if len(self.inter_lang2vec_layers) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)
        else:
            intermediate_lang2vec_preds = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)
                olens = masks.squeeze(1).sum(1) # NOTE: because there are subsampling before transformer, so olens != ilens

                if layer_idx in self.inter_lang2vec_layers:
                    if isinstance(xs_pad, tuple):
                        encoder_out = xs_pad[0]
                    else:
                        encoder_out = xs_pad
                    
                    frame_level_feats = self.ecapa_encoder(encoder_out)
                    utt_level_feat = self.pooling(frame_level_feats, feat_lengths=olens)

                    if self.projector is not None:
                        lang_embd = self.projector(utt_level_feat)
                    else:
                        lang_embd = utt_level_feat
    
                    lang2vec_pred = self.lang2vec_head(lang_embd)
                    if self.lang2vec_type in ["phonology_knn", "syntax_knn", "inventory_knn"]:
                        lang2vec_pred = torch.sigmoid(lang2vec_pred)
                    
                    intermediate_lang2vec_preds.append(lang2vec_pred)

                    if self.use_lang2vec_condition:
                        lang2vec_condition = self.conditioning_layer(lang2vec_pred) # (B, transformer_dim)
                        lang2vec_condition = lang2vec_condition.unsqueeze(1) # (B, 1, transformer_dim)
                        if self.lang2vec_merge_type == "concat_dim":
                            encoder_out = torch.cat([encoder_out, lang2vec_condition.expand(-1, encoder_out.size(1), -1)], dim=-1)
                            encoder_out = self.lang2vec_merge_layer(encoder_out)
                        elif self.lang2vec_merge_type == "concat_time_start":
                            encoder_out = torch.cat([lang2vec_condition, encoder_out], dim=1) # (B, T + 1, transformer_dim)
                            masks = torch.cat(
                                [torch.ones(masks.size(0), 1, 1).to(masks.device), masks],
                                dim=2,
                            ) # (B, 1, T + 1)
                        elif self.lang2vec_merge_type == "add":
                            encoder_out = encoder_out + lang2vec_condition
                        else:
                            raise ValueError(
                                f"Unknown lang2vec merge type: {self.lang2vec_merge_type}, "
                                "support lang2vec merge types: concat_dim, concat_time_start, add"
                            )
                        
                        if isinstance(xs_pad, tuple):
                            xs_pad = (encoder_out, xs_pad[1])
                        else:
                            xs_pad = encoder_out

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        xs_pad = self.ecapa_encoder(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if intermediate_lang2vec_preds is not None and len(intermediate_lang2vec_preds) > 0:
            return (xs_pad, intermediate_lang2vec_preds), olens
        return xs_pad, olens
