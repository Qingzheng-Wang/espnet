# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv1dSubsampling2,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder


class TransformerECAPAEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        transformer_dim: dimension of transformer
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed

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
        # === Transformer Related Arguments ===
        input_size: int,
        transformer_dim: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
        # === ECAPA Related Arguments ===
        ecapa_scale: int = 8,
        ecapa_dim: int = 512,
        ecapa_output_size: int = 1536,
        # === Intermediate Lang2Vec Related Arguments ===
        inter_lang2vec_layers: List[int] = [],
        lang2vec_merge_type: str = "concat_dim", # concat_dim, concat_time_start, add
    ):
        super().__init__()
        self._transformer_dim = transformer_dim

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, transformer_dim),
                torch.nn.LayerNorm(transformer_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(transformer_dim, positional_dropout_rate),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_size,
                transformer_dim,
                dropout_rate,
                pos_enc_class(transformer_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, transformer_dim, dropout_rate)
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(input_size, transformer_dim, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, transformer_dim, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, transformer_dim, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, transformer_dim, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, transformer_dim, padding_idx=padding_idx),
                pos_enc_class(transformer_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == transformer_dim:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, transformer_dim)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                transformer_dim,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                transformer_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                transformer_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # Default to flash attention unless overrided by user
        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn
            except Exception:
                use_flash_attn = False

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                transformer_dim,
                MultiHeadedAttention(
                    attention_heads,
                    transformer_dim,
                    attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    False,
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(transformer_dim)

        self.inter_lang2vec_layers = inter_lang2vec_layers
        if len(inter_lang2vec_layers) > 0:
            assert 0 <= min(inter_lang2vec_layers) and max(inter_lang2vec_layers) < num_blocks
        
        # ECAPA
        self.ecapa_encoder = EcapaTdnnEncoder(
            input_size=transformer_dim,
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
                transformer_dim * 2,
                transformer_dim,
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
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)
                    
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
                            xs_pad = torch.cat([xs_pad, lang2vec_condition.expand(-1, xs_pad.size(1), -1)], dim=-1)
                            xs_pad = self.lang2vec_merge_layer(xs_pad)
                        elif self.lang2vec_merge_type == "concat_time_start":
                            xs_pad = torch.cat([lang2vec_condition, xs_pad], dim=1) # (B, T + 1, transformer_dim)
                            masks = torch.cat(
                                [torch.ones(masks.size(0), 1, 1, dtype=masks.dtype).to(masks.device), masks],
                                dim=2,
                            ) # (B, 1, T + 1)
                        elif self.lang2vec_merge_type == "add":
                            xs_pad = xs_pad + lang2vec_condition
                        else:
                            raise ValueError(
                                f"Unknown lang2vec merge type: {self.lang2vec_merge_type}, "
                                "support lang2vec merge types: concat_dim, concat_time_start, add"
                            )

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        
        xs_pad = self.ecapa_encoder(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if intermediate_lang2vec_preds is not None and len(intermediate_lang2vec_preds) > 0:
            return (xs_pad, intermediate_lang2vec_preds), olens
        return xs_pad, olens
