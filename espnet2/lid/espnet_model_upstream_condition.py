# Copyright 2024 Qingzheng Wang
# This is different from espnet_model_upstream_lang2vec_condition.py, this supports
# both lang2vec conditioning and self-conditioning LID on the upstream model.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# This is different with espnet_model_upstream_lang2vec_condition.py, 
# this code provide not only the lang2vec conditioning, but also the 
# self-conditioning LID on the upstream model.

from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.lid.espnet_model import ESPnetLIDModel
from espnet2.lid.loss.aamsoftmax_sc_topk_lang2vec import AAMSoftmaxSCTopKLang2Vec


class ESPnetLIDUpstreamConditionModel(ESPnetLIDModel):
    """ESPnet LID model
    Support for language identification and language embedding extraction.
    This model is modified from ESPnetSpeakerModel.
    """

    @typechecked
    def __init__(
        self,
        # ======== Model Architecture ========
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
        # ======== Conditioning Modules ========
        encoder_condition: Optional[nn.ModuleDict] = None,
        pooling_condition: Optional[nn.ModuleDict] = None,
        projector_condition: Optional[nn.ModuleDict] = None,
        # ======== Conditioning Configs ========
        lang2vec_conditioning_layers: List[int] = None,
        lid_conditioning_layers: List[int] = None,
        frozen_ecapa: bool = True,
        apply_intermediate_lang2vec_loss: bool = False,
        apply_intermediate_lid_class_loss: bool = False,
        apply_intermediate_lang2vec_condition: bool = True,
        apply_intermediate_lid_class_condition: bool = True,
        inter_lang2vec_loss_weight: float = 0.0,
        inter_lid_class_loss_weight: float = 0.0,
        cutoff_gradient_from_backbone: Optional[bool] = True,
        cutoff_gradient_before_condtrans: Optional[bool] = False,
        independent_module: Optional[bool] = True,
        use_gate: Optional[bool] = False,
        gate_type: Optional[str] = "hidden_only", # hidden_only, hidden_lang2vec_add, hidden_lang2vec_cat
        shared_conditioning_proj: Optional[bool] = False,
    ):

        super().__init__(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            pooling=pooling,
            projector=projector,
            loss=loss,
        )

        self.frontend.upstream.upstream.model.encoder.independent_module = independent_module

        if independent_module:
            # If use independent module each layer, these condition 
            # modules are defined in tasks/lid.py in build model
            self.frontend.upstream.upstream.model.encoder.ecapa_encoder = encoder_condition
            self.frontend.upstream.upstream.model.encoder.pooling = pooling_condition
            self.frontend.upstream.upstream.model.encoder.projector = projector_condition

            if lang2vec_conditioning_layers is not None:
                lang2vec_head = nn.ModuleDict()
                for layer_idx in lang2vec_conditioning_layers:
                    projector_condition_output_size = projector_condition[str(layer_idx)].output_size()
                    lang2vec_head[str(layer_idx)] = nn.Sequential(
                        nn.Linear(projector_condition_output_size, self.loss.lang2vec_dim),
                        # nn.GELU(),
                        # nn.Dropout(0.1),
                        # nn.Linear(lang2vec_dim, lang2vec_dim),
                    )
                self.frontend.upstream.upstream.model.encoder.lang2vec_head = lang2vec_head
            
            if lid_conditioning_layers is not None:
                aamsoftmax_weight = nn.ParameterDict()
                for layer_idx in lid_conditioning_layers:
                    _aamsoftmax_weight = nn.Parameter(
                        torch.FloatTensor(self.loss.K * self.loss.out_features, self.loss.in_features)
                    )
                    nn.init.xavier_uniform_(_aamsoftmax_weight)
                    aamsoftmax_weight[str(layer_idx)] = _aamsoftmax_weight
                self.frontend.upstream.upstream.model.encoder.aamsoftmax_weight = aamsoftmax_weight
        else:
            # Here use the shared module, which mean shared across downstream 
            # and upstream conditioning layers.
            self.frontend.upstream.upstream.model.encoder.ecapa_encoder = self.encoder
            self.frontend.upstream.upstream.model.encoder.pooling = self.pooling
            self.frontend.upstream.upstream.model.encoder.projector = self.projector
            if lang2vec_conditioning_layers is not None:
                self.frontend.upstream.upstream.model.encoder.lang2vec_head = getattr(self.loss, "lang2vec_head", None)
            if lid_conditioning_layers is not None:
                self.frontend.upstream.upstream.model.encoder.aamsoftmax_weight = getattr(self.loss, "weight", None)
        
        self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_layers = lang2vec_conditioning_layers
        self.frontend.upstream.upstream.model.encoder.lid_conditioning_layers = lid_conditioning_layers
        if lang2vec_conditioning_layers is None and lid_conditioning_layers is None:
            self.frontend.upstream.upstream.model.encoder.conditioning_layers = []
        else:
            self.frontend.upstream.upstream.model.encoder.conditioning_layers = sorted(
                set((lang2vec_conditioning_layers or []) + (lid_conditioning_layers or []))
            )
        
        self.frontend.upstream.upstream.model.encoder.apply_intermediate_lang2vec_condition = apply_intermediate_lang2vec_condition
        self.frontend.upstream.upstream.model.encoder.apply_intermediate_lid_class_condition = apply_intermediate_lid_class_condition

        # Each conditioning projection layer is independent
        if lang2vec_conditioning_layers is not None and apply_intermediate_lang2vec_condition:
            if shared_conditioning_proj:
                lay2vec_conditioning_projs = nn.Linear(
                    self.loss.lang2vec_dim,
                    self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                )
            else:
                lay2vec_conditioning_projs = nn.ModuleDict()
                for layer_idx in lang2vec_conditioning_layers:
                    lay2vec_conditioning_projs[str(layer_idx)] = nn.Sequential(
                        nn.Linear(
                            self.loss.lang2vec_dim,
                            self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                        ),
                    )
            self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_projs = lay2vec_conditioning_projs
            self.frontend.upstream.upstream.model.encoder.shared_conditioning_proj = shared_conditioning_proj

            if use_gate:
                self.frontend.upstream.upstream.model.encoder.use_gate = use_gate
                self.frontend.upstream.upstream.model.encoder.gate_type = gate_type
                lang2vec_gate_projs = nn.ModuleDict()
                for layer_idx in lang2vec_conditioning_layers:
                    if gate_type in ("hidden_only", "hidden_lang2vec_add"):
                        lang2vec_gate_projs[str(layer_idx)] = nn.Sequential(
                            nn.LayerNorm(
                                self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                                eps=self.frontend.upstream.upstream.model.encoder.config.layer_norm_eps,
                            ), # stable input, prevent gradient explosion at the beginning
                            nn.Linear(
                                self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                                self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                            )
                        )
                    elif gate_type == "hidden_lang2vec_cat":
                        lang2vec_gate_projs[str(layer_idx)] = nn.Sequential(
                            nn.LayerNorm(
                                self.frontend.upstream.upstream.model.encoder.config.hidden_size * 2,
                                eps=self.frontend.upstream.upstream.model.encoder.config.layer_norm_eps,
                            ),
                            nn.Linear(
                                self.frontend.upstream.upstream.model.encoder.config.hidden_size * 2,
                                self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                            )
                        )
                    else:
                        raise ValueError(f"Unknown gate type: {gate_type}")
                self.frontend.upstream.upstream.model.encoder.lang2vec_gate_projs = lang2vec_gate_projs
        
        if lid_conditioning_layers is not None and apply_intermediate_lid_class_condition:
            lid_conditioning_projs = nn.ModuleDict()
            for layer_idx in lid_conditioning_layers:
                lid_conditioning_projs[str(layer_idx)] = nn.Linear(
                    self.loss.out_features,
                    self.frontend.upstream.upstream.model.encoder.config.hidden_size,
                )
            self.frontend.upstream.upstream.model.encoder.lid_conditioning_projs = lid_conditioning_projs

        self.frontend.upstream.upstream.model.encoder.frozen_ecapa = frozen_ecapa
        self.frontend.upstream.upstream.model.encoder.cutoff_gradient_from_backbone = cutoff_gradient_from_backbone
        self.frontend.upstream.upstream.model.encoder.cutoff_gradient_before_condtrans = cutoff_gradient_before_condtrans

        assert isinstance(self.loss, AAMSoftmaxSCTopKLang2Vec), (
            f"The loss should be AAMSoftmaxSCTopKLang2Vec, but got {type(self.loss)}"
        )
        self.frontend.upstream.upstream.model.encoder.aamsoftmax_loss = self.loss

        self.apply_intermediate_lang2vec_loss = apply_intermediate_lang2vec_loss
        self.apply_intermediate_lid_class_loss = apply_intermediate_lid_class_loss
        self.inter_lang2vec_loss_weight = inter_lang2vec_loss_weight
        self.inter_lid_class_loss_weight = inter_lid_class_loss_weight

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        lang2vecs: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
        torch.Tensor,
    ]:
        """Feed-forward through encoder layers and aggregate into utterance-level

        feature.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,)
            extract_embd: a flag which doesn't go through the classification
                head when set True
            lid_labels: (Batch, )
            one-hot speaker labels used in the train phase
            task_tokens: (Batch, )
            task tokens used in case of token-based trainings
        """

        if lid_labels is not None:
            assert speech.shape[0] == lid_labels.shape[0], (
                speech.shape,
                lid_labels.shape,
            )
        if task_tokens is not None:
            assert speech.shape[0] == task_tokens.shape[0], (
                speech.shape,
                task_tokens.shape,
            )
        batch_size = speech.shape[0]
        stats = dict()

        # 1. extract feats
        # Must transfer speech_lengths to extract_feats to get correct feat_lengths
        feats, feat_lengths, intermediate_lang2vec_preds, intermediate_lid_logits = self.extract_feats(speech, speech_lengths, lid_labels)
        frame_level_feats = self.encode_frame(feats)

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, task_tokens, feat_lengths)

        # 3. (optionally) go through further projection(s)
        lang_embd = self.project_lang_embd(utt_level_feat)

        # 4. calculate loss
        # NOTE: if lid_labels is None, loss and accuracy are None
        if lang2vecs is not None:
            loss, accuracy, pred_lids, class_loss, lang2vec_loss = self.loss(lang_embd, lid_labels, lang2vecs)
            lang2vec_type = self.loss.lang2vec_type
            stats["class_loss"] = class_loss.detach()
            if lang2vec_loss is not None: # lang2vec_loss is None when setting apply_last to False in the loss
                stats[f"{lang2vec_type}_loss_downstream"] = lang2vec_loss.detach()

            if (
                intermediate_lang2vec_preds is not None and 
                self.inter_lang2vec_loss_weight > 0 and 
                self.apply_intermediate_lang2vec_loss and
                self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_layers is not None
            ):
                inter_lang2vec_losses = self._calc_intermediate_lang2vec_pred_loss(
                    intermediate_lang2vec_preds,
                    lang2vecs,
                )

                inter_lang2vec_loss_mean = 0.0
            
                for layer_idx, inter_lang2vec_loss in zip(
                    self.frontend.upstream.upstream.model.encoder.lang2vec_conditioning_layers, 
                    inter_lang2vec_losses
                ):
                    stats[f"inter_{lang2vec_type}_loss_layer{layer_idx}"] = inter_lang2vec_loss.detach()
                    inter_lang2vec_loss_mean += inter_lang2vec_loss
                
                inter_lang2vec_loss_mean /= len(inter_lang2vec_losses)
                stats[f"inter_{lang2vec_type}_loss_mean"] = inter_lang2vec_loss_mean.detach()

                lang2vec_loss_all = 0.0
                if lang2vec_loss is not None: # which means do not apply lang2vec loss on the last layer
                    lang2vec_loss_all += (1 - self.inter_lang2vec_loss_weight) * lang2vec_loss
                    lang2vec_loss_all += self.inter_lang2vec_loss_weight * inter_lang2vec_loss_mean
                else:
                    lang2vec_loss_all = inter_lang2vec_loss_mean
                
                stats[f"{lang2vec_type}_loss_all"] = lang2vec_loss_all.detach()
            else:
                lang2vec_loss_all = lang2vec_loss

            if (
                intermediate_lid_logits is not None and
                self.inter_lid_class_loss_weight > 0 and 
                self.apply_intermediate_lid_class_loss and 
                self.frontend.upstream.upstream.model.encoder.lid_conditioning_layers is not None
            ):
                inter_lid_class_losses = self._calc_intermediate_lid_class_loss(
                    intermediate_lid_logits,
                    lid_labels,
                )

                inter_lid_class_loss_mean = 0.0
                
                for layer_idx, inter_lid_class_loss in zip(
                    self.frontend.upstream.upstream.model.encoder.lid_conditioning_layers, 
                    inter_lid_class_losses
                ):
                    stats[f"inter_lid_class_loss_layer{layer_idx}"] = inter_lid_class_loss.detach()
                    inter_lid_class_loss_mean += inter_lid_class_loss
                
                inter_lid_class_loss_mean /= len(inter_lid_class_losses)
                stats[f"inter_lid_class_loss_mean"] = inter_lid_class_loss_mean.detach()

                lid_class_loss_all = 0.0
                lid_class_loss_all += (1 - self.inter_lid_class_loss_weight) * class_loss
                lid_class_loss_all += self.inter_lid_class_loss_weight * inter_lid_class_loss_mean

                stats[f"lid_class_loss_all"] = lid_class_loss_all.detach()
            else:
                lid_class_loss_all = class_loss

            # NOTE(qingzheng): recaulculate the loss, loss fomula:
            # loss = (1 - lang2vec_weight) * (
            #     (1 - inter_lid_class_loss_weight) * class_loss + inter_lid_class_loss_weight * inter_lid_class_loss_mean
            # ) + lang2vec_weight * (
            #     (1 - inter_lang2vec_loss_weight) * lang2vec_loss + inter_lang2vec_loss_weight * inter_lang2vec_loss_mean
            # )
            loss = (1- self.loss.lang2vec_weight) * lid_class_loss_all + self.loss.lang2vec_weight * lang2vec_loss_all
        else:
            # NOTE(qingzheng): if use aamsoftmax_sc_topk_lang2vec loss but not 
            # specify the lang2vec in preprocessor, it will jump to this branch,
            # but aamsoftmax_sc_topk_lang2vec default return 5 outputs, so use [:3]
            # to restrict here only retrieve the first 3 outputs
            outputs = self.loss(lang_embd, lid_labels)
            loss, accuracy, pred_lids = outputs[:3]
            if loss is not None:
                stats["class_loss"] = loss.detach()

        if extract_embd:
            return lang_embd, pred_lids

        stats["loss"] = loss.detach()
        if accuracy is not None: # if not provide labels, accuracy is None
            stats["accuracy"] = accuracy.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self, 
        speech: torch.Tensor, 
        speech_lengths: torch.Tensor,
        lid_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        # 1. extract feats
        if self.frontend is not None:
            feats, feat_lengths, intermediate_lang2vec_preds, intermediate_lid_logits = self.frontend(
                speech, speech_lengths, lid_labels
            )
        else:
            feats = speech
            feat_lengths = None
            intermediate_lang2vec_preds = None
            intermediate_lid_logits = None

        # 2. apply augmentations
        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, feat_lengths)

        # 3. normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, feat_lengths)

        return feats, feat_lengths, intermediate_lang2vec_preds, intermediate_lid_logits

    def _calc_intermediate_lang2vec_pred_loss(
        self,
        intermediate_lang2vec_preds: List[torch.Tensor],
        lang2vecs: torch.Tensor,
    ) -> List[torch.Tensor]:
        inter_lang2vec_losses = []
        for intermediate_lang2vec_pred in intermediate_lang2vec_preds:
            inter_lang2vec_loss = self.loss.lang2vec_loss(
                intermediate_lang2vec_pred,
                lang2vecs,
            )
            inter_lang2vec_losses.append(inter_lang2vec_loss)

        return inter_lang2vec_losses

    def _calc_intermediate_lid_class_loss(
        self,
        intermediate_lid_logits: List[torch.Tensor],
        lid_labels: torch.Tensor,
    ) -> List[torch.Tensor]:
        inter_lid_class_losses = []
        for intermediate_lid_logit in intermediate_lid_logits:
            if len(lid_labels.size()) == 2:
                lid_labels = lid_labels.squeeze(1)
            inter_lid_class_loss = self.loss.ce(
                intermediate_lid_logit,
                lid_labels,
            )
            inter_lid_class_losses.append(inter_lid_class_loss)

        return inter_lid_class_losses
