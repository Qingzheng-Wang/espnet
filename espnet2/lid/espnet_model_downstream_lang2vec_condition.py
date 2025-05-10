# Copyright 2023 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, Optional, Tuple, Union, List

import torch
import logging
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.lid.loss.abs_loss import AbsLoss
from espnet2.lid.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetLIDDownstreamLang2VecConditionModel(AbsESPnetModel):
    """ESPnet LID model
    Support for language identification and language embedding extraction.
    This model is modified from ESPnetSpeakerModel.
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
        use_lang2vec_condition: bool = False,
        inter_lang2vec_loss_weight: float = 0.0,
    ):

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss

        assert len(self.encoder.lang2vec_condition_layer_idx) > 0, (
            "if use_lang2vec_condition is True, the lang2vec_condition_layer_idx "
            "should be set in the encoder."
            f"Got {self.encoder.lang2vec_condition_layer_idx}"
        )
        self.encoder.pooling = self.pooling
        self.encoder.projector = self.projector
        try:
            self.encoder.lang2vec_head = self.loss.lang2vec_head
            self.encoder.lang2vec_type = self.loss.lang2vec_type
        except AttributeError:
            raise ValueError(
                f"The loss type {self.loss.__class__.__name__} does not have lang2vec_head."
            )
        
        # NOTE(qingzheng): if use_lang2vec_condition is True, use self-conditioning layer
        # else, just compute the lang2vec loss according to the intermediate language embedding outputs
        if use_lang2vec_condition:
            self.encoder.conditioning_layer = torch.nn.Linear(
                self.loss.lang2vec_dim,
                self.encoder._transformer_ndim,
            )
        
        self.inter_lang2vec_loss_weight = inter_lang2vec_loss_weight

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
        feats, feat_lengths = self.extract_feats(speech, speech_lengths)
        frame_level_feats = self.encode_frame(feats)
        intermediate_lang_embds = None
        if isinstance(frame_level_feats, tuple):
            frame_level_feats, intermediate_lang_embds = frame_level_feats

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, task_tokens, feat_lengths)

        # 3. (optionally) go through further projection(s)
        lang_embd = self.project_lang_embd(utt_level_feat)

        # 4. calculate loss
        # NOTE: if lid_labels is None, loss and accuracy are None
        if lang2vecs is not None:
            loss, accuracy, pred_lids, class_loss, lang2vec_loss  = self.loss(lang_embd, lid_labels, lang2vecs)
            lang2vec_type = self.loss.lang2vec_type
            stats[f"{lang2vec_type}_loss"] = lang2vec_loss.detach()
            stats["class_loss"] = class_loss.detach()

            if intermediate_lang_embds is not None and self.inter_lang2vec_loss_weight > 0:
                inter_lang2vec_losses = self._calc_intermediate_lang2vec_pred_loss(intermediate_lang_embds, lang2vecs)
                inter_lang2vec_loss_mean = 0.0
                
                for layer_idx, inter_lang2vec_loss in zip(self.encoder.inter_lang2vec_layers, inter_lang2vec_losses):
                    inter_lang2vec_loss = inter_lang2vec_loss.detach()
                    stats[f"inter_{lang2vec_type}_layer{layer_idx}"] = inter_lang2vec_loss
                    inter_lang2vec_loss_mean += inter_lang2vec_loss
                
                inter_lang2vec_loss_mean /= len(inter_lang2vec_losses)
                stats[f"inter_{lang2vec_type}_loss_mean"] = inter_lang2vec_loss_mean
                lang2vec_loss = (
                    (1 - self.inter_lang2vec_loss_weight) * lang2vec_loss + 
                    self.inter_lang2vec_loss_weight * inter_lang2vec_loss_mean
                )

                # recaulculate the loss
                loss = (1- self.loss.lang2vec_weight) * class_loss + self.loss.lang2vec_weight * lang2vec_loss
        else:
            loss, accuracy, pred_lids = self.loss(lang_embd, lid_labels)
            stats["class_loss"] = loss.detach()

        if extract_embd:
            return lang_embd, pred_lids

        stats["loss"] = loss.detach()
        if accuracy is not None: # if not provide labels, accuracy is None
            stats["accuracy"] = accuracy.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        # 1. extract feats
        if self.frontend is not None:
            feats, feat_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech
            feat_lengths = None

        # 2. apply augmentations
        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, feat_lengths)

        # 3. normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, feat_lengths)

        return feats, feat_lengths

    def encode_frame(self, feats: torch.Tensor) -> torch.Tensor:
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def project_lang_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            lang_embd = self.projector(utt_level_feat)
        else:
            lang_embd = utt_level_feat

        return lang_embd

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lid_labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}

    def _calc_intermediate_lang2vec_pred_loss(
        self,
        intermediate_lang_embds: List[torch.Tensor],
        lang2vecs: torch.Tensor,
    ) -> torch.Tensor:
        inter_lang2vec_losses = []
        for intermediate_lang_embd in intermediate_lang_embds:
            inter_lang2vec_loss = self.loss.lang2vec_loss(
                self.loss.lang2vec_head(intermediate_lang_embd),
                lang2vecs,
            )
            inter_lang2vec_losses.append(inter_lang2vec_loss)

        return inter_lang2vec_losses
