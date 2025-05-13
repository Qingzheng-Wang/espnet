import copy
import logging
from typing import Optional, Tuple, Union, List

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlFrontend(AbsFrontend):
    """
    Speech Pretrained Representation frontend structure for ASR.
    if multilayer_feature is True, layer is either a list of 
    layer indices (weight sum selected layers) or None (weight sum all layers).
    if multilayer_feature is False, layer is a single layer index 
    (output the specific layer hidden state) or None (output the last layer hidden state).

    NOTE(qingzheng):
    layer: int or list of int, for the selected hidden states, the layer index starting
    from 0, and at least for the hf_wav2vec2 model, 0 represents the output of the conv
    encoder (the input to the first layer of the context encoder), and 1 represents the 
    output of the first layer of the context encoder, and so on. The layer index range is 
    [0, num_layers]
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: Optional[str] = None,
        multilayer_feature: bool = False,
        layer: Optional[Union[int, List[int]]] = None,
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer, S3PRLUpstream
        except Exception as e:
            print("Error: S3PRL is not properly installed.")
            print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
            raise e

        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning(
                "All the upstream models in S3PRL now only support 16 kHz audio."
            )

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert frontend_conf.get("upstream", None) in S3PRLUpstream.available_names()
        upstream = S3PRLUpstream(
            frontend_conf.get("upstream"),
            path_or_url=frontend_conf.get("path_or_url", None),
            normalize=frontend_conf.get("normalize", False),
            extra_conf=frontend_conf.get("extra_conf", None),
        )
        if getattr(upstream.upstream, "model", None):
            if getattr(upstream.upstream.model, "feature_grad_mult", None) is not None:
                upstream.upstream.model.feature_grad_mult = 1.0
        upstream.eval()

        if layer is not None and isinstance(layer, int):
            assert (
                layer >= 0 and layer <= upstream.num_layers
            ), f"Invalid layer index: {layer}, should be in [0, {upstream.upstream.num_layers}]"
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when a specific layer used"
        elif layer is not None and isinstance(layer, list):
            assert (
                all([l >= 0 and l <= upstream.num_layers for l in layer])
            ), f"Invalid layer index: {layer}, should all be in [0, {upstream.upstream.num_layers}]"
            layer_selections = layer
            assert (
                multilayer_feature
            ), "multilayer feature will be activated, when a list of specific layers used"
        else:
            layer_selections = None
        featurizer = Featurizer(upstream, layer_selections=layer_selections)

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.upstream, self.featurizer = upstream, featurizer
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "s3prl"
        self.hop_length = self.featurizer.downsample_rate
        self.tile_factor = frontend_conf.get("tile_factor", 1)

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lens = self.upstream(input, input_lengths)
        if self.layer is not None and isinstance(self.layer, int):
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return feats, feats_lens

    def reload_pretrained_parameters(self):
        self.upstream.load_state_dict(self.pretrained_params)
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
    

class S3prlFrontendLang2VecCondition(AbsFrontend):
    """
    This is a modified version of S3prlFrontend for output with intermediate lang2vec predictions.
    """
    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: Optional[str] = None,
        multilayer_feature: bool = False,
        layer: Optional[Union[int, List[int]]] = None,
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer, S3PRLUpstreamLang2VecCondition
        except Exception as e:
            print("Error: S3PRLUpstreamLang2VecCondition are not found.")
            raise e

        super().__init__()

        assert frontend_conf.get("upstream", None) in S3PRLUpstreamLang2VecCondition.available_names()
        upstream = S3PRLUpstreamLang2VecCondition(
            frontend_conf.get("upstream"),
            path_or_url=frontend_conf.get("path_or_url", None),
            normalize=frontend_conf.get("normalize", False),
            extra_conf=frontend_conf.get("extra_conf", None),
        )
        if getattr(upstream.upstream, "model", None):
            if getattr(upstream.upstream.model, "feature_grad_mult", None) is not None:
                upstream.upstream.model.feature_grad_mult = 1.0
        upstream.eval()

        if layer is not None and isinstance(layer, int):
            assert (
                layer >= 0 and layer <= upstream.num_layers
            ), f"Invalid layer index: {layer}, should be in [0, {upstream.upstream.num_layers}]"
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when a specific layer used"
        elif layer is not None and isinstance(layer, list):
            assert (
                all([l >= 0 and l <= upstream.num_layers for l in layer])
            ), f"Invalid layer index: {layer}, should all be in [0, {upstream.upstream.num_layers}]"
            layer_selections = layer
            assert (
                multilayer_feature
            ), "multilayer feature will be activated, when a list of specific layers used"
        else:
            layer_selections = None
        featurizer = Featurizer(upstream, layer_selections=layer_selections)

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.upstream, self.featurizer = upstream, featurizer
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "s3prl"
        self.hop_length = self.featurizer.downsample_rate
        self.tile_factor = frontend_conf.get("tile_factor", 1)

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, feats_lens, intermediate_lang2vec_preds = self.upstream(input, input_lengths)
        if self.layer is not None and isinstance(self.layer, int):
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return feats, feats_lens, intermediate_lang2vec_preds
