# OWSM Audio Tower to process unlimited length audio 
# by splitting or padding into 30s chunks and processing 
# each chunk separately.

# Copyright 2025 Qingzheng Wang (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from espnet2.tasks.s2t import S2TTask
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OWSMAudioTower(nn.Module):
    def __init__(
        self,
        owsm_train_config: str,
        owsm_model_file: str,
        chunk_length: int = 3000,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):
        """
        OWSM Audio Tower to process long audio.

        Args:
            owsm_train_config: Path to OWSM training configuration.
            owsm_model_file: Path to OWSM model file.
            chunk_length: Number of frames in each chunk. Default: 3000.
                (Hint: OWSM hop length 10ms, so 30s = 3000 frames.)
        """

        super().__init__()
        full_model, owsm_train_args = S2TTask.build_model_from_file(
            owsm_train_config, owsm_model_file, device
        )
        self.d_model = full_model.encoder.output_size()
        self.encoder = full_model.encoder.to(device)
        del full_model

        self.chunk_length = chunk_length
        self.dtype = dtype
        self.device = device
    
    def _downsampling_length(
        self,
        length: Union[int, torch.Tensor],
        kernel_sizes: List[int],
        strides: List[int] = None,
        paddings: List[int] = None,
        dilations: List[int] = None,
    ) -> Union[int, torch.Tensor]:
        """Calculate output length after convolution/pooling downsampling.

        Args:
            length: Input length tensor of shape [batch]
            kernel_sizes: List of kernel sizes
            strides: List of strides
            paddings: List of paddings
            dilations: List of dilations

        Returns:
            Output length tensor of shape [batch]
        """
        if strides is None:
            strides = [1] * len(kernel_sizes)
        if paddings is None:
            paddings = [0] * len(kernel_sizes)
        if dilations is None:
            dilations = [1] * len(kernel_sizes)

        assert len(kernel_sizes) == len(strides) == len(paddings) == len(dilations), (
            "kernel_sizes, strides, paddings, and dilations must have the same length",
        )
        for kernel_size, stride, padding, dilation in zip(
            kernel_sizes, strides, paddings, dilations
        ):
            effective_kernel_size = dilation * (kernel_size - 1) + 1
            length = (length + 2 * padding - effective_kernel_size) // stride + 1

        return length
    
    def _padded_and_mask_function(
        self,
        tensor_list: List[torch.Tensor],
        tensor_len: torch.Tensor,
        padding_value: float = 0,
    ):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.

        Reference: Qwen2.5 Omni implementation in transformers.

        Args:
            tensor_list: List of tensors to pad. (batch_size, dim, num_frames)
            tensor_len: Length of each tensor. (batch_size,)
            padding_value: Value to pad with. Default: 0.0.

        Returns:
            Tuple of (padded_tensor, batch_mask, batch_mask_after_cnn).
                - padded_tensor: Padded tensor. (batch_size_chunk, dim, max_len=3000)
                - batch_mask: Mask for the padded tensor. (batch_size_chunk, 1, max_len=3000)
                - batch_mask_after_cnn: Mask for the padded tensor after CNN. (batch_size_chunk, max_len_after_cnn=1500)
        """
        # Whisper encoder requires fixed input length of chunk_length (3000 frames = 30s)
        max_len = self.chunk_length
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        # OWSM downsample by 8x
        feature_lens_after_cnn = self._downsampling_length(
            tensor_len,
            kernel_sizes=[3, 3, 3],
            strides=[2, 2, 2],
            paddings=[0, 0, 0],
            dilations=[1, 1, 1],
        )
        max_len_after_cnn = self._downsampling_length(
            self.chunk_length,
            kernel_sizes=[3, 3, 3],
            strides=[2, 2, 2],
            paddings=[0, 0, 0],
            dilations=[1, 1, 1],
        )
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn,
        )
    
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ):
        # flatten input into 1 batch without padding
        # [████████████████████░░░░░░░░░░] 3500 frames valid + 1500 padding
        # [████████░░░░░░░░░░░░░░░░░░░░░░] 2000 frames valid + 3000 padding
        # =>
        # [████████████████████████████████████████] 5500 frames (3500+2000)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0) # (mel_freqs, num_frames)
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        output_features_list, _ = self.forward(input_features, audio_feature_lengths)

        return output_features_list
    
    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ):
        """
        Extract audio features by chunking or padding into 30s chunks.

        Args:
            input_features: Input features. (mel_freqs, num_frames)
            feature_lens: Length of each input feature. (batch_size,)
        """

        chunk_num = torch.ceil(feature_lens / self.chunk_length).long()
        chunk_lengths = torch.tensor(
            [self.chunk_length] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % self.chunk_length
        chunk_lengths = torch.where(chunk_lengths == 0, self.chunk_length, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self._padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0.0,
        ) # (batch_size_chunk, dim, max_len)

        padded_feature = padded_feature.permute(0, 2, 1) # (batch_size_chunk, max_len, dim)

        output_features, _, _ = self.encoder(
            padded_feature,
            chunk_lengths,  # Use chunk_lengths directly as feat_lens
        )

        chunk_lengths_after_cnn = padded_mask_after_cnn.sum(1) # (batch_size_chunk,)
        output_features_list = []
        chunk_idx = 0
        for num_chunks in chunk_num.tolist():
            sample_chunks = []
            for i in range(num_chunks):
                valid_len = chunk_lengths_after_cnn[chunk_idx + i]
                sample_chunks.append(output_features[chunk_idx + i, :valid_len, :])
            
            sample_feature = torch.cat(sample_chunks, dim=0) # (total_frames, hidden_dim)
            output_features_list.append(sample_feature)
            
            chunk_idx += num_chunks

        output_lengths = torch.tensor([f.shape[0] for f in output_features_list], device=feature_lens.device)

        return output_features_list, output_lengths

class OWSMFeatureExtractor:
    """
    Feature extractor for OWSM models. Extracts log-mel spectrogram features
    from raw audio, similar to Whisper's feature extractor design.
    
    This is a lightweight class that handles:
    - Mel-spectrogram extraction via OWSM's DefaultFrontend
    """
    
    def __init__(
        self,
        owsm_train_config: str,
        owsm_model_file: str = None,
        sampling_rate: int = 16000,
        hop_length: int = 160,  # 10ms at 16kHz
        n_samples: int = 480000,  # 30 seconds at 16kHz
        device: str = "cpu",
    ):
        """
        Initialize the OWSM feature extractor.
        
        Args:
            owsm_train_config: Path to OWSM training config yaml file.
            owsm_model_file: Path to OWSM model file (optional, only frontend is used).
            sampling_rate: Expected audio sampling rate (default: 16000).
            hop_length: Hop length in samples (default: 160 = 10ms at 16kHz).
            n_samples: Maximum number of audio samples (default: 480000 = 30s).
            device: Device for computation (default: "cpu").
        """
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_samples = n_samples
        self.device = device
        
        # Load frontend from OWSM model
        full_model, train_args = S2TTask.build_model_from_file(
            owsm_train_config, owsm_model_file, device
        )
        self.frontend = full_model.frontend.to(device)
        self.normalize = full_model.normalize.to(device)
        self.feature_size = full_model.frontend.output_size()
        del full_model
    
    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging channels."""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # (channels, samples) or (samples, channels)
            if audio.shape[0] <= audio.shape[1]:
                # Likely (channels, samples)
                return np.mean(audio, axis=0)
            else:
                # Likely (samples, channels)
                return np.mean(audio, axis=1)
        return audio
    
    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        sampling_rate: int = 16000,
        return_tensors: str = "np",
        return_attention_mask: bool = True,
        **kwargs,
    ) -> dict:
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            audio: Raw audio waveform(s). Can be:
                - np.ndarray of shape (samples,) for single audio
                - List of np.ndarray for multiple audios
                - torch.Tensor
            sampling_rate: Sampling rate of input audio.
            return_tensors: Output format ("np" or "pt").
            return_attention_mask: Whether to return attention mask.
            
        Returns:
            Dictionary with:
                - "input_features": (batch, mel_dim, frames) mel features
                - "attention_mask": (batch, frames) attention mask
        """
        # Standardize input to list of numpy arrays
        if isinstance(audio, torch.Tensor):
            audio = [audio.cpu().numpy()]
        elif isinstance(audio, np.ndarray):
            audio = [audio]
        elif not isinstance(audio, list):
            audio = list(audio)
            
        input_features_list = []
        attention_mask_list = []
        
        for wav in audio:
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # Convert to mono
            wav = self._to_mono(wav)
            
            # Prepare tensor for frontend: (batch=1, samples)
            wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)
            input_lens = torch.tensor([wav_tensor.shape[1]], device=self.device)
            
            # Extract mel features using OWSM frontend
            with torch.no_grad():
                feats, feat_lens = self.frontend(wav_tensor, input_lens)
                feats, feat_lens = self.normalize(feats, feat_lens)
            
            # feats: (1, frames, mel_dim) -> store as (1, mel_dim, frames)
            input_features_list.append(feats.transpose(1, 2))
            
            # Create attention mask
            mask = torch.zeros((1, feats.shape[1]), dtype=torch.long, device=self.device)
            mask[:, :feat_lens[0]] = 1
            attention_mask_list.append(mask)
        
        # Collate batch
        if len(input_features_list) == 1:
            input_features = input_features_list[0]
            attention_mask = attention_mask_list[0]
        else:
            # Pad to max length in batch
            max_frames = max(f.shape[2] for f in input_features_list)
            mel_dim = input_features_list[0].shape[1]
            
            input_features = torch.zeros(
                len(input_features_list), mel_dim, max_frames,
                dtype=input_features_list[0].dtype, device=self.device
            )
            attention_mask = torch.zeros(
                len(attention_mask_list), max_frames,
                dtype=torch.long, device=self.device
            )
            
            for i, (feat, mask) in enumerate(zip(input_features_list, attention_mask_list)):
                input_features[i, :, :feat.shape[2]] = feat
                attention_mask[i, :mask.shape[1]] = mask
        
        # Convert to requested format
        if return_tensors == "np":
            input_features = input_features.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
        
        result = {"input_features": input_features}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
            
        return result


class OWSMProcessor:
    """
    OWSM Processor for audio feature extraction, compatible with HuggingFace
    processor interface used by ContinuousAudioIO.
    
    This is a thin wrapper around OWSMFeatureExtractor that provides the same
    interface as WhisperProcessor.feature_extractor.
    """
    
    def __init__(
        self,
        owsm_train_config: str,
        owsm_model_file: str = None,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_samples: int = 480000,
        device: str = "cpu",
    ):
        """
        Initialize the OWSM processor.
        
        Args:
            owsm_train_config: Path to OWSM training config yaml file.
            owsm_model_file: Path to OWSM model file.
            sampling_rate: Expected audio sampling rate.
            hop_length: Hop length in samples.
            n_samples: Maximum number of audio samples.
            device: Device for computation.
        """
        self.feature_extractor = OWSMFeatureExtractor(
            owsm_train_config=owsm_train_config,
            owsm_model_file=owsm_model_file,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            n_samples=n_samples,
            device=device,
        )
        
        # Expose attributes for ContinuousAudioIO compatibility
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_samples = n_samples
    
    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        sampling_rate: int = None,
        return_tensors: str = "np",
        return_attention_mask: bool = True,
        **kwargs,
    ) -> dict:
        """
        Process audio and extract features.
        
        Args:
            audio: Raw audio waveform(s).
            sampling_rate: Audio sampling rate (uses default if None).
            return_tensors: Output format ("np" or "pt").
            return_attention_mask: Whether to return attention mask.
            
        Returns:
            Dictionary with input_features and attention_mask.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            
        return self.feature_extractor(
            audio=audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

