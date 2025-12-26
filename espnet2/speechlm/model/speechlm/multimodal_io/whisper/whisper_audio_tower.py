# Whisper Audio Tower to process unlimited length audio 
# by splitting or padding into 30s chunks and processing 
# each chunk separately.

# Copyright 2025 Qingzheng Wang (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from transformers import WhisperModel
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperAudioTower(nn.Module):
    def __init__(
        self,
        hf_model_tag: str,
        chunk_length: int = 3000,
        dtype: str = "bfloat16",
        device: str = "cpu",
    ):
        """
        Whisper Audio Tower to process long audio.

        Args:
            hf_model_tag: Hugging Face model tag.
            chunk_length: Number of frames in each chunk. Default: 3000.
                (Hint: Whisper hop length 10ms, so 30s = 3000 frames.)
        """

        super().__init__()
        self.hf_model_tag = hf_model_tag
        full_model = WhisperModel.from_pretrained(hf_model_tag, dtype=dtype)
        self.d_model = full_model.config.d_model

        self.encoder = full_model.encoder.to(device)
        del full_model.decoder

        self.chunk_length = chunk_length
        self.dtype = dtype
        self.device = device
    
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

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        # CNN downsamples by 2x, so fixed output length is chunk_length // 2
        max_len_after_cnn = (self.chunk_length - 1) // 2 + 1
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
        # [████████████████████████████████████████] 5500帧 (3500+2000)
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
        )

        output_features = self.encoder(
            padded_feature,
            padded_mask,
            return_dict=True,
        )["last_hidden_state"]

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

