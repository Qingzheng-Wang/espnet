#!/usr/bin/env python3
# Copyright 2026 Qingzheng Wang (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Interleave data loading utilities for alternating audio-text sequences."""

import json
from pathlib import Path
from typing import Any, Dict, ItemsView, KeysView, List, Optional, ValuesView

import numpy as np
import soundfile as sf


class InterleaveReader:
    """Reader for interleaved audio-text JSONL format.

    Expected JSONL format:
    {"example_id": "001", "segments": [{"audio": "a1.wav"}, {"text": "t2"}, ...]}

    Segments alternate: audio at even indices (0, 2, 4, ...),
                        text at odd indices (1, 3, 5, ...)
    """

    def __init__(self, data_file: str, valid_ids: Optional[List[str]] = None):
        self.data: Dict[str, List[Dict]] = {}
        valid_ids_set = set(valid_ids) if valid_ids is not None else None

        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                record = json.loads(line)

                if not ("example_id" in record and "segments" in record):
                    raise ValueError(f"Line {idx} of file {data_file} is invalid")

                if valid_ids_set is not None and record["example_id"] not in valid_ids_set:
                    continue

                # Validate segment structure
                segments = record["segments"]
                if len(segments) < 2:
                    raise ValueError(
                        f"Line {idx}: segments must have at least 2 elements (audio, text)"
                    )
                if len(segments) % 2 != 0:
                    raise ValueError(
                        f"Line {idx}: segments must have even length for audio-text pairs"
                    )

                for i, seg in enumerate(segments):
                    if i % 2 == 0:  # Even index: audio
                        if "audio" not in seg:
                            raise ValueError(
                                f"Line {idx}, segment {i}: expected 'audio' key"
                            )
                    else:  # Odd index: text
                        if "text" not in seg:
                            raise ValueError(
                                f"Line {idx}, segment {i}: expected 'text' key"
                            )

                self.data[record["example_id"]] = segments

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Get data by ID, returning a dict with audio{i} and text{i} entries.

        Returns format: {audio1: (data, sr), text2: str, audio3: (data, sr), text4: str, ...}
        """
        segments = self.data[key]
        result = {}

        for i, seg in enumerate(segments):
            idx = i + 1  # 1-based index

            if i % 2 == 0:  # Audio (indices 1, 3, 5, ...)
                audio_path = Path(seg["audio"])
                audio_data, sample_rate = sf.read(audio_path, dtype="float32")

                if audio_data.ndim == 1:
                    audio_data = audio_data[np.newaxis, :]
                elif audio_data.ndim == 2:
                    audio_data = audio_data.T

                result[f"audio{idx}"] = (audio_data, sample_rate)
            else:  # Text (indices 2, 4, 6, ...)
                result[f"text{idx}"] = seg["text"]

        return result

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> KeysView[str]:
        return self.data.keys()

    def values(self) -> ValuesView[List[Any]]:
        return self.data.values()

    def items(self) -> ItemsView[str, List[Any]]:
        return self.data.items()
