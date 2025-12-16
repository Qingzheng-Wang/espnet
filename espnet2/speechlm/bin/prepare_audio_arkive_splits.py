#!/usr/bin/env python3
# Copyright 2025 William Chen (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Script for preparing audio metadata from Kaldi format using Arkive."""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import glob

import pandas as pd

try:
    from arkive import Arkive
except ImportError:
    raise ImportError(
        "arkive is not installed. Please install at https://github.com/wanchichen/arkive"
    )


def parse_segments_file(
    segments_path: str,
) -> List[Tuple[str, str, float, float]]:

    segments = []
    invalid_count = 0

    with open(segments_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                logging.warning(f"Line {line_num}: Invalid format in segments: {line}")
                invalid_count += 1
                continue

            segment_id, recording_id, start_str, end_str = parts

            # Parse times
            try:
                start = float(start_str)
                end = float(end_str)
            except ValueError:
                logging.warning(
                    f"Line {line_num}: Invalid time values: {start_str}, {end_str}"
                )
                invalid_count += 1
                continue

            # Validate times
            if start < 0:
                logging.warning(
                    f"Line {line_num}: Negative start time for {segment_id}: {start}"
                )
                invalid_count += 1
                continue

            if end <= start:
                logging.warning(
                    f"Line {line_num}: End time <= start time for {segment_id}: "
                    f"{start} -> {end}"
                )
                invalid_count += 1
                continue

            segments.append((segment_id, recording_id, start, end))

    if invalid_count > 0:
        logging.warning(f"Skipped {invalid_count} invalid segments")

    return segments


def prepare_audio_arkive(
    wav_scp_files: List[str],
    segments_files: Optional[List[str]],
    output_dir: str,
    target_format: str = "wav",
    target_bit_depth: int = 16,
    num_workers: int = 64,
    flush_interval: int = 100,
    log_level: str = "INFO",
):
    """Process Kaldi wav.scp and segments to create Arkive manifests.

    Args:
        wav_scp_files: List of paths to Kaldi wav.scp files
        segments_files: List of paths to Kaldi segments files (optional)
        output_dir: Directory to save arkive manifest files
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if segments_files is not None:
        assert len(wav_scp_files) == len(segments_files), "Number of wav.scp files and segments files must be the same"
    
    ark = Arkive(str(output_dir))

    for i in range(len(wav_scp_files)):
        logging.info(f"Processing wav.scp file {i} of {len(wav_scp_files)}")
 
        wav_scp = wav_scp_files[i]
        segments = segments_files[i] if segments_files is not None else None
        logging.info(f"Reading wav.scp file: {wav_scp}")
        logging.info(f"Reading segments file: {segments}")

        # Read Kaldi wav.scp file
        audio_paths = {}
        with open(wav_scp, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    logging.warning(f"Line {line_num}: Invalid format in wav.scp: {line}")
                    continue
                recording_id, path = parts
                audio_paths[recording_id] = os.path.abspath(path)

        if not audio_paths:
            raise ValueError(f"No valid recordings found in wav.scp file {wav_scp}")

        logging.info(f"Found {len(audio_paths)} recordings in wav.scp file {wav_scp}")

        paths = [audio_paths[recording_id] for recording_id in audio_paths]
        ark.append(
            paths,
            target_format=target_format,
            show_progress=True,
            target_bit_depth=target_bit_depth,
            flush_interval=flush_interval,
            num_workers=num_workers
        )

        logging.info(
            f"Successfully processed {len(ark.data)}/{len(audio_paths)} recordings"
        )

        df = ark.data
        df_unprocessed = df[df["original_file_path"].isin(paths)]
        
        cols = df_unprocessed.columns.tolist()
        cols.remove("original_file_path")
        cols_to_add = [df_unprocessed[col] for col in cols]
        data = dict(zip(df_unprocessed["original_file_path"], zip(*cols_to_add)))

        new_data = []
        if segments is not None:
            logging.info(f"Reading segments from: {segments}")

            # Parse and validate segments
            segment_data = parse_segments_file(segments)

            if not segment_data:
                raise ValueError("No valid segments found in segments file")

            for segment in segment_data:
                segment_id, recording_id, start, end = segment
                audio_path = audio_paths[recording_id]

                dump_data = data[audio_path]
                dump_data = [audio_path, segment_id, recording_id, start, end] + list(
                    dump_data
                )

                new_data.append(dump_data)

        else:
            for recording_id in audio_paths:
                audio_path = audio_paths[recording_id]
                dump_data = data[audio_path]
                dump_data = [audio_path, recording_id, recording_id, None, None] + list(
                    dump_data
                )

                new_data.append(dump_data)

        out_df = pd.DataFrame(
            new_data,
            columns=["original_file_path", "utt_id", "doc_id", "start_time", "end_time"]
            + cols,
        )
        
        # Save processed parquet file with backup mechanism
        parquet_path = output_dir / "metadata_processed.parquet"

        if parquet_path.exists():
            df_processed = pd.read_parquet(str(parquet_path))
            df_processed = pd.concat([df_processed, out_df], ignore_index=True)
            df_processed.to_parquet(str(parquet_path))
        else:
            out_df.to_parquet(str(parquet_path))


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare audio metadata from Kaldi wav.scp and segments files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected_splits",
        type=str,
        nargs="+",
        default=None,
        help="Selected splits to process",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory containing the splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for arkive manifest files",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="Logging level",
    )
    parser.add_argument(
        "--target_format",
        type=str,
        default="wav",
        help="Target format for the audio files",
    )
    parser.add_argument(
        "--target_bit_depth",
        type=int,
        default=16,
        help="Target bit depth for the audio files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of workers for the audio processing",
    )
    parser.add_argument(
        "--flush_interval",
        type=int,
        default=100,
        help="Flush interval for the audio processing",
    )
    return parser


def main(cmd=None):
    """Run the main function."""
    parser = get_parser()
    args = parser.parse_args(cmd)
    splits_dir = args.splits_dir
    output_dir = args.output_dir

    wav_scp_dir = os.path.join(splits_dir, "wav.scp")
    wav_scp_files = sorted(glob.glob(os.path.join(wav_scp_dir, "*split.*")))

    selected_splits = args.selected_splits if args.selected_splits is not None else range(len(wav_scp_files))
    selected_splits = [int(i) for i in selected_splits]
    wav_scp_files = [wav_scp_files[i] for i in selected_splits]

    segments_dir = os.path.join(splits_dir, "segments")
    if os.path.exists(segments_dir):
        segments_files = sorted(glob.glob(os.path.join(segments_dir, "*split.*")))
        segments_files = [segments_files[i] for i in selected_splits]
    else:
        segments_files = None

    prepare_audio_arkive(
        wav_scp_files, segments_files, output_dir,
        args.target_format,
        args.target_bit_depth,
        args.num_workers,
        args.flush_interval
    )


if __name__ == "__main__":
    main()