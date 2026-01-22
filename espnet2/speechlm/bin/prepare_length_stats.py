#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Script for collecting sequence length statistics for efficient batching."""

import argparse
import json
import logging
from multiprocessing import Pool, Manager
from pathlib import Path
from threading import Thread
from typing import Dict, Set, Tuple

import yaml

from espnet2.speechlm.dataloader.iterator import DataIteratorFactory
from espnet2.speechlm.model import _all_job_types


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for length statistics preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare Length Statistics for SpeechLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/stats"),
        help="Directory to save length statistics",
    )

    # Data specifiers - simplified with helper function
    for split in ["train", "valid"]:
        for spec_type in ["unregistered", "registered"]:
            if spec_type == "unregistered":
                format_str = "task:name:data_json[:factor]"
                example = "audio_to_text:librispeech:train.json:2.0"
            else:
                format_str = "task:name[:factor]"
                example = "text_to_audio:ljspeech:1.5"

            parser.add_argument(
                f"--{split}-{spec_type}-specifier",
                type=str,
                default="",
                help=f"{spec_type.capitalize()} {split} data specifier. "
                f"Format: '{format_str}' (e.g., '{example}')",
            )

    # Processing options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=1000,
        help="Number of samples between flushes to disk (for resume support)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode: overwrite existing files instead of continuing",
    )

    return parser


def worker(
    preprocessor,
    rank: int,
    world_size: int,
    unregistered_spec: str = "",
    registered_spec: str = "",
    processed_keys: Set[str] = None,
    result_queue=None,
    flush_interval: int = 1000,
):
    """Worker function to collect length statistics for a data shard.

    Args:
        preprocessor: Data preprocessor
        rank: Worker rank
        world_size: Total number of workers
        unregistered_spec: Unregistered data specifier
        registered_spec: Registered data specifier
        processed_keys: Set of already processed example_ids (for resume)
        result_queue: Queue to send intermediate results (for flush)
        flush_interval: Number of samples between flushes
    """
    if processed_keys is None:
        processed_keys = set()

    # Create iterator with appropriate specifier
    iterator = DataIteratorFactory(
        unregistered_specifier=unregistered_spec,
        registered_specifier=registered_spec,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        sequential_load=True,
        num_workers=0,
        collate_fn=lambda x: x[0],
    ).build_iter()

    # Collect statistics for this shard
    stats = {}
    skipped = 0
    for key, data_dict in iterator:
        key = tuple(key)
        example_id = key[2]  # (task, data_name, example_id)

        # Skip already processed keys (resume support)
        if example_id in processed_keys:
            skipped += 1
            continue

        stats[key] = preprocessor.find_length(key, data_dict)

        if len(stats) % flush_interval == 0 and len(stats) > 0:
            logging.getLogger(__name__).info(
                f"Worker {rank}: Processed {len(stats)} entries, skipped {skipped}"
            )
            # Flush intermediate results to queue
            if result_queue is not None:
                result_queue.put(dict(stats))
                stats.clear()

    # Send remaining results
    if result_queue is not None and stats:
        result_queue.put(dict(stats))
        stats.clear()

    return stats

def worker_wrapper(args):
    return worker(*args)


def load_existing_stats(output_file: Path) -> Set[str]:
    """Load already processed example_ids from existing stats file.

    Args:
        output_file: Path to the existing JSONL stats file

    Returns:
        Set of already processed example_ids
    """
    processed_keys = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    processed_keys.update(obj.keys())
                except json.JSONDecodeError:
                    continue
    return processed_keys


def flush_writer(result_queue, output_file: Path, stop_event):
    """Background thread to write results from queue to file.

    Args:
        result_queue: Queue containing stats dictionaries from workers
        output_file: Output file path
        stop_event: Event to signal writer to stop
    """
    logger = logging.getLogger(__name__)
    total_written = 0

    with open(output_file, "a") as f:
        while not stop_event.is_set() or not result_queue.empty():
            try:
                # Use timeout to periodically check stop_event
                stats = result_queue.get(timeout=0.5)
                for key_tuple, value in stats.items():
                    example_id = key_tuple[2]
                    json_obj = {example_id: value}
                    f.write(json.dumps(json_obj) + "\n")
                f.flush()
                total_written += len(stats)
                logger.info(f"Flushed {len(stats)} entries (total: {total_written})")
            except Exception:
                # Queue.Empty or other exceptions
                continue

def collect_length_stats(
    preprocessor,
    num_workers: int,
    spec_type: str,
    specifier: str,
    output_file: Path,
    flush_interval: int = 1000,
) -> int:
    """Collect length statistics from all worker processes.

    Args:
        preprocessor: Data preprocessor
        num_workers: Number of parallel workers
        spec_type: Either "unregistered" or "registered"
        specifier: The data specifier string
        output_file: Path to output file (for resume and flush)
        flush_interval: Number of samples between flushes

    Returns:
        Number of new entries processed
    """
    from threading import Event

    logger = logging.getLogger(__name__)

    # Load existing processed keys for resume
    processed_keys = load_existing_stats(output_file)
    if processed_keys:
        logger.info(f"Resume mode: found {len(processed_keys)} already processed keys")

    # Set up keyword arguments based on spec type
    kwargs = {
        "unregistered_spec": specifier if spec_type == "unregistered" else "",
        "registered_spec": specifier if spec_type == "registered" else "",
    }

    # Create shared queue and stop event for flush writer
    manager = Manager()
    result_queue = manager.Queue()
    stop_event = Event()

    # Start background writer thread
    writer_thread = Thread(
        target=flush_writer,
        args=(result_queue, output_file, stop_event),
        daemon=True,
    )
    writer_thread.start()

    # Run workers in parallel
    args_list = [
        (
            preprocessor,
            rank,
            num_workers,
            kwargs["unregistered_spec"],
            kwargs["registered_spec"],
            processed_keys,
            result_queue,
            flush_interval,
        )
        for rank in range(num_workers)
    ]

    total_new = 0
    try:
        with Pool(num_workers) as pool:
            for result in pool.imap_unordered(worker_wrapper, args_list):
                # Any remaining results (shouldn't happen with queue mode)
                if result:
                    result_queue.put(result)
                    total_new += len(result)
    finally:
        # Signal writer to stop and wait for it
        stop_event.set()
        writer_thread.join(timeout=10)

    return total_new

def summarize_stats(output_file: Path) -> None:
    """Print summary statistics for the output file."""
    logger = logging.getLogger(__name__)

    if not output_file.exists():
        return

    total_entries = 0
    total_frames = 0

    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for value in obj.values():
                    total_entries += 1
                    total_frames += value
            except json.JSONDecodeError:
                continue

    if total_entries > 0:
        logger.info(
            f"Stats summary for {output_file}: "
            f"{total_entries} entries | "
            f"Total: {total_frames} frames | "
            f"Avg: {total_frames / total_entries:.1f} frames/entry"
        )


def main():
    """Main entry point for length statistics preparation."""
    # Parse arguments
    args = get_parser().parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s (%(module)s:%(lineno)d) [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting length statistics preparation")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build preprocessor from config
    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    # Add skip_init_encoder=True to all multimodal_io configs
    # This prevents loading heavy models during prepare_stats
    for io_name, io_kwargs in config["multimodal_io"].items():
        if io_name == "continuous_audio" or io_name == "discrete_audio":
            io_kwargs["skip_init_encoder"] = True

    job_template = _all_job_types[config["job_type"]](config, is_train=True)
    preprocessor = job_template.build_preprocessor()

    # Collect all specifiers to process
    specifiers = []
    for split in ["train", "valid"]:
        for spec_type in ["unregistered", "registered"]:
            spec_str = getattr(args, f"{split}_{spec_type}_specifier")
            if spec_str:
                for spec in spec_str.split():
                    specifiers.append((spec_type, spec))

    # Process each specifier
    for spec_type, specifier in specifiers:
        # Parse specifier and generate output filename
        parts = specifier.split(":")
        task, data_name = parts[0], parts[1]
        output_file = args.output_dir / f"stats_{task}_{data_name}.jsonl"

        # Handle existing file
        if output_file.exists():
            if args.no_resume:
                logger.info(f"Overwriting existing file: {output_file}")
                output_file.unlink()
            else:
                existing_count = len(load_existing_stats(output_file))
                logger.info(
                    f"Resume mode for {specifier}: "
                    f"found {existing_count} existing entries"
                )

        # Collect statistics (with resume and flush support)
        logger.info(f"Processing {spec_type} specifier: {specifier}")
        new_count = collect_length_stats(
            preprocessor,
            args.num_workers,
            spec_type,
            specifier,
            output_file,
            args.flush_interval,
        )
        logger.info(f"Added {new_count} new entries for {specifier}")

        # Print summary
        summarize_stats(output_file)


if __name__ == "__main__":
    main()
