#!/usr/bin/env python3
"""
Script to count total hours of audio datasets by reading utt2num_samples files.
Calculates hours based on 16kHz sampling rate with caching mechanism.
"""

import os
import json
from pathlib import Path

def extract_dataset_name(folder_name):
    """
    Extract dataset name from folder name by removing prefix.
    
    Args:
        folder_name (str): Full folder name
    
    Returns:
        str: Cleaned dataset name
    """
    prefix = "audio_raw_audio_text_dialogue_"
    if folder_name.startswith(prefix):
        return folder_name[len(prefix):]
    return folder_name

def count_hours_from_utt2num_samples(utt2num_samples_path, sample_rate=16000):
    """
    Count total hours from utt2num_samples file.
    
    Args:
        utt2num_samples_path (str): Path to utt2num_samples file
        sample_rate (int): Sample rate in Hz (default: 16000)
    
    Returns:
        tuple: (total_samples, total_hours, num_utterances)
    """
    total_samples = 0
    num_utterances = 0
    
    try:
        with open(utt2num_samples_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        utterance_id = parts[0]
                        num_samples = int(parts[1])
                        total_samples += num_samples
                        num_utterances += 1
    except FileNotFoundError:
        print(f"Warning: File not found: {utt2num_samples_path}")
        return 0, 0.0, 0
    except Exception as e:
        print(f"Error reading {utt2num_samples_path}: {e}")
        return 0, 0.0, 0
    
    # Calculate hours: samples / (sample_rate * 3600)
    total_hours = total_samples / (sample_rate * 3600)
    
    return total_samples, total_hours, num_utterances

def load_cached_stats(dataset_dir):
    """
    Load cached statistics from dataset directory.
    
    Args:
        dataset_dir (str): Path to dataset directory
    
    Returns:
        dict or None: Cached statistics or None if not found
    """
    cache_file = os.path.join(dataset_dir, "dataset_stats.json")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache from {cache_file}: {e}")
    return None

def save_cached_stats(dataset_dir, total_hours, total_samples, num_utterances, sample_rate=16000):
    """
    Save statistics to cache file in dataset directory.
    
    Args:
        dataset_dir (str): Path to dataset directory
        total_hours (float): Total hours of audio
        total_samples (int): Total number of samples
        num_utterances (int): Number of utterances
        sample_rate (int): Sample rate in Hz
    """
    cache_file = os.path.join(dataset_dir, "dataset_stats.json")
    stats = {
        "total_hours": total_hours,
        "total_samples": total_samples,
        "num_utterances": num_utterances,
        "sample_rate": sample_rate,
        "cached": True
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  - Cached statistics to {cache_file}")
    except Exception as e:
        print(f"  - Warning: Could not save cache to {cache_file}: {e}")

def find_all_datasets(dump_audio_dir):
    """
    Find all dataset directories containing utt2num_samples files.
    
    Args:
        dump_audio_dir (str): Path to dump_audio directory
    
    Returns:
        list: List of dataset directory paths
    """
    datasets = []
    dump_audio_path = Path(dump_audio_dir)
    
    if not dump_audio_path.exists():
        print(f"Error: Directory not found: {dump_audio_dir}")
        return datasets
    
    # Find all subdirectories that contain utt2num_samples files
    for item in dump_audio_path.iterdir():
        if item.is_dir():
            utt2num_samples_path = item / "utt2num_samples"
            if utt2num_samples_path.exists():
                datasets.append(str(item))
    
    return sorted(datasets)

def create_readme(dataset_dir, total_hours, total_samples, num_utterances, sample_rate=16000):
    """
    Create a README file with dataset statistics.
    
    Args:
        dataset_dir (str): Path to dataset directory
        total_hours (float): Total hours of audio
        total_samples (int): Total number of samples
        num_utterances (int): Number of utterances
        sample_rate (int): Sample rate in Hz
    """
    readme_path = os.path.join(dataset_dir, "README.md")
    
    content = f"""# Dataset Statistics

- **Total Hours**: {total_hours:.2f}
- **Total Utterances**: {num_utterances:,}
"""
    
    try:
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"  - Created README.md in {dataset_dir}")
    except Exception as e:
        print(f"  - Error creating README.md in {dataset_dir}: {e}")

def create_summary_table(dump_audio_dir, dataset_stats):
    """
    Create a summary table with all dataset statistics in CSV format for Google Sheets.
    
    Args:
        dump_audio_dir (str): Path to dump_audio directory
        dataset_stats (list): List of tuples (dataset_name, hours, samples, utterances)
    """
    summary_file = os.path.join(dump_audio_dir, "dataset_hours_stat.csv")
    
    try:
        with open(summary_file, 'w') as f:
            f.write("Dataset Name,Hours\n")
            
            for dataset_name, hours, samples, utterances in dataset_stats:
                f.write(f"{dataset_name},{hours:.1f}\n")
        
        print(f"\nCreated summary table: {summary_file}")
        
        # Also print the table to console
        print("\n" + "=" * 50)
        print("DATASET HOURS SUMMARY")
        print("=" * 50)
        print(f"{'Dataset Name':<30} {'Hours':<10}")
        print("-" * 50)
        for dataset_name, hours, samples, utterances in dataset_stats:
            print(f"{dataset_name:<30} {hours:<10.1f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error creating summary table: {e}")

def main():
    """Main function to process all datasets and create statistics."""
    # Base directory containing all datasets
    dump_audio_dir = "/work/nvme/bbjs/qwang20/espnet_speechlm3_jinchuan/egs2/qwen_audio/speechlm1/dump_kaldi"
    
    print("=" * 80)
    print("Dataset Hours Counter with Caching")
    print("=" * 80)
    
    # Find all datasets
    datasets = find_all_datasets(dump_audio_dir)
    
    if not datasets:
        print("No datasets found!")
        return
    
    print(f"Found {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"  - {os.path.basename(dataset)}")
    print()
    
    # Process each dataset
    dataset_stats = []
    total_all_hours = 0.0
    total_all_samples = 0
    total_all_utterances = 0
    
    for dataset_dir in datasets:
        dataset_name = extract_dataset_name(os.path.basename(dataset_dir))
        utt2num_samples_path = os.path.join(dataset_dir, "utt2num_samples")
        
        print(f"Processing {dataset_name}...")
        
        # Check if we have cached statistics
        cached_stats = load_cached_stats(dataset_dir)
        
        if cached_stats and cached_stats.get("cached", False):
            print(f"  - Using cached statistics")
            total_hours = cached_stats["total_hours"]
            total_samples = cached_stats["total_samples"]
            num_utterances = cached_stats["num_utterances"]
        else:
            print(f"  - Calculating statistics from utt2num_samples...")
            # Count hours for this dataset
            total_samples, total_hours, num_utterances = count_hours_from_utt2num_samples(utt2num_samples_path)
            
            # Save to cache
            save_cached_stats(dataset_dir, total_hours, total_samples, num_utterances)
        
        print(f"  - Utterances: {num_utterances:,}")
        print(f"  - Total samples: {total_samples:,}")
        print(f"  - Duration: {total_hours:.2f} hours ({total_hours/24:.2f} days)")
        
        # Create README for this dataset
        create_readme(dataset_dir, total_hours, total_samples, num_utterances)
        
        # Add to stats list
        dataset_stats.append((dataset_name, total_hours, total_samples, num_utterances))
        
        # Add to totals
        total_all_hours += total_hours
        total_all_samples += total_samples
        total_all_utterances += num_utterances
        
        print()
    
    # Create summary table
    create_summary_table(dump_audio_dir, dataset_stats)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total utterances: {total_all_utterances:,}")
    print(f"Total samples: {total_all_samples:,}")
    print(f"Total duration: {total_all_hours:.2f} hours ({total_all_hours/24:.2f} days)")
    print("=" * 80)

if __name__ == "__main__":
    main()