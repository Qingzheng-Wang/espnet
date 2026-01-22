# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Rules

### Language
- ALWAYS respond in Simplified Chinese (简体中文)

### Code Comments
- MUST be in English
- MUST be concise and clear
- NEVER include reasoning process in comments
- ❌ BAD: `# I think we should use a for loop here because we need to iterate through...`
- ✅ GOOD: `# Extract features from audio files`

### Log Management
- ALL generated scripts and SLURM logs MUST be stored in `logs/` directory
- SLURM log file naming format: `{slurm_id}_{task_name}_{timestamp}`
  - Example: `12345678_train_asr_20250113_143052.log`
- When writing SLURM scripts, set output/error paths accordingly:
```bash
#SBATCH --output=logs/%j_%x_%Y%m%d_%H%M%S.out
#SBATCH --error=logs/%j_%x_%Y%m%d_%H%M%S.err
```

### Git Workflow
- After EACH code change, IMMEDIATELY:
  1. Summarize the change briefly
  2. `git add` the modified files
  3. `git commit -m "<concise message in English>"`
  4. `git push`
- Commit messages should be clear and conventional, e.g.:
  - `fix: resolve OOM issue in dataloader`
  - `feat: add language identification module`
  - `refactor: simplify feature extraction pipeline`

### Response Format
- At the END of each response, add a summary block:
```
---
**本轮总结：**
- 改动内容：[简述完成了什么]

**Git 记录：**
| Commit Message | Branch | Files Changed |
|----------------|--------|---------------|
| fix: xxx       | main   | src/model.py  |
---
```

### Priority
If rules conflict, follow this order: LANGUAGE > GIT WORKFLOW > LOG MANAGEMENT > CODE COMMENTS > RESPONSE FORMAT

---

## Overview

SpeechLM is a multimodal speech-language model training framework within ESPnet2. It supports training LLMs on audio-text data with both discrete (tokenized) and continuous (feature-based) audio representations. The codebase is self-contained with no dependencies on other ESPnet folders.

## Common Commands

### Environment Setup
```bash
conda activate espnet_whisper2
```

### Training Pipeline (from recipe directory)
```bash
cd <espnet_root>/egs2/librispeech/speechlm1

# Stage 1: Prepare length statistics for efficient batching
python3 ../../../espnet2/speechlm/bin/prepare_length_stats.py \
    --train-config conf/train_speechlm_asr.yaml \
    --output-dir ./exp/speechlm_stats \
    --train-registered-specifier 'audio_to_text:dataset_name:1.0'

# Stage 2: Train with DeepSpeed
deepspeed --num_gpus=N ../../../espnet2/speechlm/bin/train.py \
    --train-config conf/train_speechlm_asr.yaml \
    --output-dir ./exp/train_output \
    --train-registered-specifier "specifier1 specifier2..." \
    --stats-dir ./exp/speechlm_stats

# Stage 3: Inference
python3 ../../../espnet2/speechlm/bin/inference.py \
    --config conf/inference_simple.yaml \
    --checkpoint ./exp/train_output/checkpoint
```

### Dataset Specifier Format
```
task:data_name:resampling_factor
# Examples:
audio_to_text:librispeech:1.0
text_only:wikipedia:2.0  # 2x oversampling
```

### Environment Variable
```bash
export ESPNET_DATASET_REGISTRY="/path/to/dataset_registry.yaml"  # Required for registered datasets
```

## Architecture

### Core Design Pattern: Job Template + Preprocessor

The system separates concerns via `AbsJobTemplate` (model/abs_job.py):
- **Preprocessor**: CPU-side data transformations (runs in DataLoader workers)
  - `preprocessing()`: Single sample → training-ready example
  - `collate_fn()`: Batch samples, passed to PyTorch DataLoader
  - `find_length()`: Quick length computation for batch construction
- **Model**: GPU-side computation (forward pass, loss)

### Multimodal I/O System

`AbsIO` (model/speechlm/multimodal_io/abs_io.py) provides uniform interface for all modalities:

| Method | Purpose | Runs On |
|--------|---------|---------|
| `preprocess()` | Single item preprocessing | CPU (DataLoader workers) |
| `encode_batch()` | Batch encoding | GPU |
| `decode_batch()` | Batch decoding | GPU |
| `find_length()` | Length statistics | CPU |
| `copy_for_worker()` | Lightweight copy for multiprocessing | CPU |

Key implementations:
- **ContinuousAudioIO** (audio.py): Audio features via audio towers (OWSM, Whisper)
- **DiscreteAudioIO** (audio.py): Multi-stream tokenization (codec + SSL)
- **HuggingFaceTextIO** (text.py): Text tokenization

### Data Flow

```
Dataset JSON → DataIteratorFactory → DataLoader → Preprocessor → Model
                     │
                     ├── Load CombinedDataset (rank-based sharding)
                     ├── Load stats files (pre-computed token counts)
                     └── batchfy: bucket (sort by length) or pack (sequence packing)
```

### Token Sequence Structure

```
<bos> <user> <audio> [placeholders...] <eos> <assistant> <text> [tokens...] <eos>
       └── loss_mask = 0 ──────────────┘     └── loss_mask = 1 ────────────┘
```

For continuous audio, placeholders are replaced with audio tower features during forward pass.

### Vocabulary Management

- Special tokens: 0-255 reserved
- Modality tokens: allocated after special tokens
- `vocab_intervals`: maps token ranges to modalities for loss computation

## Key Files

| Component | File |
|-----------|------|
| Training script | bin/train.py |
| Inference script | bin/inference.py |
| Length stats preparation | bin/prepare_length_stats.py |
| Dataset JSON preparation | bin/prepare_dataset_json.py |
| Job template base | model/abs_job.py |
| SpeechLM job | model/speechlm/speechlm_job.py |
| Multimodal I/O base | model/speechlm/multimodal_io/abs_io.py |
| Audio I/O | model/speechlm/multimodal_io/audio.py |
| Text I/O | model/speechlm/multimodal_io/text.py |
| OWSM audio tower | model/speechlm/multimodal_io/audio_tower/owsm_audio_tower.py |
| Whisper audio tower | model/speechlm/multimodal_io/audio_tower/whisper_audio_tower.py |
| DeepSpeed trainer | trainer/deepspeed_trainer.py |
| Data iterator | dataloader/iterator.py |
| Batching algorithms | dataloader/batch.py |
| Parallel LLM wrapper | model/speechlm/lm/parallel.py |

## Configuration Structure

Training configs (YAML) contain:
- `job_type`: Task type (e.g., "speechlm")
- `multimodal_io`: Audio/text encoder configs (model tags, dtypes, attention impl)
- `model`: LLM backbone config (HuggingFace model tag, torch_dtype)
- `preprocessor`: Input/output modality settings, loss region
- `data_loading`: Batching method, batch size, num workers
- `trainer`: Steps, intervals, DeepSpeed config path, freeze parameters

## Task Types

Defined in `dataloader/task_conf.py` and `model/speechlm/task_conf_speechlm.py`:
- `audio_to_text`: Speech recognition, audio captioning
- `text_to_audio`: TTS, audio generation
- `text_only`: Text-only LM training
- `dialogue`: Multimodal conversation

## Batching Methods

- **bucket**: Groups similar-length sequences, minimizes padding
- **pack**: Concatenates multiple samples into one sequence (sequence packing), reduces padding waste further

## Dependencies

Core: `torch`, `transformers==4.57.1`, `deepspeed`, `lhotse`, `wandb`, `polars`, `librosa`

## Logging

Training supports:
- `--wandb-mode`: Weights & Biases logging
- `--swanlab-*`: SwanLab logging options
