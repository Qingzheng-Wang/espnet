#!/bin/bash
# Training script for SpeechLM on LibriSpeech
set -e
set -u
set -o pipefail

# Configuration
stage=1
stop_stage=3
ngpu=2  # Number of GPUs to use (start with 2 for testing)

# Paths
manifest_dir=./manifest
stats_dir=./exp/speechlm_stats
output_dir=./exp/train_speechlm_$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "SpeechLM Training on LibriSpeech"
echo "============================================"

# Stage 1: Check manifest data
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Checking manifest data..."
    
    if [ ! -f ${manifest_dir}/train_960/dataset.json ]; then
        echo "Error: Manifest data not found!"
        echo "Please run: ./prep.sh"
        exit 1
    fi
    
    echo "✓ Manifest data found"
    echo "  - Train: ${manifest_dir}/train_960/dataset.json"
    echo "  - Dev: ${manifest_dir}/dev/dataset.json"
fi

# Stage 2: Prepare length statistics
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo ""
    echo "Stage 2: Preparing length statistics..."
    
    if [ -f ${stats_dir}/stats_audio_to_text_libri960.jsonl ]; then
        echo "Stats already exist, skipping..."
    else
        mkdir -p ${stats_dir}
        
        python3 ../../../espnet2/speechlm/bin/prepare_length_stats.py \
            --train-config conf/train_speechlm_asr.yaml \
            --output-dir ${stats_dir} \
            --train-unregistered-specifier "audio_to_text:libri960:${manifest_dir}/train_960/dataset.json:1.0" \
            --valid-unregistered-specifier "audio_to_text:libri_dev:${manifest_dir}/dev/dataset.json:1.0" \
            --num-workers 32 \
            --log-level INFO
        
        echo "✓ Length statistics prepared"
    fi
fi

# Stage 3: Start training
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ""
    echo "Stage 3: Starting training with ${ngpu} GPUs..."
    echo "Output directory: ${output_dir}"
    
    # Check CUDA availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: CUDA not available"
        exit 1
    fi
    
    # Set a random master port to avoid port conflicts
    MASTER_PORT=$((29500 + RANDOM % 1000))
    echo "Using master port: $MASTER_PORT"
    
    # Launch training with DeepSpeed
    mkdir -p ${output_dir}
    deepspeed --num_gpus=${ngpu} \
        --master_port=$MASTER_PORT \
        ../../../espnet2/speechlm/bin/train.py \
        --train-config conf/train_speechlm_asr.yaml \
        --output-dir ${output_dir} \
        --train-unregistered-specifier "audio_to_text:libri960:${manifest_dir}/train_960/dataset.json:1.0" \
        --valid-unregistered-specifier "audio_to_text:libri_dev:${manifest_dir}/dev/dataset.json:1.0" \
        --stats-dir ${stats_dir} \
        --wandb-name "speechlm_librispeech_asr" \
        --wandb-tags baseline librispeech \
        --log-level INFO
    
    echo ""
    echo "============================================"
    echo "Training completed!"
    echo "Checkpoints saved in: ${output_dir}/checkpoints/"
    echo "Logs saved in: ${output_dir}/wandb/"
    echo "============================================"
fi

echo ""
echo "To resume training, use:"
echo "  --resume_path ${output_dir}/checkpoints/step_XXXXX"

