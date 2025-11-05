#!/bin/bash
# Test script for single GPU training (easier debugging)
set -e
set -u

# Set CUDA library path for flash attention compatibility
# Add nvidia CUDA runtime library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

manifest_dir=./manifest
stats_dir=./exp/speechlm_stats
output_dir=./exp/test_single_gpu

echo "=== Testing with Single GPU (easier debugging) ==="

# Use smaller dataset for quick test
train_spec="audio_to_text:libri_dev:${manifest_dir}/dev/dataset.json:1.0"
valid_spec="audio_to_text:libri_dev_clean:${manifest_dir}/dev_clean/dataset.json:1.0"

# Check if stats exist
if [ ! -f ${stats_dir}/stats_audio_to_text_libri_dev.jsonl ]; then
    echo "Generating stats for dev set..."
    mkdir -p ${stats_dir}
    
    python3 ../../../espnet2/speechlm/bin/prepare_length_stats.py \
        --train-config conf/train_speechlm_asr.yaml \
        --output-dir ${stats_dir} \
        --train-unregistered-specifier "${train_spec}" \
        --valid-unregistered-specifier "${valid_spec}" \
        --num-workers 2 \
        --log-level INFO
fi

echo ""
echo "Starting single-GPU training..."
echo "This will help identify any model/config issues"

# Set a random master port to avoid port conflicts
MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"

# Single GPU training (no DeepSpeed distributed complications)
deepspeed --num_gpus=1 \
    --master_port=$MASTER_PORT \
    ../../../espnet2/speechlm/bin/train.py \
    --train-config conf/train_speechlm_asr.yaml \
    --output-dir ${output_dir} \
    --train-unregistered-specifier "${train_spec}" \
    --valid-unregistered-specifier "${valid_spec}" \
    --stats-dir ${stats_dir} \
    --wandb-name "test_single_gpu" \
    --wandb-tags test debug \
    --log-level INFO 2>&1 | tee ${output_dir}/train.log

echo ""
echo "Check logs at: ${output_dir}/train.log"

