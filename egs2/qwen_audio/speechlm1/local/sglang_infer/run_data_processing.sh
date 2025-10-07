#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/data_process_%j.out
#SBATCH --error=logs/data_process_%j.err

# 创建日志目录
mkdir -p logs

# 加载环境
source ~/.bashrc
conda activate espnet2

echo "=================================================="
echo "Data Processing Job Starting"
echo "=================================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# 读取 SGLang 服务器信息
if [ -f "sglang_server_info.txt" ]; then
    echo "Reading SGLang server info..."
    source sglang_server_info.txt
    echo "  Server URL: $SGLANG_URL"
    echo "  Server Node: $NODE_HOSTNAME"
    echo ""
else
    echo "Warning: sglang_server_info.txt not found"
    echo "Make sure SGLang server is running first!"
    echo ""
fi

# 设置参数
DATA_DIR="data"
DUMP_DIR="dump"
DUMP_AUDIO_DIR="dump_audio"
PROMPT_JSON="osr_prompt.json"

# 处理所有数据集分割
for SPLIT in librimix2_train librimix2_dev librimix2_test; do
    echo "=================================================="
    echo "Processing: $SPLIT"
    echo "=================================================="
    
    python local/osr/prepare_librimix2_dialogue.py \
        --data_dir ${DATA_DIR} \
        --dump_dir ${DUMP_DIR} \
        --dump_audio_dir ${DUMP_AUDIO_DIR} \
        --split ${SPLIT} \
        --prompt_json ${PROMPT_JSON} \
        --use_punctuation_restoration \
        --model_name Qwen/Qwen3-4B \
        --use_sglang \
        --sglang_url ${SGLANG_URL} \
        --batch_size 1
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $SPLIT"
    else
        echo "✗ Error processing $SPLIT"
        exit 1
    fi
    echo ""
done

echo "=================================================="
echo "All processing completed at $(date)"
echo "=================================================="