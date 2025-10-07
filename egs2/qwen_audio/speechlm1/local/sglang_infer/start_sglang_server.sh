#!/bin/bash
#SBATCH --job-name=sglang_server
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # 使用 1 个 GPU
#SBATCH --mem=32G
#SBATCH --time=24:00:00           # 运行 24 小时
#SBATCH --output=logs/sglang_server_%j.out
#SBATCH --error=logs/sglang_server_%j.err

# 创建日志目录
mkdir -p logs

. ./path.sh

# 设置端口（可以改成其他端口）
PORT=30000

# 获取节点的主机名和 IP
NODE_HOSTNAME=$(hostname)
NODE_IP=$(hostname -i)

echo "=================================================="
echo "SGLang Server Starting"
echo "=================================================="
echo "Node: $NODE_HOSTNAME"
echo "IP: $NODE_IP"
echo "Port: $PORT"
echo "Time: $(date)"
echo "=================================================="
echo ""
echo "To connect from your data processing script:"
echo "  --sglang_url http://${NODE_IP}:${PORT}"
echo ""
echo "=================================================="

# 将服务器信息写入文件，供其他作业读取
SERVER_INFO_FILE="sglang_server_info.txt"
echo "SGLANG_URL=http://${NODE_IP}:${PORT}" > $SERVER_INFO_FILE
echo "NODE_HOSTNAME=${NODE_HOSTNAME}" >> $SERVER_INFO_FILE
echo "NODE_IP=${NODE_IP}" >> $SERVER_INFO_FILE
echo "PORT=${PORT}" >> $SERVER_INFO_FILE
echo "JOB_ID=${SLURM_JOB_ID}" >> $SERVER_INFO_FILE

echo "Server info saved to: $SERVER_INFO_FILE"
echo ""

module load cuda/12.3.0

# 启动 SGLang 服务器

# 如果要使用 Qwen3-4B，取消下面的注释
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port ${PORT} \
    --host 0.0.0.0 \
    --disable-cuda-graph \
    --mem-fraction-static 0.8

echo ""
echo "SGLang server stopped at $(date)"