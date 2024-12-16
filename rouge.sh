#!/bin/bash
#SBATCH --job-name=5k_quality # 作业名称
#SBATCH --output=5k_quality.out     # 标准输出和错误输出文件
#SBATCH --error=5k_quality.err      # 错误输出文件
#SBATCH --time=1-23:00:00           # 运行时间 (HH:MM:SS)
#SBATCH --partition=scavenger         # 分区名称
#SBATCH --account=scavenger       # 账号名称
#SBATCH --gres=gpu:rtxa5000:1 
#SBATCH --mem=32G                 # 申请32G内存
#SBATCH --cpus-per-task=4         # 每个任务使用的CPU核数

module load cuda/11.8.0

CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TRITON_CACHE_DIR="/fs/clip-projects/geoguesser/zheyuan/cache/triton"
export HF_DATASETS_CACHE="/fs/clip-projects/geoguesser/zheyuan/cache/huggingface/dataset"
export TRANSFORMERS_CACHE="/fs/clip-projects/geoguesser/zheyuan/cache/transformers"
export HUGGINGFACE_HUB_CACHE="/fs/clip-projects/geoguesser/zheyuan/cache/huggingface/hub"
export XDG_CACHE_HOME="/fs/clip-projects/geoguesser/zheyuan/cache/huggingface/xdg"


source ~/.bashrc
conda activate geo

python rouge.py --model qwen_sft --reasoning_path "/fs/clip-projects/geoguesser/reasoning_experiments" --results_file_Name "5k_qwen_sft_quality.jsonl" 
