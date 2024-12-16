#!/bin/bash
#SBATCH --job-name=4kFramecpm # 作业名称
#SBATCH --output=4kFramecpm.out     # 标准输出和错误输出文件
#SBATCH --error=4kFramecpm.err      # 错误输出文件
#SBATCH --time=1-12:00:00           # 运行时间 (HH:MM:SS)
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1 
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

python evaluation.py --model "cpm" --model_path /fs/clip-projects/geoguesser/vlms/cpm/MiniCPM-V-2_6 --ckpt_dir /fs/clip-projects/geoguesser/vlms/cpm/output/minicpm-v-v2_6-chat/v3-20241108-065955/checkpoint-534 --dataset_path "/fs/clip-projects/geoguesser/runze/data/yfcc4k" --reasoning_path "/fs/clip-projects/geoguesser/runze/data/results/yfcc4k_GuesserPro(cpm)" --results_file_Name "results_s6_cpm.jsonl"  --crop_box_treshold 0.5 --crop_text_treshold 0.5
