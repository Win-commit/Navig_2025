#!/bin/bash
#SBATCH --job-name=inference # 作业名称
#SBATCH --output=inference.out     # 标准输出和错误输出文件
#SBATCH --error=inference.err      # 错误输出文件
#SBATCH --time=1:00:00           # 运行时间 (HH:MM:SS)
#SBATCH --partition=scavenger         # 分区名称
#SBATCH --account=scavenger       # 账号名称
#SBATCH --gres=gpu:rtx6000ada:1 
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

python inference.py --model "qwen"  --model_path /fs/clip-projects/geoguesser/vlms/qwen/Qwen2-VL-7B-Instruct --ckpt_dir /fs/clip-projects/geoguesser/vlms/qwen/output/qwen2-vl-7b-instruct/v5-20241108-053635/checkpoint-534  --image_path "examples/-33.3741232_-70.661976.jpg"   --crop_box_treshold 0.3 --crop_text_treshold 0.25 

# "/fs/clip-projects/geoguesser/vlms/llava/output/llava1_6-vicuna-7b-instruct/v10-20241108-045625/checkpoint-534"
# "/fs/clip-projects/geoguesser/vlms/cpm/output/minicpm-v-v2_6-chat/v3-20241108-065955/checkpoint-534"
