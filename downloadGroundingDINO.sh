#!/bin/bash
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=0:30:0
#SBATCH --output=geo.out
#SBATCH --error=geo.err
source ~/.bashrc
module load cuda/11.8.0
module load gcc/11.2.0  
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

conda activate geo
pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 11.8
if [ ! -d "GroundingDINO" ]; then
	    git clone https://github.com/IDEA-Research/GroundingDINO.git
    else
	        echo "GroundingDINO directory already exists. Skipping clone."
		    cd GroundingDINO
		        git pull
			    cd ..
		    fi

cd GroundingDINO/

pip install -e .
pip install openai
pip install cnocr
pip install ftfy regex tqdm

#IF failed with  'GLIBCXX_3.4.30' not found ,Run:conda install -c conda-forge libstdcxx-ng
mkdir -p weights
cd weights
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
	    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    else
	        echo "Pre-trained weights already downloaded."
	fi
	cd ..

	cd ..


