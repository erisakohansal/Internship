#!/bin/bash -l
#SBATCH --job-name=if_rl_verl_fraction
#SBATCH --account=p201382
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erisa.kohansal@linguacustodia.com

set -xeuo pipefail

module load env/staging/2024.1
module load CUDA-Python/12.6.0-gfbf-2024a-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load CMake/3.29.3-GCCcore-13.3.0
module load Ninja/1.12.1-GCCcore-13.3.0

source /project/scratch/p201382/erisa/Internship/Cascade2/verl-version/.venv/bin/activate
cd /project/scratch/p201382/erisa/Internship/Cascade2/verl-version
export PYTHONPATH=""

./launch.sh
