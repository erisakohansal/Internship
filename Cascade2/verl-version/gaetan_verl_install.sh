module load env/staging/2024.1
module load CUDA-Python/12.6.0-gfbf-2024a-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load CMake/3.29.3-GCCcore-13.3.0
module load Ninja/1.12.1-GCCcore-13.3.0

VENV_PATH="/mnt/tier1/project/p201382/erisa/Internship/Cascade2/.venv" 
python -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

export PYTHONPATH=""

pip install --upgrade pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate datasets tensorboard liger-kernel hydra-core peft einops

export FLASH_ATTN_CUDA_ARCHS="80"
MAX_JOBS=16 pip install flash-attn --no-build-isolation --no-deps

# vllm and verl, download rustup for vllm
# configure RUSTUP_HOME and CARGO_HOME to a directory where you have write permissions, e.g., your home directory or a project-specific directory. This will allow you to install Rust without requiring root access.
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# . "/mnt/tier1/project/p201382/erisa/cargo/env"
pip install -v vllm --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu126

cd verl/ pip install -e . --no-deps
cd verifiable-instructions/ pip install -e . --no-deps
pip install -U "ray[data,train,tune,serve]"
pip install codetiming
pip install pybind11 pylatexenc torchdata wandb
pip install tensordict==0.10.0 # Error in _run_prompt: Storing non-tensor data in TensorDict at least requires tensordict version 0.10
pip install --no-deps TransferQueue 

export VLLM_VERSION="0.24.0"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export CUDA_VERSION_VLLM="129" # or 126
export CUDA_VERSION_TORCH="126" # or 126
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION_VLLM}-cp38-abi3-manylinux_2_28_x86_64.whl--extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_TORCH}

# verifiable instructions dependencies
pip install langdetect
pip install immutabledict
pip install nltk
# then for nltk, in a python terminal
# import nltk
# nltk.download('punkt_tab', download_dir='/mnt/tier1/project/p201382/erisa/nltk')
# also set the NLTK_DATA environment variable to point to the directory where you downloaded the data, e.g., export NLTK_DATA=/mnt/tier1/project/p201382/erisa/nltk