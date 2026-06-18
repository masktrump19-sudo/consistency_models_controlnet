#!/usr/bin/env bash
# Create the `cm` conda environment for the edge-ControlNet project.
# Run this in YOUR interactive shell (where conda can write env dirs and the
# GPU driver is available) -- NOT inside the restricted assistant sandbox.
#
#   bash scripts/setup_cm_env.sh
#   conda activate cm
#
# Adjust CU_TAG to match your NVIDIA driver (A6000 works with cu117/cu118/cu121).
set -e

ENV_NAME=${ENV_NAME:-cm}
CU_TAG=${CU_TAG:-cu118}
TORCH=${TORCH:-2.0.1}
TV=${TV:-0.15.2}

echo ">> creating conda env: $ENV_NAME (python 3.9)"
conda create -y -n "$ENV_NAME" python=3.9

echo ">> installing mpi4py + openmpi from conda-forge (no system MPI needed)"
conda install -y -n "$ENV_NAME" -c conda-forge mpi4py openmpi

echo ">> installing torch $TORCH ($CU_TAG) + project deps via pip"
# --no-capture-output streams pip's progress; without it `conda run` buffers the
# output and the ~2.3GB torch download looks frozen. flash-attn==0.2.8 is skipped
# (won't build on modern stacks); the UNet falls back to plain-torch attention.
# China network? export CN_MIRROR=1 to use SJTU (torch) + Tsinghua (PyPI) mirrors.
RUN="conda run --no-capture-output -n $ENV_NAME python -m pip install"

if [ "${CN_MIRROR:-0}" = "1" ]; then
    echo "   (CN_MIRROR=1: using SJTU torch wheels + Tsinghua PyPI)"
    $RUN \
        "https://mirror.sjtu.edu.cn/pytorch-wheels/${CU_TAG}/torch-${TORCH}%2B${CU_TAG}-cp39-cp39-linux_x86_64.whl" \
        "https://mirror.sjtu.edu.cn/pytorch-wheels/${CU_TAG}/torchvision-${TV}%2B${CU_TAG}-cp39-cp39-linux_x86_64.whl"
    $RUN -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "numpy<2" piq==0.7.0 blobfile einops opencv-python-headless pillow \
        scipy tqdm matplotlib requests gradio==4.44.1 "huggingface_hub<1.0"
else
    $RUN --extra-index-url "https://download.pytorch.org/whl/${CU_TAG}" \
        "torch==${TORCH}" "torchvision==${TV}"
    $RUN "numpy<2" piq==0.7.0 blobfile einops opencv-python-headless pillow \
        scipy tqdm matplotlib requests gradio==4.44.1 "huggingface_hub<1.0"
fi

echo ">> making the in-repo \`cm\` package importable (skips flash-attn dep)"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e . --no-deps

echo ">> verifying"
conda run -n "$ENV_NAME" python -c "import torch, cv2, piq, blobfile, einops, mpi4py, gradio; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

echo "Done. Activate with:  conda activate $ENV_NAME"
