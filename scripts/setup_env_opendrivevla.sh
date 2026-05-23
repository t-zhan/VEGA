#!/usr/bin/env bash
# =============================================================================
# setup_env_opendrivevla.sh
# 搭建 OpenDriveVLA (drivevla) conda 环境
#
# 约束：不修改项目代码，所有依赖冲突通过固定包版本解决
# 成功标准：4 卡 torchrun 推理脚本跑通
#
# 用法：
#   bash scripts/setup_env_opendrivevla.sh
#
# 可选环境变量：
#   CUDA_VERSION   torch wheel 版本，支持 11.8 / 12.1（默认 12.1）
#   CUDA_HOME      CUDA 安装路径（默认 /usr/local/cuda）
# =============================================================================
set -euo pipefail

# ---------- 可配置项 ----------
ENV_NAME="drivevla"
PYTHON_VERSION="3.10"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

# ---------- CUDA_HOME ----------
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d "/usr/local/cuda-12.4" ]]; then
        export CUDA_HOME="/usr/local/cuda-12.4"
    elif [[ -d "/usr/local/cuda" ]]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "[ERROR] CUDA_HOME not set and no CUDA found under /usr/local/."
        echo "        Please set: export CUDA_HOME=/path/to/cuda"
        exit 1
    fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "[INFO] CUDA_HOME=${CUDA_HOME}"

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENDRIVEVLA_ROOT="${REPO_ROOT}/third_party/OpenDriveVLA"

# ---------- conda init ----------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# 1. 创建 conda 环境
# =============================================================================
echo ""
echo "[1/7] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"
pip install --upgrade pip

# =============================================================================
# 2. PyTorch（必须在所有编译步骤之前）
# =============================================================================
echo ""
echo "[2/7] Installing PyTorch 2.1.2 (CUDA ${CUDA_VERSION})..."
if [[ "${CUDA_VERSION}" == "12.1" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi
pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url "${TORCH_INDEX}"

# =============================================================================
# 3. mmcv-full 源码编译（依赖 torch，需 CUDA_HOME）
# =============================================================================
echo ""
echo "[3/7] Building mmcv-full from source (mmcv_1_7_2)..."
# conda 打包的 setuptools 82.x 不含 pkg_resources，mmcv setup.py 依赖该模块
# 降级到 69.5.1（已知兼容版本），不影响其他包运行
pip install --force-reinstall "setuptools==69.5.1"
cd "${OPENDRIVEVLA_ROOT}/third_party/mmcv_1_7_2"
pip install ninja psutil   # optional.txt
# pip install 26.x 的 metadata 子进程不暴露 pkg_resources，改用 setup.py install
# setup.py 会通过 easy_install 自动安装 yapf/addict/pyyaml 等依赖
MMCV_WITH_OPS=1 python setup.py install

# =============================================================================
# 4. mmdet / mmseg / mmengine 及辅助包
# =============================================================================
echo ""
echo "[4/7] Installing mmdet, mmseg, mmengine and auxiliaries..."
pip install \
    mmdet==2.26.0 \
    mmsegmentation==0.29.1 \
    mmengine==0.9.0 \
    motmetrics==1.4.0 \
    casadi==3.6.0

# =============================================================================
# 5. mmdet3d runtime deps + 源码编译
#    显式固定版本，避免与 LLaVA deps 冲突
# =============================================================================
echo ""
echo "[5/7] Building mmdet3d from source (mmdetection3d_1_0_0rc6)..."
# 注意：mmdet3d 的 requirements.txt 可能将 numpy 升级至 2.x
# 先显式安装必要包，mmdet3d 编译后再重新钉住 numpy==1.26.4
pip install \
    "numpy==1.26.4" \
    "scipy>=1.10.1,<1.14" \
    "scikit-image>=0.19.3" \
    "numba>=0.59.0" \
    "trimesh>=2.35.39,<2.35.40" \
    networkx \
    plyfile \
    nuscenes-devkit \
    tensorboard \
    fsspec

cd "${OPENDRIVEVLA_ROOT}/third_party/mmdetection3d_1_0_0rc6"
FORCE_CUDA=1 pip install --no-build-isolation .

# mmdet3d requirements.txt 可能将 numpy 升级至 2.x，重新钉住至 1.x（torch 2.1.2 ABI 兼容）
pip install "numpy==1.26.4"

# =============================================================================
# 6. LLaVA / OpenDriveVLA Python 依赖
#    所有版本显式固定，防止 pip 升级破坏已编译包
# =============================================================================
echo ""
echo "[6/7] Installing LLaVA / OpenDriveVLA Python dependencies..."
pip install \
    "pydantic==1.10.8" \
    "tokenizers~=0.15.2" \
    "transformers>=4.36.0,<4.40.0" \
    "sentencepiece~=0.1.99" \
    "peft==0.4.0" \
    "accelerate==0.29.3" \
    "deepspeed==0.14.2" \
    "bitsandbytes==0.41.0" \
    "datasets==2.16.1" \
    "scikit-learn==1.2.2" \
    "einops==0.6.1" \
    "einops-exts==0.0.4" \
    "pytorch-lightning==1.2.5" \
    "torchmetrics==0.11.4" \
    "httpx==0.24.0" \
    "gradio==3.35.2" \
    "gradio_client==0.2.9" \
    "urllib3<=2.0.0" \
    open_clip_torch \
    timm \
    hf_transfer \
    opencv-python-headless \
    av \
    decord \
    tyro \
    shortuuid \
    ftfy \
    wandb \
    markdown2 \
    "uvicorn" \
    "fastapi" \
    "requests"

# LLaVA 依赖安装后重新钉住 numpy（部分包可能再次拉升 numpy 至 2.x）
pip install "numpy==1.26.4"

# transformers: pinned commit（LLaVA 依赖定制版，不可替换为 PyPI 版本）
pip install \
    "transformers @ git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4"

# flash-attn: 直接用 URL 安装预编译 wheel（torch 2.1.x + CUDA 12.x + cp310）
# 注：wheel 名中含 + 号（PEP 440 local version），不能用 pip install /path/to/.whl；
#     --no-build-isolation 方式会因 os.rename() 跨设备 (EXDEV errno 18) 失败；
#     直接 URL 安装是最可靠的方式。
pip install flash-attn==2.5.7

# =============================================================================
# 7. 安装 OpenDriveVLA 本体（editable，--no-deps 防止覆盖已固定的包）
# =============================================================================
echo ""
echo "[7/7] Installing OpenDriveVLA (editable)..."
cd "${OPENDRIVEVLA_ROOT}"
pip install -e . --no-deps

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "============================================================"
echo " drivevla 环境安装完成！"
echo ""
echo " 激活方式:  conda activate ${ENV_NAME}"
echo ""
echo " 验证（4 卡推理）:"
echo "   cd ${OPENDRIVEVLA_ROOT}"
echo "   bash scripts/eval_drivevla.sh checkpoints/OpenDriveVLA-0.5B 4"
echo "============================================================"
