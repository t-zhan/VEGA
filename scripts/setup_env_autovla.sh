#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="autovla"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"

if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d "/usr/local/cuda-12.4" ]]; then
        export CUDA_HOME="/usr/local/cuda-12.4"
    elif [[ -d "/usr/local/cuda" ]]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "[ERROR] CUDA_HOME not found. Set: export CUDA_HOME=/path/to/cuda"
        exit 1
    fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "[INFO] CUDA_HOME=${CUDA_HOME}"

# 网络代理（用于访问 GitHub 等；步骤5下载模型前会 unset）
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
echo "[INFO] Proxy: ${https_proxy}"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "[1/5] Creating conda environment '${ENV_NAME}' (Python 3.9)..."
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    conda create -n "${ENV_NAME}" python=3.9 -y
fi
conda activate "${ENV_NAME}"
pip install --upgrade pip

echo ""
echo "[2/5] Installing PyTorch 2.4.0 (cu121 wheel)..."
pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[3/5] Installing requirements (excluding torch/torchvision)..."
REQ_FILTERED=$(mktemp)
grep -vE '^(torch|torchvision)==' "${AUTOVLA_ROOT}/requirements.txt" > "${REQ_FILTERED}"
pip install -r "${REQ_FILTERED}"
rm -f "${REQ_FILTERED}"

# nuplan-devkit 在模块级 import pytest，需显式安装
pip install pytest -q

echo ""
echo "[4/5] Installing AutoVLA and optional extras..."
cd "${AUTOVLA_ROOT}"
pip install -e .

# flash-attn: 用 curl 下载预编译 wheel（先尝试直连，再走国内镜像代理），安装前验证文件大小
CXX11ABI=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
_FLASH_FNAME="flash_attn-2.7.4.post1+cu12torch2.4cxx11abi${CXX11ABI}-cp39-cp39-linux_x86_64.whl"
_FLASH_LOCAL="/tmp/${_FLASH_FNAME}"
_FLASH_GITHUB="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${_FLASH_FNAME}"
_FLASH_MIRROR="https://ghproxy.com/${_FLASH_GITHUB}"
echo "  Downloading flash-attn 2.7.4.post1 (cxx11abi=${CXX11ABI})..."
_flash_ok=false
for _url in "${_FLASH_GITHUB}" "${_FLASH_MIRROR}"; do
    echo "    Trying ${_url}..."
    rm -f "${_FLASH_LOCAL}"
    curl -fsSL --retry 2 --retry-delay 15 --connect-timeout 30 --max-time 600 \
        -o "${_FLASH_LOCAL}" "${_url}" || true
    _sz=$(stat -c%s "${_FLASH_LOCAL}" 2>/dev/null || echo 0)
    if [[ "${_sz}" -gt 50000000 ]]; then
        echo "    Downloaded ${_sz} bytes, installing..."
        if pip install "${_FLASH_LOCAL}"; then
            _flash_ok=true; break
        fi
    else
        echo "    Download failed or incomplete (${_sz} bytes), trying next source..."
    fi
done
rm -f "${_FLASH_LOCAL}"
${_flash_ok} || { echo "[ERROR] flash-attn install failed"; exit 1; }

# autoawq: 关闭 pip 构建隔离，允许使用当前环境已安装的 torch/numpy 作为构建依赖
# 注意：autoawq 0.2.8 声明 transformers<=4.47.1，但实际运行兼容 4.49.0；
# 安装后强制恢复 AutoVLA 所需的 transformers==4.49.0
echo "  Installing autoawq 0.2.8..."
pip install autoawq==0.2.8 --no-build-isolation
pip install "transformers==4.49.0"

# waymo-open-dataset 1.6.7 要求 jaxlib==0.4.13，cp39 最低可用版本为 0.4.18；
# 修补 wheel 将所有精确约束（==）改为最低约束（>=），使已安装的更新版本被接受
_WMO_DIR=$(mktemp -d)
pip download waymo-open-dataset-tf-2-12-0==1.6.7 --no-deps -d "${_WMO_DIR}" -q
_WMO_SRC=$(ls "${_WMO_DIR}"/*.whl | head -1)
_WMO_DST="${_WMO_SRC%.whl}_patched.whl"
# 修补 wheel：将 METADATA 中所有 Requires-Dist 的精确约束 == 改为 >=
_PATCH_TMP=$(mktemp -d)
cp "${_WMO_SRC}" "${_WMO_DST}"
_META_PATH=$(unzip -l "${_WMO_DST}" | awk '/METADATA$/{print $NF}')
unzip -q "${_WMO_DST}" "${_META_PATH}" -d "${_PATCH_TMP}"
sed -i 's/\(Requires-Dist: [^=]*\)==/\1>=/g' "${_PATCH_TMP}/${_META_PATH}"
pushd "${_PATCH_TMP}" >/dev/null
zip -q "${_WMO_DST}" "${_META_PATH}"
popd >/dev/null
rm -rf "${_PATCH_TMP}"
pip install "${_WMO_DST}"
rm -rf "${_WMO_DIR}"
# waymo 会将 protobuf 升至 6.x，AutoVLA 需要 4.25.3，强制回退
pip install "protobuf==4.25.3"

pip install "numpy<2"

echo ""
echo "[5/5] Checking Qwen2.5-VL-3B-Instruct model..."
cd "${AUTOVLA_ROOT}"
if ls Qwen2.5-VL-3B-Instruct/*.safetensors &>/dev/null; then
    echo "  Model already exists, skipping download."
else
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export HF_ENDPOINT=https://hf-mirror.com
    bash scripts/download_qwen.sh
fi

echo ""
echo "============================================================"
echo " autovla_codeclean 环境安装完成！"
echo " 激活方式: conda activate ${ENV_NAME}"
echo "============================================================"
