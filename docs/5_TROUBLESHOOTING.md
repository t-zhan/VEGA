# Troubleshooting

## 子模块目录为空

执行 `git clone` 后未初始化子模块：

```shell
git submodule update --init --recursive
```

## `ImportError: libGL.so.1: cannot open shared object file`

缺少系统图形库（常见于无头服务器环境）：

```shell
apt-get update && apt-get install libgl1
```

## `libgfortran.so.5: cannot open shared object file`

```shell
sudo apt-get install libgfortran5
# 或
conda install libgfortran
```

## 编译 mmcv / mmdet3d 时找不到 CUDA

```shell
export CUDA_HOME=/usr/local/cuda   # 替换为实际路径
```

## nuScenes 预处理与主环境依赖冲突

`nuscenes-devkit` 与 AutoVLA 主环境存在冲突，建议新建独立 conda 环境进行预处理：

```shell
conda create -n nuscenes_preprocess python=3.10 -y
conda activate nuscenes_preprocess
pip install nuscenes-devkit
```

## navtrain 数据集下载后文件损坏

用 MD5 校验各分卷，详见 [third_party/AutoVLA/navsim/docs/splits.md](third_party/AutoVLA/navsim/docs/splits.md)：

```bash
echo "6f92f38d5f03ed852da7872a7122bdd2  navtrain_current_1.tgz" | md5sum -c -
echo "7a72f0a758b5df6cbe4c565920a4869f  navtrain_current_2.tgz" | md5sum -c -
```
