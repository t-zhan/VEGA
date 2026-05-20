# 开发者待办事项

> **VEGA = VLA Embedding Geometry of Action**
> 
> 研究目标：分析 VLA 模型中 action 表示的几何性质（各向同性、内在维度、流形结构、跨模型对齐性……）；
> 
> 研究对象： AutoVLA（离散 codebook token）和 OpenDriveVLA（连续轨迹坐标）。

---

## 阶段一：基础设施——提取 Action 表示

### 1.1 环境搭建
- [ ] 搭建 `autovla` conda 环境，验证 Qwen2.5-VL-3B 推理流水线可运行
- [ ] 搭建 `drivevla` conda 环境，完成 mmcv/mmdet3d 源码编译，验证 nuScenes 推理可运行
- [ ] 编写 `scripts/autovla/1_setup_env.sh`（conda 环境 + navsim 安装）
- [ ] 编写 `scripts/opendrivevla/1_setup_env.sh`（conda 环境 + 编译步骤）

### 1.2 AutoVLA Action 表示提取
- [ ] Hook Qwen2.5-VL 最后几层 hidden states，在生成 `<action_i>` token 时保存
  - 目标：得到每个 action token 对应的上下文表示向量 `h ∈ R^d`
  - 实现：`src/extraction/autovla_hidden_extractor.py`
- [ ] 提取 `ActionTokenizer` codebook 向量（`codebook_cache/*.pkl`）
  - codebook shape: `(n_bins, 6, 4, 2)`，需 flatten 为 `(n_bins, D_traj)` 后分析
  - 实现：`src/extraction/autovla_codebook_loader.py`
- [ ] 批量推理 nuPlan/nuScenes 验证集，保存 (scene_id, action_token_id, hidden_state) 三元组

### 1.3 OpenDriveVLA Action 表示提取
- [ ] Hook LLaVA backbone 最后隐层，在输出轨迹坐标 token 时保存隐向量
  - 实现：`src/extraction/opendrivevla_hidden_extractor.py`
- [ ] 提取预测轨迹坐标（`pred_trajs_dict.json`），构建 (scene_id, traj_coords, hidden_state) 三元组
- [ ] 批量推理 nuScenes 验证集，保存提取结果

---

## 阶段二：Action 空间几何分析

### 2.1 AutoVLA Codebook 几何分析
- [ ] **覆盖率分析**：codebook 对实际 nuPlan/nuScenes 轨迹空间的量化误差（重构误差分布）
  - `src/analysis/codebook_coverage.py`
- [ ] **簇内/簇间结构**：codebook 向量的 inter-cluster distance matrix，检验 K-means 是否产生均匀分割
- [ ] **轨迹语义一致性**：相邻 codebook token（`<action_i>` 与 `<action_i+1>`）对应轨迹是否语义连续（直行→转弯的过渡是否平滑）
- [ ] **Token embedding 各向同性**：计算 `<action_i>` token embedding 在 LLM embedding table 中的各向同性指标（average cosine similarity、singular value spectrum）

### 2.2 AutoVLA 隐空间几何分析
- [ ] **内在维度估计**：对 hidden states 使用 TwoNN 或 MLE 估计内在维度，比较 fast-thinking（无 CoT）vs slow-thinking（有 CoT）的差异
  - `src/analysis/intrinsic_dim.py`
- [ ] **表示各向同性**：计算 action hidden states 的 isotropy score（参考 Ethayarajh 2019）
- [ ] **CoT 对 action 表示的影响**：对比有/无 CoT 时同一场景的 action hidden vector，量化差异（cosine distance、L2 distance）

### 2.3 OpenDriveVLA 轨迹流形分析
- [ ] **轨迹分布降维**：对预测轨迹坐标用 PCA/UMAP 降维，可视化轨迹流形结构
  - `src/analysis/traj_manifold.py`
- [ ] **BEV 特征与轨迹的对齐性**：计算 BEV feature vector 与对应 hidden state 的 CKA（Centered Kernel Alignment）相似度
- [ ] **连续轨迹隐空间内在维度**：与 AutoVLA 离散 hidden states 对比

### 2.4 跨模型对齐分析
- [ ] **同场景 action 表示对比**（需构建 nuScenes 上两模型的共同子集）
  - AutoVLA 输出：`(hidden_state, action_token_id)` 
  - OpenDriveVLA 输出：`(hidden_state, traj_coords)`
- [ ] **表示空间 CKA 对齐矩阵**：逐层计算两模型隐层的 CKA，定位 action 语义对齐最强的层
  - `src/analysis/cross_model_cka.py`
- [ ] **动作语义几何一致性**：相同驾驶行为（如左转直行）在两模型 hidden space 中的欧氏距离分布是否一致

---

## 阶段三：可视化与消融

- [ ] **Codebook Voronoi 可视化**：将 nuPlan 轨迹投影到 2D，画出 codebook 的 Voronoi 分割
  - `src/visualization/codebook_voronoi.py`
- [ ] **Hidden state PCA/UMAP 轨迹**：按场景类型（直行/转弯/变道）着色，检验几何可分性
  - `src/visualization/hidden_umap.py`
- [ ] **Isotropy 随训练 step 的变化曲线**（需要 AutoVLA checkpoints，若可获取）
- [ ] **消融：RFT 是否改善 action 表示几何性质**（SFT checkpoint vs RFT checkpoint 对比）

---

## 阶段四：文档与复现包

- [ ] 填充 `docs/1_INSTALL.md`（两套环境安装 + 分析工具依赖）
- [ ] 填充 `docs/2_DATA_PREP.md`（nuPlan + nuScenes 数据准备，重点是推理输出缓存）
- [ ] 填充 `docs/3_TRAIN.md`（本项目无新训练，此节记录如何复现两模型推理）
- [ ] 填充 `docs/4_EVAL.md`（几何指标计算流程）

---

## 已完成

- [x] 初始化项目结构
- [x] 配置 git submodule（AutoVLA、OpenDriveVLA）
- [x] 配置 `.gitignore`
- [x] 生成中文 README 与 docs/ 文档框架
- [x] 确定 scripts/ 目录结构（`scripts/autovla/`、`scripts/opendrivevla/`）
- [x] 分析两子模块 action 表示方式与技术架构
