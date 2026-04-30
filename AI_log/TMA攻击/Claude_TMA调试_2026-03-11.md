# TMA 攻击调试日志

**日期**: 2026-03-11
**任务**: 运行 TMA 攻击并解决显存问题

---

## 已解决的问题

### 1. 模型加载问题
- ✅ 修复 `local_files_only=True` 参数传递
- ✅ 修复数据集路径（从 `openvla-main/dataset` 改为 `dataset`）
- ✅ 修复数据集名称（从 `libero_spatial` 改为 `libero_spatial_no_noops`）

### 2. 数据集下载问题
- ✅ 识别 Git LFS 指针文件问题
- ✅ 用户已下载完整数据集（文件大小从 134B 变为 100MB+）

### 3. 代码兼容性问题
- ✅ 移除 `colorjitter` 参数（函数不支持）
- ✅ 修复 DataParallel 兼容性（添加 `self.device` 属性）

---

## 当前问题：显存不足

### 问题描述
即使使用 DataParallel 双 GPU，仍然 OOM：
- GPU 0: 23.52 GiB / 23.54 GiB (几乎满载)
- GPU 1: 未充分利用

### 原因分析
`torch.nn.DataParallel` 的限制：
1. 模型主副本在 GPU 0
2. 前向传播在多 GPU 并行
3. **梯度聚合在 GPU 0**（导致 GPU 0 显存压力大）

### 尝试过的方案
1. ❌ batch_size=8 → OOM
2. ❌ batch_size=4 → OOM
3. ❌ batch_size=2 → OOM
4. ❌ DataParallel (bs=8) → OOM (GPU 0 满载)
5. ❌ batch_size=1 → OOM (22.43 GiB / 23.54 GiB)
6. ❌ batch_size=1 + 梯度检查点 → OOM (22.52 GiB / 23.54 GiB)
7. ❌ DDP (bs=2, 双 GPU) → OOM (GPU 0: 22.88 GB, GPU 1: 22.88 GB)

**结论**：OpenVLA-7B 模型本身占用约 22.5-22.9 GB 显存。即使使用 DDP，每个 GPU 都需要加载完整模型副本，仍然无法运行。

## 最终结论

**24GB 显存的 GPU 无法运行 OpenVLA-7B 的训练**，无论使用单 GPU 还是 DDP 多 GPU。

需要：
- 更大显存的 GPU (如 A100 40GB/80GB)
- 或使用模型并行（将模型切分到多个 GPU）
- 或使用 8-bit/4-bit 量化

---

## 解决方案

### 方案一：使用 DistributedDataParallel (推荐)
使用 DDP 替代 DataParallel，每个 GPU 独立维护模型副本和优化器状态。

参考 UADA_ddp 实现：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
    VLAAttacker/TMA_wrapper_ddp.py [参数]
```

需要创建 `TMA_wrapper_ddp.py`。

### 方案二：减小模型或使用梯度累积
- 使用更小的 batch size (bs=1) + 更大的梯度累积步数
- 启用梯度检查点（gradient checkpointing）

### 方案三：使用单 GPU + 优化
- batch_size=1
- 启用混合精度训练
- 减小 patch size

---

## 建议

**推荐使用方案一（DDP）**，因为：
1. 显存分布更均匀
2. 训练速度更快
3. 代码库已有 UADA_ddp 参考实现

是否需要我创建 `TMA_wrapper_ddp.py`？

---

## 更新：2026-03-11 22:01

### DDP 测试结果
✅ 已创建 `TMA_wrapper_ddp.py` 和 `TMA_ddp.py`
❌ DDP 仍然 OOM：
- GPU 0: 22.88 GB
- GPU 1: 22.88 GB
- 原因：每个 GPU 都需要加载完整模型副本

### 8-bit 量化测试
❌ 需要先安装 `accelerate` 库：
```bash
pip install accelerate
```

### 最终结论

**24GB GPU 无法运行 OpenVLA-7B 训练**，即使使用 DDP。

**唯一可行方案：8-bit 量化**
- 需要安装：`pip install accelerate`
- 预期显存：~11-12 GB（可在 24GB GPU 上运行）
- 已修改 `TMA_wrapper.py` 支持 8-bit 量化

**其他方案：**
- 使用更大显存 GPU（A100 40GB/80GB）
- 使用 4-bit 量化（需要进一步修改代码）
- 使用模型并行（DeepSpeed ZeRO-3）
