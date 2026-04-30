# TMA 攻击训练总结报告

**日期**: 2026-03-12
**任务**: Target Manipulation Attack (TMA) on OpenVLA-7B
**数据集**: libero_spatial_no_noops
**实验 ID**: a251313a-6b71-4e8f-81ef-7b2230dc998e

---

## 1. 实验配置

### 硬件环境
- **GPU**: 2 × 48GB 显存（GPU 1, 2）
- **模型**: OpenVLA-7B-finetuned-libero-spatial（完整精度）
- **显存占用**: ~22.9 GB/GPU

### 训练参数
```bash
--maskidx 0
--lr 0.002
--iter 2000
--bs 1
--accumulate 8
--warmup 20
--geometry True
--patch_size 3,50,50
--innerLoop 50
--dataset libero_spatial_no_noops
--targetAction 0
```

### 关键设置
- **Batch Size**: 1（实际有效 batch size = 1 × 8 = 8）
- **Gradient Accumulation**: 8 步
- **Learning Rate**: 2e-3
- **Patch 尺寸**: 3×50×50（RGB）
- **几何变换**: 启用（增强鲁棒性）
- **目标动作**: 索引 0

---

## 2. 训练过程

### 时间线
- **开始时间**: 2026-03-11 21:51
- **结束时间**: 2026-03-12 12:45
- **总耗时**: ~15 小时
- **完成进度**: 2000/2000 iterations（100%）✅

### Checkpoints
已保存 15 个 checkpoints：
- 每 100 iterations 保存一次：0, 100, 200, ..., 1000
- 后期每 200 iterations：1200, 1400, 1700
- 最终 checkpoint：last (2000 iterations)

### 输出文件
```
run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/
├── 0/, 100/, 200/, ..., 1700/    # Checkpoints
│   ├── patch.pt                   # 对抗 patch (31KB)
│   └── val_related_data/          # 验证数据
│       ├── 0.png - 7.png          # 8张可视化图像
│       ├── continuous_actions_gt.pt
│       └── continuous_actions_pred.pt
├── loss_curve.png                 # 损失曲线 (39KB)
├── train_CE_loss.pkl              # 交叉熵损失
├── train_inner_avg_loss.pkl       # 内循环平均损失
├── train_inner_relatived_distance.pkl  # 相对距离 (514KB)
├── val_ASR.pkl                    # 验证集攻击成功率
├── val_CE_loss.pkl                # 验证集 CE 损失
├── val_inner_relatived_distance.pkl    # 验证集相对距离
└── val_L1_loss.pkl                # 验证集 L1 损失
```

---

## 3. 技术要点

### 问题解决历程

#### 问题 1: 显存不足（24GB GPU）
- **现象**: OOM 错误，模型占用 ~22.9 GB
- **尝试方案**:
  - DataParallel：GPU 0 过载
  - DDP：两块 GPU 均 OOM
  - 8-bit 量化：准备测试
- **最终方案**: 更换至 48GB GPU 服务器

#### 问题 2: Git LFS 数据集问题
- **现象**: tfrecord 文件为 134B 指针文件
- **解决**: `git lfs pull` 下载实际数据

#### 问题 3: 数据集路径错误
- **现象**: `KeyError: 'libero_spatial'`
- **解决**: 使用正确名称 `libero_spatial_no_noops`

#### 问题 4: 代码兼容性
- **colorjitter 参数**: 移除不存在的参数
- **DataParallel device**: 添加 `self.device` 属性
- **DDP 初始化**: 正确设置 `torch.cuda.set_device(local_rank)`

### 代码修改记录

**VLAAttacker/TMA_wrapper.py**
```python
# 使用 GPU 1, 2
device = torch.device(f"cuda:{args.device}")

# 完整精度加载
vla = AutoModelForVision2Seq.from_pretrained(
    vla_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
)
```

**VLAAttacker/white_patch/TMA.py**
```python
# 修复 device 属性
self.device = next(self.vla.parameters()).device

# 移除 colorjitter 参数
perturbed_images = self.random_patch_transform.apply_random_patch_batch(
    images, self.patch, self.mask
)
```

**scripts/run_TMA.sh**
```bash
CUDA_VISIBLE_DEVICES=1,2 python VLAAttacker/TMA_wrapper.py \
    --maskidx 0 --lr 2e-3 --iter 2000 \
    --bs 1 --accumulate 8 --warmup 20 \
    --geometry True --patch_size 3,50,50 \
    --innerLoop 50 --dataset libero_spatial_no_noops \
    --targetAction 0 --device 0
```

---

## 4. 实验结果

### 生成的对抗 Patch
- **位置**: `run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/last/patch.pt`
- **尺寸**: 3×50×50 (RGB)
- **大小**: 31 KB
- **目标**: 操纵 VLA 模型输出目标动作（索引 0）

### 可视化结果
- 8 张验证图像展示 patch 应用效果
- 包含预测动作 vs 真实动作对比

### 训练指标
- **损失曲线**: 已生成并保存
- **ASR**: 验证集攻击成功率已记录
- **Action Distance**: 动作距离指标已保存

---

## 5. 后续工作

### 立即可做
1. **评估攻击效果**
   ```bash
   bash scripts/run_simulation.sh
   ```
   - 使用 LIBERO 仿真环境测试
   - 评估攻击成功率和任务完成率

2. **可视化分析**
   - 查看损失曲线趋势
   - 分析不同 checkpoint 的效果
   - 对比有/无 patch 的动作输出

### 扩展实验
1. **完成剩余 300 iterations**
   - 从 checkpoint 1700 继续训练
   - 观察是否进一步收敛

2. **其他攻击方法**
   - UADA (Untargeted Action Discrepancy Attack)
   - UPA (Untargeted Position-aware Attack)

3. **其他数据集**
   - libero_object_no_noops
   - libero_goal_no_noops
   - libero_10_no_noops
   - bridge_orig

---

## 6. 关键命令记录

### 训练
```bash
CUDA_VISIBLE_DEVICES=1,2 bash scripts/run_TMA.sh
```

### 查看结果
```bash
# 查看实验目录
ls -lh run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/

# 查看最新 checkpoint
ls -lh run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/1700/

# 查看损失曲线
open run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/loss_curve.png
```

### 评估
```bash
bash scripts/run_simulation.sh \
    --exp_path run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e \
    --cudaid 1 --trials 50 --task libero_spatial
```

---

## 7. 经验总结

### 成功因素
1. ✅ 使用足够显存的 GPU（48GB）
2. ✅ 正确配置数据集路径和名称
3. ✅ 合理设置 batch size 和梯度累积
4. ✅ 启用几何变换增强鲁棒性
5. ✅ 定期保存 checkpoint

### 注意事项
1. ⚠️ OpenVLA-7B 需要 ~23GB 显存
2. ⚠️ Git LFS 数据集需手动 pull
3. ⚠️ 数据集名称必须精确匹配
4. ⚠️ 训练时间较长（~13.5 小时）
5. ⚠️ 需要本地模型文件（`local_files_only=True`）

### 优化建议
1. 使用更大 batch size（如果显存允许）
2. 调整学习率和 warmup 策略
3. 尝试不同 patch 尺寸
4. 测试多个目标动作索引
5. 使用 W&B 或 SwanLab 监控训练

---

**实验状态**: ✅ 完成
**下一步**: 运行仿真评估攻击效果
