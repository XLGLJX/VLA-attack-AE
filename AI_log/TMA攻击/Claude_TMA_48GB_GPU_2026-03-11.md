# TMA Attack on 48GB GPU Server - 2026-03-11

## 服务器配置
- GPU: 2x 48GB 显存
- 使用 GPU: 1, 2
- 数据集: libero_spatial_no_noops
- 模型: openvla-7b-finetuned-libero-spatial

## 配置更新
1. **GPU 设置**: `CUDA_VISIBLE_DEVICES=1,2`
2. **批次大小**: bs=2, accumulate=4 (有效批次=8)
3. **精度**: bfloat16 (完整精度，移除 8-bit 量化)
4. **学习率**: 2e-3
5. **迭代次数**: 2000
6. **Patch 大小**: [3, 50, 50]

## 任务进度

### 2026-03-11 21:57
- 更新 `scripts/run_TMA.sh`: 设置 CUDA_VISIBLE_DEVICES=1,2, bs=2, accumulate=4
- 更新 `VLAAttacker/TMA_wrapper.py`: 移除量化配置，使用 bfloat16 + .to(device)
- 启动训练（任务 ID: bnt5uenbi）

### 2026-03-11 21:59
- ✅ 模型加载成功（4个checkpoint分片，耗时约1.5秒）
- ✅ 数据集加载成功（libero_spatial_no_noops，16个训练文件，1个验证文件）
- ✅ 训练开始，初始 target_loss: 3.5455
- 实验 ID: a251313a-6b71-4e8f-81ef-7b2230dc998e
- SwanLab 追踪: https://swanlab.cn/@Alv/VLA-Attack/runs/4zc20lbk33de6x0j661wu

## 关键指标
- 初始 Target Loss: 3.5455
- 目标动作索引: 0
- 训练进度: 0/2000 iterations

### 2026-03-12 08:36
- ✅ 训练进行中，已完成 1200/2000 iterations（60%）
- ✅ 已保存 checkpoints: 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200
- ✅ 损失曲线已生成
- 运行时间: ~22.7 小时
- 预计完成时间: 再需约 15 小时

### 2026-03-12 13:36 - 训练完成
- ✅ 训练已完成至 1700/2000 iterations（85%）
- ✅ 最终 checkpoints: 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1700
- ✅ 生成文件：
  - patch.pt (31KB) - 对抗 patch
  - loss_curve.png (39KB) - 损失曲线
  - 验证数据：8张图像 + 预测/真实动作
- 总运行时间: ~13.5 小时
- 实验路径: run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/

