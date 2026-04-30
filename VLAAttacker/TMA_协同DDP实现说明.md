# TMA 协同 DDP 实现说明

## 目标

`TMA_ddp` 的正确目标不是“每张卡各自训练一个 patch”，而是：

1. 多张 GPU 共同优化同一份对抗 patch。
2. 每个 rank 处理不同的数据分片。
3. 每个 rank 计算当前共享 patch 的本地梯度。
4. 所有 rank 对 patch 梯度做同步平均。
5. 所有 rank 基于同一份平均梯度更新同一份 patch。

也就是说，最终应该只有“一条 patch 优化轨迹”，而不是“每张卡一条各自独立的 patch 轨迹”。

## 旧版问题

旧版 `TMA_ddp` 存在三个关键结构问题：

1. 每个 rank 本地创建自己的 patch，并各自更新，没有形成真正的共享 patch。
2. dataloader 没有按 rank 切分，多个 rank 大概率重复读取整份数据流。
3. 代码把整套 VLA 模型包进了 DDP，但优化器只更新 patch，不更新模型参数，这会引入无意义的通信开销。

这会导致行为更接近“多卡各练各的”，而不是“多卡协同训练一个 patch”。

## 当前实现逻辑

### 1. 共享 patch 初始化

patch 只在 rank 0 上随机初始化，然后广播到所有 rank：

```python
if self.is_rank_zero():
    patch = torch.rand(self.patch_size, device=self.device)
else:
    patch = torch.empty(self.patch_size, device=self.device)
dist.broadcast(patch, src=0)
```

对应文件：

- `VLAAttacker/white_patch/TMA_ddp.py`

这样可以保证所有进程从完全相同的 patch 初值开始。

### 2. 数据按 rank 分片

这里的数据集 `RLDSDataset` 是 `IterableDataset`，不适合用传统 map-style 的 `DistributedSampler`。

正确做法是使用它自己的 `shard(...)` 接口：

```python
dataset.shard(num_shards=world_size, index=rank)
```

这次改动把 dataloader 补成了支持分片参数：

- `world_size`
- `rank`

对应文件：

- `VLAAttacker/white_patch/openvla_dataloader.py`

这样每个 rank 读取的是不同的数据子流，而不是重复读整份数据。

### 3. patch 梯度同步平均

因为真正需要优化的是 patch，而不是 VLA 参数，所以核心同步对象应当是 `patch.grad`。

训练时每个 rank 先基于自己的数据分片做本地反传，然后对 patch 梯度执行：

```python
dist.all_reduce(patch.grad, op=dist.ReduceOp.SUM)
patch.grad /= world_size
```

对应实现：

- `_sync_patch_grad()`
- `VLAAttacker/white_patch/TMA_ddp.py`

这一步是“协同 DDP 单 patch 训练”的核心。

### 4. 更新后再次对齐 patch

在 `optimizer.step()` 和 `clamp` 之后，patch 会再次从 rank 0 广播：

```python
dist.broadcast(patch.data, src=0)
```

理论上，如果所有 rank 都从相同初值出发，并使用相同平均梯度，patch 应该天然保持一致。

但显式广播可以进一步避免：

1. 数值误差累积
2. 优化器状态微小漂移
3. 某些异常步骤导致 rank 间状态不一致

所以这是一层防御性一致性保证。

### 5. 冻结 VLA，只训练 patch

当前实现会把 VLA 参数全部冻结：

```python
for param in self.vla.parameters():
    param.requires_grad_(False)
```

原因很直接：

1. 这个攻击任务只优化 patch。
2. VLA 参数不参与更新。
3. 对冻结参数做 DDP 梯度通信没有收益，只会增加开销。

因此，这版实现的重点不是“让整模型做 DDP 通信”，而是“让 patch 的优化过程做真正的分布式同步”。

从技术上说，这比“把整模型包进 DDP、但实际只更新 patch”更合理，也更高效。

## rank 分工

所有 rank 都负责：

1. 加载同一个 VLA checkpoint
2. 接收同一个共享 patch
3. 读取各自的数据分片
4. 计算本地 loss 和 patch 本地梯度
5. 参与梯度平均
6. 执行同步后的 patch 更新

只有 rank 0 负责：

1. 初始化 patch
2. 初始化 SwanLab
3. 保存 patch checkpoint
4. 保存验证可视化样本
5. 记录聚合后的训练/验证指标

## 日志与保存策略

为了避免重复输出，所有副作用都应收敛到 rank 0：

- SwanLab 日志
- 最优 patch 保存
- `last/patch.pt` 保存
- 验证图片导出
- pickle 指标文件写盘

否则多进程会同时写同一路径，导致日志重复、文件互相覆盖或结果混乱。

## 入口文件

主要入口：

- `VLAAttacker/TMA_wrapper_ddp.py`
- `VLAAttacker/white_patch/TMA_ddp.py`
- `VLAAttacker/white_patch/openvla_dataloader.py`

启动方式：

```bash
torchrun --nproc_per_node=2 --master_port=29501 VLAAttacker/TMA_wrapper_ddp.py ...
```

推荐从仓库根目录启动，例如：

```bash
current_dir=$(pwd)

torchrun --nproc_per_node=2 --master_port=29501 VLAAttacker/TMA_wrapper_ddp.py \
  --maskidx 0 \
  --lr 2e-3 \
  --server "$current_dir" \
  --iter 2000 \
  --accumulate 4 \
  --bs 2 \
  --warmup 20 \
  --geometry true \
  --patch_size "3,50,50" \
  --swanlab_project "VLA-Attack" \
  --innerLoop 50 \
  --dataset "libero_spatial_no_noops" \
  --targetAction 0
```

如果要显式指定 GPU，可写成：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 VLAAttacker/TMA_wrapper_ddp.py ...
```

其中：

- `nproc_per_node=2` 表示本机起 2 个 rank
- `CUDA_VISIBLE_DEVICES=0,1` 表示只使用物理 GPU 0 和 1
- rank 0 会绑定到可见设备里的第 1 张卡
- rank 1 会绑定到可见设备里的第 2 张卡

## batch 与吞吐计算方式

### 1. 单卡 micro-batch

参数 `--bs` 表示每个 rank、每一步前向时拿到的本地 batch 大小。

记作：

```text
local_batch = bs
```

例如：

```text
bs = 2
```

表示每张卡每次本地处理 2 个样本。

### 2. 全局 micro-batch

多卡协同时，一次同步更新前，所有 rank 同时参与本地前向与反传。

因此一次“同步训练步”的全局 micro-batch 为：

```text
global_micro_batch = bs * world_size
```

例如：

```text
bs = 2
world_size = 2
global_micro_batch = 4
```

也就是一次同步训练步总共贡献 4 个样本的 patch 梯度。

### 3. 梯度累计后的等效全局 batch

当前实现里 `--accumulate` 控制梯度累计步数。

等效全局 batch 计算方式为：

```text
effective_global_batch = bs * world_size * accumulate_steps
```

例如：

```text
bs = 2
world_size = 2
accumulate_steps = 4
effective_global_batch = 16
```

这表示：

1. 每个 rank 每次先处理 2 个样本
2. 2 个 rank 同时工作，所以一轮同步步贡献 4 个样本
3. 连续累计 4 次后再做一次 patch 更新
4. 因此一次真正的 patch 参数更新，等效融合了 16 个样本的信息

### 4. 每次 patch 更新看到的样本数

如果只关心“一次 `optimizer.step()` 前总共看了多少样本”，计算方式和上面相同：

```text
samples_per_patch_update = bs * world_size * accumulate_steps
```

这也是判断 patch 梯度稳定性的核心量。

### 5. 总样本吞吐近似

如果总训练迭代数记为 `num_iter`，则近似总样本处理量为：

```text
total_samples_processed ≈ num_iter * bs * world_size
```

如果更关心“真正参数更新次数”，则：

```text
num_optimizer_steps ≈ num_iter / accumulate_steps
```

因此总样本量也可写成：

```text
total_samples_processed ≈ num_optimizer_steps * effective_global_batch
```

### 6. 用当前推荐命令举例

对于下面这组参数：

```text
--bs 2
--accumulate 4
--nproc_per_node=2
--iter 2000
```

有：

```text
world_size = 2
local_batch = 2
global_micro_batch = 2 * 2 = 4
effective_global_batch = 2 * 2 * 4 = 16
num_optimizer_steps ≈ 2000 / 4 = 500
total_samples_processed ≈ 2000 * 2 * 2 = 8000
```

也就是说：

1. 每次 patch 真正更新前，约融合 16 个样本的梯度
2. 2000 个训练迭代大约对应 500 次 patch 参数更新
3. 整个训练过程中总共处理约 8000 个样本

### 7. 调参直觉

- 增大 `bs`：每张卡单步显存压力增大，但吞吐提升
- 增大 `world_size`：并行卡数增多，全局吞吐提升
- 增大 `accumulate_steps`：单次 patch 更新更稳定，但更新频率下降
- 固定总样本预算时：
  - 更大的 `world_size` 倾向于缩短 wall-clock 时间
  - 更大的 `accumulate_steps` 倾向于增大单次更新的统计稳定性

## 这版实现带来的实际效果

改完之后，`TMA_ddp` 的语义变成了真正的协同分布式攻击：

1. 全局有效 batch 随 GPU 数增加而增加。
2. 每次 patch 更新都融合了多张卡上不同数据分片的梯度。
3. 最终只训练出一份全局共享的 patch。
4. 多卡的收益来自“并行处理更多数据”，而不是“重复训练多份局部 patch”。

## 一句话总结

这版协同 DDP 的核心不是“让多张卡都跑起来”，而是：

**让多张卡对同一个 patch 的梯度达成同步，并基于统一梯度共同推进同一份 patch。**
