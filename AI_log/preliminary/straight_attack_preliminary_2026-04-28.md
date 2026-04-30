# Straight Attack Preliminary Summary

**日期**: 2026-04-28
**状态**: preliminary
**目标**: 搭建一条平行于 `TMA` 的独立攻击链路，用于后续验证新 idea，当前命名为 `straight attack`

## 1. 背景

为了避免后续实验直接污染现有 `TMA` 基线，实现上先复制并拆出一条新的攻击文件链路。该链路当前保持与 `TMA` / `TMA_ddp` 的主流程兼容，但把后续方向控制相关的评估和扩展点收敛到独立文件中。

## 2. 当前文件链路

单卡入口：

- `VLAAttacker/straight_attack_wrapper.py`
- `VLAAttacker/white_patch/straight_attack.py`

DDP 入口：

- `VLAAttacker/straight_attack_wrapper_ddp.py`
- `VLAAttacker/white_patch/straight_attack_ddp.py`

方向评估辅助模块：

- `VLAAttacker/white_patch/straight_attack_metrics.py`

启动脚本：

- `scripts/run_straight_attack.sh`
- `scripts/run_straight_attack_ddp.sh`

输出目录：

- `run/straight_attack/<exp_id>`

## 3. 已完成内容

### 3.1 平行攻击链路搭建

已建立 `straight attack` 的单卡和 DDP 两条入口，并与原 `TMA` 链路分离命名：

- 实验名前缀改为 `straight_attack_*`
- 运行产物写入 `run/straight_attack/...`
- 单卡和 DDP 都有独立 wrapper

### 3.2 目标方向评估接口

为避免 `maskidx`、`targetAction` 和 `targetDirection` 三者出现配置冲突，当前实现已经改成：

- `targetAction` 改为长度 `<= 7` 的数组输入
- 未填写的位置自动补成 `0`
- 真正生效的攻击维度不是单独的 `maskidx`，而是：
  `effective_maskidx = maskidx ∩ 已填写 targetAction 的下标`
- 由 `maskidx + targetAction` 自动推导 `targetDirection`

例如：

- `maskidx = [0,1,2], targetAction = [1.0]`
  -> 实际只攻击 `x`
  -> `targetDirection = [1.0, 0.0, 0.0]`
- `maskidx = [0,1,2], targetAction = [1.0, -0.5]`
  -> 实际攻击 `x,y`
  -> `targetDirection = [1.0, -0.5, 0.0]`
- `maskidx = [2], targetAction = [0.0, 0.0, -0.5]`
  -> 实际攻击 `z`
  -> `targetDirection = [0.0, 0.0, -0.5]`

因此当前不需要手动传 `--targetDirection`。

补充约束：

- `targetAction` 最长只能提供 `7` 个值
- 如果 `maskidx` 中的维度在 `targetAction` 里没有显式填写，则该维度不会进入攻击
- `targetDirection` 只从最终生效维度中的 `x/y/z` 轴自动提取
- 如果最终在 `x/y/z` 上没有非零目标值，则不会形成合法的方向评估目标

### 3.3 方向偏移评估函数

`straight_attack_metrics.py` 中新增了目标方向评估逻辑，当前会根据 `maskidx` 中命中的平移轴，对预测动作进行方向一致性分析。

目前指标分成两类：

- 单轴 `xyz` 攻击：
  - `target_direction_scalar_offset_mae`
  - `target_direction_scalar_sign_hit_rate`
  - `target_direction_x_mae / y_mae / z_mae`
  - `target_direction_x_bias / y_bias / z_bias`

- 多轴 `xyz` 攻击：
  - `target_direction_offset_l2`
  - `target_direction_cosine_similarity`
  - `target_direction_angle_deg`
  - `target_direction_projection_gap`
  - `target_direction_x_mae / y_mae / z_mae`
  - `target_direction_x_bias / y_bias / z_bias`

说明：

- `mae` 表示该轴与目标方向分量之间的绝对误差
- `bias` 表示该轴的带符号偏差，便于判断整体偏向正方向还是负方向
- `scalar_offset_mae` 是单轴攻击下的标量偏移绝对误差
- `scalar_sign_hit_rate` 是单轴攻击下预测方向与目标方向同号的比例
- 只有 `maskidx` 中包含的 xyz 轴才会进入该评估

#### 3.3.1 新增函数说明

`straight_attack_metrics.py` 里这次新增的函数主要分成 6 类：

- `normalize_target_direction(target_direction)`
  作用：对传入的目标方向做合法性检查。
  约束：必须是 3 维向量，对应 `x,y,z`；不能是零向量。
  输出：`torch.float32` 的方向向量，供后续评估直接使用。
- `resolve_direction_axes(maskidx)`
  作用：根据当前攻击的 `maskidx`，判断这次评估到底涉及哪些平移轴。
  逻辑：只会从 `0,1,2` 中筛选，也就是只关注 `x/y/z` 三个方向，不处理旋转维度或 gripper。
- `empty_direction_metrics()` 与 `accumulate_direction_metrics(accumulator, update)`
  作用：提供方向指标的初始化和累计逻辑。
  用途：训练循环和验证循环里，每个 batch / inner loop 先算局部结果，再累计成阶段性统计量。
- `build_ideal_straight_trajectory(trajectory, maskidx, target_action)`
  作用：根据当前攻击目标，构造一条“理想直线轨迹”。
  规则：
  - 起点使用输入轨迹的第一个点
  - 被攻击维度按 `target_action` 指定的每步位移线性推进
  - 未被攻击的维度保持原轨迹值不变
  当前状态：
  - 该函数已经写好
  - 但目前还没有接入 `straight attack` 主训练评估流程
- `calculate_direction_offset_metrics(pred_actions, maskidx, target_direction, eps=1e-8)`
  作用：这是核心评估函数。
  输入：
  `pred_actions`：模型解码后的连续动作预测
  `maskidx`：当前攻击目标的动作维度索引
  `target_direction`：希望模型输出逼近的目标方向
  输出：
  返回一个原始统计字典，里面是累加量而不是最终平均值，便于后续在单卡和 DDP 下统一归约。
- `build_direction_log_payload(prefix, metrics, maskidx)`
  作用：把原始统计量转换成真正写入 SwanLab 的日志字段。
  输入中的 `prefix` 目前用于区分：
  `TRAIN`
  `VAL`
  输出：标准化后的日志字典。
  对于单轴攻击，会生成 `TRAIN_target_direction_scalar_offset_mae`、`VAL_target_direction_scalar_sign_hit_rate` 这类标量指标；
  对于多轴攻击，会生成 `TRAIN_target_direction_offset_l2`、`VAL_target_direction_x_mae` 这类向量指标。

#### 3.3.2 `calculate_direction_offset_metrics` 内部实现逻辑

这一部分结合当前代码来说明，代码主体如下：

下面按代码顺序说明：

1. 初始化统计字典并做早返回

```python
metrics = empty_direction_metrics()
if target_direction is None or pred_actions is None or pred_actions.numel() == 0 or len(maskidx) == 0:
    return metrics
```

这里先创建一个全 0 的统计字典，然后检查输入是否可计算。如果没有目标方向、没有预测结果、预测张量为空，或者根本没有攻击维度，就直接返回全 0 结果。

这样做的好处是：

- 训练循环不需要额外写很多 `if/else`
- DDP 和单卡都可以统一使用这套返回格式
- 后面 `accumulate_direction_metrics()` 可以无脑做累加

1. 先从 `maskidx` 里筛出真正参与“方向评估”的轴

```python
available_axes = resolve_direction_axes(maskidx)
if not available_axes:
    return metrics
```

`resolve_direction_axes(maskidx)` 只保留 `0,1,2`，也就是 `x/y/z` 三个平移轴。

这说明当前的方向评估逻辑是：

- 只关心平移方向
- 不关心旋转维度
- 不关心 gripper

例如：

- `maskidx = [0]` -> `available_axes = [0]`
- `maskidx = [0, 1, 2]` -> `available_axes = [0, 1, 2]`
- `maskidx = [6]` -> `available_axes = []`，直接返回全 0

1. 把一维展开的预测结果重组为二维张量

```python
pred_actions = pred_actions.detach().to(torch.float32)
if pred_actions.numel() % len(maskidx) != 0:
    return metrics

pred_actions = pred_actions.view(pred_actions.shape[0] // len(maskidx), len(maskidx))
```

这里的核心目的是“恢复样本结构”。

进入这个函数前，`pred_actions` 往往已经是按 mask 提取后的连续值序列。比如：

- `maskidx = [0]` 时，每个样本只保留 1 个值
- `maskidx = [0,1,2]` 时，每个样本会保留 3 个值

因此这里先检查总元素数能否被 `len(maskidx)` 整除，然后 reshape 成：

- 行：样本数
- 列：当前 `maskidx` 对应的目标维度

例如：

- `maskidx = [0]`，`pred_actions` 长度为 `8`，会变成 `8 x 1`
- `maskidx = [0,1,2]`，`pred_actions` 长度为 `24`，会变成 `8 x 3`

1. 从重组后的预测中提取方向分量，并对齐目标方向

```python
selected_cols = [maskidx.index(axis) for axis in available_axes]
pred_offsets = pred_actions[:, selected_cols]
target_offsets = target_direction[available_axes].to(pred_offsets.device, dtype=pred_offsets.dtype)
target_offsets = target_offsets.unsqueeze(0).expand(pred_offsets.shape[0], -1)
```

这一步很关键，它决定了“评估到底比的是什么”。

`selected_cols` 的作用是：

- 在 `pred_actions` 的列里找到 `x/y/z` 对应的位置
- 因为 `maskidx` 的顺序不一定天然就是 `[0,1,2]`

例如：

- `maskidx = [2]` 时，`available_axes = [2]`
- `maskidx.index(2) = 0`
- 所以虽然攻击的是 z 轴，但在当前重组张量里，它仍然是第 0 列

`pred_offsets` 表示：

- 当前 batch 中，每个样本在目标平移轴上的预测偏移量

`target_offsets` 表示：

- 目标方向在这些轴上的目标值

再通过：

```python
target_offsets.unsqueeze(0).expand(pred_offsets.shape[0], -1)
```

把目标方向扩展成和 batch 同样的形状，这样后面可以逐样本逐列比较。

1. 单轴和多轴的统计逻辑不同

```python
if len(available_axes) == 1:
    ...
    return metrics

vector_gap = torch.linalg.norm(pred_offsets - target_offsets, dim=1)
...
```

当前实现里，单轴攻击和多轴攻击不是同一套指标。

单轴攻击时：

- 不再计算 `cosine_similarity / angle_deg / projection_gap`
- 因为单一标量方向下，用向量夹角没有太大意义
- 只保留：
  - 标量绝对偏差
  - 方向同号统计
  - 该轴的 `mae / bias`

多轴攻击时，才会进入下面这套向量方向评估。

这一段对应四类核心方向指标：

- `vector_gap`
  直接比较预测向量和目标向量的欧氏距离，衡量“整体差了多少”
- `cosine_similarity`
  只看方向是否一致，越接近 `1` 越表示朝着同一个方向
- `angle_deg`
  是余弦相似度的角度版本，更直观地表示“夹角多大”
- `projection_gap`
  看预测结果在目标方向上的投影和目标长度之间差多少，适合判断“沿目标方向推进得够不够”

这里的两个 `clamp_min(eps)` 很重要：

- 防止目标向量长度为 0 时除零
- 防止预测向量长度为 0 时除零

而：

```python
cosine_similarity = cosine_similarity.clamp(-1.0, 1.0)
```

是为了避免浮点误差导致 `acos` 输入越界。

1. 把逐样本结果累加成原始统计量

```python
metrics["count"] = float(pred_offsets.shape[0])
metrics["vector_gap_sum"] = float(vector_gap.sum().item())
metrics["cosine_sum"] = float(cosine_similarity.sum().item())
metrics["angle_sum"] = float(angle_deg.sum().item())
metrics["projection_gap_sum"] = float(projection_gap.sum().item())
```

这里不直接求平均，而是先保存“总和”。

原因是：

- 单卡训练里，一个 step 内可能会多次调用
- DDP 场景下还要跨卡做归约

如果过早做平均，后面全局聚合会不够自然；保留 `sum + count` 则更稳定。

1. 再逐轴统计绝对误差和带符号误差

```python
for column_idx, axis in enumerate(available_axes):
    axis_name = AXIS_NAMES[axis]
    axis_error = pred_offsets[:, column_idx] - target_offsets[:, column_idx]
    metrics[f"{axis_name}_abs_error_sum"] = float(torch.abs(axis_error).sum().item())
    metrics[f"{axis_name}_signed_error_sum"] = float(axis_error.sum().item())
```

这里按每个参与评估的轴分别计算：

- `abs_error_sum`
  用来衡量误差大小，不关心方向正负
- `signed_error_sum`
  用来衡量整体偏差方向，能看出模型是整体偏大、偏小，还是沿相反方向移动

例如在 `x` 轴上：

- 如果 `x_signed_error_sum > 0`，说明整体上预测值偏向目标值上方
- 如果 `x_signed_error_sum < 0`，说明整体上预测值偏向目标值下方

1. 最后返回的是“原始累计统计”，不是最终日志值

```python
return metrics
```

这个返回值不会直接上 SwanLab，而是会在后面继续经过：

- `accumulate_direction_metrics()`：累计多个局部统计
- `build_direction_log_payload()`：用 `count` 归一化，生成真正要记录的 `TRAIN_* / VAL_*` 指标

#### 3.3.3 为什么返回“sum”而不是直接返回平均值

当前函数返回的是累加量，而不是已经归一化的均值，主要有两个原因：

- 训练/验证循环内部会多次调用这个函数，需要先把多个 batch / inner loop 的结果累计起来
- DDP 模式下还要跨卡做归约，如果一开始就先算平均值，后续再做全局聚合会更容易失真

因此设计上是：

1. `calculate_direction_offset_metrics()` 只返回原始累计量
2. `accumulate_direction_metrics()` 负责把多个局部统计累加起来
3. `build_direction_log_payload()` 再用 `count` 做归一化，生成最终日志指标

#### 3.3.4 各指标的含义

单轴攻击：

- `target_direction_scalar_offset_mae`
  表示预测标量偏移与目标标量偏移之间的平均绝对误差。越小越好。
- `target_direction_scalar_sign_hit_rate`
  表示预测方向和目标方向同号的比例。越接近 `1` 越好。
- `target_direction_x_mae / y_mae / z_mae`
  表示被攻击轴上与目标方向分量的绝对误差。
- `target_direction_x_bias / y_bias / z_bias`
  表示被攻击轴上的带符号平均偏差。

多轴攻击：

- `target_direction_offset_l2`
  表示预测方向向量与目标方向向量之间的欧氏距离。越小越接近目标方向。
- `target_direction_cosine_similarity`
  表示预测方向与目标方向的夹角一致性。越接近 `1`，方向越一致。
- `target_direction_angle_deg`
  表示预测方向与目标方向之间的夹角，单位是度。越小越好。
- `target_direction_projection_gap`
  表示预测方向在目标方向单位向量上的投影，与目标方向长度之间的差距。它更强调“沿目标方向前进了多少”。
- `target_direction_x_mae / y_mae / z_mae`
  表示每个轴上与目标方向分量的绝对误差。
- `target_direction_x_bias / y_bias / z_bias`
  表示每个轴上的带符号平均偏差。正值和负值能帮助判断模型整体是偏大、偏小，还是朝相反方向跑。

#### 3.3.5 这些函数接在什么位置

- `straight_attack_wrapper.py` 和 `straight_attack_wrapper_ddp.py`
  负责根据 `maskidx + targetAction` 自动生成 `targetDirection`，并把它传入攻击器实例。
- `straight_attack.py`
  在单卡训练与验证循环中调用 `calculate_direction_offset_metrics()`，然后通过 `build_direction_log_payload()` 写入 SwanLab。
- `straight_attack_ddp.py`
  在 DDP 场景下复用同一套方向评估逻辑，并额外做一次跨卡归约，确保最终记录的是全局平均指标。

#### 3.3.6 代码层面的边界

目前这些新增函数的定位是“评估函数”而不是“优化目标函数”：

- 它们不会直接影响 patch 更新方向
- 它们只负责度量当前攻击结果是否朝设定的目标方向偏移
- 如果后续要把方向目标真正纳入 loss，可以在这套评估函数基础上继续扩展

### 3.4 SwanLab 可视化接入

方向评估已经接入训练和验证日志：

- `TRAIN_target_direction_*`
- `VAL_target_direction_*`

其中单轴 `xyz` 攻击时，重点看：

- `TRAIN/VAL_target_direction_scalar_offset_mae`
- `TRAIN/VAL_target_direction_scalar_sign_hit_rate`

多轴攻击时，重点看：

- `TRAIN/VAL_target_direction_offset_l2`
- `TRAIN/VAL_target_direction_cosine_similarity`
- `TRAIN/VAL_target_direction_angle_deg`
- `TRAIN/VAL_target_direction_projection_gap`

同时，这些阶段性指标也会额外保存为 `pkl`，写入实验目录，便于后处理分析。

### 3.5 DDP 链路修正

`straight_attack_ddp.py` 已补充自己的 `_attack_entry`，确保 DDP 模式下实例化的是 `straight attack` 子类本身，而不是退回基类实现。否则目标方向评估不会生效。

## 4. 当前验证状态

已完成：

- 代码接线完成
- 单卡 / DDP wrapper 参数已打通
- `targetAction` 已改成数组输入，并与 `maskidx` 做生效维度对齐
- 轻量语法解析已通过

未完成：

- 尚未实际运行一次 `straight attack` 冒烟实验
- 目前方向评估只做“预测动作 vs 目标方向”的偏移分析，尚未把该目标直接写入攻击优化目标本身
- `build_ideal_straight_trajectory()` 还未真正接入主评测链路

## 5. 后续建议

下一步可以按下面顺序推进：

1. 先用一个最简单方向做冒烟，例如 `--maskidx 0 --targetAction 1.0`
2. 检查 SwanLab 中单轴指标：
   `VAL_target_direction_scalar_offset_mae`
   `VAL_target_direction_scalar_sign_hit_rate`
3. 如果要测试多轴，再改成类似 `--maskidx 0,1,2 --targetAction 1.0,-0.5,0.25`
4. 再检查多轴指标：
   `VAL_target_direction_offset_l2`
   `VAL_target_direction_cosine_similarity`
   `VAL_target_direction_angle_deg`
   `VAL_target_direction_projection_gap`
5. 再决定是否把“方向一致性”从评估指标进一步改成训练目标或 loss 项

## 6. 示例参数

单卡：

```bash
python VLAAttacker/straight_attack_wrapper.py \
  --maskidx 0 \
  --targetAction "1.0"
```

DDP：

```bash
torchrun --nproc_per_node=2 --master_port=29501 VLAAttacker/straight_attack_wrapper_ddp.py \
  --maskidx 0 \
  --targetAction "1.0"
```
