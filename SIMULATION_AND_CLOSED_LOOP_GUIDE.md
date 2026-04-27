# Simulation And Closed-Loop Guide

这份文档整理当前项目中与以下内容相关的现成实现：

- OpenVLA 单步动作推理
- LIBERO / Bridge 闭环执行
- 对抗 patch 的闭环仿真评估

目标是回答 4 个问题：

1. 当前项目里哪些脚本是在做闭环执行
2. 闭环执行时的数据流是怎样的
3. patch 是在什么位置插入仿真链路的
4. 实际运行时应该从哪个脚本开始

## 1. 现有文档情况

当前仓库里已经有一些零散说明，但没有一份完整串起“闭环执行 + 仿真 + patch 评估”的独立文档：

- [README.md](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/README.md)
  只给出了 `bash scripts/run_simulation.sh` 这一层入口，没有展开调用链。
- [CLAUDE.md](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/CLAUDE.md)
  说明了评估入口和部分参数，但没有系统解释闭环流程。
- [AI_log/Claude_TMA_总结_2026-03-12.md](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/AI_log/Claude_TMA_总结_2026-03-12.md)
  包含实验记录和命令示例，但属于实验日志，不是使用指南。

因此这份文档补的是“工程说明书”这一块。

## 2. 先区分三种层级

### 2.1 单步动作推理

这一级只负责：

- 输入当前图像和任务文本
- 输出当前一步动作

相关文件：

- [experiments/robot/openvla_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/openvla_utils.py)
- [experiments/robot/robot_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/robot_utils.py)
- [prismatic/models/vlas/openvla.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/prismatic/models/vlas/openvla.py)

核心接口：

- `get_vla_action(...)`
- `get_action(...)`
- `vla.predict_action(...)`

### 2.2 普通闭环执行

这一级负责把“单步动作推理”放进环境循环里：

- 取当前观测
- 调模型出动作
- `env.step(action)`
- 得到下一帧观测
- 重复直到成功或超时

这是标准 rollout / closed-loop evaluation。

相关文件：

- [experiments/robot/libero/run_libero_eval.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval.py)
- [experiments/robot/bridge/run_bridgev2_eval.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/bridge/run_bridgev2_eval.py)

### 2.3 带 patch 的闭环仿真

这是当前攻击项目最重要的一层。它和普通闭环评估的区别只有一个：

- 在每一步把环境图像送给模型前，先贴上 patch

相关文件：

- [experiments/robot/libero/run_libero_eval_args_geo_batch.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval_args_geo_batch.py)
- [VLAAttacker/white_patch/appply_random_transform.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/VLAAttacker/white_patch/appply_random_transform.py)
- [evaluation_tool/eval_queue_single_four_spec.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/evaluation_tool/eval_queue_single_four_spec.py)
- [scripts/run_simulation.sh](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/scripts/run_simulation.sh)

## 3. 普通闭环执行链路

以 LIBERO 为例，主入口是：

- [experiments/robot/libero/run_libero_eval.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval.py)

它的闭环结构可以概括成：

1. 初始化任务套件和环境
2. 取一帧图像观测
3. 调 OpenVLA 得到当前一步动作
4. 规范 gripper 动作格式
5. `env.step(action.tolist())`
6. 重复直到 `done` 或最大步数

代码里的关键位置：

- 获取环境：
  [experiments/robot/libero/libero_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/libero_utils.py)
- 获取图像：
  [experiments/robot/libero/libero_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/libero_utils.py)
- 调模型：
  [experiments/robot/robot_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/robot_utils.py)
- 真正执行：
  [experiments/robot/libero/run_libero_eval.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval.py)

可以抽象成：

```text
obs_t -> image_t -> model -> action_t -> env.step(action_t) -> obs_{t+1}
```

这就是闭环。

## 4. OpenVLA 在闭环里如何出动作

OpenVLA 在仿真里并不是一次性输出整段轨迹，而是每个时刻输出一步动作。

单步调用链如下：

1. [experiments/robot/robot_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/robot_utils.py) 的 `get_action(...)`
2. 调 [experiments/robot/openvla_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/openvla_utils.py) 的 `get_vla_action(...)`
3. 内部再调 OpenVLA 模型的 `predict_action(...)`

也就是：

```text
get_action -> get_vla_action -> predict_action
```

注意：

- OpenVLA 输入的是当前图像和任务描述
- 输出的是当前一步动作
- 动态过程体现在 rollout 中重复调用，而不是单次前向里显式建模整个序列

## 5. 带 patch 的闭环仿真实验链路

主入口不是 `run_libero_eval.py`，而是：

- [experiments/robot/libero/run_libero_eval_args_geo_batch.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval_args_geo_batch.py)

它和普通 LIBERO 评估的区别是多了两步：

1. 从磁盘读取 `patch.pt`
2. 在每一步观测图像上贴 patch

关键代码流程：

1. 加载 patch：
   `patch = torch.load(cfg.patchroot)`
2. 初始化 patch 变换器：
   `randomPatchTransform = RandomPatchTransform('cpu', False)`
3. 每一步从环境获得原始图像：
   `img = get_libero_image(obs, resize_size)`
4. 把 patch 贴到图像上：
   `img = randomPatchTransform.simulation_random_patch(...)`
5. 再把这张图送给 `get_action(...)`
6. 得到动作后继续 `env.step(...)`

可以抽象成：

```text
obs_t
  -> image_t
  -> patched_image_t
  -> model
  -> action_t
  -> env.step(action_t)
  -> obs_{t+1}
```

所以 patch 不只是影响单步推理，而是在整个 rollout 的每一时刻持续干预。

## 6. patch 在仿真里插入的位置

带 patch 评估最关键的问题，是“patch 插在哪个位置”。

答案是：

- patch 插在环境观测图像和模型推理之间
- 不改环境状态本身
- 不直接改动作
- 只改模型看到的视觉输入

也就是：

```text
env observation -> patch transform -> attacked observation -> policy
```

这个设计和训练阶段保持一致：

- 训练时 patch 也是贴到图像上
- 推理时 patch 继续贴到图像上

区别只是：

- 训练里 patch 需要反向传播优化
- 仿真评估里 patch 已经固定，只负责测试效果

## 7. 为什么它叫闭环 patch 仿真

因为 patch 的作用不是只影响一步，而是：

1. 干扰当前观测
2. 导致当前动作偏移
3. 当前动作改变环境状态
4. 环境状态改变下一帧观测
5. patch 再继续干扰下一帧

所以误差会沿着闭环逐步积累。

这也是为什么：

- 单步动作偏一点
- 任务最终可能明显失败

## 8. 顶层评估入口是什么

当前项目用于批量跑 patch 仿真的入口是：

- [scripts/run_simulation.sh](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/scripts/run_simulation.sh)

它会调用：

- [evaluation_tool/eval_queue_single_four_spec.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/evaluation_tool/eval_queue_single_four_spec.py)

这个脚本的作用是：

1. 读取某个 patch 实验目录
2. 选出要评估的任务集
3. 自动拼出 `run_libero_eval_args_geo_batch.py` 的命令
4. 对不同数据集配置对应的 patch 放置参数
5. 批量执行带 patch 的 LIBERO 闭环评估

所以顶层关系是：

```text
run_simulation.sh
  -> eval_queue_single_four_spec.py
    -> run_libero_eval_args_geo_batch.py
      -> get_action / env.step
```

## 9. 普通闭环评估和 patch 闭环评估的区别

### 9.1 普通闭环评估

入口：

- [experiments/robot/libero/run_libero_eval.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval.py)

特点：

- 不加载 patch
- 图像直接送模型
- 用来测正常策略表现

### 9.2 带 patch 的闭环评估

入口：

- [experiments/robot/libero/run_libero_eval_args_geo_batch.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval_args_geo_batch.py)

特点：

- 加载训练好的 `patch.pt`
- 在每帧图像上贴 patch
- 用来测攻击是否能在 rollout 中破坏任务

## 10. 常见输出结果在哪里

### 10.1 patch 训练输出

通常位于：

- `run/white_patch_attack/<exp_id>/`

典型内容：

- `patch.pt`
- `loss_curve.png`
- `val_*.pkl`
- `last/`
- 若干 checkpoint 目录

### 10.2 闭环仿真输出

普通或带 patch 的 LIBERO 评估通常会产出：

- 本地日志 `EVAL-*.txt`
- rollout 视频 `rollouts/.../*.mp4`

具体视频保存逻辑在：

- [experiments/robot/libero/libero_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/libero_utils.py)

Bridge 评估则还可能保存：

- rollout 数据 `.npz`

## 11. 最常用的三条运行方式

### 11.1 跑普通 LIBERO 闭环评估

参考：

- [experiments/robot/libero/eval_LIBERO.sh](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/eval_LIBERO.sh)

核心命令形态：

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <checkpoint> \
  --task_suite_name libero_spatial
```

### 11.2 跑 patch 闭环评估

核心脚本：

- [experiments/robot/libero/run_libero_eval_args_geo_batch.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval_args_geo_batch.py)

核心命令形态：

```bash
python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
  --pretrained_checkpoint <checkpoint> \
  --task_suite_name libero_spatial \
  --patchroot <path_to_patch.pt> \
  --cudaid 0 \
  --x 120 --y 160 --angle 0 --shx 0 --shy 0
```

### 11.3 用项目默认入口批量跑 patch 仿真

```bash
bash scripts/run_simulation.sh
```

它内部再去调：

- [evaluation_tool/eval_queue_single_four_spec.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/evaluation_tool/eval_queue_single_four_spec.py)

## 12. 当前实现里需要注意的地方

### 12.1 `run_simulation.sh` 需要手动改路径

当前 [scripts/run_simulation.sh](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/scripts/run_simulation.sh) 里的 `--exp_path` 还是占位符：

```bash
--exp_path PATH TO/fe28658a-4a27-4ffa-82c4-94d44ffc9d48
```

实际运行前需要改成真实 patch 实验目录。

### 12.2 patch 仿真脚本里有硬编码路径提示

[experiments/robot/libero/run_libero_eval_args_geo_batch.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval_args_geo_batch.py) 里有：

```python
sys.path.append("PATH TO/white_patch")
```

如果环境里没有正确处理 import 路径，这里需要改成项目里的真实 `white_patch` 目录。

### 12.3 LIBERO 环境对 gripper 动作有额外变换

闭环执行前会对 gripper 做：

- `[0,1] -> [-1,+1]` 归一化
- 再做一次符号翻转

对应代码在：

- [experiments/robot/robot_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/robot_utils.py)

所以如果你在分析 rollout 动作和训练输出的差异，要把这一步考虑进去。

## 13. 推荐理解顺序

如果你是第一次读这条链路，建议按下面顺序看：

1. [experiments/robot/openvla_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/openvla_utils.py)
   先看单步动作是怎么从图像和指令生成的
2. [experiments/robot/robot_utils.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/robot_utils.py)
   再看通用的 `get_action()` 和 gripper 处理
3. [experiments/robot/libero/run_libero_eval.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval.py)
   看普通闭环 rollout
4. [experiments/robot/libero/run_libero_eval_args_geo_batch.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/experiments/robot/libero/run_libero_eval_args_geo_batch.py)
   看 patch 是如何插进去的
5. [evaluation_tool/eval_queue_single_four_spec.py](/mnt/home/lvmingyuan/marionette/VLA-attack-AE/evaluation_tool/eval_queue_single_four_spec.py)
   最后看批量评估组织器

## 14. 一句话总结

当前项目已经实现了完整的闭环评估链路：

- OpenVLA 负责单步动作预测
- LIBERO / Bridge 脚本负责环境闭环 rollout
- patch 仿真脚本负责在每步观测上插入对抗 patch
- `run_simulation.sh` 和 `eval_queue_single_four_spec.py` 负责把 patch 评估组织成批量实验

如果后续你想继续补文档，最值得补的是：

- 一个“最小可运行 patch 仿真命令示例”
- 一个“普通闭环 vs patch 闭环”的对照实验说明
- 一个“常见报错与修复”章节
