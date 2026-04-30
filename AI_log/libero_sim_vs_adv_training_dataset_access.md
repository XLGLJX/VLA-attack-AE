# 仿真阶段访问 LIBERO 与训练对抗样本访问逐帧数据集的区别

## 结论先说

- 仿真阶段不是把 `libero` 训练集逐帧喂给模型，而是：
  - 先从 `LIBERO benchmark/task suite` 里拿任务定义与默认初始状态；
  - 然后在环境里 `reset -> set_init_state -> env.step(...)` 在线 rollout；
  - 每一步图像都来自仿真环境当前观测，而不是本地 TFDS/RLDS 文件。
- 训练对抗样本时，虽然底层 RLDS 管线一开始以“轨迹”为单位做标准化和变换，但最终经过 `frame_map` 和 `as_numpy_iterator()` 后，优化器实际消费的是“单步 frame / transition”的 batch。

---

## 一、仿真阶段：按任务/轨迹 rollout 访问 LIBERO

### 调用入口

`scripts/run_simulation.sh`
-> `evaluation_tool/eval_queue_single_four_spec.py`
-> `experiments/robot/libero/run_libero_eval_args_geo_batch.py`

### 访问链路

1. `eval_queue_single_four_spec.py` 根据 `--task` 选择 `libero_spatial / libero_object / libero_goal / libero_10` 等任务套件，并拼出评测命令。
2. `run_libero_eval_args_geo_batch.py` 中：
   - `benchmark.get_benchmark_dict()[cfg.task_suite_name]()` 初始化 task suite；
   - `task_suite.get_task(task_id)` 取任务；
   - `task_suite.get_task_init_states(task_id)` 取该任务的一组默认初始状态；
   - `get_libero_env(...)` 创建 `OffScreenRenderEnv`；
   - 每个 episode 走 `env.reset()` 和 `env.set_init_state(initial_states[episode_idx])`；
   - 后续每一步都靠 `env.step(...)` 推进环境。
3. 图像来自 `obs["agentview_image"]`，由 `get_libero_image(...)` 做：
   - 180 度旋转；
   - JPEG encode/decode；
   - resize 到模型期望输入尺寸。
4. patch 也是在线贴到当前观测图像上：
   - `simulation_random_patch(img, patch, ...)`
   - 然后才拿去做动作推理。

### 这里“访问 LIBERO 轨迹”到底指什么

更准确地说，仿真阶段访问的是：

- `task suite` 里的任务集合；
- 每个任务对应的一组 `initial_states`。

它**没有**像训练 dataloader 那样从 RLDS 文件里逐帧读 `image/action/language`。  
它只是借用了 demo 轨迹对应的初始状态，随后所有帧都由仿真环境在线生成。

### 仿真阶段的采样单位

- 外层单位：`task`
- 中层单位：`episode`
- 内层单位：`environment step`

这是一条典型的在线闭环评测路径。

### 与训练集的关系

`run_libero_eval_args_geo_batch.py` 里为不同 task suite 写死了 `max_steps`，注释直接说明这些值来自“该训练 demo 中最长轨迹长度”：

- `libero_spatial`: 193
- `libero_object`: 254
- `libero_goal`: 270
- `libero_10`: 505
- `libero_90`: 373

所以仿真评测会参考训练 demo 的轨迹长度上界，但**不会**把训练集逐帧回放给模型。

---

## 二、训练对抗样本：按 frame / transition 消费本地 RLDS 数据

### 调用入口

- `scripts/run_UADA.sh`
- `scripts/run_TMA_ddp.sh`

它们分别进入：

- `VLAAttacker/UADA_wrapper.py`
- `VLAAttacker/TMA_wrapper_ddp.py`

再调用：

- `VLAAttacker/white_patch/openvla_dataloader.py`

### 访问链路

1. `get_dataloader(...)` 把逻辑数据集名映射到本地 RLDS builder：
   - `libero_spatial -> libero_spatial_no_noops`
   - `libero_object -> libero_object_no_noops`
   - `libero_goal -> libero_goal_no_noops`
   - `libero_10 -> libero_10_no_noops`
2. 数据根目录固定是：
   - `Path(f"{server}/dataset")`
3. `RLDSDataset(...)` 被创建为训练集和验证集。
4. `RLDSDataset` 内部流程：
   - `make_dataset_from_rlds(...)`：先以“轨迹”为单位读取 TFDS/RLDS；
   - `apply_trajectory_transforms(...)`：做轨迹级标准化/切块；
   - `apply_frame_transforms(...)`：做逐帧 decode/resize；
   - `self.dataset.as_numpy_iterator()`：最终吐出的是逐步样本流。
5. `RLDSBatchTransform.__call__` 只取当前 step：
   - `rlds_batch["observation"]["image_primary"][0]`
   - `rlds_batch["action"][0]`
   - `rlds_batch["task"]["language_instruction"]`
6. 然后构造一条监督样本：
   - 输入：当前图像 + 指令 prompt
   - 标签：当前 step 的 action token
7. PyTorch `DataLoader` 再把这些 step-level 样本拼成 batch。

### 为什么说“训练时访问的是每一帧数据集”

关键不在于最底层存储是不是按 episode 组织，而在于**优化器看到的样本单位**是什么。

这里最终喂给训练循环的是：

- 一张图 `pixel_values`
- 一句语言指令
- 一个 step 对应的 action label

也就是标准的 transition / frame 监督样本。

### 训练循环里如何消费这些 frame batch

`UADA_ddp.py` 中直接是：

- `for i, data in enumerate(train_dataloader):`
- `pixel_values = data["pixel_values"]`
- 对当前 batch 的图像贴 patch；
- 前向；
- 反传更新 patch。

`TMA_ddp.py` 中也是：

- `train_iterator = iter(train_dataloader)`
- `data = next(train_iterator)`
- `pixel_values = data["pixel_values"]`
- 对 batch 图像贴 patch；
- 做目标动作攻击损失优化。

也就是说，对抗 patch 的优化对象是“当前 batch 里的若干帧”，而不是一整条 rollout 轨迹。

### 训练时实际加载了哪些字段

虽然原始 RLDS step 里有：

- `image`
- `wrist_image`
- `state`
- `joint_state`
- `language_instruction`
- `action`

但当前训练 dataloader 只显式请求了：

- `load_camera_views=("primary",)`
- `load_proprio=False`
- `load_language=True`

因此训练实际使用的是：

- 主视角 RGB
- 语言指令
- action

而不是完整观测。

---

## 三、两条路径的本质区别

| 维度 | 仿真评测 | 对抗样本训练 |
| --- | --- | --- |
| 数据来源 | `LIBERO` 环境在线观测 | 本地 `dataset/` 下 RLDS/TFDS |
| 访问入口 | `task suite -> env.reset/set_init_state/step` | `get_dataloader -> RLDSDataset -> DataLoader` |
| 采样单位 | task / episode / env step | frame / transition batch |
| 是否在线闭环 | 是 | 否 |
| 图像产生方式 | 环境渲染得到当前帧 | 从 TFRecord 解码得到历史数据帧 |
| patch 施加位置 | 当前 rollout 帧 | 当前 batch 的历史样本帧 |
| 优化/评测目标 | 成功率、轨迹执行结果 | action token loss / 目标动作偏移 |
| 与 demo 的关系 | 借 demo 初始状态与长度上界 | 直接消费 demo 逐步样本 |

---

## 四、本地数据集规模

### 1. 当前本地实际存在的 LIBERO RLDS builder

`dataset` 是一个符号链接，指向：

- `/mnt/home/lvmingyuan/data/VLA_Datasets`

当前本地可见的 LIBERO 训练数据目录有：

- `dataset/libero_spatial_no_noops`
- `dataset/libero_object_no_noops`
- `dataset/libero_goal_no_noops`
- `dataset/libero_10_no_noops`

另外还有：

- `dataset/modified_libero_rlds`

它更像是附带的修改版 LIBERO RLDS 仓库副本，不是当前训练代码直接读取的 builder 路径。

### 2. `_no_noops` 的含义

这些数据不是原始 LIBERO demo 直接转出来的。  
`experiments/robot/libero/regenerate_libero_dataset.py` 说明了其生成逻辑：

- 重放原始 demo；
- 过滤 no-op / zero action；
- 只保留成功 episode；
- 之后再转成 RLDS。

因此训练用的是“重放后、去 no-op、仅成功轨迹”的版本。

### 3. 本地规模统计

下面的数字分成三类：

- `目录大小`：来自本机 `du -sh`
- `shards / trajectories / TFDS numBytes`：来自各自 `dataset_info.json`
- `num_transitions`：只有 `libero_spatial_no_noops` 在当前仓库里找到了现成缓存统计；其余三个当前仓库未发现对应缓存文件

| 数据集 | 目录大小 | TFRecord shards | 轨迹数 trajectories | TFDS 记录字节数 numBytes | 已缓存 transition 数 |
| --- | --- | ---: | ---: | ---: | ---: |
| `libero_spatial_no_noops` | 1.8G | 16 | 432 | 1,914,619,638 | 52,970 |
| `libero_object_no_noops` | 2.7G | 32 | 454 | 2,817,311,159 | 未在仓库中找到缓存值 |
| `libero_goal_no_noops` | 1.8G | 16 | 428 | 1,841,891,826 | 未在仓库中找到缓存值 |
| `libero_10_no_noops` | 3.5G | 32 | 379 | 3,656,799,026 | 未在仓库中找到缓存值 |

补充观察：

- `libero_object_no_noops` 轨迹数最多：454
- `libero_10_no_noops` 单目录最大：3.5G
- `libero_spatial_no_noops` 在现成缓存里可确认有 `52,970` 个 step-level transitions
- 从评测脚本里的 `max_steps` 注释看，`libero_10` 单轨迹上界最长

### 4. step 特征规模

以 `libero_spatial_no_noops` 的 `features.json` 为例，每个 step 至少包含：

- `action`: `float32[7]`
- `observation.image`: `256 x 256 x 3` JPEG
- `observation.wrist_image`: `256 x 256 x 3` JPEG
- `observation.state`: `float32[8]`
- `observation.joint_state`: `float32[7]`
- `language_instruction`
- `is_first / is_last / is_terminal / reward / discount`

但请注意，当前训练代码不会把这些字段全部读进模型。

### 5. 本次未统计到的内容

- 当前本地 `dataset/` 下**没有看到** `bridge_orig` 对应的 builder 目录，所以这次没有给出 `bridge_orig` 的本地实际规模。
- `libero_object_no_noops / libero_goal_no_noops / libero_10_no_noops` 的精确 `num_transitions`，当前仓库里没有现成缓存文件；若需要，可以专门在带 `tensorflow_datasets` 的环境里遍历 builder 再补一版。

---

## 五、最容易混淆的一句话总结

- 仿真评测：用 `LIBERO` 环境在线生成当前轨迹的每一步观测。
- 对抗样本训练：用本地 RLDS 历史数据把每一步观测当监督样本喂给 patch 优化。

所以二者一个是“在线 rollout”，一个是“离线逐帧监督”。

---

## 六、对应代码位置

- 仿真入口：`scripts/run_simulation.sh`
- 评测调度：`evaluation_tool/eval_queue_single_four_spec.py`
- LIBERO 仿真主循环：`experiments/robot/libero/run_libero_eval_args_geo_batch.py`
- LIBERO 环境与图像处理：`experiments/robot/libero/libero_utils.py`
- 训练入口：
  - `scripts/run_UADA.sh`
  - `scripts/run_TMA_ddp.sh`
- dataloader 入口：`VLAAttacker/white_patch/openvla_dataloader.py`
- RLDS dataset 封装：`prismatic/vla/datasets/datasets.py`
- RLDS 核心管线：`prismatic/vla/datasets/rlds/dataset.py`
- LIBERO 数据重生成：`experiments/robot/libero/regenerate_libero_dataset.py`
