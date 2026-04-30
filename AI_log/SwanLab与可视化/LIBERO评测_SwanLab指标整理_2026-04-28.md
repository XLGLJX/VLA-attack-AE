# LIBERO评测 SwanLab 指标整理

## 1. 文档目的

本文整理当前 `run_libero_eval.py` 中保存到 SwanLab 的指标与可视化内容，并说明这些记录如何通过抽象出的
`experiments/robot/libero/libero_swanlab.py` 进行复用。

注意：

- 这里实际抽象出来的文件是 `libero_swanlab.py`
- 不是 `libero_utils.py`
- `libero_utils.py` 主要负责环境、图像和视频等基础评测工具
- `libero_swanlab.py` 才负责 LIBERO 评测的 SwanLab 记录接口

## 2. 当前代码结构

当前 SwanLab 记录相关代码分成两层：

### 2.1 通用层

文件：

- `experiments/robot/swanlab_utils.py`

职责：

- `maybe_init_swanlab(...)`
  - 在 `use_swanlab=False` 时跳过初始化
  - 在环境未安装 `swanlab` 时给出统一报错
  - 对 `config` 做基础清洗

- `maybe_log_swanlab(...)`
  - 统一包装 `swanlab.log(...)`
  - 支持无 `step` 和带 `step` 两种情况

### 2.2 LIBERO评测语义层

文件：

- `experiments/robot/libero/libero_swanlab.py`

职责：

- 针对 LIBERO 评测语义，封装轨迹、视频、task 汇总、全局汇总等记录逻辑

### 2.3 调用层

文件：

- `experiments/robot/libero/run_libero_eval.py`

职责：

- 真正执行 LIBERO rollout
- 在合适的时机调用 `libero_swanlab.py` 里的接口

## 3. `run_libero_eval.py` 的记录流程

`run_libero_eval.py` 中 SwanLab 的记录流程如下：

1. 评测开始前，通过 `maybe_init_swanlab(...)` 初始化实验
2. 每个 episode 结束后：
   - 记录 rollout GIF
   - 记录单条 EEF 轨迹
   - 记录当前 task 下累计的多条 EEF 轨迹
3. 每个 task 结束后：
   - 记录 task 成功率柱状图
   - 记录 task 结果表格
4. 全部 task 结束后：
   - 记录总成功率 gauge
   - 记录成功/失败分布饼图
   - 记录总体汇总表

## 4. 当前保存到 SwanLab 的内容

### 4.1 Episode 级视频

接口：

- `libero_swanlab.log_rollout_video(...)`

SwanLab key：

- `rollout_video/{task_description}`

内容：

- 每个 episode 的 GIF 视频
- caption 中包含：
  - `Episode {episode_idx}`
  - `Success={done}`

说明：

- 这是最直接的行为回放
- 适合人工观察 patch 或策略是否导致异常动作

## 4.2 Episode 级 EEF 轨迹

接口：

- `libero_swanlab.log_episode_trajectory(...)`

SwanLab keys：

- `eef_trajectory/{task_description}`
- `eef_path_length/{task_description}`

内容：

- `eef_trajectory/{task_description}`
  - 单个 episode 的 3D 末端执行器轨迹
  - 使用 `Line3D` 展示
  - 颜色随时间渐变

- `eef_path_length/{task_description}`
  - 当前 episode 轨迹总长度
  - 通过相邻位置差的 L2 距离累加得到

说明：

- `eef_trajectory` 用于观察轨迹形状
- `eef_path_length` 用于观察动作是否变得更绕、更抖或更长

## 4.3 Task 内多 Episode 轨迹汇总

接口：

- `libero_swanlab.update_multi_episode_trajectory(...)`

SwanLab key：

- `eef_trajectory_multi/{task_description}`

内容：

- 同一 task 下多条 episode 的轨迹叠加图
- 每条轨迹按成功/失败着色：
  - 成功：绿色
  - 失败：红色
- 图例中会显示 `Episode {episode_idx} | success/fail`

说明：

- 这个图适合对比：
  - 成功轨迹是否收敛到某一类路径
  - 失败轨迹是否集中偏向某个方向

## 4.4 Task 级汇总结果

接口：

- `libero_swanlab.log_task_results(...)`

SwanLab keys：

- `task_success_rate_bar`
- `task_results_table`

内容：

- `task_success_rate_bar`
  - 每个 task 的成功率柱状图
  - 横轴标签格式为 `Task {task_id} (n={num_episodes})`

- `task_results_table`
  - 表格字段包括：
    - `Task ID`
    - `Task Description`
    - `Success Rate`
    - `Num Episodes`

说明：

- 这部分是 task 粒度的主汇总视图
- 适合横向比较不同 task 的攻击效果

## 4.5 总体汇总结果

接口：

- `libero_swanlab.log_final_summary(...)`

SwanLab keys：

- `total_success_rate_gauge`
- `episode_distribution_pie`
- `total_summary_table`

内容：

- `total_success_rate_gauge`
  - 总成功率仪表盘

- `episode_distribution_pie`
  - 成功/失败 episode 分布饼图

- `total_summary_table`
  - 汇总表字段包括：
    - `Total Episodes`
    - `Successes`
    - `Failures`
    - `Success Rate`

说明：

- 这部分用于最终汇总展示
- 一般适合直接用于实验结果截图和比较

## 4.6 直接以标量形式写入的记录

虽然主要可视化接口已经抽到 `libero_swanlab.py`，但仍有一部分简单标量会在调用侧直接记录。

在 `run_libero_eval_args_geo_batch.py` 中，当前还直接记录：

- `success_rate/{task_description}`
- `num_episodes/{task_description}`
- `success_rate/total`
- `num_episodes/total`

说明：

- 这些是简单数值
- 目前直接用 `maybe_log_swanlab(...)` 就足够
- 不一定需要再包成额外 helper

## 5. 当前存在但未在 `run_libero_eval.py` 主流程中启用的接口

`libero_swanlab.py` 里还提供了：

- `eef_trajectory_to_object3d(...)`

作用：

- 可将单条 EEF 轨迹转为 `Object3D` 点云格式

当前状态：

- 该接口已经可用
- 但 `run_libero_eval.py` 目前主流程没有调用它
- 当前主流程默认展示的是 `Line3D`

说明：

- 如果后续想看离散点云式轨迹展示，可以直接复用这个接口

## 6. 不属于 SwanLab、但和评测相关的本地输出

下面这些内容是评测过程中的本地保存，不是 SwanLab 记录：

- 本地文本日志：
  - `local_log_dir/{run_id}.txt`

- 本地 rollout 视频：
  - `save_rollout_video(...)` 生成的 mp4

- patch eval 汇总文件：
  - `*_summary.json`

说明：

- 这些文件主要用于离线排查与复现实验
- 它们和 SwanLab 面板是互补关系

## 7. 推荐理解方式

可以将当前 LIBERO 评测的 SwanLab 体系理解为三层：

1. 基础层：`swanlab_utils.py`
   - 负责 init 和 log 的通用包装

2. 语义层：`libero_swanlab.py`
   - 负责“LIBERO 评测应该记录什么”

3. 执行层：`run_libero_eval.py`
   - 负责“在 rollout 的哪个阶段触发记录”

这样做的好处是：

- 新评测文件可以直接复用现有记录接口
- 调用侧不需要再手写图表构造逻辑
- 轨迹、视频、task 汇总、final summary 的记录方式保持一致

## 8. 后续扩展建议

如果后续还要扩展 SwanLab 记录，建议优先按下面方式做：

- 通用初始化或基础 `log` 包装：
  - 放在 `swanlab_utils.py`

- LIBERO 专属图表或轨迹展示：
  - 放在 `libero_swanlab.py`

- 评测脚本中的时机控制：
  - 保留在 `run_libero_eval.py` 或其他评测入口

不建议把所有逻辑重新塞回单个评测文件，否则会重新出现：

- 图表构造代码重复
- 多评测文件日志行为不一致
- 后续维护成本变高

