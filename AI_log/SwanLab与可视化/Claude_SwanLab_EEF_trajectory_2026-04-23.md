# Claude - SwanLab EEF 3D 轨迹记录集成

**日期**: 2026-04-23
**任务**: 在 `run_libero_eval.py` 中集成 SwanLab Object3D，保存机械臂末端执行器（EEF）在 3D 空间中的运动轨迹

---

## 修改内容

### 1. 文件
- `experiments/robot/libero/run_libero_eval.py`

### 2. 新增 import
```python
import swanlab
```

### 3. 新增 helper 函数
```python
def eef_trajectory_to_object3d(trajectory: list, episode_idx: int, done: bool) -> swanlab.Object3D:
    """将 EEF xyz 轨迹转为 SwanLab Object3D 点云，整条轨迹使用统一颜色。"""
    traj = np.stack(trajectory, axis=0)  # (T, 3)
    T = len(traj)
    # 统一颜色：成功用绿色，失败用红色
    color = [50, 205, 50] if done else [220, 20, 60]
    colors = np.tile(np.array(color, dtype=np.uint8), (T, 1))
    points_xyzrgb = np.concatenate([traj, colors], axis=1)
    status = "success" if done else "fail"
    return swanlab.Object3D(
        points_xyzrgb,
        caption=f"Episode {episode_idx} | {status} | Steps {T}",
    )
```

### 4. episode 循环内修改
- **初始化**: `eef_trajectory = []` （每 episode 开始时清空）
- **dummy action 阶段**: `eef_trajectory.append(obs["robot0_eef_pos"].copy())`
- **real action 阶段**: `eef_trajectory.append(obs["robot0_eef_pos"].copy())`
- **episode 结束后上传**:
  ```python
  if swanlab_enabled and len(eef_trajectory) > 0:
      traj_obj = eef_trajectory_to_object3d(eef_trajectory, episode_idx, done)
      traj = np.stack(eef_trajectory, axis=0)
      path_length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
      maybe_log_swanlab(
          swanlab_enabled,
          {
              f"eef_trajectory/{task_description}": traj_obj,
              f"eef_path_length/{task_description}": path_length,
          },
          step=total_episodes,
      )
  ```

---

## SwanLab 上报指标

| 指标路径 | 类型 | 说明 |
|---|---|---|
| `eef_trajectory/{task_description}` | `swanlab.Object3D` | 3D 点云轨迹，可交互旋转查看 |
| `eef_path_length/{task_description}` | float | 该 episode 的 EEF 路径总长度 |

## 颜色约定
- **绿色** (`[50, 205, 50]`): 该 episode 成功完成 (`done=True`)
- **红色** (`[220, 20, 60]`): 该 episode 失败 (`done=False`)

---

## 使用方式

```bash
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_swanlab True \
    --swanlab_project VLA-Attack-Eval \
    --num_trials_per_task 10
```

运行后，在 SwanLab Web 端的 **Media** 标签下查看 `eef_trajectory` 分组，可拖拽旋转 3D 点云。
