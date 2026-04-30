import torch


AXIS_NAMES = {0: "x", 1: "y", 2: "z"}
DIRECTION_METRIC_KEYS = [
    "count",  # 参与方向指标统计的样本数。
    "vector_gap_sum",  # 预测偏移与目标方向偏移之间的 L2 距离总和。
    "cosine_sum",  # 预测方向与目标方向的余弦相似度总和。
    "angle_sum",  # 预测方向与目标方向夹角（度数）的总和。
    "projection_gap_sum",  # 预测结果在目标方向上的投影差总和。
    "sign_match_sum",  # 单轴攻击时，预测方向与目标方向同号的样本数。
    "x_abs_error_sum",  # x 轴分量绝对误差总和。
    "y_abs_error_sum",  # y 轴分量绝对误差总和。
    "z_abs_error_sum",  # z 轴分量绝对误差总和。
    "x_signed_error_sum",  # x 轴分量带符号误差总和。
    "y_signed_error_sum",  # y 轴分量带符号误差总和。
    "z_signed_error_sum",  # z 轴分量带符号误差总和。
]


def normalize_target_direction(target_direction):
    if target_direction is None:
        return None
    if len(target_direction) != 3:
        raise ValueError("target_direction must contain exactly 3 values for x,y,z.")
    normalized = torch.tensor(target_direction, dtype=torch.float32)
    if torch.linalg.norm(normalized).item() == 0:
        raise ValueError("target_direction cannot be the zero vector.")
    return normalized


def resolve_direction_axes(maskidx):
    return [axis for axis in (0, 1, 2) if axis in maskidx]


def empty_direction_metrics():
    return {key: 0.0 for key in DIRECTION_METRIC_KEYS}


def accumulate_direction_metrics(accumulator, update):
    for key in DIRECTION_METRIC_KEYS:
        accumulator[key] += update.get(key, 0.0)


def _resolve_target_action_by_axis(target_action, maskidx, axis):
    if isinstance(target_action, torch.Tensor):
        target_action = target_action.detach().flatten().tolist()

    if isinstance(target_action, (list, tuple)):
        if len(target_action) == len(maskidx):
            return float(target_action[maskidx.index(axis)])
        if axis < len(target_action):
            return float(target_action[axis])
        raise ValueError(
            f"target_action length {len(target_action)} cannot resolve attacked axis {axis} for maskidx={maskidx}."
        )

    return float(target_action)


def build_ideal_straight_trajectory(trajectory, maskidx, target_action):
    """
    Build an ideal straight-attack trajectory from a full trajectory.

    Rules:
    - The first point of the input trajectory is used as the shared start point.
    - For attacked axes in `maskidx`, the ideal trajectory follows a constant per-step displacement
      defined by `target_action`.
    - For axes not included in `maskidx`, the original trajectory coordinates are kept unchanged.

    `target_action` may be:
    - a scalar: applied to every attacked axis
    - a sequence aligned with `maskidx`
    - a full action vector where `target_action[axis]` is used for each attacked axis
    """
    if trajectory is None:
        raise ValueError("trajectory must not be None.")

    input_is_tensor = isinstance(trajectory, torch.Tensor)
    input_dtype = trajectory.dtype if input_is_tensor else None
    input_device = trajectory.device if input_is_tensor else None

    traj = torch.as_tensor(trajectory, dtype=torch.float32)
    if traj.ndim != 2:
        raise ValueError(f"trajectory must have shape [T, D], got {tuple(traj.shape)}.")
    if traj.shape[0] == 0:
        raise ValueError("trajectory must contain at least one point.")

    ideal = traj.clone()
    start_point = traj[0]
    attacked_axes = [axis for axis in maskidx if 0 <= axis < traj.shape[1]]
    if attacked_axes:
        step_index = torch.arange(traj.shape[0], dtype=traj.dtype, device=traj.device)
        for axis in attacked_axes:
            per_step_delta = _resolve_target_action_by_axis(target_action, maskidx, axis)
            ideal[:, axis] = start_point[axis] + step_index * per_step_delta

    if input_is_tensor:
        return ideal.to(device=input_device, dtype=input_dtype)
    return ideal.cpu().numpy()


def calculate_direction_offset_metrics(pred_actions, maskidx, target_direction, eps=1e-8):
    metrics = empty_direction_metrics()
    if target_direction is None or pred_actions is None or pred_actions.numel() == 0 or len(maskidx) == 0:
        return metrics

    available_axes = resolve_direction_axes(maskidx)
    if not available_axes:
        return metrics

    pred_actions = pred_actions.detach().to(torch.float32)
    target_direction = torch.as_tensor(target_direction, dtype=torch.float32)
    if pred_actions.numel() % len(maskidx) != 0:
        return metrics

    pred_actions = pred_actions.view(pred_actions.shape[0] // len(maskidx), len(maskidx))
    selected_cols = [maskidx.index(axis) for axis in available_axes]
    pred_offsets = pred_actions[:, selected_cols]
    target_offsets = target_direction[available_axes].to(pred_offsets.device, dtype=pred_offsets.dtype)
    target_offsets = target_offsets.unsqueeze(0).expand(pred_offsets.shape[0], -1)

    metrics["count"] = float(pred_offsets.shape[0])

    if len(available_axes) == 1:
        axis = available_axes[0]
        axis_name = AXIS_NAMES[axis]
        axis_error = pred_offsets[:, 0] - target_offsets[:, 0]

        # Single-axis attacks only need scalar error and direction-sign statistics.
        abs_error = torch.abs(axis_error)
        sign_match = (pred_offsets[:, 0] * target_offsets[:, 0]) > eps

        metrics["vector_gap_sum"] = float(abs_error.sum().item())
        metrics["sign_match_sum"] = float(sign_match.to(torch.float32).sum().item())
        metrics[f"{axis_name}_abs_error_sum"] = float(abs_error.sum().item())
        metrics[f"{axis_name}_signed_error_sum"] = float(axis_error.sum().item())
        return metrics

    vector_gap = torch.linalg.norm(pred_offsets - target_offsets, dim=1)
    target_norm = torch.linalg.norm(target_offsets[0]).clamp_min(eps)
    target_unit = target_offsets[0] / target_norm
    pred_norm = torch.linalg.norm(pred_offsets, dim=1).clamp_min(eps)
    cosine_similarity = torch.sum(pred_offsets * target_unit.unsqueeze(0), dim=1) / pred_norm
    cosine_similarity = cosine_similarity.clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cosine_similarity))
    projection_gap = torch.abs(torch.sum(pred_offsets * target_unit.unsqueeze(0), dim=1) - target_norm)

    metrics["vector_gap_sum"] = float(vector_gap.sum().item())
    metrics["cosine_sum"] = float(cosine_similarity.sum().item())
    metrics["angle_sum"] = float(angle_deg.sum().item())
    metrics["projection_gap_sum"] = float(projection_gap.sum().item())

    for column_idx, axis in enumerate(available_axes):
        axis_name = AXIS_NAMES[axis]
        axis_error = pred_offsets[:, column_idx] - target_offsets[:, column_idx]
        metrics[f"{axis_name}_abs_error_sum"] = float(torch.abs(axis_error).sum().item())
        metrics[f"{axis_name}_signed_error_sum"] = float(axis_error.sum().item())

    return metrics


def build_direction_log_payload(prefix, metrics, maskidx):
    count = metrics["count"]
    if count <= 0:
        return {}

    available_axes = resolve_direction_axes(maskidx)
    if len(available_axes) == 1:
        axis_name = AXIS_NAMES[available_axes[0]]
        return {
            f"{prefix}_target_direction_scalar_offset_mae": metrics["vector_gap_sum"] / count,
            f"{prefix}_target_direction_scalar_sign_hit_rate": metrics["sign_match_sum"] / count,
            f"{prefix}_target_direction_{axis_name}_mae": metrics[f"{axis_name}_abs_error_sum"] / count,
            f"{prefix}_target_direction_{axis_name}_bias": metrics[f"{axis_name}_signed_error_sum"] / count,
        }

    payload = {
        f"{prefix}_target_direction_offset_l2": metrics["vector_gap_sum"] / count,
        f"{prefix}_target_direction_cosine_similarity": metrics["cosine_sum"] / count,
        f"{prefix}_target_direction_angle_deg": metrics["angle_sum"] / count,
        f"{prefix}_target_direction_projection_gap": metrics["projection_gap_sum"] / count,
    }

    for axis in available_axes:
        axis_name = AXIS_NAMES[axis]
        payload[f"{prefix}_target_direction_{axis_name}_mae"] = metrics[f"{axis_name}_abs_error_sum"] / count
        payload[f"{prefix}_target_direction_{axis_name}_bias"] = metrics[f"{axis_name}_signed_error_sum"] / count

    return payload
