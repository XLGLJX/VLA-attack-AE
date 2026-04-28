"""Reusable SwanLab helpers for LIBERO evaluation scripts."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pyecharts import options as pyopts

from experiments.robot.swanlab_utils import maybe_log_swanlab, swanlab


def _require_swanlab() -> Any:
    if swanlab is None:
        raise ImportError("SwanLab helpers were requested, but `swanlab` is not installed.")
    return swanlab


def eef_trajectory_to_object3d(trajectory: list, episode_idx: int, done: bool) -> Any:
    swanlab_module = _require_swanlab()
    traj = np.stack(trajectory, axis=0)
    point_count = len(traj)
    color = [50, 205, 50] if done else [220, 20, 60]
    colors = np.tile(np.array(color, dtype=np.uint8), (point_count, 1))
    points_xyzrgb = np.concatenate([traj, colors], axis=1)
    status = "success" if done else "fail"
    return swanlab_module.Object3D(
        points_xyzrgb,
        caption=f"Episode {episode_idx} | {status} | Steps {point_count}",
    )


def eef_trajectory_to_line3d(trajectory: list, episode_idx: int, done: bool) -> Any:
    swanlab_module = _require_swanlab()
    point_count = len(trajectory)
    data = [
        [float(point[1]), float(point[0]), float(point[2]), idx / max(point_count - 1, 1)]
        for idx, point in enumerate(trajectory)
    ]
    status = "success" if done else "fail"
    line3d = swanlab_module.echarts.Line3D()
    line3d.add(
        "",
        data,
        encode={"x": 1, "y": 0, "z": 2},
        xaxis3d_opts=pyopts.Axis3DOpts(name="Y (Front/Back)", type_="value"),
        yaxis3d_opts=pyopts.Axis3DOpts(name="X (Left/Right)", type_="value"),
        zaxis3d_opts=pyopts.Axis3DOpts(name="Z (Up/Down)", type_="value"),
        grid3d_opts=pyopts.Grid3DOpts(width=100, height=100, depth=100),
        label_opts={"is_show": False},
    )
    line3d.set_global_opts(
        visualmap_opts=pyopts.VisualMapOpts(
            min_=0,
            max_=1,
            dimension=3,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
            ],
            is_show=False,
        ),
        title_opts=pyopts.TitleOpts(title=f"Episode {episode_idx} | {status}"),
    )
    options = line3d.get_options()
    options["series"][0]["lineStyle"] = {"width": 12}
    return line3d


def create_multi_episode_line3d() -> Any:
    swanlab_module = _require_swanlab()
    line3d = swanlab_module.echarts.Line3D()
    line3d.set_global_opts(
        legend_opts=pyopts.LegendOpts(is_show=True),
        title_opts=pyopts.TitleOpts(title="Multi-Episode Trajectories"),
    )
    return line3d


def add_episode_to_line3d(line3d: Any, trajectory: list, episode_idx: int, done: bool) -> Any:
    data = [[float(point[1]), float(point[0]), float(point[2])] for point in trajectory]
    status = "success" if done else "fail"
    color = "#32CD32" if done else "#DC143C"
    line3d.add(
        f"Episode {episode_idx} | {status}",
        data,
        encode={"x": 1, "y": 0, "z": 2},
        xaxis3d_opts=pyopts.Axis3DOpts(name="Y (Front/Back)", type_="value"),
        yaxis3d_opts=pyopts.Axis3DOpts(name="X (Left/Right)", type_="value"),
        zaxis3d_opts=pyopts.Axis3DOpts(name="Z (Up/Down)", type_="value"),
        grid3d_opts=pyopts.Grid3DOpts(width=100, height=100, depth=100),
        label_opts={"is_show": False},
    )
    options = line3d.get_options()
    options["series"][-1]["lineStyle"] = {"color": color, "width": 12}
    return line3d


def log_rollout_video(
    enabled: bool,
    task_description: str,
    gif_path: str,
    episode_idx: int,
    done: bool,
    *,
    step: Optional[int] = None,
) -> None:
    swanlab_module = _require_swanlab()
    maybe_log_swanlab(
        enabled,
        {
            f"rollout_video/{task_description}": swanlab_module.Video(
                gif_path, caption=f"Episode {episode_idx} | Success={done}"
            )
        },
        step=step,
    )


def log_episode_trajectory(
    enabled: bool,
    task_description: str,
    trajectory: list,
    episode_idx: int,
    done: bool,
    *,
    step: Optional[int] = None,
) -> float:
    traj_line3d = eef_trajectory_to_line3d(trajectory, episode_idx, done)
    traj = np.stack(trajectory, axis=0)
    path_length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    maybe_log_swanlab(
        enabled,
        {
            f"eef_trajectory/{task_description}": traj_line3d,
            f"eef_path_length/{task_description}": path_length,
        },
        step=step,
    )
    return path_length


def update_multi_episode_trajectory(
    enabled: bool,
    task_description: str,
    current_line3d: Optional[Any],
    trajectory: list,
    episode_idx: int,
    done: bool,
    *,
    step: Optional[int] = None,
) -> Any:
    if current_line3d is None:
        current_line3d = create_multi_episode_line3d()
    current_line3d = add_episode_to_line3d(current_line3d, trajectory, episode_idx, done)
    maybe_log_swanlab(
        enabled,
        {f"eef_trajectory_multi/{task_description}": current_line3d},
        step=step,
    )
    return current_line3d


def log_task_results(enabled: bool, task_results: list[dict[str, Any]], *, step: Optional[int] = None) -> None:
    swanlab_module = _require_swanlab()
    bar = swanlab_module.echarts.Bar()
    x_labels = [f"Task {result['task_id']}\n(n={result['num_episodes']})" for result in task_results]
    bar.add_xaxis(x_labels)
    bar.add_yaxis("Success Rate", [round(result["success_rate"], 4) for result in task_results])

    table = swanlab_module.echarts.Table()
    table.add(
        ["Task ID", "Task Description", "Success Rate", "Num Episodes"],
        [
            [
                str(result["task_id"]),
                result["task_description"],
                f"{result['success_rate']:.4f}",
                str(result["num_episodes"]),
            ]
            for result in task_results
        ],
    )
    maybe_log_swanlab(
        enabled,
        {"task_success_rate_bar": bar, "task_results_table": table},
        step=step,
    )


def log_final_summary(enabled: bool, total_episodes: int, total_successes: int) -> None:
    swanlab_module = _require_swanlab()
    total_success_rate = float(total_successes) / float(total_episodes)
    total_failures = total_episodes - total_successes

    gauge = swanlab_module.echarts.Gauge()
    gauge.add(
        "Total Success Rate",
        [("Success Rate", round(total_success_rate * 100, 2))],
        detail_label_opts={"formatter": "{value}%"},
    )

    pie = swanlab_module.echarts.Pie()
    pie.add("Episode Distribution", [("Success", total_successes), ("Failure", total_failures)])

    summary_table = swanlab_module.echarts.Table()
    summary_table.add(
        ["Metric", "Value"],
        [
            ["Total Episodes", str(total_episodes)],
            ["Successes", str(total_successes)],
            ["Failures", str(total_failures)],
            ["Success Rate", f"{total_success_rate:.4f}"],
        ],
    )
    maybe_log_swanlab(
        enabled,
        {
            "total_success_rate_gauge": gauge,
            "episode_distribution_pie": pie,
            "total_summary_table": summary_table,
        },
    )
