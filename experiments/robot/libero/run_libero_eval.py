"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_swanlab [ True | False ] \
        --swanlab_project <PROJECT>
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 3 \
        --run_id_note server_smoke \
        --use_swanlab True \
        --swanlab_project VLA-Attack-Eval
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from pyecharts import options as pyopts

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.swanlab_utils import maybe_init_swanlab, maybe_log_swanlab
import swanlab


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


def eef_trajectory_to_line3d(trajectory: list, episode_idx: int, done: bool):
    """将 EEF xyz 轨迹转为 swanlab.echarts.Line3D 3D 折线图，颜色从轨迹起点到终点渐变。"""
    T = len(trajectory)
    # data: [y, x, z, normalized_time]，通过 encode 显式映射
    # 确保 ECharts X 轴 = MuJoCo X (左右)，Y 轴 = MuJoCo Y (前后)
    data = [
        [float(p[1]), float(p[0]), float(p[2]), i / max(T - 1, 1)]
        for i, p in enumerate(trajectory)
    ]
    status = "success" if done else "fail"
    ec = swanlab.echarts

    line3d = ec.Line3D()
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

    # 颜色从起点到终点渐变: 深蓝 -> 青 -> 黄 -> 红
    range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9",
                   "#e0f3f8", "#fee090", "#fdae61", "#f46d43", "#d73027"]
    line3d.set_global_opts(
        visualmap_opts=pyopts.VisualMapOpts(
            min_=0,
            max_=1,
            dimension=3,
            range_color=range_color,
            is_show=False,
        ),
        title_opts=pyopts.TitleOpts(title=f"Episode {episode_idx} | {status}"),
    )
    # 加粗线宽
    opts = line3d.get_options()
    opts["series"][0]["lineStyle"] = {"width": 6}
    return line3d


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    exp_name: str = "origin"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = "spatial1"                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_swanlab: bool = False                        # Whether to also log results in SwanLab
    swanlab_project: str = "VLA-Attack"              # Name of SwanLab project to log to

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    swanlab_enabled = maybe_init_swanlab(
        cfg.use_swanlab,
        cfg.swanlab_project,
        run_id,
        vars(cfg),
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []  # 收集所有 task 的结果用于 SwanLab 汇总展示
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            eef_trajectory = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        # eef_trajectory.append(obs["robot0_eef_pos"].copy())
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)    

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    eef_trajectory.append(obs["robot0_eef_pos"].copy())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            mp4_path = save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file,exp_name=cfg.exp_name
            )

            # Save GIF and upload to SwanLab
            if swanlab_enabled and replay_images:
                import imageio
                processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
                gif_dir = f"./rollouts/{cfg.exp_name}/{DATE}/gifs"
                os.makedirs(gif_dir, exist_ok=True)
                gif_path = f"{gif_dir}/{DATE_TIME}--episode={total_episodes}--success={done}--task={processed_task}.gif"
                imageio.mimsave(gif_path, replay_images, fps=10)
                maybe_log_swanlab(
                    swanlab_enabled,
                    {
                        f"rollout_video/{task_description}": swanlab.Video(
                            gif_path, caption=f"Episode {episode_idx} | Success={done}"
                        ),
                    },
                    step=total_episodes,
                )

            # Upload EEF trajectory to SwanLab (Line3D)
            if swanlab_enabled and len(eef_trajectory) > 0:
                traj_line3d = eef_trajectory_to_line3d(eef_trajectory, episode_idx, done)
                traj = np.stack(eef_trajectory, axis=0)
                path_length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
                maybe_log_swanlab(
                    swanlab_enabled,
                    {
                        f"eef_trajectory/{task_description}": traj_line3d,
                        f"eef_path_length/{task_description}": path_length,
                    },
                    step=total_episodes,
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        # 收集当前 task 结果
        task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "success_rate": float(task_successes) / float(task_episodes),
            "num_episodes": task_episodes,
        })

        # 每完成一个 task，立即上传当前累积的柱状图与数据表（step=task_id，支持渐进增长）
        if swanlab_enabled and task_results:
            ec = swanlab.echarts

            bar = ec.Bar()
            x_labels = [f"Task {r['task_id']}\n(n={r['num_episodes']})" for r in task_results]
            bar.add_xaxis(x_labels)
            bar.add_yaxis("Success Rate", [round(r["success_rate"], 4) for r in task_results])

            table = ec.Table()
            headers = ["Task ID", "Task Description", "Success Rate", "Num Episodes"]
            rows = [
                [str(r["task_id"]), r["task_description"], f"{r['success_rate']:.4f}", str(r["num_episodes"])]
                for r in task_results
            ]
            table.add(headers, rows)

            maybe_log_swanlab(
                swanlab_enabled,
                {
                    "task_success_rate_bar": bar,
                    "task_results_table": table,
                },
                step=task_id,
            )

    # Save local log file
    log_file.close()

    # 汇总仪表盘与饼图展示总成功率
    if swanlab_enabled and total_episodes > 0:
        ec = swanlab.echarts
        total_success_rate = float(total_successes) / float(total_episodes)
        total_failures = total_episodes - total_successes

        gauge = ec.Gauge()
        gauge.add(
            "Total Success Rate",
            [("Success Rate", round(total_success_rate * 100, 2))],
            detail_label_opts={"formatter": "{value}%"},
        )

        pie = ec.Pie()
        pie.add(
            "Episode Distribution",
            [
                ("Success", total_successes),
                ("Failure", total_failures),
            ],
        )

        summary_table = ec.Table()
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
            swanlab_enabled,
            {
                "total_success_rate_gauge": gauge,
                "episode_distribution_pie": pie,
                "total_summary_table": summary_table,
            },
        )


if __name__ == "__main__":
    eval_libero()
