"""
run_libero_eval_realtime.py

Runs an OpenVLA model in a LIBERO simulation environment with on-screen rendering enabled.

Usage:
    python experiments/robot/libero/run_libero_eval_realtime.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --task_id 0 \
        --center_crop True \
        --num_trials_per_task 1
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.swanlab_utils import maybe_init_swanlab, maybe_log_swanlab  # noqa: E402


def get_libero_render_env(
    task,
    resolution=256,
    render_camera="frontview",
    render_gpu_device_id=-1,
):
    """Initializes a LIBERO environment with both on-screen and off-screen rendering enabled."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "render_camera": render_camera,
        "render_gpu_device_id": render_gpu_device_id,
    }
    env = ControlEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def render_env_frame(env):
    """Refresh the on-screen viewer if the wrapped environment exposes one."""
    if hasattr(env, "render"):
        env.render()
    elif hasattr(env, "env") and hasattr(env.env, "render"):
        env.env.render()


def get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unsupported task suite: {task_suite_name}")


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    exp_name: str = "origin"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    task_id: Optional[int] = 0
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    resolution: int = 256
    render_camera: str = "frontview"
    render_gpu_device_id: int = -1
    save_rollout: bool = False

    run_id_note: Optional[str] = "realtime"
    local_log_dir: str = "./experiments/logs"
    use_swanlab: bool = False
    swanlab_project: str = "VLA-Attack"
    seed: int = 7
    # fmt: on


@draccus.wrap()
def eval_libero_realtime(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)

    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    run_id = f"EVAL-REALTIME-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    if cfg.task_id is None:
        task_ids = range(num_tasks_in_suite)
    else:
        assert 0 <= cfg.task_id < num_tasks_in_suite, f"task_id {cfg.task_id} out of range [0, {num_tasks_in_suite - 1}]"
        task_ids = [cfg.task_id]

    resize_size = get_image_resize_size(cfg)
    max_steps = get_max_steps(cfg.task_suite_name)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_render_env(
            task,
            resolution=cfg.resolution,
            render_camera=cfg.render_camera,
            render_gpu_device_id=cfg.render_gpu_device_id,
        )

        try:
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                print(f"\nTask {task_id}: {task_description}")
                log_file.write(f"\nTask {task_id}: {task_description}\n")

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                render_env_frame(env)

                t = 0
                done = False
                replay_images = []

                print(f"Starting episode {task_episodes + 1}...")
                log_file.write(f"Starting episode {task_episodes + 1}...\n")
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                            render_env_frame(env)
                            t += 1
                            continue

                        img = get_libero_image(obs, resize_size)
                        replay_images.append(img)

                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )
                        action = normalize_gripper_action(action, binarize=True)

                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, _, done, _ = env.step(action.tolist())
                        render_env_frame(env)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as exc:
                        print(f"Caught exception: {exc}")
                        log_file.write(f"Caught exception: {exc}\n")
                        break

                task_episodes += 1
                total_episodes += 1

                if cfg.save_rollout:
                    save_rollout_video(
                        replay_images,
                        total_episodes,
                        success=done,
                        task_description=task_description,
                        log_file=log_file,
                        exp_name=cfg.exp_name,
                    )

                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()

            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
            log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
            log_file.flush()
            maybe_log_swanlab(
                swanlab_enabled,
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                },
            )
        finally:
            env.close()

    maybe_log_swanlab(
        swanlab_enabled,
        {
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        },
    )
    log_file.close()


if __name__ == "__main__":
    eval_libero_realtime()
