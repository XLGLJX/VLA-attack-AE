"""
run_libero_eval_viser.py

Runs an OpenVLA model in a headless LIBERO simulation environment and streams frames to a Viser web UI.

Usage:
    python experiments/robot/libero/run_libero_eval_viser.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 1
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval_viser.py \
  --model_family openvla \
  --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --center_crop True \
  --num_trials_per_task 1
"""

import os
import sys
import time
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
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


def format_action(action: Optional[np.ndarray]) -> str:
    if action is None:
        return "N/A"
    return np.array2string(np.asarray(action), precision=3, suppress_small=True)


def format_vector(values: Optional[np.ndarray]) -> str:
    if values is None:
        return "N/A"
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def format_action_table(action: Optional[np.ndarray]) -> str:
    if action is None:
        return "N/A"
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    action = np.asarray(action).reshape(-1)
    return ", ".join(f"{label}={value:.3f}" for label, value in zip(labels, action))


def format_gripper_status(obs: Optional[dict], action: Optional[np.ndarray]) -> str:
    if obs is None:
        return "N/A"
    qpos = format_vector(obs.get("robot0_gripper_qpos"))
    if action is None:
        command = "N/A"
    else:
        gripper_action = float(np.asarray(action).reshape(-1)[-1])
        if gripper_action > 0:
            command = f"close ({gripper_action:.3f})"
        elif gripper_action < 0:
            command = f"open ({gripper_action:.3f})"
        else:
            command = f"neutral ({gripper_action:.3f})"
    return f"qpos={qpos}, command={command}"


def format_eef_pose(obs: Optional[dict]) -> str:
    if obs is None:
        return "N/A"
    pos = format_vector(obs.get("robot0_eef_pos"))
    quat = format_vector(obs.get("robot0_eef_quat"))
    return f"pos={pos}, quat={quat}"


def html_rows(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td style='padding:4px 10px 4px 0; white-space:nowrap; vertical-align:top;'><b>{html.escape(label)}</b></td>"
        f"<td style='padding:4px 0; white-space:pre-wrap; word-break:break-word;'>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return (
        "<div style='line-height:1.5;'>"
        "<table style='width:100%; border-collapse:collapse; table-layout:fixed;'>"
        f"{body}</table></div>"
    )


def format_run_status_html(
    task_description: str,
    task_id: int,
    episode_idx: int,
    step_idx: int,
    done: bool,
    total_episodes: int,
    total_successes: int,
) -> str:
    success_rate = (total_successes / total_episodes * 100.0) if total_episodes > 0 else 0.0
    return html_rows(
        [
            ("Task ID", str(task_id)),
            ("Task", task_description),
            ("Episode", str(episode_idx)),
            ("Step", str(step_idx)),
            ("Done", str(done)),
            ("Completed Episodes", str(total_episodes)),
            ("Successes", f"{total_successes} ({success_rate:.1f}%)"),
        ]
    )


def format_action_status_html(action: Optional[np.ndarray]) -> str:
    return html_rows(
        [
            ("Last Action (Vector)", format_action(action)),
            ("Action 7D", format_action_table(action)),
        ]
    )


def format_robot_status_html(obs: Optional[dict], action: Optional[np.ndarray]) -> str:
    return html_rows(
        [
            ("Gripper State", format_gripper_status(obs, action)),
            ("EEF Pose", format_eef_pose(obs)),
        ]
    )


def format_io_status_html(local_log_filepath: str, save_rollout: bool) -> str:
    return html_rows(
        [
            ("Save Rollout Video", str(save_rollout)),
            ("Log File", local_log_filepath),
        ]
    )


def env_image_for_viser(obs: dict) -> np.ndarray:
    """Prepare the raw environment frame for browser display."""
    return np.ascontiguousarray(obs["agentview_image"][::-1])


def compute_image_plane_size(image: np.ndarray, max_width: float) -> tuple[float, float]:
    height, width = image.shape[:2]
    render_width = max_width
    render_height = max_width * float(height) / float(width)
    return render_width, render_height


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
    task_id: Optional[int] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    resolution: int = 256
    save_rollout: bool = False

    host: str = "127.0.0.1"
    port: int = 38765
    jpeg_quality: int = 80
    stream_model_input: bool = True
    control_width: str = "large"
    main_image_max_width: float = 12.0

    run_id_note: Optional[str] = "viser"
    local_log_dir: str = "./experiments/logs"
    use_swanlab: bool = False
    swanlab_project: str = "VLA-Attack"
    seed: int = 7
    # fmt: on


@draccus.wrap()
def eval_libero_viser(cfg: GenerateConfig) -> None:
    try:
        import viser
    except ImportError as exc:
        raise ImportError("`viser` is required for run_libero_eval_viser.py. Install it with `pip install viser`.") from exc

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

    run_id = f"EVAL-VISER-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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

    server = viser.ViserServer(host=cfg.host, port=cfg.port)
    server.gui.configure_theme(control_layout="collapsible", control_width=cfg.control_width, show_share_button=False)
    server.gui.set_panel_label("OpenVLA LIBERO Monitor")
    server.scene.world_axes.visible = False

    blank_frame = np.zeros((cfg.resolution, cfg.resolution, 3), dtype=np.uint8)
    image_plane_width, image_plane_height = compute_image_plane_size(blank_frame, cfg.main_image_max_width)
    main_image_handle = server.scene.add_image(
        "/main_env_image",
        blank_frame,
        render_width=image_plane_width,
        render_height=image_plane_height,
        format="jpeg",
        jpeg_quality=cfg.jpeg_quality,
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    with server.gui.add_folder("Frames"):
        model_image_handle = server.gui.add_image(
            blank_frame,
            label="Model Input Frame",
            format="jpeg",
            jpeg_quality=cfg.jpeg_quality,
            visible=cfg.stream_model_input,
        )

    with server.gui.add_folder("Controls"):
        pause_checkbox = server.gui.add_checkbox("Pause", initial_value=False)
        stop_button = server.gui.add_button("Stop After Episode", color="red")

    with server.gui.add_folder("Status"):
        run_status_html = server.gui.add_html("<div>Waiting for the first frame...</div>")
        action_status_html = server.gui.add_html(html_rows([("Last Action (Vector)", "Waiting for the first action...")]))
        robot_status_html = server.gui.add_html(html_rows([("Robot State", "Waiting for the first observation...")]))
        io_status_html = server.gui.add_html(
            format_io_status_html(local_log_filepath, cfg.save_rollout)
        )
        access_markdown = server.gui.add_markdown(
            "\n".join(
                [
                    f"**Bind Host**: {cfg.host}",
                    f"**Bind Port**: {cfg.port}",
                    "**SSH Port Forward**: `ssh -L {port}:localhost:{port} <user>@<server>`".replace("{port}", str(cfg.port)),
                    f"**Browser URL**: `http://localhost:{cfg.port}`",
                ]
            )
        )

    print(f"Viser server listening on http://{cfg.host}:{cfg.port}")
    log_file.write(f"Viser server listening on http://{cfg.host}:{cfg.port}\n")
    log_file.flush()

    fixed_camera_position = (0.0, 0.0, -image_plane_height)
    fixed_camera_look_at = (0.0, 0.0, 0.0)
    fixed_camera_up = (0.0, -1.0, 0.0)

    server.initial_camera.position = fixed_camera_position
    server.initial_camera.look_at = fixed_camera_look_at
    server.initial_camera.up_direction = fixed_camera_up

    def apply_fixed_camera(camera) -> None:
        camera.position = fixed_camera_position
        camera.look_at = fixed_camera_look_at
        camera.up_direction = fixed_camera_up

    @server.on_client_connect
    def _(client) -> None:
        apply_fixed_camera(client.camera)

        @client.camera.on_update
        def _lock_camera(camera) -> None:
            apply_fixed_camera(camera)

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
    stop_requested = False
    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.resolution)

        try:
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                print(f"\nTask {task_id}: {task_description}")
                log_file.write(f"\nTask {task_id}: {task_description}\n")

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                main_image_handle.image = env_image_for_viser(obs)
                model_frame = get_libero_image(obs, resize_size)
                if cfg.stream_model_input:
                    model_image_handle.image = model_frame

                t = 0
                done = False
                replay_images = []
                last_action = None

                run_status_html.content = format_run_status_html(
                    task_description,
                    task_id,
                    episode_idx,
                    t,
                    done,
                    total_episodes,
                    total_successes,
                )
                action_status_html.content = format_action_status_html(last_action)
                robot_status_html.content = format_robot_status_html(obs, last_action)
                print(f"Starting episode {task_episodes + 1}...")
                log_file.write(f"Starting episode {task_episodes + 1}...\n")

                while t < max_steps + cfg.num_steps_wait:
                    if stop_button.value:
                        stop_button.value = False
                        stop_requested = True

                    while pause_checkbox.value and not stop_requested:
                        time.sleep(0.1)
                        if stop_button.value:
                            stop_button.value = False
                            stop_requested = True
                            break
                    if stop_requested:
                        break

                    try:
                        if t < cfg.num_steps_wait:
                            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                            main_image_handle.image = env_image_for_viser(obs)
                            model_frame = get_libero_image(obs, resize_size)
                            if cfg.stream_model_input:
                                model_image_handle.image = model_frame
                            t += 1
                            run_status_html.content = format_run_status_html(
                                task_description,
                                task_id,
                                episode_idx,
                                t,
                                done,
                                total_episodes,
                                total_successes,
                            )
                            action_status_html.content = format_action_status_html(last_action)
                            robot_status_html.content = format_robot_status_html(obs, last_action)
                            continue

                        model_frame = get_libero_image(obs, resize_size)
                        replay_images.append(model_frame)

                        observation = {
                            "full_image": model_frame,
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

                        last_action = np.asarray(action)
                        obs, _, done, _ = env.step(action.tolist())
                        main_image_handle.image = env_image_for_viser(obs)
                        if cfg.stream_model_input:
                            model_image_handle.image = model_frame
                        t += 1

                        run_status_html.content = format_run_status_html(
                            task_description,
                            task_id,
                            episode_idx,
                            t,
                            done,
                            total_episodes,
                            total_successes,
                        )
                        action_status_html.content = format_action_status_html(last_action)
                        robot_status_html.content = format_robot_status_html(obs, last_action)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
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

                if stop_requested:
                    break

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

        if stop_requested:
            break

    maybe_log_swanlab(
        swanlab_enabled,
        {
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        },
    )
    print("Simulation finished. Viser UI will remain available until you stop the process.")
    log_file.write("Simulation finished.\n")
    log_file.close()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down Viser server.")


if __name__ == "__main__":
    eval_libero_viser()
