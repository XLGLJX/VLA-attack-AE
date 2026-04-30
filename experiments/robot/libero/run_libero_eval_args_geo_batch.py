"""Patch-based LIBERO evaluation with both CLI and importable interfaces.

直接命令行调用示例：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
    --patchroot run/white_patch_attack/a251313a-6b71-4e8f-81ef-7b2230dc998e/last/patch.pt \
    --task_suite_name libero_spatial \
    --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
    --num_trials_per_task 3 \
    --run_id_note patch_eval_smoke \
    --local_log_dir ./experiments/logs \
    --use_swanlab true \
    --swanlab_project VLA-Attack-Eval
```

攻击结束后在 Python 中直接调用示例：

```python
from experiments.robot.libero.run_libero_eval_args_geo_batch import evaluate_saved_patch_from_attack

summary = evaluate_saved_patch_from_attack(
    attack_args=args,
    patchroot=f"{path}/last/patch.pt",
    num_trials_per_task=10,
)
```
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import imageio
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from VLAAttacker.white_patch.appply_random_transform import RandomPatchTransform
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero import libero_swanlab
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
from experiments.robot.swanlab_utils import maybe_init_swanlab

DEFAULT_CHECKPOINTS = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
    "libero_90": "openvla/openvla-7b",
}
DEFAULT_LOCAL_MODEL_DIRS = {
    "libero_spatial": "models/openvla-7b-finetuned-libero-spatial",
    "libero_object": "models/openvla-7b-finetuned-libero-object",
    "libero_goal": "models/openvla-7b-finetuned-libero-goal",
    "libero_10": "models/openvla-7b-finetuned-libero-10",
    "libero_90": "models/openvla-7b",
}
DEFAULT_PATCH_TRANSFORMS = {
    "libero_spatial": {"x": 120, "y": 160, "angle": 0.0, "shx": 0.0, "shy": 0.0},
    "libero_object": {"x": 30, "y": 150, "angle": 0.0, "shx": 0.0, "shy": 0.0},
    "libero_goal": {"x": 15, "y": 158, "angle": 0.0, "shx": 0.0, "shy": 0.0},
    "libero_10": {"x": 5, "y": 160, "angle": 0.0, "shx": 0.0, "shy": 0.0},
    "libero_90": {"x": 5, "y": 160, "angle": 0.0, "shx": 0.0, "shy": 0.0},
}
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 193,
    "libero_object": 254,
    "libero_goal": 270,
    "libero_10": 505,
    "libero_90": 373,
}


@dataclass
class PatchLiberoEvalConfig:
    model_family: str = "openvla"
    exp_name: str = "patch_eval"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    task_suite_name: str = "libero_object"
    num_steps_wait: int = 10
    num_trials_per_task: int = 100
    run_id_note: Optional[str] = "test_libero_object"
    local_log_dir: Union[str, Path] = "./experiments/logs"
    use_swanlab: bool = True
    swanlab_project: str = "VLA-Attack"
    swanlab_note: str = ""
    seed: int = 7
    patchroot: Union[str, Path] = ""
    x: int = 30
    y: int = 150
    angle: float = 0.0
    shx: float = 0.0
    shy: float = 0.0
    cudaid: int = 0
    resize_patch: bool = False
    geometry: bool = True
    colorjitter: bool = False
    source_attack_name: str = ""
    source_attack_run_dir: str = ""
    source_attack_dataset: str = ""
    source_attack_swanlab_note: str = ""


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def infer_task_suite_name(dataset_name: Optional[str]) -> str:
    if not dataset_name:
        raise ValueError("dataset_name is required to infer a LIBERO task suite.")
    normalized = dataset_name.replace("_no_noops", "")
    for suite_name in DEFAULT_CHECKPOINTS:
        if suite_name in normalized:
            return suite_name
    raise ValueError(f"Cannot infer LIBERO task suite from dataset name: {dataset_name}")


def resolve_patch_path(patchroot: Union[str, Path]) -> Path:
    patchroot = Path(patchroot).expanduser().resolve()
    if patchroot.is_file():
        return patchroot
    if (patchroot / "patch.pt").is_file():
        return patchroot / "patch.pt"
    if (patchroot / "last" / "patch.pt").is_file():
        return patchroot / "last" / "patch.pt"

    iter_dirs = [path for path in patchroot.iterdir() if path.is_dir() and (path / "patch.pt").is_file()] if patchroot.is_dir() else []
    if iter_dirs:
        numeric_dirs = sorted(
            iter_dirs,
            key=lambda item: int(item.name) if item.name.isdigit() else item.name,
        )
        return numeric_dirs[-1] / "patch.pt"

    raise FileNotFoundError(f"Could not resolve a patch.pt from: {patchroot}")


def infer_pretrained_checkpoint(task_suite_name: str, server_root: Optional[Union[str, Path]] = None) -> str:
    if server_root is not None:
        local_candidate = Path(server_root).expanduser().resolve() / DEFAULT_LOCAL_MODEL_DIRS[task_suite_name]
        if local_candidate.exists():
            return str(local_candidate)
    return DEFAULT_CHECKPOINTS[task_suite_name]


def get_default_patch_transform(task_suite_name: str) -> dict[str, float]:
    return DEFAULT_PATCH_TRANSFORMS[task_suite_name].copy()


def ensure_eval_config(cfg: Union[PatchLiberoEvalConfig, argparse.Namespace, dict]) -> PatchLiberoEvalConfig:
    if isinstance(cfg, PatchLiberoEvalConfig):
        return cfg
    if isinstance(cfg, argparse.Namespace):
        return PatchLiberoEvalConfig(**vars(cfg))
    if isinstance(cfg, dict):
        return PatchLiberoEvalConfig(**cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)}")


def build_patch_eval_config(
    patchroot: Union[str, Path],
    task_suite_name: str,
    pretrained_checkpoint: Optional[Union[str, Path]] = None,
    local_log_dir: Optional[Union[str, Path]] = None,
    run_id_note: Optional[str] = None,
    exp_name: Optional[str] = None,
    swanlab_project: str = "VLA-Attack",
    swanlab_note: str = "",
    use_swanlab: bool = True,
    num_trials_per_task: int = 100,
    num_steps_wait: int = 10,
    x: Optional[int] = None,
    y: Optional[int] = None,
    angle: Optional[float] = None,
    shx: Optional[float] = None,
    shy: Optional[float] = None,
    cudaid: int = 0,
    model_family: str = "openvla",
    center_crop: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    seed: int = 7,
    resize_patch: bool = False,
    geometry: bool = True,
    colorjitter: bool = False,
    source_attack_name: str = "",
    source_attack_run_dir: str = "",
    source_attack_dataset: str = "",
    source_attack_swanlab_note: str = "",
) -> PatchLiberoEvalConfig:
    patch_path = resolve_patch_path(patchroot)
    transform_defaults = get_default_patch_transform(task_suite_name)
    run_dir = patch_path.parent.parent if patch_path.parent.name == "last" else patch_path.parent
    local_log_dir = Path(local_log_dir).expanduser().resolve() if local_log_dir else run_dir
    exp_name = exp_name or f"{run_dir.name}_{task_suite_name}"
    run_id_note = run_id_note or patch_path.parent.name

    return PatchLiberoEvalConfig(
        model_family=model_family,
        exp_name=exp_name,
        pretrained_checkpoint=str(pretrained_checkpoint or infer_pretrained_checkpoint(task_suite_name)),
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        center_crop=center_crop,
        task_suite_name=task_suite_name,
        num_steps_wait=num_steps_wait,
        num_trials_per_task=num_trials_per_task,
        run_id_note=run_id_note,
        local_log_dir=str(local_log_dir),
        use_swanlab=use_swanlab,
        swanlab_project=swanlab_project,
        swanlab_note=swanlab_note,
        seed=seed,
        patchroot=str(patch_path),
        x=transform_defaults["x"] if x is None else x,
        y=transform_defaults["y"] if y is None else y,
        angle=transform_defaults["angle"] if angle is None else angle,
        shx=transform_defaults["shx"] if shx is None else shx,
        shy=transform_defaults["shy"] if shy is None else shy,
        cudaid=cudaid,
        resize_patch=resize_patch,
        geometry=geometry,
        colorjitter=colorjitter,
        source_attack_name=source_attack_name,
        source_attack_run_dir=source_attack_run_dir,
        source_attack_dataset=source_attack_dataset,
        source_attack_swanlab_note=source_attack_swanlab_note,
    )


def build_patch_eval_config_from_attack(
    attack_args,
    patchroot: Union[str, Path],
    task_suite_name: Optional[str] = None,
    pretrained_checkpoint: Optional[Union[str, Path]] = None,
    local_log_dir: Optional[Union[str, Path]] = None,
    run_id_note: Optional[str] = None,
    exp_name: Optional[str] = None,
    num_trials_per_task: int = 50,
    num_steps_wait: int = 10,
    x: Optional[int] = None,
    y: Optional[int] = None,
    angle: Optional[float] = None,
    shx: Optional[float] = None,
    shy: Optional[float] = None,
    cudaid: Optional[int] = None,
    use_swanlab: Optional[bool] = None,
) -> PatchLiberoEvalConfig:
    dataset_name = getattr(attack_args, "dataset", "")
    task_suite_name = task_suite_name or infer_task_suite_name(dataset_name)
    server_root = getattr(attack_args, "server", REPO_ROOT)
    patch_path = resolve_patch_path(patchroot)
    attack_run_dir = patch_path.parent.parent if patch_path.parent.name == "last" else patch_path.parent
    attack_project = getattr(attack_args, "swanlab_project", "VLA-Attack")
    attack_note = getattr(attack_args, "swanlab_note", "")
    eval_note_parts = [part for part in [attack_note, "libero_patch_eval"] if part]

    if use_swanlab is None:
        use_swanlab = str(attack_project).lower() != "false"

    return build_patch_eval_config(
        patchroot=patch_path,
        task_suite_name=task_suite_name,
        pretrained_checkpoint=pretrained_checkpoint or infer_pretrained_checkpoint(task_suite_name, server_root),
        local_log_dir=local_log_dir or attack_run_dir,
        run_id_note=run_id_note or f"{attack_run_dir.name}-patch-eval",
        exp_name=exp_name or attack_run_dir.name,
        swanlab_project=attack_project if attack_project != "false" else "VLA-Attack",
        swanlab_note=" | ".join(eval_note_parts),
        use_swanlab=use_swanlab,
        num_trials_per_task=num_trials_per_task,
        num_steps_wait=num_steps_wait,
        x=x,
        y=y,
        angle=angle,
        shx=shx,
        shy=shy,
        cudaid=cudaid if cudaid is not None else getattr(attack_args, "device", 0),
        model_family="openvla",
        center_crop=True,
        load_in_8bit=False,
        load_in_4bit=False,
        seed=getattr(attack_args, "seed", 7),
        resize_patch=getattr(attack_args, "resize_patch", False),
        geometry=getattr(attack_args, "geometry", True),
        colorjitter=False,
        source_attack_name=getattr(attack_args, "__class__", SimpleNamespace(__name__="attack")).__name__,
        source_attack_run_dir=str(attack_run_dir),
        source_attack_dataset=dataset_name,
        source_attack_swanlab_note=attack_note,
    )


def evaluate_saved_patch_from_attack(attack_args, patchroot: Union[str, Path], **kwargs) -> dict[str, Any]:
    cfg = build_patch_eval_config_from_attack(attack_args, patchroot=patchroot, **kwargs)
    return evaluate_patch_on_libero(cfg)


def _build_run_id(cfg: PatchLiberoEvalConfig) -> str:
    run_id = f"EVAL-{cfg.task_suite_name}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def _prepare_config_for_runtime(cfg: PatchLiberoEvalConfig) -> PatchLiberoEvalConfig:
    cfg = ensure_eval_config(cfg)
    cfg.patchroot = str(resolve_patch_path(cfg.patchroot))
    cfg.local_log_dir = str(Path(cfg.local_log_dir).expanduser().resolve())
    if not cfg.pretrained_checkpoint:
        cfg.pretrained_checkpoint = infer_pretrained_checkpoint(cfg.task_suite_name)
    return cfg


def _load_patch(cfg: PatchLiberoEvalConfig) -> torch.Tensor:
    patch = torch.load(cfg.patchroot, map_location="cpu")
    if not isinstance(patch, torch.Tensor):
        raise TypeError(f"Expected patch tensor at {cfg.patchroot}, got {type(patch)}")
    return patch.detach().cpu()


def _apply_patch_to_image(img: np.ndarray, patch: torch.Tensor, patch_transform: RandomPatchTransform, cfg: PatchLiberoEvalConfig):
    return patch_transform.simulation_random_patch(
        img,
        patch,
        geometry=cfg.geometry,
        colorjitter=cfg.colorjitter,
        angle=cfg.angle,
        shx=cfg.shx,
        shy=cfg.shy,
        position=(cfg.x, cfg.y),
    )


def evaluate_patch_on_libero(cfg: Union[PatchLiberoEvalConfig, argparse.Namespace, dict]) -> dict[str, Any]:
    cfg = _prepare_config_for_runtime(cfg)
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty."
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting center_crop=True because model was trained with image augmentations."
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization."

    device = f"cuda:{cfg.cudaid}" if torch.cuda.is_available() else "cpu"
    patch_transform = RandomPatchTransform("cpu", cfg.resize_patch)
    patch = _load_patch(cfg)
    set_seed_everywhere(cfg.seed)

    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg, device=device)
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in model norm_stats."
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    run_id = _build_run_id(cfg)
    local_log_dir = Path(cfg.local_log_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    local_log_filepath = local_log_dir / f"{run_id}.txt"
    print(f"Logging to local log file: {local_log_filepath}")
    log_file = open(local_log_filepath, "w", encoding="utf-8")
    swanlab_enabled = maybe_init_swanlab(
        cfg.use_swanlab,
        cfg.swanlab_project,
        run_id,
        asdict(cfg),
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    resize_size = get_image_resize_size(cfg)

    total_episodes = 0
    total_successes = 0
    task_results = []
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes = 0
        task_successes = 0
        multi_line3d = None
        max_steps = MAX_STEPS_BY_SUITE[cfg.task_suite_name]
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []
            eef_trajectory = []
            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    img = _apply_patch_to_image(img, patch, patch_transform, cfg)
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
                        device=device,
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    eef_trajectory.append(obs["robot0_eef_pos"].copy())
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
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                exp_name=cfg.exp_name,
            )

            if swanlab_enabled and replay_images:
                processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
                gif_dir = f"./rollouts/{cfg.exp_name}/{DATE}/gifs"
                Path(gif_dir).mkdir(parents=True, exist_ok=True)
                gif_path = f"{gif_dir}/{DATE_TIME}--episode={total_episodes}--success={done}--task={processed_task}.gif"
                imageio.mimsave(gif_path, replay_images, fps=10)
                libero_swanlab.log_rollout_video(
                    swanlab_enabled,
                    task_description,
                    gif_path,
                    episode_idx,
                    done,
                    step=total_episodes,
                )

            if swanlab_enabled and eef_trajectory:
                libero_swanlab.log_episode_trajectory(
                    swanlab_enabled,
                    task_description,
                    eef_trajectory,
                    episode_idx,
                    done,
                    step=total_episodes,
                )
                multi_line3d = libero_swanlab.update_multi_episode_trajectory(
                    swanlab_enabled,
                    task_description,
                    multi_line3d,
                    eef_trajectory,
                    episode_idx,
                    done,
                    step=episode_idx,
                )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes else 0.0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {total_success_rate}")
        log_file.write(f"Current task success rate: {task_success_rate}\n")
        log_file.write(f"Current total success rate: {total_success_rate}\n")
        log_file.flush()
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "num_episodes": task_episodes,
                "num_successes": task_successes,
                "success_rate": task_success_rate,
            }
        )
        if swanlab_enabled and task_results:
            libero_swanlab.log_task_results(swanlab_enabled, task_results, step=task_id)

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    summary = {
        "run_id": run_id,
        "task_suite_name": cfg.task_suite_name,
        "patchroot": cfg.patchroot,
        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
        "local_log_filepath": str(local_log_filepath),
        "local_log_dir": str(local_log_dir),
        "num_episodes_total": total_episodes,
        "num_successes_total": total_successes,
        "success_rate_total": total_success_rate,
        "task_results": task_results,
        "patch_transform": {
            "x": cfg.x,
            "y": cfg.y,
            "angle": cfg.angle,
            "shx": cfg.shx,
            "shy": cfg.shy,
            "geometry": cfg.geometry,
            "colorjitter": cfg.colorjitter,
        },
    }
    summary_path = local_log_dir / f"{run_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    if swanlab_enabled and total_episodes > 0:
        libero_swanlab.log_final_summary(swanlab_enabled, total_episodes, total_successes)
    with open(local_log_dir / f"{cfg.task_suite_name}.txt", "a", encoding="utf-8") as file:
        file.write(
            f"success_rate/total:{total_success_rate}, "
            f"num_episodes/total:{total_episodes} "
            f"position_info:{cfg.angle}_{cfg.shx}_{cfg.shy}_{cfg.x}_{cfg.y}\n"
        )

    log_file.close()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-based LIBERO evaluation.")
    parser.add_argument("--model_family", type=str, default="openvla")
    parser.add_argument("--exp_name", type=str, default="patch_eval")
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--center_crop", type=str2bool, default=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_object")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=100)
    parser.add_argument("--run_id_note", type=str, default="test_libero_object")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs")
    parser.add_argument("--use_swanlab", type=str2bool, default=True)
    parser.add_argument("--swanlab_project", type=str, default="VLA-Attack")
    parser.add_argument("--swanlab_note", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patchroot", type=str, required=True)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument("--shx", type=float, default=None)
    parser.add_argument("--shy", type=float, default=None)
    parser.add_argument("--cudaid", type=int, default=0)
    parser.add_argument("--resize_patch", type=str2bool, default=False)
    parser.add_argument("--geometry", type=str2bool, default=True)
    parser.add_argument("--colorjitter", type=str2bool, default=False)
    args = parser.parse_args()
    return build_patch_eval_config(
        patchroot=args.patchroot,
        task_suite_name=args.task_suite_name,
        pretrained_checkpoint=args.pretrained_checkpoint or None,
        local_log_dir=args.local_log_dir,
        run_id_note=args.run_id_note,
        exp_name=args.exp_name,
        swanlab_project=args.swanlab_project,
        swanlab_note=args.swanlab_note,
        use_swanlab=args.use_swanlab,
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        x=args.x,
        y=args.y,
        angle=args.angle,
        shx=args.shx,
        shy=args.shy,
        cudaid=args.cudaid,
        model_family=args.model_family,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        resize_patch=args.resize_patch,
        geometry=args.geometry,
        colorjitter=args.colorjitter,
    )


if __name__ == "__main__":
    evaluate_patch_on_libero(parse_args())
