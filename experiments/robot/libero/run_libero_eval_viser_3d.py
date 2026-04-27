"""
run_libero_eval_viser_3d.py

Runs an OpenVLA model in a headless LIBERO simulation environment and streams a 3D state view to Viser.

Usage:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0,1 python experiments/robot/libero/run_libero_eval_viser_3d.py \
        --model_family openvla \
        --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 1
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval_viser_3d.py \
  --model_family openvla \
  --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
"""

import html
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import draccus
import mujoco
import numpy as np
import tqdm
from libero.libero import benchmark
from scipy.spatial.transform import Rotation

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

SKIP_FIXTURE_NAMES = {"main_table", "floor", "countertop", "coffee_table"}
COLOR_EEF = (72, 201, 176)
COLOR_OBJ = (88, 166, 255)
COLOR_INTEREST = (255, 174, 66)
COLOR_FIXTURE = (176, 176, 176)
COLOR_SITE = (190, 120, 255)
COLOR_TRAIL = (72, 201, 176)
COLOR_LINK = (255, 174, 66)
COLOR_OBJECT_BOX = (88, 166, 255)
COLOR_INTEREST_BOX = (255, 174, 66)
COLOR_FIXTURE_BOX = (176, 176, 176)
COLOR_WORKSPACE_BOX = (120, 210, 120)
COLOR_ROBOT_COLLISION = (244, 122, 96)
COLOR_OBJECT_COLLISION = (88, 166, 255)
COLOR_INTEREST_COLLISION = (255, 174, 66)
COLOR_FIXTURE_COLLISION = (150, 150, 150)


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
    snapshot: Optional["SceneSnapshot"],
) -> str:
    success_rate = (total_successes / total_episodes * 100.0) if total_episodes > 0 else 0.0
    interest_list = ", ".join(snapshot.obj_of_interest) if snapshot is not None else "N/A"
    movable_count = len(snapshot.objects) if snapshot is not None else 0
    fixture_count = len(snapshot.fixtures) if snapshot is not None else 0
    site_count = len(snapshot.sites) if snapshot is not None else 0
    return html_rows(
        [
            ("Task ID", str(task_id)),
            ("Task", task_description),
            ("Episode", str(episode_idx)),
            ("Step", str(step_idx)),
            ("Done", str(done)),
            ("Completed Episodes", str(total_episodes)),
            ("Successes", f"{total_successes} ({success_rate:.1f}%)"),
            ("Objects of Interest", interest_list or "N/A"),
            ("Movable Objects", str(movable_count)),
            ("Fixtures", str(fixture_count)),
            ("Sites", str(site_count)),
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
    return np.ascontiguousarray(obs["agentview_image"][::-1])


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def normalize_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat_wxyz)
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def site_matrix_to_wxyz(site_mat: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_matrix(np.asarray(site_mat, dtype=np.float64).reshape(3, 3))
    return normalize_quaternion(xyzw_to_wxyz(rotation.as_quat()))


def trail_to_segments(points: list[np.ndarray]) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    points_arr = np.asarray(points, dtype=np.float32)
    return np.stack([points_arr[:-1], points_arr[1:]], axis=1)


def get_workspace_focus_point(snapshot: Optional["SceneSnapshot"], fallback: Optional[np.ndarray] = None) -> np.ndarray:
    if snapshot is not None and snapshot.workspace_box is not None:
        lower, upper = snapshot.workspace_box
        focus = (np.asarray(lower, dtype=np.float64) + np.asarray(upper, dtype=np.float64)) / 2.0
        focus[2] = float(upper[2])
        return focus
    if snapshot is not None:
        return snapshot.eef.position.copy()
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float64).copy()
    return np.array([0.35, 0.0, 0.2], dtype=np.float64)


def aabb_to_segments(bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lower, upper = bounds
    lower = np.asarray(lower, dtype=np.float32).reshape(3)
    upper = np.asarray(upper, dtype=np.float32).reshape(3)
    corners = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=np.float32,
    )
    edge_indices = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )
    return corners[edge_indices]


def sanitize_path_component(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(name))


def get_domain_env(env: Any) -> Any:
    return env.env if hasattr(env, "env") else env


def iter_descendant_body_ids(model: Any, root_body_id: int) -> list[int]:
    body_parentid = np.asarray(model.body_parentid, dtype=np.int32)
    descendants = []
    stack = [root_body_id]
    while stack:
        body_id = stack.pop()
        descendants.append(body_id)
        child_ids = np.where(body_parentid == body_id)[0]
        for child_id in child_ids:
            if child_id != body_id:
                stack.append(int(child_id))
    return descendants


def iter_body_geom_ids(model: Any, root_body_id: int, *, include_descendants: bool) -> list[int]:
    body_ids = iter_descendant_body_ids(model, root_body_id) if include_descendants else [root_body_id]
    geom_ids = []
    for body_id in body_ids:
        geom_start = int(model.body_geomadr[body_id])
        geom_count = int(model.body_geomnum[body_id])
        geom_ids.extend(range(geom_start, geom_start + geom_count))
    return geom_ids


def safe_geom_ids_from_names(model: Any, geom_names: list[str] | tuple[str, ...]) -> list[int]:
    geom_ids = []
    seen = set()
    for geom_name in geom_names:
        if geom_name is None:
            continue
        try:
            geom_id = int(model.geom_name2id(geom_name))
        except Exception:
            continue
        if geom_id >= 0 and geom_id not in seen:
            geom_ids.append(geom_id)
            seen.add(geom_id)
    return geom_ids


def get_geom_name(model: Any, geom_idx: int) -> str:
    try:
        geom_name = model.geom_id2name(int(geom_idx))
    except Exception:
        geom_name = None
    if geom_name:
        return str(geom_name)
    return f"geom_{int(geom_idx):04d}"


def is_supported_collision_geom(model: Any, geom_idx: int) -> bool:
    geom_type = int(model.geom_type[geom_idx])
    geom_group = int(model.geom_group[geom_idx])
    return geom_group != 1 and geom_type != int(mujoco.mjtGeom.mjGEOM_NONE)


def get_entity_collision_geom_ids(model: Any, entity_model: Any, body_id: int) -> list[int]:
    geom_names = list(getattr(entity_model, "contact_geoms", []) or [])
    geom_ids = safe_geom_ids_from_names(model, geom_names)
    if len(geom_ids) > 0:
        return sorted(set(geom_ids))
    return sorted(
        set(
            geom_idx
            for geom_idx in iter_body_geom_ids(model, body_id, include_descendants=True)
            if is_supported_collision_geom(model, geom_idx)
        )
    )


def get_robot_collision_geom_ids(domain_env: Any) -> list[int]:
    model = domain_env.sim.model
    geom_names = []
    body_prefixes = set()
    for robot in getattr(domain_env, "robots", []):
        robot_model = getattr(robot, "robot_model", None)
        if robot_model is not None:
            geom_names.extend(list(getattr(robot_model, "contact_geoms", []) or []))
            naming_prefix = getattr(robot_model, "naming_prefix", None)
            if naming_prefix:
                body_prefixes.add(str(naming_prefix))
            for gripper_model in getattr(robot_model, "grippers", {}).values():
                geom_names.extend(list(getattr(gripper_model, "contact_geoms", []) or []))
                gripper_prefix = getattr(gripper_model, "naming_prefix", None)
                if gripper_prefix:
                    body_prefixes.add(str(gripper_prefix))
        gripper = getattr(robot, "gripper", None)
        if isinstance(gripper, dict):
            gripper_values = list(gripper.values())
        elif gripper is None:
            gripper_values = []
        else:
            gripper_values = [gripper]
        for gripper_model in gripper_values:
            geom_names.extend(list(getattr(gripper_model, "contact_geoms", []) or []))
            gripper_prefix = getattr(gripper_model, "naming_prefix", None)
            if gripper_prefix:
                body_prefixes.add(str(gripper_prefix))

    geom_ids = safe_geom_ids_from_names(model, geom_names)
    if len(geom_ids) > 0:
        return geom_ids

    if len(body_prefixes) == 0:
        body_prefixes = {"robot0_", "gripper0_", "mount0_"}

    geom_ids = []
    for geom_idx in range(int(model.ngeom)):
        body_name = model.body_id2name(int(model.geom_bodyid[geom_idx])) or ""
        if any(str(body_name).startswith(prefix) for prefix in body_prefixes) and is_supported_collision_geom(
            model, geom_idx
        ):
            geom_ids.append(geom_idx)
    return sorted(set(geom_ids))


def compute_body_bounds(sim: Any, model: Any, body_id: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    geom_positions = []
    geom_radii = []
    for descendant_body_id in iter_descendant_body_ids(model, body_id):
        geom_start = int(model.body_geomadr[descendant_body_id])
        geom_count = int(model.body_geomnum[descendant_body_id])
        for geom_idx in range(geom_start, geom_start + geom_count):
            geom_positions.append(np.asarray(sim.data.geom_xpos[geom_idx], dtype=np.float64).copy())
            geom_radii.append(float(model.geom_rbound[geom_idx]))

    if len(geom_positions) == 0:
        return None

    lowers = [position - radius for position, radius in zip(geom_positions, geom_radii)]
    uppers = [position + radius for position, radius in zip(geom_positions, geom_radii)]
    return np.min(np.stack(lowers, axis=0), axis=0), np.max(np.stack(uppers, axis=0), axis=0)


def get_workspace_box(domain_env: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    size_attrs = [
        "table_full_size",
        "kitchen_table_full_size",
        "study_table_full_size",
        "living_room_table_full_size",
        "coffee_table_full_size",
    ]
    for attr in size_attrs:
        if hasattr(domain_env, attr):
            full_size = np.asarray(getattr(domain_env, attr), dtype=np.float64).reshape(3)
            center = np.asarray(domain_env.workspace_offset, dtype=np.float64).reshape(3).copy()
            center[2] -= full_size[2] / 2.0
            half_size = full_size / 2.0
            return center - half_size, center + half_size
    return None


@dataclass
class CollisionGeomState:
    pose: "PoseState"
    geom_type: int
    size: np.ndarray
    rbound: float
    color: tuple[int, int, int]


@dataclass
class PoseState:
    position: np.ndarray
    wxyz: np.ndarray


@dataclass
class SceneSnapshot:
    eef: PoseState
    objects: dict[str, PoseState]
    fixtures: dict[str, PoseState]
    sites: dict[str, PoseState]
    obj_of_interest: tuple[str, ...]
    robot_collision_geoms: dict[str, CollisionGeomState]
    object_collision_geoms: dict[str, CollisionGeomState]
    fixture_collision_geoms: dict[str, CollisionGeomState]
    object_boxes: dict[str, tuple[np.ndarray, np.ndarray]]
    fixture_boxes: dict[str, tuple[np.ndarray, np.ndarray]]
    workspace_box: Optional[tuple[np.ndarray, np.ndarray]]


@dataclass
class PoseNodeHandles:
    frame: Any
    marker: Any
    label: Any


@dataclass
class CollisionGeomHandles:
    frame: Any
    primary: Any
    extras: tuple[Any, ...]


@dataclass
class Scene3DHandles:
    root: Any
    eef: PoseNodeHandles
    objects: dict[str, PoseNodeHandles]
    fixtures: dict[str, PoseNodeHandles]
    sites: dict[str, PoseNodeHandles]
    robot_collision_geoms: dict[str, CollisionGeomHandles]
    object_collision_geoms: dict[str, CollisionGeomHandles]
    fixture_collision_geoms: dict[str, CollisionGeomHandles]
    eef_trail: Any
    target_links: Any
    object_boxes: dict[str, Any]
    fixture_boxes: dict[str, Any]
    workspace_box: Any


@dataclass(eq=True, frozen=True)
class SceneRenderSettings:
    show_labels: bool
    show_markers: bool
    show_eef_frame: bool
    show_object_frames: bool
    show_fixture_frames: bool
    show_site_frames: bool
    show_robot_collision: bool
    show_object_collision: bool
    show_fixture_collision: bool
    show_trail: bool
    show_interest_links: bool
    show_object_boxes: bool
    show_fixture_boxes: bool
    show_workspace_box: bool
    stream_env_frame: bool
    stream_model_input: bool
    collision_opacity: float
    trail_visible_points: int


@dataclass
class GuiControls:
    pause_checkbox: Any
    stop_button: Any
    show_labels_checkbox: Any
    show_markers_checkbox: Any
    show_eef_frame_checkbox: Any
    show_object_frames_checkbox: Any
    show_fixture_frames_checkbox: Any
    show_site_frames_checkbox: Any
    show_robot_collision_checkbox: Any
    show_object_collision_checkbox: Any
    show_fixture_collision_checkbox: Any
    show_trail_checkbox: Any
    show_interest_links_checkbox: Any
    show_object_boxes_checkbox: Any
    show_fixture_boxes_checkbox: Any
    show_workspace_box_checkbox: Any
    stream_env_frame_checkbox: Any
    stream_model_input_checkbox: Any
    trail_visible_points_slider: Any
    collision_opacity_slider: Any
    focus_workspace_button: Any


def extract_snapshot(env: Any, obs: dict, show_fixtures: bool, show_sites: bool) -> SceneSnapshot:
    domain_env = get_domain_env(env)
    sim = domain_env.sim
    model = domain_env.sim.model

    eef = PoseState(
        position=np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
        wxyz=normalize_quaternion(xyzw_to_wxyz(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))),
    )

    obj_of_interest = tuple(domain_env.obj_of_interest)

    robot_collision_geoms: dict[str, CollisionGeomState] = {}
    for geom_idx in get_robot_collision_geom_ids(domain_env):
        geom_key = f"{int(geom_idx):04d}_{sanitize_path_component(get_geom_name(model, geom_idx))}"
        robot_collision_geoms[geom_key] = CollisionGeomState(
            pose=PoseState(
                position=np.asarray(sim.data.geom_xpos[geom_idx], dtype=np.float64).copy(),
                wxyz=site_matrix_to_wxyz(np.asarray(sim.data.geom_xmat[geom_idx], dtype=np.float64)),
            ),
            geom_type=int(model.geom_type[geom_idx]),
            size=np.asarray(model.geom_size[geom_idx], dtype=np.float64).copy(),
            rbound=float(model.geom_rbound[geom_idx]),
            color=COLOR_ROBOT_COLLISION,
        )

    objects: dict[str, PoseState] = {}
    object_collision_geoms: dict[str, CollisionGeomState] = {}
    object_boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for object_name in sorted(domain_env.objects_dict.keys()):
        if object_name not in domain_env.obj_body_id:
            continue
        body_id = domain_env.obj_body_id[object_name]
        objects[object_name] = PoseState(
            position=np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy(),
            wxyz=normalize_quaternion(np.asarray(sim.data.body_xquat[body_id], dtype=np.float64).copy()),
        )
        bounds = compute_body_bounds(sim, model, body_id)
        if bounds is not None:
            object_boxes[object_name] = bounds
        color = COLOR_INTEREST_COLLISION if object_name in obj_of_interest else COLOR_OBJECT_COLLISION
        for geom_idx in get_entity_collision_geom_ids(model, domain_env.objects_dict[object_name], body_id):
            geom_key = (
                f"{sanitize_path_component(object_name)}__"
                f"{int(geom_idx):04d}_{sanitize_path_component(get_geom_name(model, geom_idx))}"
            )
            object_collision_geoms[geom_key] = CollisionGeomState(
                pose=PoseState(
                    position=np.asarray(sim.data.geom_xpos[geom_idx], dtype=np.float64).copy(),
                    wxyz=site_matrix_to_wxyz(np.asarray(sim.data.geom_xmat[geom_idx], dtype=np.float64)),
                ),
                geom_type=int(model.geom_type[geom_idx]),
                size=np.asarray(model.geom_size[geom_idx], dtype=np.float64).copy(),
                rbound=float(model.geom_rbound[geom_idx]),
                color=color,
            )

    fixtures: dict[str, PoseState] = {}
    fixture_collision_geoms: dict[str, CollisionGeomState] = {}
    fixture_boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if show_fixtures:
        for fixture_name in sorted(domain_env.fixtures_dict.keys()):
            if fixture_name in SKIP_FIXTURE_NAMES or fixture_name not in domain_env.obj_body_id:
                continue
            body_id = domain_env.obj_body_id[fixture_name]
            fixtures[fixture_name] = PoseState(
                position=np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy(),
                wxyz=normalize_quaternion(np.asarray(sim.data.body_xquat[body_id], dtype=np.float64).copy()),
            )
            bounds = compute_body_bounds(sim, model, body_id)
            if bounds is not None:
                fixture_boxes[fixture_name] = bounds
            for geom_idx in get_entity_collision_geom_ids(model, domain_env.fixtures_dict[fixture_name], body_id):
                geom_key = (
                    f"{sanitize_path_component(fixture_name)}__"
                    f"{int(geom_idx):04d}_{sanitize_path_component(get_geom_name(model, geom_idx))}"
                )
                fixture_collision_geoms[geom_key] = CollisionGeomState(
                    pose=PoseState(
                        position=np.asarray(sim.data.geom_xpos[geom_idx], dtype=np.float64).copy(),
                        wxyz=site_matrix_to_wxyz(np.asarray(sim.data.geom_xmat[geom_idx], dtype=np.float64)),
                    ),
                    geom_type=int(model.geom_type[geom_idx]),
                    size=np.asarray(model.geom_size[geom_idx], dtype=np.float64).copy(),
                    rbound=float(model.geom_rbound[geom_idx]),
                    color=COLOR_FIXTURE_COLLISION,
                )

    sites: dict[str, PoseState] = {}
    if show_sites:
        for site_name in sorted(domain_env.object_sites_dict.keys()):
            try:
                sites[site_name] = PoseState(
                    position=np.asarray(sim.data.get_site_xpos(site_name), dtype=np.float64).copy(),
                    wxyz=site_matrix_to_wxyz(np.asarray(sim.data.get_site_xmat(site_name), dtype=np.float64)),
                )
            except Exception:
                continue

    return SceneSnapshot(
        eef=eef,
        objects=objects,
        fixtures=fixtures,
        sites=sites,
        obj_of_interest=obj_of_interest,
        robot_collision_geoms=robot_collision_geoms,
        object_collision_geoms=object_collision_geoms,
        fixture_collision_geoms=fixture_collision_geoms,
        object_boxes=object_boxes,
        fixture_boxes=fixture_boxes,
        workspace_box=get_workspace_box(domain_env),
    )


def add_pose_node(
    scene: Any,
    path: str,
    label_text: str,
    pose: PoseState,
    color: tuple[int, int, int],
    *,
    axes_length: float,
    axes_radius: float,
    marker_radius: float,
    label_offset: float,
) -> PoseNodeHandles:
    frame = scene.add_frame(
        f"{path}/frame",
        axes_length=axes_length,
        axes_radius=axes_radius,
        origin_color=color,
        position=pose.position,
        wxyz=pose.wxyz,
    )
    marker = scene.add_icosphere(
        f"{path}/frame/marker",
        radius=marker_radius,
        color=color,
        opacity=0.9,
        position=(0.0, 0.0, 0.0),
    )
    label = scene.add_label(
        f"{path}/frame/label",
        label_text,
        position=(0.0, 0.0, label_offset),
        anchor="bottom-center",
        font_size_mode="screen",
        font_screen_scale=1.0,
        depth_test=False,
    )
    return PoseNodeHandles(frame=frame, marker=marker, label=label)


def update_pose_node(node: PoseNodeHandles, pose: PoseState) -> None:
    node.frame.position = pose.position
    node.frame.wxyz = pose.wxyz


def set_pose_node_visibility(node: PoseNodeHandles, show_frame: bool, show_marker: bool, show_label: bool) -> None:
    node.frame.visible = show_frame
    node.marker.visible = show_marker
    node.label.visible = show_label


def set_pose_collection_visibility(
    nodes: dict[str, PoseNodeHandles],
    *,
    show_frame: bool,
    show_marker: bool,
    show_label: bool,
) -> None:
    for node in nodes.values():
        set_pose_node_visibility(node, show_frame=show_frame, show_marker=show_marker, show_label=show_label)


def set_line_handle_visibility(handle: Any, visible: bool) -> None:
    if handle is not None:
        handle.visible = visible


def add_collision_geom(
    scene: Any,
    path: str,
    state: CollisionGeomState,
    *,
    opacity: float,
) -> CollisionGeomHandles:
    frame = scene.add_frame(
        path,
        show_axes=False,
        position=state.pose.position,
        wxyz=state.pose.wxyz,
    )
    color = state.color
    geom_type = int(state.geom_type)
    size = np.asarray(state.size, dtype=np.float64).reshape(3)

    primary = None
    extras = []
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        primary = scene.add_box(
            f"{path}/shape",
            color=color,
            dimensions=(2.0 * size[0], 2.0 * size[1], 2.0 * size[2]),
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        primary = scene.add_icosphere(
            f"{path}/shape",
            radius=float(size[0]),
            color=color,
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        primary = scene.add_icosphere(
            f"{path}/shape",
            radius=1.0,
            scale=(float(size[0]), float(size[1]), float(size[2])),
            color=color,
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        primary = scene.add_cylinder(
            f"{path}/shape",
            radius=float(size[0]),
            height=max(2.0 * float(size[1]), 1e-4),
            color=color,
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius = float(size[0])
        half_length = float(size[1])
        primary = scene.add_cylinder(
            f"{path}/shape",
            radius=radius,
            height=max(2.0 * half_length, 1e-4),
            color=color,
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
        )
        extras.append(
            scene.add_icosphere(
                f"{path}/cap_pos",
                radius=radius,
                color=color,
                opacity=opacity,
                position=(0.0, 0.0, half_length),
            )
        )
        extras.append(
            scene.add_icosphere(
                f"{path}/cap_neg",
                radius=radius,
                color=color,
                opacity=opacity,
                position=(0.0, 0.0, -half_length),
            )
        )
    else:
        fallback_dim = max(2.0 * float(state.rbound), 1e-4)
        primary = scene.add_box(
            f"{path}/shape",
            color=color,
            dimensions=(fallback_dim, fallback_dim, fallback_dim),
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
        )

    return CollisionGeomHandles(frame=frame, primary=primary, extras=tuple(extras))


def update_collision_geom_handle(handle: CollisionGeomHandles, state: CollisionGeomState) -> None:
    handle.frame.position = state.pose.position
    handle.frame.wxyz = state.pose.wxyz


def set_collision_geom_visibility(handle: CollisionGeomHandles, visible: bool) -> None:
    handle.frame.visible = visible


def set_collision_geom_opacity(handle: CollisionGeomHandles, opacity: float) -> None:
    if handle.primary is not None and hasattr(handle.primary, "opacity"):
        handle.primary.opacity = opacity
    for extra in handle.extras:
        if hasattr(extra, "opacity"):
            extra.opacity = opacity


def set_collision_geom_collection_visibility(handles: dict[str, CollisionGeomHandles], visible: bool) -> None:
    for handle in handles.values():
        set_collision_geom_visibility(handle, visible)


def set_collision_geom_collection_opacity(handles: dict[str, CollisionGeomHandles], opacity: float) -> None:
    for handle in handles.values():
        set_collision_geom_opacity(handle, opacity)


def apply_scene_visibility(
    handles: Scene3DHandles,
    *,
    show_eef_frame: bool,
    show_object_frames: bool,
    show_fixture_frames: bool,
    show_site_frames: bool,
    show_robot_collision: bool,
    show_object_collision: bool,
    show_fixture_collision: bool,
    show_markers: bool,
    show_labels: bool,
    show_trail: bool,
    show_interest_links: bool,
    show_object_boxes: bool,
    show_fixture_boxes: bool,
    show_workspace_box: bool,
) -> None:
    set_pose_node_visibility(handles.eef, show_frame=show_eef_frame, show_marker=show_markers, show_label=show_labels)
    set_pose_collection_visibility(
        handles.objects,
        show_frame=show_object_frames,
        show_marker=show_markers,
        show_label=show_labels,
    )
    set_pose_collection_visibility(
        handles.fixtures,
        show_frame=show_fixture_frames,
        show_marker=show_markers,
        show_label=show_labels,
    )
    set_pose_collection_visibility(
        handles.sites,
        show_frame=show_site_frames,
        show_marker=show_markers,
        show_label=show_labels,
    )
    set_collision_geom_collection_visibility(handles.robot_collision_geoms, show_robot_collision)
    set_collision_geom_collection_visibility(handles.object_collision_geoms, show_object_collision)
    set_collision_geom_collection_visibility(handles.fixture_collision_geoms, show_fixture_collision)
    handles.eef_trail.visible = show_trail
    handles.target_links.visible = show_interest_links
    for handle in handles.object_boxes.values():
        set_line_handle_visibility(handle, show_object_boxes)
    for handle in handles.fixture_boxes.values():
        set_line_handle_visibility(handle, show_fixture_boxes)
    set_line_handle_visibility(handles.workspace_box, show_workspace_box)


def build_scene_handles(server: Any, snapshot: SceneSnapshot, cfg: "GenerateConfig") -> Scene3DHandles:
    root = server.scene.add_frame("/task_scene", show_axes=False)
    server.scene.add_grid(
        "/task_scene/grid",
        width=2.0,
        height=2.0,
        plane="xy",
        cell_size=0.1,
        section_size=0.5,
        cell_color=(220, 220, 220),
        section_color=(180, 180, 180),
        plane_color=(245, 245, 245),
        plane_opacity=0.12,
        shadow_opacity=0.0,
        position=(0.0, 0.0, 0.0),
    )

    eef = add_pose_node(
        server.scene,
        "/task_scene/eef",
        "eef",
        snapshot.eef,
        COLOR_EEF,
        axes_length=cfg.eef_axes_length,
        axes_radius=cfg.eef_axes_radius,
        marker_radius=cfg.eef_marker_radius,
        label_offset=cfg.label_offset,
    )

    objects: dict[str, PoseNodeHandles] = {}
    for object_name, pose in snapshot.objects.items():
        color = COLOR_INTEREST if object_name in snapshot.obj_of_interest else COLOR_OBJ
        objects[object_name] = add_pose_node(
            server.scene,
            f"/task_scene/objects/{object_name}",
            object_name,
            pose,
            color,
            axes_length=cfg.object_axes_length,
            axes_radius=cfg.object_axes_radius,
            marker_radius=cfg.object_marker_radius,
            label_offset=cfg.label_offset,
        )

    fixtures: dict[str, PoseNodeHandles] = {}
    for fixture_name, pose in snapshot.fixtures.items():
        fixtures[fixture_name] = add_pose_node(
            server.scene,
            f"/task_scene/fixtures/{fixture_name}",
            fixture_name,
            pose,
            COLOR_FIXTURE,
            axes_length=cfg.fixture_axes_length,
            axes_radius=cfg.fixture_axes_radius,
            marker_radius=cfg.fixture_marker_radius,
            label_offset=cfg.label_offset,
        )

    sites: dict[str, PoseNodeHandles] = {}
    for site_name, pose in snapshot.sites.items():
        sites[site_name] = add_pose_node(
            server.scene,
            f"/task_scene/sites/{site_name}",
            site_name,
            pose,
            COLOR_SITE,
            axes_length=cfg.site_axes_length,
            axes_radius=cfg.site_axes_radius,
            marker_radius=cfg.site_marker_radius,
            label_offset=cfg.label_offset * 0.7,
        )

    robot_collision_geoms: dict[str, CollisionGeomHandles] = {}
    for geom_key, state in snapshot.robot_collision_geoms.items():
        robot_collision_geoms[geom_key] = add_collision_geom(
            server.scene,
            f"/task_scene/collision/robot/{geom_key}",
            state,
            opacity=cfg.collision_opacity,
        )

    object_collision_geoms: dict[str, CollisionGeomHandles] = {}
    for geom_key, state in snapshot.object_collision_geoms.items():
        object_collision_geoms[geom_key] = add_collision_geom(
            server.scene,
            f"/task_scene/collision/objects/{geom_key}",
            state,
            opacity=cfg.collision_opacity,
        )

    fixture_collision_geoms: dict[str, CollisionGeomHandles] = {}
    for geom_key, state in snapshot.fixture_collision_geoms.items():
        fixture_collision_geoms[geom_key] = add_collision_geom(
            server.scene,
            f"/task_scene/collision/fixtures/{geom_key}",
            state,
            opacity=cfg.collision_opacity,
        )

    eef_trail = server.scene.add_line_segments(
        "/task_scene/debug/eef_trail",
        points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=COLOR_TRAIL,
        line_width=2.5,
    )
    target_links = server.scene.add_line_segments(
        "/task_scene/debug/interest_links",
        points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=COLOR_LINK,
        line_width=2.0,
    )
    object_boxes: dict[str, Any] = {}
    for object_name, bounds in snapshot.object_boxes.items():
        color = COLOR_INTEREST_BOX if object_name in snapshot.obj_of_interest else COLOR_OBJECT_BOX
        object_boxes[object_name] = server.scene.add_line_segments(
            f"/task_scene/debug/object_boxes/{object_name}",
            points=aabb_to_segments(bounds),
            colors=color,
            line_width=1.5,
        )
    fixture_boxes: dict[str, Any] = {}
    for fixture_name, bounds in snapshot.fixture_boxes.items():
        fixture_boxes[fixture_name] = server.scene.add_line_segments(
            f"/task_scene/debug/fixture_boxes/{fixture_name}",
            points=aabb_to_segments(bounds),
            colors=COLOR_FIXTURE_BOX,
            line_width=1.0,
        )
    workspace_box = None
    if snapshot.workspace_box is not None:
        workspace_box = server.scene.add_line_segments(
            "/task_scene/debug/workspace_box",
            points=aabb_to_segments(snapshot.workspace_box),
            colors=COLOR_WORKSPACE_BOX,
            line_width=2.0,
        )
    return Scene3DHandles(
        root=root,
        eef=eef,
        objects=objects,
        fixtures=fixtures,
        sites=sites,
        robot_collision_geoms=robot_collision_geoms,
        object_collision_geoms=object_collision_geoms,
        fixture_collision_geoms=fixture_collision_geoms,
        eef_trail=eef_trail,
        target_links=target_links,
        object_boxes=object_boxes,
        fixture_boxes=fixture_boxes,
        workspace_box=workspace_box,
    )


def update_scene_handles(
    handles: Scene3DHandles,
    snapshot: SceneSnapshot,
    eef_trail_points: deque[np.ndarray],
    trail_visible_points: int,
) -> None:
    update_pose_node(handles.eef, snapshot.eef)

    for object_name, pose in snapshot.objects.items():
        node = handles.objects.get(object_name)
        if node is not None:
            update_pose_node(node, pose)

    for fixture_name, pose in snapshot.fixtures.items():
        node = handles.fixtures.get(fixture_name)
        if node is not None:
            update_pose_node(node, pose)

    for site_name, pose in snapshot.sites.items():
        node = handles.sites.get(site_name)
        if node is not None:
            update_pose_node(node, pose)

    for geom_key, state in snapshot.robot_collision_geoms.items():
        handle = handles.robot_collision_geoms.get(geom_key)
        if handle is not None:
            update_collision_geom_handle(handle, state)

    for geom_key, state in snapshot.object_collision_geoms.items():
        handle = handles.object_collision_geoms.get(geom_key)
        if handle is not None:
            update_collision_geom_handle(handle, state)

    for geom_key, state in snapshot.fixture_collision_geoms.items():
        handle = handles.fixture_collision_geoms.get(geom_key)
        if handle is not None:
            update_collision_geom_handle(handle, state)

    eef_trail_points.append(snapshot.eef.position.copy())
    if trail_visible_points <= 1:
        handles.eef_trail.points = np.zeros((0, 2, 3), dtype=np.float32)
    else:
        visible_points = list(eef_trail_points)[-trail_visible_points:]
        handles.eef_trail.points = trail_to_segments(visible_points)

    interest_segments = []
    for object_name in snapshot.obj_of_interest:
        pose = snapshot.objects.get(object_name)
        if pose is None:
            pose = snapshot.fixtures.get(object_name)
        if pose is None:
            pose = snapshot.sites.get(object_name)
        if pose is None:
            continue
        interest_segments.append([snapshot.eef.position, pose.position])
    handles.target_links.points = (
        np.asarray(interest_segments, dtype=np.float32).reshape(-1, 2, 3)
        if len(interest_segments) > 0
        else np.zeros((0, 2, 3), dtype=np.float32)
    )
    for object_name, bounds in snapshot.object_boxes.items():
        handle = handles.object_boxes.get(object_name)
        if handle is not None:
            handle.points = aabb_to_segments(bounds)
    for fixture_name, bounds in snapshot.fixture_boxes.items():
        handle = handles.fixture_boxes.get(fixture_name)
        if handle is not None:
            handle.points = aabb_to_segments(bounds)
    if handles.workspace_box is not None and snapshot.workspace_box is not None:
        handles.workspace_box.points = aabb_to_segments(snapshot.workspace_box)


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
    stream_env_frame: bool = True
    stream_model_input: bool = True
    control_width: str = "large"

    show_fixtures: bool = True
    show_sites: bool = False
    show_labels: bool = True
    show_markers: bool = True
    show_eef_frame: bool = True
    show_object_frames: bool = True
    show_fixture_frames: bool = True
    show_site_frames: bool = True
    show_trail: bool = True
    show_interest_links: bool = True
    show_robot_collision: bool = True
    show_object_collision: bool = True
    show_fixture_collision: bool = True
    show_object_boxes: bool = False
    show_fixture_boxes: bool = False
    show_workspace_box: bool = True
    auto_focus_workspace: bool = True
    collision_opacity: float = 0.32
    reset_trail_each_episode: bool = True
    max_trail_points: int = 128
    trail_visible_points: int = 128
    camera_focus_height_offset: float = 0.03
    eef_axes_length: float = 0.10
    eef_axes_radius: float = 0.005
    eef_marker_radius: float = 0.012
    object_axes_length: float = 0.08
    object_axes_radius: float = 0.004
    object_marker_radius: float = 0.010
    fixture_axes_length: float = 0.06
    fixture_axes_radius: float = 0.003
    fixture_marker_radius: float = 0.008
    site_axes_length: float = 0.04
    site_axes_radius: float = 0.002
    site_marker_radius: float = 0.006
    label_offset: float = 0.035

    run_id_note: Optional[str] = "viser-3d"
    local_log_dir: str = "./experiments/logs"
    use_swanlab: bool = False
    swanlab_project: str = "VLA-Attack"
    seed: int = 7
    # fmt: on


@draccus.wrap()
def eval_libero_viser_3d(cfg: GenerateConfig) -> None:
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "`viser` is required for run_libero_eval_viser_3d.py. Install it with `pip install viser`."
        ) from exc

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

    run_id = f"EVAL-VISER-3D-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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
    server.gui.set_panel_label("OpenVLA LIBERO 3D Monitor")
    server.scene.world_axes.visible = True

    blank_frame = np.zeros((cfg.resolution, cfg.resolution, 3), dtype=np.uint8)
    with server.gui.add_folder("Frames"):
        env_image_handle = server.gui.add_image(
            blank_frame,
            label="Environment Frame",
            format="jpeg",
            jpeg_quality=cfg.jpeg_quality,
            visible=cfg.stream_env_frame,
        )
        model_image_handle = server.gui.add_image(
            blank_frame,
            label="Model Input Frame",
            format="jpeg",
            jpeg_quality=cfg.jpeg_quality,
            visible=cfg.stream_model_input,
        )
        stream_env_frame_checkbox = server.gui.add_checkbox("Stream Environment Frame", initial_value=cfg.stream_env_frame)
        stream_model_input_checkbox = server.gui.add_checkbox(
            "Stream Model Input Frame", initial_value=cfg.stream_model_input
        )

    with server.gui.add_folder("Controls"):
        pause_checkbox = server.gui.add_checkbox("Pause", initial_value=False)
        stop_button = server.gui.add_button("Stop After Episode", color="red")
        show_labels_checkbox = server.gui.add_checkbox("Show Labels", initial_value=cfg.show_labels)
        show_markers_checkbox = server.gui.add_checkbox("Show Markers", initial_value=cfg.show_markers)
        show_eef_frame_checkbox = server.gui.add_checkbox("Show EEF Frame", initial_value=cfg.show_eef_frame)
        show_object_frames_checkbox = server.gui.add_checkbox("Show Object Frames", initial_value=cfg.show_object_frames)
        show_fixture_frames_checkbox = server.gui.add_checkbox(
            "Show Fixture Frames", initial_value=cfg.show_fixture_frames and cfg.show_fixtures
        )
        show_site_frames_checkbox = server.gui.add_checkbox(
            "Show Site Frames", initial_value=cfg.show_site_frames and cfg.show_sites
        )
        show_robot_collision_checkbox = server.gui.add_checkbox(
            "Show Robot Collision", initial_value=cfg.show_robot_collision
        )
        show_object_collision_checkbox = server.gui.add_checkbox(
            "Show Object Collision", initial_value=cfg.show_object_collision
        )
        show_fixture_collision_checkbox = server.gui.add_checkbox(
            "Show Fixture Collision", initial_value=cfg.show_fixture_collision and cfg.show_fixtures
        )
        show_trail_checkbox = server.gui.add_checkbox("Show Trail", initial_value=cfg.show_trail)
        show_interest_links_checkbox = server.gui.add_checkbox(
            "Show Interest Links", initial_value=cfg.show_interest_links
        )
        show_object_boxes_checkbox = server.gui.add_checkbox("Show Object Boxes", initial_value=cfg.show_object_boxes)
        show_fixture_boxes_checkbox = server.gui.add_checkbox(
            "Show Fixture Boxes", initial_value=cfg.show_fixture_boxes and cfg.show_fixtures
        )
        show_workspace_box_checkbox = server.gui.add_checkbox("Show Workspace Box", initial_value=cfg.show_workspace_box)
        trail_visible_points_slider = server.gui.add_slider(
            "Visible Trail Points",
            min=2,
            max=max(2, cfg.max_trail_points),
            step=1,
            initial_value=max(2, min(cfg.trail_visible_points, cfg.max_trail_points)),
        )
        collision_opacity_slider = server.gui.add_slider(
            "Collision Opacity",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=cfg.collision_opacity,
        )
        focus_workspace_button = server.gui.add_button("Focus Workspace", color="blue")

    with server.gui.add_folder("Status"):
        run_status_html = server.gui.add_html("<div>Waiting for the first scene snapshot...</div>")
        action_status_html = server.gui.add_html(
            html_rows([("Last Action (Vector)", "Waiting for the first action...")])
        )
        robot_status_html = server.gui.add_html(html_rows([("Robot State", "Waiting for the first observation...")]))
        io_status_html = server.gui.add_html(format_io_status_html(local_log_filepath, cfg.save_rollout))
        access_markdown = server.gui.add_markdown(
            "\n".join(
                [
                    f"**Bind Host**: {cfg.host}",
                    f"**Bind Port**: {cfg.port}",
                    "**SSH Port Forward**: `ssh -L {port}:localhost:{port} <user>@<server>`".replace(
                        "{port}", str(cfg.port)
                    ),
                    f"**Browser URL**: `http://localhost:{cfg.port}`",
                ]
            )
        )

    default_camera_offset = np.array([0.9, -1.05, 0.9], dtype=np.float64)
    camera_focus_state = {
        "look_at": np.array([0.35, 0.0, 0.20], dtype=np.float64),
        "offset": default_camera_offset.copy(),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    server.initial_camera.position = tuple(camera_focus_state["look_at"] + camera_focus_state["offset"])
    server.initial_camera.look_at = tuple(camera_focus_state["look_at"])
    server.initial_camera.up_direction = tuple(camera_focus_state["up"])

    print(f"Viser server listening on http://{cfg.host}:{cfg.port}")
    log_file.write(f"Viser server listening on http://{cfg.host}:{cfg.port}\n")
    log_file.flush()

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
    task_scene_handles: Optional[Scene3DHandles] = None
    last_render_settings: Optional[SceneRenderSettings] = None

    controls = GuiControls(
        pause_checkbox=pause_checkbox,
        stop_button=stop_button,
        show_labels_checkbox=show_labels_checkbox,
        show_markers_checkbox=show_markers_checkbox,
        show_eef_frame_checkbox=show_eef_frame_checkbox,
        show_object_frames_checkbox=show_object_frames_checkbox,
        show_fixture_frames_checkbox=show_fixture_frames_checkbox,
        show_site_frames_checkbox=show_site_frames_checkbox,
        show_robot_collision_checkbox=show_robot_collision_checkbox,
        show_object_collision_checkbox=show_object_collision_checkbox,
        show_fixture_collision_checkbox=show_fixture_collision_checkbox,
        show_trail_checkbox=show_trail_checkbox,
        show_interest_links_checkbox=show_interest_links_checkbox,
        show_object_boxes_checkbox=show_object_boxes_checkbox,
        show_fixture_boxes_checkbox=show_fixture_boxes_checkbox,
        show_workspace_box_checkbox=show_workspace_box_checkbox,
        stream_env_frame_checkbox=stream_env_frame_checkbox,
        stream_model_input_checkbox=stream_model_input_checkbox,
        trail_visible_points_slider=trail_visible_points_slider,
        collision_opacity_slider=collision_opacity_slider,
        focus_workspace_button=focus_workspace_button,
    )

    def apply_camera_focus(camera: Any) -> None:
        camera.look_at = tuple(camera_focus_state["look_at"])
        camera.position = tuple(camera_focus_state["look_at"] + camera_focus_state["offset"])
        camera.up_direction = tuple(camera_focus_state["up"])

    def apply_camera_focus_to_all_clients() -> None:
        get_clients = getattr(server, "get_clients", None)
        if not callable(get_clients):
            return
        try:
            clients = get_clients()
        except Exception:
            return
        client_iter = clients.values() if isinstance(clients, dict) else clients
        for client in client_iter:
            camera = getattr(client, "camera", None)
            if camera is not None:
                apply_camera_focus(camera)

    @server.on_client_connect
    def _(client) -> None:
        apply_camera_focus(client.camera)

    @controls.focus_workspace_button.on_click
    def _(_) -> None:
        apply_camera_focus_to_all_clients()

    def get_render_settings() -> SceneRenderSettings:
        return SceneRenderSettings(
            show_labels=controls.show_labels_checkbox.value,
            show_markers=controls.show_markers_checkbox.value,
            show_eef_frame=controls.show_eef_frame_checkbox.value,
            show_object_frames=controls.show_object_frames_checkbox.value,
            show_fixture_frames=controls.show_fixture_frames_checkbox.value and cfg.show_fixtures,
            show_site_frames=controls.show_site_frames_checkbox.value and cfg.show_sites,
            show_robot_collision=controls.show_robot_collision_checkbox.value,
            show_object_collision=controls.show_object_collision_checkbox.value,
            show_fixture_collision=controls.show_fixture_collision_checkbox.value and cfg.show_fixtures,
            show_trail=controls.show_trail_checkbox.value,
            show_interest_links=controls.show_interest_links_checkbox.value,
            show_object_boxes=controls.show_object_boxes_checkbox.value,
            show_fixture_boxes=controls.show_fixture_boxes_checkbox.value and cfg.show_fixtures,
            show_workspace_box=controls.show_workspace_box_checkbox.value,
            stream_env_frame=controls.stream_env_frame_checkbox.value,
            stream_model_input=controls.stream_model_input_checkbox.value,
            collision_opacity=float(controls.collision_opacity_slider.value),
            trail_visible_points=int(controls.trail_visible_points_slider.value),
        )

    def apply_render_settings_if_needed(settings: SceneRenderSettings) -> None:
        nonlocal last_render_settings
        if task_scene_handles is None:
            last_render_settings = settings
            return
        if settings != last_render_settings:
            apply_scene_visibility(
                task_scene_handles,
                show_eef_frame=settings.show_eef_frame,
                show_object_frames=settings.show_object_frames,
                show_fixture_frames=settings.show_fixture_frames,
                show_site_frames=settings.show_site_frames,
                show_robot_collision=settings.show_robot_collision,
                show_object_collision=settings.show_object_collision,
                show_fixture_collision=settings.show_fixture_collision,
                show_markers=settings.show_markers,
                show_labels=settings.show_labels,
                show_trail=settings.show_trail,
                show_interest_links=settings.show_interest_links,
                show_object_boxes=settings.show_object_boxes,
                show_fixture_boxes=settings.show_fixture_boxes,
                show_workspace_box=settings.show_workspace_box,
            )
            if settings.collision_opacity != (last_render_settings.collision_opacity if last_render_settings else None):
                set_collision_geom_collection_opacity(task_scene_handles.robot_collision_geoms, settings.collision_opacity)
                set_collision_geom_collection_opacity(task_scene_handles.object_collision_geoms, settings.collision_opacity)
                set_collision_geom_collection_opacity(
                    task_scene_handles.fixture_collision_geoms, settings.collision_opacity
                )
            env_image_handle.visible = settings.stream_env_frame
            model_image_handle.visible = settings.stream_model_input
            last_render_settings = settings

    def update_image_streams(obs: dict, model_frame: np.ndarray, done: bool, settings: SceneRenderSettings) -> None:
        if settings.stream_env_frame:
            env_image_handle.image = env_image_for_viser(obs)
        if settings.stream_model_input:
            model_image_handle.image = model_frame

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.resolution)
        eef_trail_points: deque[np.ndarray] = deque(maxlen=cfg.max_trail_points)

        try:
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                print(f"\nTask {task_id}: {task_description}")
                log_file.write(f"\nTask {task_id}: {task_description}\n")

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                if cfg.reset_trail_each_episode:
                    eef_trail_points.clear()
                scene_snapshot = extract_snapshot(env, obs, cfg.show_fixtures, cfg.show_sites)
                if cfg.auto_focus_workspace:
                    focus_target = get_workspace_focus_point(scene_snapshot, fallback=camera_focus_state["look_at"])
                    focus_target[2] += cfg.camera_focus_height_offset
                    camera_focus_state["look_at"] = focus_target
                    server.initial_camera.look_at = tuple(camera_focus_state["look_at"])
                    server.initial_camera.position = tuple(camera_focus_state["look_at"] + camera_focus_state["offset"])
                    server.initial_camera.up_direction = tuple(camera_focus_state["up"])
                    apply_camera_focus_to_all_clients()
                    cfg.auto_focus_workspace = False

                if task_scene_handles is None:
                    task_scene_handles = build_scene_handles(server, scene_snapshot, cfg)
                    last_render_settings = None
                t = 0
                done = False
                replay_images = []
                last_action = None

                current_render_settings = get_render_settings()
                apply_render_settings_if_needed(current_render_settings)
                update_scene_handles(
                    task_scene_handles,
                    scene_snapshot,
                    eef_trail_points,
                    current_render_settings.trail_visible_points,
                )

                model_frame = get_libero_image(obs, resize_size)
                update_image_streams(obs, model_frame, done=done, settings=current_render_settings)

                run_status_html.content = format_run_status_html(
                    task_description,
                    task_id,
                    episode_idx,
                    t,
                    done,
                    total_episodes,
                    total_successes,
                    scene_snapshot,
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
                            scene_snapshot = extract_snapshot(env, obs, cfg.show_fixtures, cfg.show_sites)
                            current_render_settings = get_render_settings()
                            apply_render_settings_if_needed(current_render_settings)
                            update_scene_handles(
                                task_scene_handles,
                                scene_snapshot,
                                eef_trail_points,
                                current_render_settings.trail_visible_points,
                            )
                            model_frame = get_libero_image(obs, resize_size)
                            update_image_streams(obs, model_frame, done=done, settings=current_render_settings)
                            t += 1
                            run_status_html.content = format_run_status_html(
                                task_description,
                                task_id,
                                episode_idx,
                                t,
                                done,
                                total_episodes,
                                total_successes,
                                scene_snapshot,
                            )
                            action_status_html.content = format_action_status_html(last_action)
                            robot_status_html.content = format_robot_status_html(obs, last_action)
                            continue

                        model_frame = get_libero_image(obs, resize_size)
                        replay_images.append(model_frame)

                        observation = {
                            "full_image": model_frame,
                            "state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
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
                        scene_snapshot = extract_snapshot(env, obs, cfg.show_fixtures, cfg.show_sites)
                        current_render_settings = get_render_settings()
                        apply_render_settings_if_needed(current_render_settings)
                        update_scene_handles(
                            task_scene_handles,
                            scene_snapshot,
                            eef_trail_points,
                            current_render_settings.trail_visible_points,
                        )
                        update_image_streams(obs, model_frame, done=done, settings=current_render_settings)
                        t += 1

                        run_status_html.content = format_run_status_html(
                            task_description,
                            task_id,
                            episode_idx,
                            t,
                            done,
                            total_episodes,
                            total_successes,
                            scene_snapshot,
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

        if task_scene_handles is not None:
            task_scene_handles.root.remove()
            task_scene_handles = None
            last_render_settings = None

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
    eval_libero_viser_3d()
