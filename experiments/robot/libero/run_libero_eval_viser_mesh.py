"""
run_libero_eval_viser_mesh.py

Runs an OpenVLA model in a headless LIBERO simulation environment and streams MuJoCo visual meshes to Viser.

This is a mesh-focused companion to run_libero_eval_viser_3d.py. It renders compiled MuJoCo mesh geoms
with simple material colors instead of approximating the scene with collision primitives.

Usage:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_viser_mesh.py \
        --model_family openvla \
        --pretrained_checkpoint models/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --task_id 0 \
        --center_crop True \
        --num_trials_per_task 1
        
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval_viser_mesh.py \
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

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero import libero_mesh_viser as mesh_viser  # noqa: E402
from experiments.robot.libero.run_libero_eval_viser_3d import (  # noqa: E402
    aabb_to_segments,
    env_image_for_viser,
    get_domain_env,
    get_max_steps,
    get_workspace_box,
    get_workspace_focus_point,
    iter_descendant_body_ids,
    site_matrix_to_wxyz,
    trail_to_segments,
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


def format_vector(values: Optional[np.ndarray]) -> str:
    if values is None:
        return "N/A"
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def format_action(action: Optional[np.ndarray]) -> str:
    if action is None:
        return "N/A"
    return np.array2string(np.asarray(action), precision=3, suppress_small=True)


def format_robot_status_html(obs: Optional[dict], action: Optional[np.ndarray]) -> str:
    if obs is None:
        return html_rows([("Robot State", "N/A")])
    gripper_action = "N/A" if action is None else f"{float(np.asarray(action).reshape(-1)[-1]):.3f}"
    return html_rows(
        [
            ("EEF Pos", format_vector(obs.get("robot0_eef_pos"))),
            ("EEF Quat", format_vector(obs.get("robot0_eef_quat"))),
            ("Gripper QPos", format_vector(obs.get("robot0_gripper_qpos"))),
            ("Gripper Cmd", gripper_action),
        ]
    )


def format_run_status_html(
    task_description: str,
    task_id: int,
    episode_idx: int,
    step_idx: int,
    done: bool,
    total_episodes: int,
    total_successes: int,
    mesh_count: int,
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
            ("Mesh Geoms", str(mesh_count)),
        ]
    )


def format_io_status_html(local_log_filepath: str, save_rollout: bool) -> str:
    return html_rows(
        [
            ("Save Rollout Video", str(save_rollout)),
            ("Log File", local_log_filepath),
        ]
    )


def parse_geom_groups(groups: str) -> set[int]:
    parsed = set()
    for item in str(groups).split(","):
        item = item.strip()
        if item:
            parsed.add(int(item))
    return parsed


def sanitize_path_component(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(name))


def get_geom_name(model: Any, geom_id: int) -> str:
    try:
        geom_name = model.geom_id2name(int(geom_id))
    except Exception:
        geom_name = None
    if geom_name:
        return str(geom_name)
    return f"geom_{int(geom_id):04d}"


def get_body_name(model: Any, body_id: int) -> str:
    try:
        body_name = model.body_id2name(int(body_id))
    except Exception:
        body_name = None
    return str(body_name) if body_name else f"body_{int(body_id):04d}"


def mesh_arrays_for_geom(model: Any, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    mesh_id = int(model.geom_dataid[geom_id])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float32).copy()
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32).copy()
    return vertices, faces


def geom_rgba(model: Any, geom_id: int) -> np.ndarray:
    rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64).copy()
    if rgba[3] <= 1e-4:
        mat_id = int(model.geom_matid[geom_id])
        if mat_id >= 0:
            rgba = np.asarray(model.mat_rgba[mat_id], dtype=np.float64).copy()
    if rgba[3] <= 1e-4:
        rgba = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float64)
    return np.clip(rgba, 0.0, 1.0)


def body_category_maps(domain_env: Any) -> dict[int, str]:
    model = domain_env.sim.model
    categories: dict[int, str] = {}
    for object_name, body_id in getattr(domain_env, "obj_body_id", {}).items():
        if object_name in getattr(domain_env, "objects_dict", {}):
            category = "object"
        elif object_name in getattr(domain_env, "fixtures_dict", {}):
            category = "fixture"
        else:
            category = "scene"
        for descendant_body_id in iter_descendant_body_ids(model, int(body_id)):
            categories[int(descendant_body_id)] = category

    for body_id in range(int(model.nbody)):
        if body_id in categories:
            continue
        body_name = get_body_name(model, body_id).lower()
        if any(token in body_name for token in ("robot", "panda", "gripper", "mount", "eef")):
            categories[body_id] = "robot"
        elif body_id == 0:
            categories[body_id] = "world"
        else:
            categories[body_id] = "scene"
    return categories


def body_id_to_entity_name(domain_env: Any) -> dict[int, tuple[str, str]]:
    model = domain_env.sim.model
    id_to_name: dict[int, tuple[str, str]] = {}
    for object_name, body_id in getattr(domain_env, "obj_body_id", {}).items():
        if object_name in getattr(domain_env, "objects_dict", {}):
            category = "object"
        elif object_name in getattr(domain_env, "fixtures_dict", {}):
            category = "fixture"
        else:
            continue
        id_to_name[int(body_id)] = (object_name, category)
        for descendant_body_id in iter_descendant_body_ids(model, int(body_id)):
            id_to_name.setdefault(int(descendant_body_id), (object_name, category))
    return id_to_name


@dataclass
class EntityLabelState:
    name: str
    category: str
    position: np.ndarray
    is_interest: bool


@dataclass
class MeshGeomState:
    geom_id: int
    name: str
    category: str
    position: np.ndarray
    wxyz: np.ndarray
    color: tuple[int, int, int]
    opacity: float


@dataclass
class MeshGeomHandle:
    mesh: Any
    category: str


@dataclass
class MeshSceneHandles:
    root: Any
    mesh_geoms: dict[int, MeshGeomHandle]
    labels: dict[str, Any]
    current_eef: Any
    eef_trail: Any
    eef_trail_points: Any
    interest_links: Any
    workspace_box: Any


def extract_mesh_states(env: Any, geom_groups: set[int]) -> list[MeshGeomState]:
    domain_env = get_domain_env(env)
    model = domain_env.sim.model
    data = domain_env.sim.data
    body_categories = body_category_maps(domain_env)

    mesh_states = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) not in geom_groups:
            continue
        if int(model.geom_dataid[geom_id]) < 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        category = body_categories.get(body_id, "scene")
        rgba = geom_rgba(model, geom_id)
        mesh_states.append(
            MeshGeomState(
                geom_id=int(geom_id),
                name=get_geom_name(model, geom_id),
                category=category,
                position=np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy(),
                wxyz=site_matrix_to_wxyz(np.asarray(data.geom_xmat[geom_id], dtype=np.float64)),
                color=tuple(int(round(channel * 255.0)) for channel in rgba[:3]),
                opacity=float(rgba[3]),
            )
        )
    return mesh_states


def extract_entity_label_states(env: Any, label_offset: float) -> dict[str, EntityLabelState]:
    domain_env = get_domain_env(env)
    sim = domain_env.sim
    obj_of_interest = set(getattr(domain_env, "obj_of_interest", []) or [])
    states: dict[str, EntityLabelState] = {}
    for object_name, body_id in getattr(domain_env, "obj_body_id", {}).items():
        if object_name in getattr(domain_env, "objects_dict", {}):
            category = "object"
        elif object_name in getattr(domain_env, "fixtures_dict", {}):
            category = "fixture"
        else:
            continue
        position = np.asarray(sim.data.body_xpos[int(body_id)], dtype=np.float64).copy()
        position[2] += label_offset
        states[object_name] = EntityLabelState(
            name=object_name,
            category=category,
            position=position,
            is_interest=object_name in obj_of_interest,
        )
    return states


def extract_eef_position(env: Any, obs: Optional[dict] = None) -> np.ndarray:
    domain_env = get_domain_env(env)
    try:
        eef_site_id = domain_env.robots[0].eef_site_id
        return np.asarray(domain_env.sim.data.site_xpos[int(eef_site_id)], dtype=np.float64).copy()
    except Exception:
        if obs is None:
            raise
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()


def interest_link_segments(eef_position: np.ndarray, label_states: dict[str, EntityLabelState]) -> np.ndarray:
    segments = []
    for state in label_states.values():
        if state.is_interest:
            segments.append([eef_position, state.position])
    if len(segments) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)


def trail_segments_with_time_gradient(
    points: list[np.ndarray],
    start_color: tuple[int, int, int] = (66, 135, 245),
    end_color: tuple[int, int, int] = (255, 176, 66),
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)

    points_arr = np.asarray(points, dtype=np.float32)
    segments = np.stack([points_arr[:-1], points_arr[1:]], axis=1)

    point_count = points_arr.shape[0]
    color_steps = np.linspace(0.0, 1.0, point_count, dtype=np.float32)[:, None]
    start = np.asarray(start_color, dtype=np.float32).reshape(1, 3)
    end = np.asarray(end_color, dtype=np.float32).reshape(1, 3)
    point_colors = np.round((1.0 - color_steps) * start + color_steps * end).astype(np.uint8)
    segment_colors = np.stack([point_colors[:-1], point_colors[1:]], axis=1)
    return segments, segment_colors


def trail_points_with_time_gradient(
    points: list[np.ndarray],
    start_color: tuple[int, int, int] = (66, 135, 245),
    end_color: tuple[int, int, int] = (255, 176, 66),
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points_arr = np.asarray(points, dtype=np.float32)
    point_count = points_arr.shape[0]
    color_steps = np.linspace(0.0, 1.0, point_count, dtype=np.float32)[:, None]
    start = np.asarray(start_color, dtype=np.float32).reshape(1, 3)
    end = np.asarray(end_color, dtype=np.float32).reshape(1, 3)
    point_colors = np.round((1.0 - color_steps) * start + color_steps * end).astype(np.uint8)
    return points_arr, point_colors


def add_mesh_scene(
    server: Any,
    env: Any,
    mesh_states: list[MeshGeomState],
    label_states: dict[str, EntityLabelState],
    cfg: "GenerateConfig",
) -> MeshSceneHandles:
    domain_env = get_domain_env(env)
    model = domain_env.sim.model
    root = server.scene.add_frame("/mesh_scene", show_axes=False)
    server.scene.add_grid(
        "/mesh_scene/grid",
        width=2.0,
        height=2.0,
        plane="xy",
        cell_size=0.1,
        section_size=0.5,
        cell_color=(220, 220, 220),
        section_color=(180, 180, 180),
        plane_color=(245, 245, 245),
        plane_opacity=0.10,
        shadow_opacity=0.0,
        position=(0.0, 0.0, 0.0),
    )

    handles: dict[int, MeshGeomHandle] = {}
    for state in mesh_states:
        vertices, faces = mesh_arrays_for_geom(model, state.geom_id)
        mesh_handle = server.scene.add_mesh_simple(
            f"/mesh_scene/geoms/{state.category}/{state.geom_id:04d}_{sanitize_path_component(state.name)}",
            vertices=vertices,
            faces=faces,
            color=state.color,
            opacity=min(state.opacity, cfg.mesh_opacity),
            material=cfg.mesh_material,
            flat_shading=cfg.flat_shading,
            side=cfg.mesh_side,
            position=state.position,
            wxyz=state.wxyz,
            visible=True,
        )
        handles[state.geom_id] = MeshGeomHandle(mesh=mesh_handle, category=state.category)

    labels: dict[str, Any] = {}
    for state in label_states.values():
        label_text = f"* {state.name}" if state.is_interest else state.name
        labels[state.name] = server.scene.add_label(
            f"/mesh_scene/labels/{state.category}/{sanitize_path_component(state.name)}",
            label_text,
            position=state.position,
            anchor="bottom-center",
            font_size_mode="screen",
            font_screen_scale=cfg.label_scale,
            depth_test=False,
        )

    current_eef = server.scene.add_icosphere(
        "/mesh_scene/debug/current_eef",
        radius=0.014,
        color=(255, 96, 96),
        opacity=0.95,
        material="toon5",
        position=extract_eef_position(env),
    )
    eef_trail = server.scene.add_line_segments(
        "/mesh_scene/debug/eef_trail",
        points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=4.0,
    )
    eef_trail_points = server.scene.add_point_cloud(
        "/mesh_scene/debug/eef_trail_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        point_size=0.012,
        point_shape="circle",
    )
    interest_links = server.scene.add_line_segments(
        "/mesh_scene/debug/interest_links",
        points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=(255, 174, 66),
        line_width=2.0,
    )

    workspace_box = None
    workspace = get_workspace_box(domain_env)
    if workspace is not None:
        workspace_box = server.scene.add_line_segments(
            "/mesh_scene/debug/workspace_box",
            points=aabb_to_segments(workspace),
            colors=(120, 210, 120),
            line_width=4.0,
        )
    return MeshSceneHandles(
        root=root,
        mesh_geoms=handles,
        labels=labels,
        current_eef=current_eef,
        eef_trail=eef_trail,
        eef_trail_points=eef_trail_points,
        interest_links=interest_links,
        workspace_box=workspace_box,
    )


def update_mesh_scene(handles: MeshSceneHandles, mesh_states: list[MeshGeomState]) -> None:
    for state in mesh_states:
        handle = handles.mesh_geoms.get(state.geom_id)
        if handle is None:
            continue
        handle.mesh.position = state.position
        handle.mesh.wxyz = state.wxyz


def update_mesh_overlays(
    handles: MeshSceneHandles,
    env: Any,
    obs: dict,
    label_states: dict[str, EntityLabelState],
    eef_trail_points: deque[np.ndarray],
    max_visible_trail_points: int,
) -> None:
    for state in label_states.values():
        label_handle = handles.labels.get(state.name)
        if label_handle is not None:
            label_handle.position = state.position
    eef_position = extract_eef_position(env, obs)
    handles.current_eef.position = eef_position
    eef_trail_points.append(eef_position)
    if max_visible_trail_points <= 1:
        handles.eef_trail.points = np.zeros((0, 2, 3), dtype=np.float32)
        handles.eef_trail.colors = np.zeros((0, 2, 3), dtype=np.uint8)
        handles.eef_trail_points.points = np.zeros((0, 3), dtype=np.float32)
        handles.eef_trail_points.colors = np.zeros((0, 3), dtype=np.uint8)
    else:
        visible_points = list(eef_trail_points)[-max_visible_trail_points:]
        trail_segments, trail_colors = trail_segments_with_time_gradient(visible_points)
        trail_points, trail_point_colors = trail_points_with_time_gradient(visible_points)
        handles.eef_trail.points = trail_segments
        handles.eef_trail.colors = trail_colors
        handles.eef_trail_points.points = trail_points
        handles.eef_trail_points.colors = trail_point_colors
    handles.interest_links.points = interest_link_segments(eef_position, label_states)


def apply_mesh_visibility(
    handles: MeshSceneHandles,
    *,
    show_robot_mesh: bool,
    show_object_mesh: bool,
    show_fixture_mesh: bool,
    show_scene_mesh: bool,
    show_labels: bool,
    show_interest_links: bool,
    show_eef_trail: bool,
    show_workspace_box: bool,
) -> None:
    visible_by_category = {
        "robot": show_robot_mesh,
        "object": show_object_mesh,
        "fixture": show_fixture_mesh,
        "scene": show_scene_mesh,
        "world": show_scene_mesh,
    }
    for handle in handles.mesh_geoms.values():
        handle.mesh.visible = visible_by_category.get(handle.category, show_scene_mesh)
    for label in handles.labels.values():
        label.visible = show_labels
    handles.current_eef.visible = show_eef_trail
    handles.interest_links.visible = show_interest_links
    handles.eef_trail.visible = show_eef_trail
    handles.eef_trail_points.visible = show_eef_trail
    if handles.workspace_box is not None:
        handles.workspace_box.visible = show_workspace_box


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

    mesh_geom_groups: str = "1"
    mesh_opacity: float = 1.0
    mesh_material: str = "standard"
    mesh_side: str = "double"
    flat_shading: bool = False
    show_robot_mesh: bool = True
    show_object_mesh: bool = True
    show_fixture_mesh: bool = True
    show_scene_mesh: bool = True
    show_labels: bool = True
    show_interest_links: bool = True
    show_eef_trail: bool = True
    show_workspace_box: bool = True
    label_offset: float = 0.05
    label_scale: float = 0.9
    max_trail_points: int = 128
    trail_visible_points: int = 128
    auto_focus_workspace: bool = True
    camera_focus_height_offset: float = 0.03

    run_id_note: Optional[str] = "viser-mesh"
    local_log_dir: str = "./experiments/logs"
    use_swanlab: bool = False
    swanlab_project: str = "VLA-Attack"
    seed: int = 7
    # fmt: on


@draccus.wrap()
def eval_libero_viser_mesh(cfg: GenerateConfig) -> None:
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

    run_id = f"EVAL-VISER-MESH-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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
    runtime = mesh_viser.create_mesh_viser_runtime(cfg, local_log_filepath)
    print(f"Viser mesh server listening on http://{cfg.host}:{cfg.port}")
    log_file.write(f"Viser mesh server listening on http://{cfg.host}:{cfg.port}\n")
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
    geom_groups = mesh_viser.parse_geom_groups(cfg.mesh_geom_groups)

    total_episodes, total_successes = 0, 0
    stop_requested = False

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.resolution)
        eef_trail_points: deque[np.ndarray] = deque(maxlen=cfg.max_trail_points)

        try:
            task_episodes, task_successes = 0, 0
            cfg.auto_focus_workspace = True  # 每个 task 开始时重置，确保重新 focus
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                print(f"\nTask {task_id}: {task_description}")
                log_file.write(f"\nTask {task_id}: {task_description}\n")

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                mesh_states = extract_mesh_states(env, geom_groups)
                label_states = extract_entity_label_states(env, cfg.label_offset)
                eef_trail_points.clear()

                if cfg.auto_focus_workspace:
                    mesh_viser.auto_focus_workspace(runtime, env, cfg)
                    cfg.auto_focus_workspace = False

                mesh_viser.sync_mesh_scene(runtime, env, obs, mesh_states, label_states, eef_trail_points, cfg)

                model_frame = get_libero_image(obs, resize_size)
                mesh_viser.update_stream_images(runtime, obs, model_frame)

                t = 0
                done = False
                replay_images = []
                last_action = None

                mesh_viser.update_status(
                    runtime,
                    task_description=task_description,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    step_idx=t,
                    done=done,
                    total_episodes=total_episodes,
                    total_successes=total_successes,
                    mesh_count=len(mesh_states),
                    obs=obs,
                    action=last_action,
                )
                print(f"Starting episode {task_episodes + 1}...")
                log_file.write(f"Starting episode {task_episodes + 1}...\n")

                while t < max_steps + cfg.num_steps_wait:
                    if runtime.controls.stop_button.value:
                        runtime.controls.stop_button.value = False
                        stop_requested = True

                    while runtime.controls.pause_checkbox.value and not stop_requested:
                        time.sleep(0.1)
                        if runtime.controls.stop_button.value:
                            runtime.controls.stop_button.value = False
                            stop_requested = True
                            break
                    if stop_requested:
                        break

                    try:
                        if t < cfg.num_steps_wait:
                            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                            mesh_states = extract_mesh_states(env, geom_groups)
                            label_states = extract_entity_label_states(env, cfg.label_offset)
                            mesh_viser.sync_mesh_scene(runtime, env, obs, mesh_states, label_states, eef_trail_points, cfg)
                            model_frame = get_libero_image(obs, resize_size)
                            mesh_viser.update_stream_images(runtime, obs, model_frame)
                            t += 1
                            mesh_viser.update_status(
                                runtime,
                                task_description=task_description,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                step_idx=t,
                                done=done,
                                total_episodes=total_episodes,
                                total_successes=total_successes,
                                mesh_count=len(mesh_states),
                                obs=obs,
                                action=last_action,
                            )
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
                        mesh_states = extract_mesh_states(env, geom_groups)
                        label_states = extract_entity_label_states(env, cfg.label_offset)
                        mesh_viser.sync_mesh_scene(runtime, env, obs, mesh_states, label_states, eef_trail_points, cfg)
                        mesh_viser.update_stream_images(runtime, obs, model_frame)
                        t += 1

                        mesh_viser.update_status(
                            runtime,
                            task_description=task_description,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            step_idx=t,
                            done=done,
                            total_episodes=total_episodes,
                            total_successes=total_successes,
                            mesh_count=len(mesh_states),
                            obs=obs,
                            action=last_action,
                        )
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

        mesh_viser.clear_mesh_scene(runtime)

    maybe_log_swanlab(
        swanlab_enabled,
        {
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        },
    )
    print("Simulation finished. Viser mesh UI will remain available until you stop the process.")
    log_file.write("Simulation finished.\n")
    log_file.close()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down Viser mesh server.")


if __name__ == "__main__":
    eval_libero_viser_mesh()
