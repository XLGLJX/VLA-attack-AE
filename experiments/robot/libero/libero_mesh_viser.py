"""Reusable Viser mesh UI helpers for LIBERO visualization scripts."""

from __future__ import annotations

import html
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import mujoco
import numpy as np

from experiments.robot.libero.run_libero_eval_viser_3d import (
    aabb_to_segments,
    env_image_for_viser,
    get_domain_env,
    get_workspace_box,
    get_workspace_focus_point,
    iter_descendant_body_ids,
    site_matrix_to_wxyz,
)


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
    return html_rows([("Save Rollout Video", str(save_rollout)), ("Log File", local_log_filepath)])


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
    return str(geom_name) if geom_name else f"geom_{int(geom_id):04d}"


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


@dataclass
class MeshViserImageHandles:
    env_image_handle: Any
    model_image_handle: Any
    stream_env_frame_checkbox: Any
    stream_model_input_checkbox: Any


@dataclass
class MeshViserControlHandles:
    pause_checkbox: Any
    stop_button: Any
    show_robot_mesh_checkbox: Any
    show_object_mesh_checkbox: Any
    show_fixture_mesh_checkbox: Any
    show_scene_mesh_checkbox: Any
    show_labels_checkbox: Any
    show_interest_links_checkbox: Any
    show_eef_trail_checkbox: Any
    show_workspace_box_checkbox: Any
    focus_workspace_button: Any


@dataclass
class MeshViserStatusHandles:
    run_status_html: Any
    action_status_html: Any
    robot_status_html: Any
    io_status_html: Any
    access_markdown: Any


@dataclass
class MeshViserRuntime:
    server: Any
    images: MeshViserImageHandles
    controls: MeshViserControlHandles
    status: MeshViserStatusHandles
    camera_focus_state: dict[str, np.ndarray]
    mesh_scene_handles: Optional[MeshSceneHandles] = None


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
    segments = [[eef_position, state.position] for state in label_states.values() if state.is_interest]
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


def add_mesh_scene(server: Any, env: Any, mesh_states: list[MeshGeomState], label_states: dict[str, EntityLabelState], cfg: Any) -> MeshSceneHandles:
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
    mesh_handles: dict[int, MeshGeomHandle] = {}
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
        mesh_handles[state.geom_id] = MeshGeomHandle(mesh=mesh_handle, category=state.category)

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
        mesh_geoms=mesh_handles,
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


def create_mesh_viser_runtime(cfg: Any, local_log_filepath: str) -> MeshViserRuntime:
    try:
        import viser
    except ImportError as exc:
        raise ImportError("`viser` is required for mesh visualization. Install it with `pip install viser`.") from exc

    server = viser.ViserServer(host=cfg.host, port=cfg.port)
    server.gui.configure_theme(control_layout="collapsible", control_width=cfg.control_width, show_share_button=False)
    server.gui.set_panel_label("OpenVLA LIBERO Mesh Monitor")
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
        stream_model_input_checkbox = server.gui.add_checkbox("Stream Model Input Frame", initial_value=cfg.stream_model_input)

    with server.gui.add_folder("Controls"):
        pause_checkbox = server.gui.add_checkbox("Pause", initial_value=False)
        stop_button = server.gui.add_button("Stop After Episode", color="red")
        show_robot_mesh_checkbox = server.gui.add_checkbox("Show Robot Mesh", initial_value=cfg.show_robot_mesh)
        show_object_mesh_checkbox = server.gui.add_checkbox("Show Object Mesh", initial_value=cfg.show_object_mesh)
        show_fixture_mesh_checkbox = server.gui.add_checkbox("Show Fixture Mesh", initial_value=cfg.show_fixture_mesh)
        show_scene_mesh_checkbox = server.gui.add_checkbox("Show Scene Mesh", initial_value=cfg.show_scene_mesh)
        show_labels_checkbox = server.gui.add_checkbox("Show Labels", initial_value=cfg.show_labels)
        show_interest_links_checkbox = server.gui.add_checkbox("Show Interest Links", initial_value=cfg.show_interest_links)
        show_eef_trail_checkbox = server.gui.add_checkbox("Show EEF Trail", initial_value=cfg.show_eef_trail)
        show_workspace_box_checkbox = server.gui.add_checkbox("Show Workspace Box", initial_value=cfg.show_workspace_box)
        focus_workspace_button = server.gui.add_button("Focus Workspace", color="blue")

    with server.gui.add_folder("Status"):
        run_status_html = server.gui.add_html("<div>Waiting for the first mesh scene...</div>")
        action_status_html = server.gui.add_html(html_rows([("Last Action", "Waiting for the first action...")]))
        robot_status_html = server.gui.add_html(html_rows([("Robot State", "Waiting for the first observation...")]))
        io_status_html = server.gui.add_html(format_io_status_html(local_log_filepath, cfg.save_rollout))
        access_markdown = server.gui.add_markdown(
            "\n".join(
                [
                    f"**Bind Host**: {cfg.host}",
                    f"**Bind Port**: {cfg.port}",
                    "**SSH Port Forward**: `ssh -L {port}:localhost:{port} <user>@<server>`".replace("{port}", str(cfg.port)),
                    f"**Browser URL**: `http://localhost:{cfg.port}`",
                    f"**Mesh Groups**: `{cfg.mesh_geom_groups}`",
                    "**Note**: This mesh view renders geometry and simple material colors, not texture maps.",
                ]
            )
        )

    runtime = MeshViserRuntime(
        server=server,
        images=MeshViserImageHandles(
            env_image_handle=env_image_handle,
            model_image_handle=model_image_handle,
            stream_env_frame_checkbox=stream_env_frame_checkbox,
            stream_model_input_checkbox=stream_model_input_checkbox,
        ),
        controls=MeshViserControlHandles(
            pause_checkbox=pause_checkbox,
            stop_button=stop_button,
            show_robot_mesh_checkbox=show_robot_mesh_checkbox,
            show_object_mesh_checkbox=show_object_mesh_checkbox,
            show_fixture_mesh_checkbox=show_fixture_mesh_checkbox,
            show_scene_mesh_checkbox=show_scene_mesh_checkbox,
            show_labels_checkbox=show_labels_checkbox,
            show_interest_links_checkbox=show_interest_links_checkbox,
            show_eef_trail_checkbox=show_eef_trail_checkbox,
            show_workspace_box_checkbox=show_workspace_box_checkbox,
            focus_workspace_button=focus_workspace_button,
        ),
        status=MeshViserStatusHandles(
            run_status_html=run_status_html,
            action_status_html=action_status_html,
            robot_status_html=robot_status_html,
            io_status_html=io_status_html,
            access_markdown=access_markdown,
        ),
        camera_focus_state={
            "look_at": np.array([0.35, 0.0, 0.20], dtype=np.float64),
            "offset": np.array([0.9, -1.05, 0.9], dtype=np.float64),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        },
    )
    initialize_mesh_camera(runtime)
    return runtime


def apply_camera_focus(runtime: MeshViserRuntime, camera: Any) -> None:
    state = runtime.camera_focus_state
    camera.position = tuple(state["look_at"] + state["offset"])
    camera.up_direction = tuple(state["up"])
    camera.look_at = tuple(state["look_at"])


def apply_camera_focus_to_all_clients(runtime: MeshViserRuntime) -> None:
    get_clients = getattr(runtime.server, "get_clients", None)
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
            apply_camera_focus(runtime, camera)


def initialize_mesh_camera(runtime: MeshViserRuntime) -> None:
    server = runtime.server
    state = runtime.camera_focus_state
    server.initial_camera.position = tuple(state["look_at"] + state["offset"])
    server.initial_camera.look_at = tuple(state["look_at"])
    server.initial_camera.up_direction = tuple(state["up"])

    @server.on_client_connect
    def _(client) -> None:
        apply_camera_focus(runtime, client.camera)

    @runtime.controls.focus_workspace_button.on_click
    def _(_) -> None:
        apply_camera_focus_to_all_clients(runtime)


def auto_focus_workspace(runtime: MeshViserRuntime, env: Any, cfg: Any) -> None:
    focus_target = get_workspace_focus_point(None, fallback=runtime.camera_focus_state["look_at"])
    workspace = get_workspace_box(get_domain_env(env))
    if workspace is not None:
        lower, upper = workspace
        focus_target = (np.asarray(lower) + np.asarray(upper)) / 2.0
        focus_target[2] = float(np.asarray(upper)[2]) + cfg.camera_focus_height_offset
    runtime.camera_focus_state["look_at"] = focus_target
    runtime.server.initial_camera.position = tuple(focus_target + runtime.camera_focus_state["offset"])
    runtime.server.initial_camera.up_direction = tuple(runtime.camera_focus_state["up"])
    runtime.server.initial_camera.look_at = tuple(focus_target)
    apply_camera_focus_to_all_clients(runtime)


def sync_mesh_scene(
    runtime: MeshViserRuntime,
    env: Any,
    obs: dict,
    mesh_states: list[MeshGeomState],
    label_states: dict[str, EntityLabelState],
    eef_trail_points: deque[np.ndarray],
    cfg: Any,
) -> None:
    if runtime.mesh_scene_handles is None:
        runtime.mesh_scene_handles = add_mesh_scene(runtime.server, env, mesh_states, label_states, cfg)
    update_mesh_scene(runtime.mesh_scene_handles, mesh_states)
    update_mesh_overlays(runtime.mesh_scene_handles, env, obs, label_states, eef_trail_points, cfg.trail_visible_points)
    apply_mesh_visibility(
        runtime.mesh_scene_handles,
        show_robot_mesh=runtime.controls.show_robot_mesh_checkbox.value,
        show_object_mesh=runtime.controls.show_object_mesh_checkbox.value,
        show_fixture_mesh=runtime.controls.show_fixture_mesh_checkbox.value,
        show_scene_mesh=runtime.controls.show_scene_mesh_checkbox.value,
        show_labels=runtime.controls.show_labels_checkbox.value,
        show_interest_links=runtime.controls.show_interest_links_checkbox.value,
        show_eef_trail=runtime.controls.show_eef_trail_checkbox.value,
        show_workspace_box=runtime.controls.show_workspace_box_checkbox.value,
    )


def update_stream_images(runtime: MeshViserRuntime, obs: dict, model_frame: np.ndarray) -> None:
    if runtime.images.stream_env_frame_checkbox.value:
        runtime.images.env_image_handle.image = env_image_for_viser(obs)
    if runtime.images.stream_model_input_checkbox.value:
        runtime.images.model_image_handle.image = model_frame


def update_status(
    runtime: MeshViserRuntime,
    *,
    task_description: str,
    task_id: int,
    episode_idx: int,
    step_idx: int,
    done: bool,
    total_episodes: int,
    total_successes: int,
    mesh_count: int,
    obs: Optional[dict],
    action: Optional[np.ndarray],
) -> None:
    runtime.status.run_status_html.content = format_run_status_html(
        task_description,
        task_id,
        episode_idx,
        step_idx,
        done,
        total_episodes,
        total_successes,
        mesh_count,
    )
    runtime.status.action_status_html.content = html_rows([("Last Action", format_action(action))])
    runtime.status.robot_status_html.content = format_robot_status_html(obs, action)


def clear_mesh_scene(runtime: MeshViserRuntime) -> None:
    if runtime.mesh_scene_handles is not None:
        runtime.mesh_scene_handles.root.remove()
        runtime.mesh_scene_handles = None
