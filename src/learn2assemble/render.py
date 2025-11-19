import math

import polyscope as ps
import polyscope.imgui as psim
from trimesh import Scene, Trimesh
import numpy as np
from scipy.spatial.transform.rotation import Rotation
from multiprocessing import Queue
import json
import trimesh
from learn2assemble import RESOURCE_DIR, get_gripper_spheres
from learn2assemble.grasp import compute_grasp_frame, compute_grasp_table
from learn2assemble.insertion import compute_insertion_drt, compute_insertion_table


def init_polyscope():
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(0)

step_id = 0
t = 0
q = None

def render_sequence(parts: list[Trimesh],
                    sequence: np.ndarray,
                    settings: dict):
    n_part = len(parts)

    env = settings.get("env", {})
    boundary_part_ids = env.get("boundary_part_ids", [])

    table_insertion, drts = compute_insertion_table(parts, settings)
    table_grasp, grasp_frames, scaled_parts = compute_grasp_table(parts, settings)
    draw_assembly(scaled_parts, sequence[0, :])

    def sequence_callback():
        global step_id, t, q

        changed, step_id = psim.SliderInt("step", v=step_id, v_min=0, v_max=sequence.shape[0] - 1)
        current_state = sequence[step_id, :]

        if changed:
            ps.remove_all_structures()
            ps.remove_all_groups()

            if step_id > 0:
                prev_state = sequence[step_id - 1, :]
            else:
                prev_state = np.zeros(len(parts))
                prev_state[boundary_part_ids] = 2

            held_state = (current_state == 2)
            held_state[boundary_part_ids] = False
            held_parts = held_state.nonzero()[0]
            for robot_id, part_id in enumerate(held_parts):
                frames = compute_grasp_frame(part_id, current_state, table_grasp, grasp_frames)
                if frames is not None:
                    draw_gripper(frames[0, :], 0.05, f"robot {robot_id}")
            draw_assembly(scaled_parts, current_state)

            to_install_part_id = np.logical_and(current_state == 2, prev_state != 2).nonzero()[0]
            q = None
            if to_install_part_id.shape[0] > 0:
                to_install_part_id = to_install_part_id[0]
                part_drts = compute_insertion_drt(to_install_part_id, current_state, table_insertion, drts)
                if part_drts is not None:
                    q = np.zeros(n_part * 6)
                    q[6 * to_install_part_id: to_install_part_id * 6 + 3] = part_drts[0, :]

        if q is not None:
            changed, t = psim.SliderFloat("time", v=t, v_min=0, v_max=1)
            if changed:
                draw_assembly_motion(scaled_parts, current_state, t * q)

    ps.set_user_callback(sequence_callback)


def render_batch_simulation(parts: list[Trimesh], boundary_part_ids: list = [], queue: Queue = None):
    global storage, step_id
    # storage = []
    # step_id = 0
    camera = {"farClipRatio": 20.0, "fov": 45.0, "nearClipRatio": 0.005, "projectionMode": "Perspective",
              "viewMat": [0.960128843784332, 0.279557704925537, -6.99097890688449e-10, -5.19866847991943,
                          -0.19538114964962, 0.671027898788452, 0.715225756168365, -0.13018886744976, 0.199946850538254,
                          -0.686708748340607, 0.698893308639526, -9.46305561065674, 0.0, 0.0, 0.0, 1.0],
              "windowHeight": 2019, "windowWidth": 3840}
    camera = json.dumps(camera)

    def callback():
        # global storage, step_id
        try:
            if queue is not None:
                queue_data = queue.get(block=False)
                # storage.append(data)
                if queue_data["type"] == "render":
                    [render_part_states, render_stability] = queue_data["data"]
                    draw_assembly_batches(parts, render_part_states, render_stability, boundary_part_ids, 2)
        except:
            pass
        # changed, step_id = psim.SliderInt("step_id", v = step_id, v_min = 0, v_max= len(storage) - 1)
        # if psim.IsKeyReleased(psim.ImGuiKey_LeftArrow):
        #     step_id = np.clip(step_id - 1, 0, len(storage) - 1)
        #     changed = True
        # if psim.IsKeyReleased(psim.ImGuiKey_RightArrow):
        #     step_id = np.clip(step_id + 1, 0, len(storage) - 1)
        #     changed = True
        # if changed:
        #     [render_part_states, render_stability] = storage[step_id]
        #     draw_assembly_batches(parts, render_part_states, render_stability, boundary_part_ids, 1.5)

    init_polyscope()
    ps.set_view_from_json(camera)
    ps.set_user_callback(callback)
    ps.show()


def draw_assembly_motion(parts: list[Trimesh], part_states: np.ndarray, q: np.ndarray):
    for part_id, part_mesh in enumerate(parts):
        enabled = True
        if part_states[part_id] == 1:
            color = [1.0, 1.0, 1.0]
        elif part_states[part_id] == 2:
            color = [0.0, 0.0, 0.0]
        else:
            enabled = False

        mesh = part_mesh.copy()
        center = mesh.center_mass
        mesh.apply_translation(-center)
        R = Rotation.from_rotvec(q[part_id * 6 + 3: part_id * 6 + 6])
        v = q[part_id * 6: part_id * 6 + 3]
        T = np.identity(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = v + center
        mesh.apply_transform(T)

        # faces
        obj = ps.register_surface_mesh(f"part_{part_id}", mesh.vertices, mesh.faces,
                                       color=color, enabled=enabled)
        # wireframe
        sharp = mesh.face_adjacency_angles > np.radians(30)
        edges = mesh.face_adjacency_edges[sharp]
        if edges is not []:
            obj = ps.register_curve_network(f"part_wire_{part_id}",
                                            nodes=mesh.vertices,
                                            edges=edges,
                                            color=(0, 0, 0), enabled=enabled)
            obj.set_radius(0.002, False)


def draw_assembly(parts: list[Trimesh],
                  part_states: np.ndarray = None):
    for part_id, part_mesh in enumerate(parts):
        enabled = True
        if part_states is None or part_states[part_id] == 1:
            color = [1.0, 1.0, 1.0]
        elif part_states[part_id] == 2:
            color = [0.0, 0.0, 0.0]
        else:
            color = [1.0, 1.0, 1.0]
            enabled = False

        # faces
        ps.register_surface_mesh(f"part_{part_id}", part_mesh.vertices, part_mesh.faces, color=color, enabled=enabled)

        # wireframe
        sharp = part_mesh.face_adjacency_angles > np.radians(30)
        edges = part_mesh.face_adjacency_edges[sharp]
        if edges is not []:
            obj = ps.register_curve_network(f"part_wire_{part_id}",
                                            nodes=part_mesh.vertices,
                                            edges=edges,
                                            color=(0, 0, 0), enabled=enabled)
            obj.set_radius(0.002, False)


def draw_assembly_batches(parts: list[Trimesh],
                          batch_part_states: np.ndarray,
                          batch_stability: np.ndarray = None,
                          boundary_parts: list = [],
                          scale=1.1):
    n_batch = batch_part_states.shape[0]
    n_col = math.ceil(math.sqrt(n_batch))

    assembly_mesh = Scene()
    for part_id, part_mesh in enumerate(parts):
        assembly_mesh.add_geometry(part_mesh)
    assembly_mesh.to_geometry()
    box = assembly_mesh.bounding_box
    offset = max(box.extents[0], box.extents[1]) * scale

    scene = Scene()
    colors = []
    for batch_id, part_states in enumerate(batch_part_states):
        i_row = batch_id // n_col
        i_col = batch_id % n_col
        t = np.array([i_row * offset, i_col * offset, 0])
        stability = batch_stability[batch_id] if batch_stability is not None else 1
        for part_id, part_mesh in enumerate(parts):
            if part_states[part_id] == 1:
                if stability == -1:
                    color = np.array([[1.0, 0.4, 0.4]]).repeat(part_mesh.faces.shape[0], axis=0)
                elif stability == 0:
                    color = np.array([[0.4, 0.4, 1.0]]).repeat(part_mesh.faces.shape[0], axis=0)
                else:
                    color = np.array([[0.4, 1.0, 0.4]]).repeat(part_mesh.faces.shape[0], axis=0)
            elif part_states[part_id] == 2:
                if part_id in boundary_parts:
                    color = np.array([[0.0, 0.0, 0.0]]).repeat(part_mesh.faces.shape[0], axis=0)
                else:
                    color = np.array([[0.2, 0.2, 0.2]]).repeat(part_mesh.faces.shape[0], axis=0)
            else:
                continue
            new_mesh = Trimesh(part_mesh.vertices, part_mesh.faces)
            new_mesh.apply_translation(t)
            scene.add_geometry(geometry=new_mesh)
            colors.append(color)

    # faces
    scene_mesh = scene.to_geometry()
    colors = np.vstack(colors)
    assembly = ps.register_surface_mesh(f"assembly",
                                        scene_mesh.vertices,
                                        scene_mesh.faces)
    assembly.add_color_quantity("face", colors, defined_on="faces", enabled=True)
    # wireframe
    sharp = scene_mesh.face_adjacency_angles > np.radians(30)
    edges = scene_mesh.face_adjacency_edges[sharp]
    if edges is not []:
        obj = ps.register_curve_network(f"assembly_wire",
                                        nodes=scene_mesh.vertices,
                                        edges=edges,
                                        color=(0, 0, 0))
        obj.set_radius(0.002, False)


def draw_contacts(contacts: list[dict],
                  part_states: np.ndarray,
                  enable=False):
    scene = Scene()
    points = []
    normals = []
    xaxes = []
    yaxes = []
    for contact in contacts:
        iA = contact['iA']
        iB = contact['iB']
        if part_states[iA] > 0 and part_states[iB] > 0:
            if 'mesh' in contact:
                scene.add_geometry(geometry=contact['mesh'])
                points.extend(contact["mesh"].vertices)
                normals.extend([contact['plane'][:3, 2] for pt in contact["mesh"].vertices])
                xaxes.extend([contact['plane'][:3, 0] for pt in contact["mesh"].vertices])
                yaxes.extend([contact['plane'][:3, 1] for pt in contact["mesh"].vertices])

    contact_mesh = scene.to_geometry()
    contact_obj = ps.register_surface_mesh(f"contacts", contact_mesh.vertices, contact_mesh.faces)
    contact_obj.set_enabled(enable)

    points = np.array(points)
    normals = np.array(normals)
    xaxes = np.array(xaxes)
    yaxes = np.array(yaxes)
    if points.shape[0] != 0:
        ps_cloud = ps.register_point_cloud(f"contact_points", points, radius=0.01, color=(0.1, 0.4, 0.4),
                                           enabled=False)
        ps_cloud.add_vector_quantity(f"normal", normals, radius=0.02, length=0.05, color=(0.2, 0.5, 0.5),
                                     enabled=False)
        ps_cloud.add_vector_quantity(f"x-axes", xaxes, radius=0.02, length=0.05, color=(0.5, 0.2, 0.5),
                                     enabled=False)
        ps_cloud.add_vector_quantity(f"y-axes", yaxes, radius=0.02, length=0.05, color=(0.5, 0.5, 0.2),
                                     enabled=False)


def draw_gripper(frame: np.ndarray,
                 open_width: float,
                 name="pick_gripper"):
    hand = trimesh.load_mesh(RESOURCE_DIR + "/gripper/hand_camera.obj")
    finger_left = trimesh.load_mesh(RESOURCE_DIR + "/gripper/finger.obj")
    finger_right = trimesh.load_mesh(RESOURCE_DIR + "/gripper/finger.obj")

    finger_left.apply_translation([0, 0.0584, -open_width / 2])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    T[:3, 3] = np.array([0, 0.0584, open_width / 2])
    finger_right.apply_transform(T)

    tcp = np.eye(4)
    tcp[:3, 3] = [0, 0, -0.1034]
    tcp[:3, 1] = [0, 0, 1]
    tcp[:3, 2] = [0, 1, 0]
    tcp[:3, 0] = [-1, 0, 0]

    sphere_centers, sphere_radius = get_gripper_spheres(np.array([open_width]))
    sphere_centers = sphere_centers.reshape(-1, 3)
    sphere_radius = sphere_radius.reshape(-1)
    sphere_cloud = trimesh.points.PointCloud(vertices=sphere_centers)

    for obj in [hand, finger_left, finger_right]:
        obj.apply_transform(tcp)

    for obj in [hand, finger_left, finger_right, sphere_cloud]:
        obj.apply_transform(frame)

    gripper_scene = trimesh.Scene()
    for obj in [hand, finger_left, finger_right]:
        gripper_scene.add_geometry(obj)
    gripper_mesh = gripper_scene.to_mesh()

    ps.register_surface_mesh(f"{name}", gripper_mesh.vertices, gripper_mesh.faces)
    pcl = ps.register_point_cloud(f"{name}_spheres", sphere_cloud.vertices, enabled=False)
    pcl.add_scalar_quantity("radius", sphere_radius)
    pcl.set_point_radius_quantity("radius", autoscale=False)
