import numpy
import numpy as np
from trimesh import Trimesh
import trimesh
from learn2assemble import *
import warp as wp
import time
from scipy.spatial.transform import Rotation


@wp.kernel
def sample_grasp_poses_kernel(mesh_id: wp.uint64,
                              face_ids: wp.array(dtype=wp.int32),
                              wp_frames: wp.array3d(dtype=wp.float32),
                              wp_open_widths: wp.array(dtype=wp.float32),
                              wp_flag: wp.array(dtype=wp.bool),
                              gripper_width: wp.float32,
                              seed: wp.int32):
    tid = wp.tid()
    state = wp.uint32(seed + tid)

    # Sample the location in that triangle using random barycentric coordinates.
    bary = wp.sample_triangle(state)
    pt0 = wp.mesh_eval_position(mesh_id,
                                face_ids[tid],
                                bary[0],
                                bary[1])

    nrm0 = wp.mesh_eval_face_normal(mesh_id, face_ids[tid])
    pt1 = pt0 - nrm0 * gripper_width
    query0 = wp.mesh_query_ray(mesh_id, pt1, -nrm0, max_t=1.0)
    query1 = wp.mesh_query_ray(mesh_id, pt1, nrm0, max_t=gripper_width)

    if not query0.result and query1.result:
        pt2 = wp.mesh_eval_position(mesh_id, query1.face, query1.u, query1.v)
        nrm2 = wp.mesh_eval_face_normal(mesh_id, query1.face)

        if wp.dot(nrm2, -nrm0) >= 0.99:

            yaxis = nrm0
            zaxis = wp.sample_unit_sphere(state)
            zaxis = wp.cross(yaxis, zaxis)
            zaxis = zaxis / wp.length(zaxis)
            xaxis = wp.cross(yaxis, zaxis)

            for dim in range(3):
                wp_frames[tid, dim, 0] = xaxis[dim]
                wp_frames[tid, dim, 1] = yaxis[dim]
                wp_frames[tid, dim, 2] = zaxis[dim]
                wp_frames[tid, dim, 3] = (pt0[dim] + pt2[dim]) / 2.0
            wp_frames[tid, 3, 3] = 1.0

            wp_open_widths[tid] = wp.length(pt2 - pt0)
            wp_flag[tid] = True


def remove_invalid(wp_frames: wp.array3d(dtype=wp.float32),
                   wp_open_widths: wp.array(dtype=wp.float32),
                   flag: np.ndarray[bool]):
    if wp_frames is not None:
        frames = wp_frames.numpy()
        frames = frames[flag, :, :]
        wp_frames = wp.from_numpy(frames, dtype=wp.float32, requires_grad=False)
    if wp_open_widths is not None:
        open_widths = wp_open_widths.numpy()
        open_widths = open_widths[flag]
        wp_open_widths = wp.from_numpy(open_widths, dtype=wp.float32, requires_grad=False)
    return wp_frames, wp_open_widths


def sample_grasp_frames(mesh,
                        wp_mesh_id,
                        num_of_samples: int = 100,
                        gripper_width: float = 0.08,
                        seed: int = 0):
    wp.set_device("cuda")

    # sample frames
    np.random.seed(seed)
    face_ids = np.random.choice(a=np.arange(mesh.faces.shape[0], dtype=np.int32),
                                size=num_of_samples,
                                replace=True,
                                p=mesh.area_faces / mesh.area_faces.sum())
    face_ids.sort()
    wp_face_ids = wp.from_numpy(face_ids, requires_grad=False)

    wp_frames = wp.zeros(shape=(num_of_samples, 4, 4), dtype=wp.float32, device="cuda")
    wp_open_widths = wp.zeros(shape=num_of_samples, dtype=wp.float32, device="cuda")
    wp_flag = wp.zeros(shape=num_of_samples, dtype=wp.bool, device="cuda")
    wp.launch(sample_grasp_poses_kernel,
              dim=num_of_samples,
              inputs=[wp_mesh_id, wp_face_ids, wp_frames, wp_open_widths, wp_flag, gripper_width, seed],
              device='cuda')

    # remove invalid
    flag = wp_flag.numpy()
    wp_frames, wp_open_widths = remove_invalid(wp_frames, wp_open_widths, flag)

    return wp_frames, wp_open_widths


@wp.kernel
def check_finger_contact_kernel(mesh_id: wp.uint64,
                                wp_frames: wp.array3d(dtype=wp.float32),
                                wp_open_widths: wp.array(dtype=wp.float32),
                                wp_points: wp.array3d(dtype=wp.float32),
                                wp_finger_dists: wp.array2d(dtype=wp.float32),
                                pad: wp.vec2f):
    # pad : width, height
    # z: height
    # x: width
    # y: between two fingers

    i, j = wp.tid()
    sgn_x = -1.0
    sgn_y = -1.0
    sgn_z = -1.0
    if j >= 4:
        sgn_y = 1.0
    if j % 4 >= 2:
        sgn_x = 1.0
    if j % 2 == 1:
        sgn_z = 1.0

    xd = wp.vec3f(wp_frames[i, 0, 0], wp_frames[i, 1, 0], wp_frames[i, 2, 0])
    yd = wp.vec3f(wp_frames[i, 0, 1], wp_frames[i, 1, 1], wp_frames[i, 2, 1])
    zd = wp.vec3f(wp_frames[i, 0, 2], wp_frames[i, 1, 2], wp_frames[i, 2, 2])
    o = wp.vec3f(wp_frames[i, 0, 3], wp_frames[i, 1, 3], wp_frames[i, 2, 3])

    pt = o + pad[0] * xd * sgn_x / 2.0 + pad[1] * zd * sgn_z / 2.0 + wp_open_widths[i] / 2.0 * yd * sgn_y
    for dim in range(3):
        wp_points[i, j, dim] = pt[dim]
    query = wp.mesh_query_point_sign_normal(mesh_id, pt, max_dist=wp_open_widths[i])
    if query.result:
        pt1 = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        wp_finger_dists[i, j] = wp.length(pt1 - pt) * query.sign
    else:
        wp_finger_dists[i, j] = -1.0


def check_finger_contact(wp_mesh_id: wp.uint64,
                         wp_frames: wp.array3d(dtype=wp.float32),
                         wp_open_widths: wp.array(dtype=wp.float32),
                         pad_width=0.02,
                         pad_height=0.02):
    num_of_samples = wp_frames.shape[0]
    # evaluate frames
    wp_points = wp.zeros(shape=(num_of_samples, 8, 3), dtype=wp.float32, device="cuda")
    wp_pad_dists = wp.zeros(shape=(num_of_samples, 8), dtype=wp.float32, device="cuda")
    wp_pad = wp.vec2f(pad_width, pad_height)
    wp.launch(check_finger_contact_kernel,
              dim=[num_of_samples, 8],
              inputs=[wp_mesh_id, wp_frames, wp_open_widths, wp_points, wp_pad_dists, wp_pad])

    flag = (np.abs(wp_pad_dists.numpy()) < 1E-6).all(axis=1)
    return remove_invalid(wp_frames, wp_open_widths, flag)


@wp.kernel
def evaluate_sphere_collisions_kernel(mesh_ids: wp.array(dtype=wp.uint64),
                                      wp_frames: wp.array(dtype=wp.mat44f),
                                      wp_sph_center: wp.array2d(dtype=wp.vec3f),
                                      wp_sph_radius: wp.array2d(dtype=wp.float32),
                                      wp_sph_dists: wp.array3d(dtype=wp.float32),
                                      check_ground: wp.bool):
    i, j, k = wp.tid()
    xframe = wp_frames[i]
    sph = wp_sph_center[i, j]
    sph = wp.transform_point(xframe, sph)
    mesh_id = mesh_ids[k]
    query = wp.mesh_query_point_sign_normal(mesh_id, sph, max_dist=1.0)
    if query.result:
        pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        wp_sph_dists[i, j, k] = wp.length(sph - pt) * query.sign - wp_sph_radius[i, j]

    if check_ground and sph[2] - wp_sph_radius[i, j] < 0.0:
        wp_sph_dists[i, j, k] = -1.0


def check_grasp_collisions(wp_mesh_ids: wp.array(dtype=wp.uint64),
                           wp_frames: wp.array(dtype=wp.mat44f),
                           wp_open_widths: wp.array(dtype=wp.float32),
                           check_ground=False):
    sph_centers, sph_radius = get_gripper_spheres(wp_open_widths.numpy())
    num_of_samples = wp_frames.shape[0]
    num_of_spheres = sph_radius.shape[1]
    num_of_parts = wp_mesh_ids.shape[0]

    wp_sph_centers = wp.from_numpy(sph_centers, dtype=wp.vec3f, requires_grad=False)
    wp_sph_radius = wp.from_numpy(sph_radius, dtype=wp.float32, requires_grad=False)
    wp_sph_dists = wp.zeros(shape=(num_of_samples, num_of_spheres, num_of_parts), dtype=wp.float32, device="cuda")

    wp.launch(evaluate_sphere_collisions_kernel,
              dim=(num_of_samples, num_of_spheres, num_of_parts),
              inputs=[wp_mesh_ids, wp_frames, wp_sph_centers, wp_sph_radius, wp_sph_dists, check_ground])

    sph_dists = wp_sph_dists.numpy()
    flag = (sph_dists > 1E-6).all(axis=1)
    return flag


def compute_grasp_table(parts: list[trimesh.Trimesh], settings: dict):

    grasp_settings = update_default_settings(settings, "grasp",
                            {
                                "n_sample": 1000,
                                "gripper_size": [0.02, 0.02, 0.08],
                                "check_ground": True,
                                "scale": 0.2,
                            })

    scale = grasp_settings["scale"]
    scaled_parts = [Trimesh(part.vertices, part.faces) for part in parts]
    for part in scaled_parts:
        part = part.apply_scale([scale, scale, scale])

    num_of_samples = grasp_settings["n_sample"]
    check_ground = grasp_settings["check_ground"]
    pad_width, pad_height, gripper_width = grasp_settings["gripper_size"]

    wp_parts = []
    wp_parts_id = []
    for part_id, part in enumerate(scaled_parts):
        wp_part = wp.Mesh(
            points=wp.array(part.vertices.flatten(), dtype=wp.vec3f),
            indices=wp.array(part.faces.flatten(), dtype=int),
        )
        wp_parts.append(wp_part)
        wp_parts_id.append(wp_part.id)

    wp_parts_id = wp.array(wp_parts_id, dtype=wp.uint64, device="cuda")

    table = []
    list_grasp_frames = []
    for part_id, part in enumerate(scaled_parts):
        # sample grasp poses
        wp_part_id = wp_parts_id.numpy()[part_id]
        wp_grasp_frames, wp_open_widths = sample_grasp_frames(part, wp_part_id, num_of_samples, gripper_width)

        # check finger contact
        wp_grasp_frames, wp_open_widths = check_finger_contact(wp_part_id, wp_grasp_frames, wp_open_widths, pad_width,
                                                               pad_height)

        # check pick collisions
        grasp_frames = wp_grasp_frames.numpy()
        wp_grasp_frames = wp.from_numpy(grasp_frames, dtype=wp.mat44f, requires_grad=False)
        flag = check_grasp_collisions(wp_parts_id, wp_grasp_frames, wp_open_widths, check_ground)
        valid_flag = (flag[:, part_id] == True)

        grasp_frames = grasp_frames[valid_flag, :]
        flag = flag[valid_flag, :]

        table.append(flag)
        list_grasp_frames.append(grasp_frames)

    return table, list_grasp_frames, scaled_parts


def check_graspability(current_states, boundary_part_ids, table):
    # check assemblability
    to_install_parts_flag = (current_states == 0)
    exist_parts_flag = (current_states > 0)
    held_parts_flag = (current_states == 2)
    held_parts_flag[:, boundary_part_ids] = False

    timer = time.perf_counter()
    graspability_flag = []
    for batch_id in range(current_states.shape[0]):
        to_install_parts = to_install_parts_flag[batch_id].nonzero()[0]
        held_parts = held_parts_flag[batch_id].nonzero()[0]
        to_check_parts = np.hstack([to_install_parts, held_parts])
        flag = True
        for part_id in to_check_parts:
            new_table = table[part_id][:, exist_parts_flag[batch_id]]
            flag = np.logical_and(flag, np.all(new_table, axis=-1).any())
        graspability_flag.append(flag)
    graspability_flag = np.array(graspability_flag)
    # print("graspability time", time.perf_counter() - timer)
    return graspability_flag


def compute_grasp_frame(part_id, current_state, table, grasp_frames):
    exist_parts_flag = (current_state > 0)
    new_table = table[part_id][:, exist_parts_flag]
    indices = new_table.all(axis=-1).nonzero()[0]
    if indices.shape[0] > 0:
        return grasp_frames[part_id][indices, :]
    else:
        return None


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import polyscope as ps

    init_polyscope()
    wp.init()
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")

    table, grasp_frames, scaled_parts = compute_grasp_table(parts, settings)
    table, grasp_frames2, scaled_parts = compute_grasp_table(parts, settings)

    print(grasp_frames2[1] - grasp_frames[1])

    part_states = np.ones((1, len(parts)))
    part_states[0, 3] = 2
    part_states[0, 5] = 1
    print(check_graspability(part_states, [0], table))
    current_state = part_states[0, :]
    draw_assembly(scaled_parts, current_state)
    held_parts = (current_state == 2).nonzero()[0]
    for robot_id, part_id in enumerate(held_parts):
        frames = compute_grasp_frame(part_id, current_state, table, grasp_frames)
        if frames is not None:
            draw_gripper(frames[0, :], 0.05, f"robot {robot_id}")

    ps.show()
