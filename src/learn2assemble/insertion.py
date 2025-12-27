import numpy
import numpy as np
from trimesh import Trimesh
import trimesh
from learn2assemble import *
import polyscope as ps
import warp as wp
import time


@wp.kernel
def insertion_kernel(mesh_ids: wp.array(dtype=wp.uint64),
                     pairs: wp.array(dtype=wp.vec2i),
                     points: wp.array2d(dtype=wp.vec3f),
                     drts: wp.array(dtype=wp.vec3f),
                     dist: wp.array3d(dtype=wp.float32),
                     max_dist: wp.float32,
                     n_dt_sample: wp.int32):
    i, j, k = wp.tid()
    drt = drts[i]
    iA = pairs[j][0]
    iB = pairs[j][1]
    pt = points[iA][k]
    mesh_id = mesh_ids[iB]

    for it in range(n_dt_sample):
        dt = float(it) / float(n_dt_sample - 1)
        pt_offset = pt + drt * dt
        query = wp.mesh_query_point_sign_normal(mesh_id, pt_offset, max_dist)
        if query.result:
            on_surface_pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            sign_dist = wp.length(pt_offset - on_surface_pt) * query.sign
            if dist[i, j, k] > sign_dist:
                dist[i, j, k] = sign_dist


def compute_insertion_table(parts, settings):
    # drt x part x part
    # i     j        k
    # table[i, j, k] = True if move j-th part along i-th direction does not cause collisions with k-th part

    insertion_settings = update_default_settings(settings, "insertion",
                                       {
                                           "n_surface_sample": 1000,
                                           "drt_length": 1,
                                           "max_dist": 1,
                                           "collision_eps": 0.01,
                                           "type": "orthogonal",
                                           "n_dt_sample": 10,
                                       })

    n_surface_sample = insertion_settings["n_surface_sample"]
    drt_length = insertion_settings["drt_length"]
    max_dist = insertion_settings["max_dist"]
    collision_eps = insertion_settings["collision_eps"]
    n_dt_sample = insertion_settings["n_dt_sample"]

    if insertion_settings["type"] == "orthogonal":
        drts = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]) * drt_length
    else:
        return None

    points = []
    wp_parts = []
    wp_parts_id = []
    for part_id, part in enumerate(parts):
        pts = trimesh.sample.sample_surface(part, n_surface_sample, None, seed = 0)
        points.append(pts[0])

        wp_part = wp.Mesh(
            points=wp.array(part.vertices.flatten(), dtype=wp.vec3f),
            indices=wp.array(part.faces.flatten(), dtype=int),
        )
        wp_parts.append(wp_part)
        wp_parts_id.append(wp_part.id)

    points = np.stack(points).astype(np.float32)

    wp_points = wp.from_numpy(points, device='cuda', dtype=wp.vec3f)
    wp_parts_id = wp.array(wp_parts_id, device='cuda', dtype=wp.uint64)
    wp_drts = wp.from_numpy(drts, device='cuda', dtype=wp.vec3f)

    ndrt = drts.shape[0]
    npart = len(parts)
    pairs = []
    for i in range(npart):
        for j in range(npart):
            if i != j:
                pairs.append([i, j])
    wp_pairs = wp.from_numpy(pairs, device='cuda', dtype=wp.vec2i)

    wp_dist = wp.ones((drts.shape[0], wp_pairs.shape[0], n_surface_sample), device='cuda', dtype=wp.float32) * 1E10

    wp.launch(insertion_kernel, (drts.shape[0], wp_pairs.shape[0], n_surface_sample),
              inputs=[wp_parts_id, wp_pairs, wp_points, wp_drts, wp_dist, max_dist, n_dt_sample])

    dist = wp_dist.numpy()
    table = numpy.all(dist > -collision_eps, axis=-1)
    new_table = np.ones((ndrt, npart, npart), dtype=bool)
    for idrt in range(drts.shape[0]):
        for ipair, pair in enumerate(pairs):
            new_table[idrt, pair[0], pair[1]] = table[idrt, ipair]
    return new_table, drts


def check_future_insertability(current_states: np.array, table: np.array):
    timer = time.perf_counter()

    # check assemblability
    to_install_part_flag = (current_states == 0)
    exist_part_flag = (current_states > 0)
    assemblability_flag = [False for batch_id in range(current_states.shape[0])]
    for batch_id in range(current_states.shape[0]):
        to_install_parts = to_install_part_flag[batch_id].nonzero()[0]
        flag = True
        for part_id in to_install_parts:
            new_table = table[:, part_id, exist_part_flag[batch_id]].astype(np.int32)
            new_table = new_table.all(axis=-1)
            flag = np.logical_and(flag, new_table.any())
        assemblability_flag[batch_id] = flag
    # print("check insertion time", time.perf_counter() - timer)
    return assemblability_flag

def compute_insertion_masks(current_states: np.array, boundart_part_ids: np.array, table: np.array):
    held_flag = (current_states == 2)
    held_flag[:, boundart_part_ids] = False
    insertion_masks = held_flag.copy()
    exist_part_flag = (current_states > 0)
    for batch_id in range(current_states.shape[0]):
        to_removed_parts = held_flag[batch_id].nonzero()[0]
        for part_id in to_removed_parts:
            new_table = table[:, part_id, exist_part_flag[batch_id]].astype(np.int32)
            new_table = new_table.all(axis=-1)
            insertion_masks[batch_id, part_id] = new_table.any()
    return insertion_masks

def compute_insertion_drt(part_id, current_state: np.array, table: np.array, drts):
    exist_part_flag = (current_state > 0)
    exist_part_flag[part_id] = False
    new_table = table[:, part_id, exist_part_flag]
    new_table = new_table.all(axis=-1)
    indices = new_table.nonzero()[0]
    if indices.shape[0] > 0:
        return drts[indices, :]
    else:
        return None


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import polyscope as ps

    init_polyscope()
    wp.init()
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-7")

    table, drts = compute_insertion_table(parts, default_settings)
    part_states = np.ones((1, len(parts)))
    part_states[0, 24] = 1
    part_states[0, 15] = 1
    part_states[0, 0] = 1
    assemblability_flag = check_future_insertability(part_states, table)
    part_id = 15
    part_drts = compute_insertion_drt(part_id, part_states.reshape(-1), table, drts)

    part_state = part_states.reshape(-1)
    draw_assembly(parts, part_state)

    if part_drts is not None:
        q = np.zeros(len(parts) * 6)
        q[6 * part_id: part_id * 6 + 3] = part_drts[0, :]
        t = 0


        def callback():
            global t
            changed, t = psim.SliderFloat("time", v=t, v_min=0, v_max=1)
            if changed:
                draw_assembly_motion(parts, part_state, q * t)


        ps.set_user_callback(callback)
    ps.show()
