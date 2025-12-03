import math

import torch
import trimesh.util
from trimesh.primitives import Box
from trimesh import Trimesh
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

plane_normals = np.array([[1, 0, 0],
               [-1, 0, 0],
               [0, 1, 0],
               [0, -1, 0],
               [0, 0, 1],
               [0, 0, -1]], dtype=int)

def create_contact_masks(part_masks: np.ndarray):
    npart, nx, ny, nz = part_masks.shape
    contact_masks = np.zeros((6, nx, ny, nz), dtype=bool)
    n = np.array([nx, ny, nz], dtype=int)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for id in range(plane_normals.shape[0]):
                    if part_masks[:, ix, iy, iz].any():
                        part_id = part_masks[:, ix, iy, iz].nonzero()[0][0]
                        dpt = np.array([ix, iy, iz], dtype=int) + plane_normals[id, :]
                        if (dpt >= 0).all() and (dpt < n).all():
                            contact_masks[id, ix, iy, iz] = (part_masks[part_id, dpt[0], dpt[1], dpt[2]] != True)
                        else:
                            contact_masks[id, ix, iy, iz] = True
    return contact_masks

def create_voxel_masks(parts: list[Trimesh], voxel_size = 0.25):
    # contacts
    # voxels
    # part_masks
    if len(parts) > 0:
        bnd = parts[0].bounds.copy()
        for part in parts:
            bnd[0, :] = np.minimum(bnd[0, :], part.bounds[0, :])
            bnd[1, :] = np.maximum(bnd[1, :], part.bounds[1, :])
        n = np.round((bnd[1, :] - bnd[0, :]) / voxel_size)
        nx, ny, nz = n.astype(int)
        part_masks = np.zeros((len(parts), nx, ny, nz), dtype=bool)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    pt = bnd[0, :] + np.array([voxel_size * (0.5 + ix), voxel_size * (0.5 + iy), voxel_size * (0.5 + iz)])
                    pt = pt.reshape(1, 3)
                    for part_id, part in enumerate(parts):
                        if part.contains(pt).any():
                            part_masks[part_id, ix, iy, iz] = True
        contact_masks = create_contact_masks(part_masks)
        part_masks = torch.tensor(part_masks, dtype=torch.float32, device='cuda')
        contact_masks = torch.tensor(contact_masks, dtype=torch.float32, device='cuda')

        return part_masks, contact_masks
    else:
        return None

def draw_voxel_meshes(voxel_feat: np.ndarray, part_masks: np.ndarray, voxel_size = 0.05):
    n_part, nx, ny, nz = part_masks.shape
    meshes = [[] for part_id in range(n_part)]
    colors = [None for part_id in range(n_part)]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for id in range(plane_normals.shape[0]):
                    if voxel_feat[id, ix, iy, iz]:
                        box = Box(extents=[voxel_size, voxel_size, voxel_size])
                        t = np.array([voxel_size * (0.5 + ix), voxel_size * (0.5 + iy), voxel_size * (0.5 + iz)])
                        box.apply_translation(t)
                        flag = np.linalg.norm(box.face_normals -  plane_normals[id, :], axis = -1) < 1E-6
                        mesh = box.to_mesh()
                        mesh.faces = mesh.faces[flag, :]
                        part_id = part_masks[:, ix, iy, iz].nonzero()[0][0]
                        meshes[part_id].append(mesh)
                        if voxel_feat[6, ix, iy, iz] == 1:
                            colors[part_id] = (1, 1, 1)
                        elif voxel_feat[6, ix, iy, iz] == 2:
                            colors[part_id] = (0, 0, 0)

    for part_id in range(n_part):
        if colors[part_id] is not None:
            mesh = trimesh.util.concatenate(meshes[part_id])
            mesh.merge_vertices(merge_tex=True, digits_vertex=5)
            ps.register_surface_mesh(f"part_{part_id}", mesh.vertices, mesh.faces, color = colors[part_id])
            sharp = mesh.face_adjacency_angles > np.radians(40)
            edges = mesh.face_adjacency_edges[sharp]
            lines = mesh.vertices[edges].reshape(-1, 3)
            edges = np.stack([2 * np.arange(len(lines)//2), 2 * np.arange(len(lines)//2) + 1]).T
            ps.register_curve_network(f"edge_{part_id}", nodes=lines, edges=edges, color = (0, 0, 0))

def get_voxel_features_2d(part_states: np.ndarray, part_masks: torch.tensor, contact_masks: torch.tensor, grid = 16):
    nbatch = part_states.shape[0]
    npart, nx, _, nz = part_masks.shape

    # contact features
    part_states = torch.tensor(part_states, device='cuda', dtype=torch.float32)
    flag = torch.einsum('cijk, bc -> bijk', part_masks, part_states)
    contact_feat = contact_masks[None, :].repeat_interleave(nbatch, dim=0)
    flag = flag[:, None, :, :].repeat_interleave(6, dim=1)
    contact_feat = ((contact_feat * flag) > 0).type(floatType)
    pw = torch.tensor([1, 2, 4, 8, 16, 32], dtype=floatType, device=device)
    contact_feat = torch.einsum('bcijk, c -> bijk', contact_feat, pw).unsqueeze(1)

    # voxel features
    state = part_states[:, :, None, None, None]
    state = state.repeat_interleave(part_masks.shape[1], dim=2)
    state = state.repeat_interleave(part_masks.shape[2], dim=3)
    state = state.repeat_interleave(part_masks.shape[3], dim=4)
    part_feats = part_masks[None, :].repeat_interleave(nbatch, dim=0)
    voxel_state = part_feats * state
    voxel_state = torch.sum(voxel_state, axis=1, keepdims=True)

    feats = torch.concatenate([contact_feat, voxel_state], axis=1)
    feats = feats.squeeze(3)
    pad_feats = torch.zeros((nbatch, feats.shape[1], grid, grid), device='cuda', dtype=floatType)
    pad_feats[:, :, :nx, :nz] = feats

    return pad_feats

def get_voxel_features(part_states: np.ndarray, part_masks: torch.tensor, contact_masks: torch.tensor, grid = 16):
    nbatch = part_states.shape[0]
    npart, nx, ny, nz = part_masks.shape

    # contact features
    part_states = torch.tensor(part_states, device='cuda', dtype=torch.float32)
    flag = torch.einsum('cijk, bc -> bijk', part_masks, part_states)
    contact_feat = contact_masks[None, :].repeat_interleave(nbatch, dim=0)
    flag = flag[:, None, :, :].repeat_interleave(6, dim=1)
    contact_feat = ((contact_feat * flag) > 0).type(floatType)
    pw = torch.tensor([1, 2, 4, 8, 16, 32], dtype=floatType, device=device)
    contact_feat = torch.einsum('bcijk, c -> bijk', contact_feat, pw).unsqueeze(1)

    # voxel features
    state = part_states[:, :, None, None, None]
    state = state.repeat_interleave(part_masks.shape[1], dim=2)
    state = state.repeat_interleave(part_masks.shape[2], dim=3)
    state = state.repeat_interleave(part_masks.shape[3], dim=4)
    part_feats = part_masks[None, :].repeat_interleave(nbatch, dim=0)
    voxel_state = part_feats * state
    voxel_state = torch.sum(voxel_state, axis = 1, keepdims = True)

    feats = torch.concatenate([contact_feat, voxel_state], axis = 1)
    pad_feats = torch.zeros((nbatch, feats.shape[1], grid, grid, grid), device='cuda', dtype=floatType)
    pad_feats[:, :, :nx, :ny, :nz] = feats

    return pad_feats


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
    from learn2assemble.assembly import load_assembly_from_files
    from learn2assemble.render import init_polyscope
    import polyscope as ps
    init_polyscope()

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    part_masks, contact_masks = create_voxel_masks(parts, 0.25)

    part_states = np.ones((2, len(parts)), dtype=int)
    part_states[0, 3] = 0
    part_states[0, 2] = 2
    part_states[1, 2] = 0
    nbatch = part_states.shape[0]

    voxel_feats, part_masks = get_voxel_features(part_states, part_masks, contact_masks)

    draw_voxel_meshes(voxel_feats[0, :], part_masks)
    ps.show()