import polyscope as ps
from trimesh import Scene, Trimesh
import numpy as np

def init_polyscope():
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(0)

def draw_assembly(parts: list[Trimesh], part_states: np.ndarray = None):
    part_group = ps.create_group("assembly")
    wire_group = ps.create_group("assembly_wire")

    for part_id, part_mesh in enumerate(parts):
        if part_states is None or part_states[part_id] == 1:
            color = [1.0, 1.0, 1.0]
        elif part_states[part_id] == 2:
            color = [0.0, 0.0, 0.0]
        else:
            continue

        # faces
        obj = ps.register_surface_mesh(f"part_{part_id}", part_mesh.vertices, part_mesh.faces, color=color)
        obj.add_to_group(part_group)

        # wireframe
        sharp = part_mesh.face_adjacency_angles > np.radians(30)
        edges = part_mesh.face_adjacency_edges[sharp]
        if edges is not []:
            obj = ps.register_curve_network(f"part_wire_{part_id}",
                                            nodes=part_mesh.vertices,
                                            edges=edges,
                                            color=(0, 0, 0))
            obj.add_to_group(wire_group)
            obj.set_radius(0.002, False)

    part_group.set_hide_descendants_from_structure_lists(True)
    wire_group.set_hide_descendants_from_structure_lists(True)


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
