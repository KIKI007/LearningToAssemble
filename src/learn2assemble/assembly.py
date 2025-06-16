from trimesh import Trimesh
from trimesh.collision import CollisionManager
import trimesh
import torch
import numpy as np

import shapely
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely import geometry
from types import SimpleNamespace

from learn2assemble import set_default

def shrink_or_swell_shapely_polygon(my_polygon, factor=0.10, swell=False):
    ''' returns the shapely polygon which is smaller or bigger by passed factor.
        If swell = True , then it returns bigger polygon, else smaller '''

    if isinstance(my_polygon, Polygon):
        xs = list(my_polygon.exterior.coords.xy[0])
        ys = list(my_polygon.exterior.coords.xy[1])
        if len(xs) > 0:
            x_center = 0.5 * min(xs) + 0.5 * max(xs)
            y_center = 0.5 * min(ys) + 0.5 * max(ys)
            min_corner = geometry.Point(min(xs), min(ys))
            max_corner = geometry.Point(max(xs), max(ys))
            center = geometry.Point(x_center, y_center)
            shrink_distance = center.distance(min_corner) * factor

            if swell:
                my_polygon_resized = my_polygon.buffer(shrink_distance)  # expand
            else:
                my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink
            return my_polygon_resized

    return my_polygon

def compute_contacts_between_two_parts(partA: Trimesh,
                                       partB: Trimesh,
                                       dist_tol: float = 1E-3,
                                       nrm_tol: float = 1E-3,
                                       area_tol: float = 1E-5,
                                       simplify_tol: float = 1E-4,
                                       shrink_ratio: float = 0.0):

    def angle(nA, nB):
        return np.arccos(
            np.clip(a=np.dot(nA, nB), a_min=-1.0, a_max=1.0))

    def from_shapely_to_numpy(data, simplify_tol):
        if isinstance(data, Polygon):
            data = data.simplify(simplify_tol, preserve_topology=False)
            xx, yy = data.exterior.coords.xy
            xx = xx.tolist()[:-1]
            yy = yy.tolist()[:-1]
            if len(xx) > 0 and len(yy) > 0:
                contact_points = np.zeros((len(xx), 3), dtype=np.float64)
                contact_points[:, 0] = xx
                contact_points[:, 1] = yy
                return contact_points
        return None

    def insert_contacts(contacts, face_nrm, face_center, project_face_contact_points, projection):
        for contact in contacts:
            angle_rad = angle(face_nrm, contact["normal"])
            dist = abs(np.dot((face_center - contact["center"]), face_nrm))
            if angle_rad < nrm_tol and dist < dist_tol:
                # two contact can have different projection matrix
                # this part aims to use the projection from the stored contact
                face_contact_points = trimesh.transformations.transform_points(project_face_contact_points,
                                                                               np.linalg.inv(projection))
                project_face_contact_points = trimesh.transformations.transform_points(face_contact_points,
                                                                                       np.linalg.inv(
                                                                                           contact["inv_proj"]))
                contact["proj_points"] = np.vstack([contact["proj_points"], project_face_contact_points])
                return

        contacts.append({
            "normal": face_nrm,
            "center": face_center,
            "proj_points": project_face_contact_points,
            "inv_proj": np.linalg.inv(projection)
        })
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    floatType = torch.float32
    normalA = torch.tensor(partA.face_normals, device=device, dtype=floatType)
    normalB = torch.tensor(partB.face_normals, device=device, dtype=floatType)
    centerA = torch.tensor(partA.triangles_center, device=device, dtype=floatType)
    centerB = torch.tensor(partB.triangles_center, device=device, dtype=floatType)
    partIDAs, partIDBs = torch.meshgrid(torch.arange(normalA.shape[0]), torch.arange(normalB.shape[0]), indexing="ij")
    angle_ij = torch.arccos(
        torch.clip(torch.einsum("ijk, ijk->ij", normalA[partIDAs, :], -normalB[partIDBs, :]), -1, 1))
    dist_ij = torch.abs(
        torch.einsum("ijk, ijk -> ij", (centerA[partIDAs, :] - centerB[partIDBs, :]), normalA[partIDAs, :]))
    flag = torch.logical_and(angle_ij < nrm_tol, dist_ij < dist_tol)
    face_pairs = flag.nonzero().cpu().numpy()

    contacts = []
    for faceA, faceB in face_pairs:
        face_nrm = normalA[faceA].cpu().numpy()
        face_center = centerA[faceA].cpu().numpy()
        project_T = trimesh.geometry.plane_transform(face_center, face_nrm)
        project_points_A = trimesh.transformations.transform_points(partA.triangles[faceA], project_T)
        project_points_B = trimesh.transformations.transform_points(partB.triangles[faceB], project_T)
        project_poly_A = Polygon(project_points_A)
        project_poly_A = shapely.geometry.polygon.orient(project_poly_A, sign=1.0)
        project_poly_B = Polygon(project_points_B)
        project_poly_B = shapely.geometry.polygon.orient(project_poly_B, sign=1.0)
        project_face_contact_poly = project_poly_A.intersection(project_poly_B)
        project_face_contact_points = from_shapely_to_numpy(project_face_contact_poly, simplify_tol)
        if project_face_contact_poly.area > area_tol and project_face_contact_points is not None:
            insert_contacts(contacts, face_nrm, face_center, project_face_contact_points, project_T)

    # convex_hull
    new_contacts = []
    for contact in contacts:
        rounded_points = np.round(contact["proj_points"], decimals=5)
        unique_points, indices = np.unique(rounded_points, axis=0, return_index=True)
        contact["proj_points"] = contact["proj_points"][indices]
        if contact["proj_points"].shape[0] >= 3:
            contact_points_2d = contact["proj_points"][:, :2]
            mpt = MultiPoint(contact_points_2d)
            convexhull = shrink_or_swell_shapely_polygon(mpt.convex_hull, shrink_ratio)
            cvxhull_points_2d = from_shapely_to_numpy(convexhull, simplify_tol=simplify_tol)
            if cvxhull_points_2d is not None:
                contact_points = trimesh.transformations.transform_points(cvxhull_points_2d, contact["inv_proj"])
                faces = []
                for fid in range(0, len(contact_points) - 2):
                    faces.append([0, fid + 1, fid + 2])
                cvxhull_contact_mesh = trimesh.Trimesh(vertices=contact_points, faces=faces)

                new_contact = {}
                new_contact["mesh"] = cvxhull_contact_mesh
                new_contact["plane"] = contact["inv_proj"]
                new_contacts.append(new_contact)
    return new_contacts


def fast_collision_check_using_convexhull(parts: list[Trimesh],
                                          collision_scale: float = 1.1):
    collision_pairs = []
    n_part = len(parts)
    try:
        collision = CollisionManager()
        for part_id in range(n_part):
            obj = parts[part_id]
            convex = trimesh.convex.convex_hull(obj)
            center = convex.center_mass
            convex.apply_translation(-center)
            convex.apply_scale(collision_scale)
            convex.apply_translation(center)
            collision.add_object(str(part_id), convex)
        pairs = collision.in_collision_internal(return_names=True)[1]
        for pair in pairs:
            collision_pairs.append([int(pair[0]), int(pair[1])])
    except:
        for part_IDA in range(n_part):
            for part_IDB in range(part_IDA + 1, n_part()):
                collision_pairs.append([part_IDA, part_IDB])
    return collision_pairs


def compute_assembly_contacts(parts: list[Trimesh],
                              settings:dict = {}):
    contact_settings = set_default(
        settings,
        "contact_settings",
        {
            "dist_tol": 1E-3,
            "nrm_tol": 1E-3,
            "area_tol": 1E-5,
            "simplify_tol": 1E-4,
            "collision_scale": 1.1,
            "shrink_ratio": 0.0
        }
    )

    n = SimpleNamespace(**contact_settings)
    collision_pairs = fast_collision_check_using_convexhull(parts, n.collision_scale)
    contacts = []
    for [partIDA, partIDB] in collision_pairs:
        two_parts_contacts = compute_contacts_between_two_parts(parts[partIDA],
                                                                parts[partIDB],
                                                                dist_tol=n.dist_tol,
                                                                nrm_tol=n.nrm_tol,
                                                                area_tol=n.area_tol,
                                                                simplify_tol=n.simplify_tol,
                                                                shrink_ratio=n.shrink_ratio)
        for contact in two_parts_contacts:
            contact["iA"] = partIDA
            contact["iB"] = partIDB
        contacts.extend(two_parts_contacts)

    return contacts


def load_assembly_from_files(obj_folder_path):
    '''
    Reading parts' meshes from a folder
    :param obj_folder_path: the folder contains part obj mesh, only files with names {part_id}.obj are parsed
    :return:
    '''
    from os import listdir
    from os.path import isfile, join

    parts_dict = {}
    num_parts = 0
    for f in listdir(obj_folder_path):
        file_path = join(obj_folder_path, f)
        if isfile(file_path) and f[: -4].isdigit() and f[-3:] == 'obj':
            trimesh_obj = trimesh.load(file_path, force='mesh')
            partID = int(f[:-4])
            parts_dict[partID] = trimesh_obj
            num_parts = max(num_parts, partID + 1)

    parts = []
    for part_id in range(num_parts):
        parts.append(parts_dict[part_id])
    return parts


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.render import draw_assembly, init_polyscope, draw_contacts
    import polyscope as ps

    init_polyscope()
    settings = {
        "contact_settings": {
            "dist_tol": 1E-3,
            "nrm_tol": 1E-3,
            "area_tol": 1E-5,
            "simplify_tol": 1E-4,
            "shrink_ratio": 0.1,
            "collision_scale": 1.1
        }
    }
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    part_states = np.ones(len(parts))
    part_states[0] = 2
    part_states[3] = 0
    contacts = compute_assembly_contacts(parts, settings)
    draw_contacts(contacts, part_states, enable=True)
    draw_assembly(parts, part_states)
    ps.show()
