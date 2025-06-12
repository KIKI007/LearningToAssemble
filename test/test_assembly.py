import os
from learn2assemble.assembly import *
slope_test_folder = os.path.abspath(os.path.dirname(__file__) + "/data/slope")

def test_load_assembly_from_files():
    parts = load_assembly_from_files(slope_test_folder)
    assert len(parts) == 2

def test_fast_collision_check_using_convexhull():
    parts = load_assembly_from_files(slope_test_folder)

    collision_pairs = fast_collision_check_using_convexhull(parts, 1.1)
    assert  collision_pairs == [[0, 1]]

    collision_pairs = fast_collision_check_using_convexhull(parts, 0.9)
    assert collision_pairs == []

    parts[0].apply_translation((0, 0, 0.01))
    collision_pairs = fast_collision_check_using_convexhull(parts, 1.1)
    assert collision_pairs == [[0, 1]]

    parts[0].apply_translation((0, 0, 1))
    collision_pairs = fast_collision_check_using_convexhull(parts, 1.1)
    assert collision_pairs == []

def test_compute_contacts_between_two_parts():
    parts = load_assembly_from_files(slope_test_folder)

    contacts = compute_contacts_between_two_parts(parts[0], parts[1])
    assert len(contacts) == 1

    contact = contacts[0]
    points = contact["mesh"].vertices
    assert points.shape[0] == 4

    # check plane normal
    plane = contact["plane"]
    normal = plane[:3, 2]
    assert np.linalg.norm(np.cross(np.array([1, 0, -1]), normal)) < 1e-6
    assert np.dot(np.array([1, 0, -1]), normal) > 0.99

    # check plane orthonormal
    assert np.linalg.norm(plane[:3, :3].T @ plane[:3, :3] - np.eye(3)) < 1e-6

    for pt in points:
        # check pt on plane
        assert np.abs(np.dot(normal, (pt - plane[:3, 3]))) < 1e-6

        # special for slope, each contact point must be a vertex of the original part mesh
        dist = np.linalg.norm(parts[0].vertices - pt[None, :], axis = 1)
        assert np.any(dist < 1e-6)

