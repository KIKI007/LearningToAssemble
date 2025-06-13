import numpy as np

from learn2assemble.RBE import *
from learn2assemble.assembly import *
import os

slope_test_folder = os.path.abspath(os.path.dirname(__file__) + "/data/slope")

def test_num_vars():
    parts = load_assembly_from_files(slope_test_folder)
    contacts = compute_assembly_contacts(parts)
    nf, nλn, nλt = num_vars(parts, contacts, nt=8)
    assert nf == len(parts) * 6
    assert nλn == 4
    assert nλt == 4 * 8

def test_compute_jacobian():
    parts = load_assembly_from_files(slope_test_folder)
    parts[1].center_mass = np.zeros(3)

    contacts = compute_assembly_contacts(parts)
    Jn, Jt, E = compute_jacobian(parts, contacts, nt=8)
    print("\n")
    normal = contacts[0]["plane"][:3, 2]
    xaxis = contacts[0]["plane"][:3, 0]
    yaxis = contacts[0]["plane"][:3, 1]
    points = contacts[0]["mesh"].vertices

    assert np.linalg.norm(Jn.todense()[:, :3] + normal[None, :]) < 1e-6
    assert np.linalg.norm(Jn.todense()[:, 6:9] - normal[None, :]) < 1e-6
    assert np.linalg.norm(Jt.todense()[:4, :3] + xaxis) < 1e-6
    assert np.linalg.norm(Jt.todense()[:4, 6:9] - xaxis) < 1e-6
    assert np.linalg.norm(Jt.todense()[8:12, :3] + yaxis) < 1e-6
    assert np.linalg.norm(Jt.todense()[8:12, 6:9] - yaxis) < 1e-6

    m = np.cross(points - parts[0].center_mass, -normal)
    assert np.linalg.norm(Jn.todense()[:, 3:6] - m) < 1e-6

    m = np.cross(points - parts[1].center_mass, normal)
    assert np.linalg.norm(Jn.todense()[:, 9:12] - m) < 1e-6

    m = np.cross(points - parts[0].center_mass, -xaxis)
    assert np.linalg.norm(Jt.todense()[:4, 3:6] - m) < 1e-6

    m = np.cross(points - parts[1].center_mass, xaxis)
    assert np.linalg.norm(Jt.todense()[:4, 9:12] - m) < 1e-6

    m = np.cross(points - parts[0].center_mass, -yaxis)
    assert np.linalg.norm(Jt.todense()[8:12, 3:6] - m) < 1e-6

    m = np.cross(points - parts[1].center_mass, yaxis)
    assert np.linalg.norm(Jt.todense()[8:12, 9:12] - m) < 1e-6

    λt = np.ones(Jt.shape[0])
    assert np.linalg.norm(E @ λt - np.ones(4) * 8) < 1e-6


def test_compute_gravity():
    parts = load_assembly_from_files(slope_test_folder)
    parts[1].center_mass = np.zeros(3)
    g = compute_gravity(parts, density=1.0, boundary_part_ids=[1])
    assert np.abs(g[2] + parts[0].volume) < 1e-6
    assert np.abs(g[8] + 1) < 1e-6
    g = compute_gravity(parts, density=1.0, boundary_part_ids=[])
    assert np.abs(g[8] + parts[1].volume) < 1e-6


def test_compute_generalized_mass():
    parts = load_assembly_from_files(slope_test_folder)
    M, invM, invML = compute_generalized_mass(parts, density=1, boundary_part_ids=[1])
    M = M.todense()
    invM = invM.todense()

    assert np.linalg.norm(M @ invM - np.eye(12)) < 1e-6
    assert np.linalg.norm(invML @ invML.T - invM) < 1e-6

    assert np.linalg.norm(M[:3, :3] - np.eye(3) * parts[0].volume) < 1e-6
    assert np.linalg.norm(M[3:6, 3:6] - parts[0].moment_inertia) < 1e-6
    assert np.linalg.norm(M[6:9, 6:9] - np.eye(3)) < 1e-6
    assert np.linalg.norm(M[9:12, 9:12] - np.eye(3)) < 1e-6


def test_compute_indices_mapping():
    parts = load_assembly_from_files(slope_test_folder)
    contacts = compute_assembly_contacts(parts)
    nf, nλn, nλt = num_vars(parts, contacts, nt=8)

    part_states = np.array([[1, 0],
                            [1, 1],
                            [1, 2],
                            [2, 2]])
    ps, cs = compute_indices_mapping(nf, nλn, contacts, part_states)

    np.linalg.norm(ps[6: 12, [0, 1, 2]]) < 1e-6
    np.linalg.norm(cs[6: 12, [0, 2]]) < 1e-6
