from trimesh import Trimesh
import numpy as np
from scipy.sparse import coo_matrix
import math
import torch
import scipy
from learn2assemble import set_default


def num_vars(parts, contacts, nt=8):
    nf = len(parts) * 6
    nλn = 0
    nλt = 0
    for contact in contacts:
        npt = contact["mesh"].vertices.shape[0]
        nλn += npt
        nλt += npt * nt
    return nf, nλn, nλt


def compute_jacobian(parts, contacts, nt=8):
    nf, nλn, nλt = num_vars(parts, contacts)
    JnT = np.zeros((nf, nλn), dtype=np.float64)
    fn = 0
    for contact in contacts:
        iA = contact["iA"]
        iB = contact["iB"]
        nrm = contact["plane"][:3, 2]
        for part_id in [iB, iA]:
            ct = parts[part_id].center_mass
            for id, pt in enumerate(contact["mesh"].vertices):
                m = np.hstack([nrm, np.cross(pt - ct, nrm)])
                JnT[part_id * 6: part_id * 6 + 6, fn + id] = m
            nrm = -nrm
        fn += contact["mesh"].vertices.shape[0]

    Jt = np.zeros((nλt, nf), dtype=np.float64)
    for k in range(nt):
        fn = 0
        angle = 2.0 * math.pi / nt * k
        JtkT = np.zeros((nf, nλn), dtype=np.float64)
        for contact in contacts:
            iA = contact["iA"]
            iB = contact["iB"]
            xaxis = contact["plane"][:3, 0]
            yaxis = contact["plane"][:3, 1]
            t = xaxis * np.cos(angle) + yaxis * np.sin(angle)
            for part_id in [iB, iA]:
                ct = parts[part_id].center_mass
                for id, pt in enumerate(contact["mesh"].vertices):
                    m = np.hstack([t, np.cross(pt - ct, t)])
                    JtkT[part_id * 6: part_id * 6 + 6, fn + id] = m
                t = -t
            fn += contact["mesh"].vertices.shape[0]
        Jt[nλn * k: nλn * (k + 1), :] = JtkT.T

    E = np.zeros((nλn, nt * nλn), dtype=np.float64)
    for i in range(nλn):
        for k in range(nt):
            E[i, nλn * k + i] = 1

    Jn = coo_matrix(JnT.T)
    Jn.eliminate_zeros()
    Jt = coo_matrix(Jt)
    Jt.eliminate_zeros()
    E = coo_matrix(E)
    E.eliminate_zeros()
    return Jn, Jt, E


def compute_gravity(parts,
                    density: float = 1.0,
                    boundary_part_ids: list[int] = []):
    nf = len(parts) * 6
    g = np.zeros(nf, dtype=np.float64)
    volumes, _ = compute_volume_and_inertial(parts, density, boundary_part_ids)
    for part_id in range(len(parts)):
        gi = np.array([0, 0, -volumes[part_id]], dtype=np.float64) * density
        g[part_id * 6: part_id * 6 + 3] = gi
    return g


def compute_volume_and_inertial(parts: list[Trimesh],
                                density: float = 1.0,
                                boundary_part_ids: list[int] = []):
    volumes = [0 for part in parts]
    moment_inertias = [None for part in parts]
    for part_id, part in enumerate(parts):
        if part_id in boundary_part_ids:
            volumes[part_id] = 1.0 / density
            moment_inertias[part_id] = np.eye(3) / density
        else:
            volumes[part_id] = part.volume
            moment_inertias[part_id] = part.moment_inertia

    return volumes, moment_inertias


def compute_generalized_mass(parts,
                             density: float = 1.0,
                             boundary_part_ids: list[int] = []):
    nf = len(parts) * 6
    volumes, moment_inertias = compute_volume_and_inertial(parts, density, boundary_part_ids)
    M = np.zeros((nf, nf), dtype=np.float64)
    for part_id, part in enumerate(parts):
        Ii = moment_inertias[part_id] * density
        Mi = np.identity(3) * volumes[part_id] * density

        M[part_id * 6: part_id * 6 + 3,
        part_id * 6: part_id * 6 + 3] = Mi

        M[part_id * 6 + 3: part_id * 6 + 6,
        part_id * 6 + 3: part_id * 6 + 6] = Ii

    Mt = torch.tensor(M, dtype=torch.float64, device='cpu')
    L = torch.linalg.cholesky(Mt)
    invM = torch.cholesky_inverse(L)
    invML = torch.linalg.cholesky(invM).numpy()

    M_sparse = coo_matrix(M)
    M_sparse.eliminate_zeros()

    invM_sparse = coo_matrix(invM.numpy())
    invM_sparse.eliminate_zeros()

    invML_sparse = coo_matrix(invML)
    invML_sparse.eliminate_zeros()
    return M_sparse, invM_sparse, invML_sparse


def Knt(Jn: scipy.sparse.coo_matrix,
        Jt: scipy.sparse.coo_matrix):
    Inf = scipy.sparse.eye_array(Jn.shape[1], dtype=np.float64)
    Knt = scipy.sparse.block_array([[Jn.T, Jt.T, -Inf]])
    return Knt


def compute_rbe_static_attribs(parts: list[Trimesh],
                               contacts: list[dict],
                               nt=8,
                               mu=0.55,
                               density: float = 1.0,
                               boundary_part_ids: list[int] = []):
    nf, nλn, nλt = num_vars(parts, contacts, nt)

    g = compute_gravity(parts, density, boundary_part_ids)
    Jn, Jt, E = compute_jacobian(parts, contacts, nt)
    M, invM, invML = compute_generalized_mass(parts, density, boundary_part_ids)
    L = invML.T @ Knt(Jn, Jt)

    Inλ = scipy.sparse.eye_array(nλn, dtype=np.float64)
    Onλ = scipy.sparse.coo_matrix((nλn, nf), dtype=np.float64)
    A = scipy.sparse.block_array([[mu * Inλ, -E, Onλ]])
    return L, A, g, Jn, Jt, invM


def compute_rbe_dynamic_attribs(g: np.ndarray,
                                ps: np.ndarray,
                                cs: np.ndarray,
                                Jn: scipy.sparse,
                                Jt: scipy.sparse,
                                invM: scipy.sparse,
                                mu: float = 0.55,
                                Ccp: float = 1E5):
    nλn = Jn.shape[0]
    nλt = Jt.shape[0]
    nt = nλt // nλn

    K = Knt(Jn, Jt)
    nbatch = cs.shape[1]

    Pg = ps * g[:, None]
    q = K.T @ invM @ Pg

    Al = np.zeros((nλn, nbatch), dtype=np.float64)
    Au = np.ones((nλn, nbatch), dtype=np.float64) * mu * Ccp
    xl = np.vstack([np.zeros((nλn, nbatch), dtype=np.float64),
                    np.zeros((nλt, nbatch), dtype=np.float64),
                    -Ccp * (1 - ps)])
    xu = np.vstack([Ccp * cs,
                    np.tile(Ccp * cs, (nt, 1)),
                    Ccp * (1 - ps)])

    return q, xl, xu, Al, Au

class IndexMapping:

    def __init__(self, nf, nλn, contacts):
        self.nf = nf
        self.nλn = nλn
        self.contacts = contacts

    def __call__(self, batch_part_states):

        if batch_part_states.ndim == 1:
            batch_part_states = batch_part_states.reshape(1, -1)

        n_batch = batch_part_states.shape[0]
        n_part = batch_part_states.shape[1]

        p = np.zeros((self.nf, n_batch), dtype=np.float64)
        for ib in range(n_batch):
            part_states = batch_part_states[ib, :]
            for part_id in range(n_part):
                p[part_id * 6: part_id * 6 + 6, ib] = (part_states[part_id] == 1)

        c = np.zeros((self.nλn, n_batch), dtype=np.float64)
        for ib in range(n_batch):
            part_states = batch_part_states[ib, :]
            nf = 0
            for contact in self.contacts:
                iA = contact["iA"]
                iB = contact["iB"]
                for pt in contact["mesh"].vertices:
                    if part_states[iA] > 0 and part_states[iB] > 0 and (part_states[iA] < 2 or part_states[iB] < 2):
                        c[nf, ib] = 1
                    else:
                        c[nf, ib] = 0
                    nf = nf + 1
        return p, c

def init_rbe(parts: list[Trimesh],
             contacts: list[dict],
             settings: dict):

    rbe = set_default(
        settings,
        "rbe",
        {
            "nt": 8,
            "mu": 0.55,
            "Ccp": 1E6,
            "density": 1e4,
            "boundary_part_ids": [len(parts) - 1],
            "velocity_tol": 1e-3,
            "verbose": False,
        }
    )

    nf, nλn, nλt = num_vars(parts, contacts, rbe["nt"])
    L, A, g, Jn, Jt, invM = compute_rbe_static_attribs(parts,
                                                       contacts,
                                                       nt=rbe["nt"],
                                                       mu=rbe["mu"],
                                                       density=rbe["density"],
                                                       boundary_part_ids=rbe["boundary_part_ids"])
    rbe["A"] = A
    rbe["L"] = L
    rbe["g"] = g
    rbe["Jn"] = Jn
    rbe["Jt"] = Jt
    rbe["invM"] = invM
    rbe["nλn"] = nλn
    rbe["nλt"] = nλt
    rbe["nf"] = nf
    rbe["mapping"] = IndexMapping(nf, nλn, contacts)
    rbe["pre-computed"] = True
    settings["rbe"] = rbe
