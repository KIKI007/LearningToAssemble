import math
import scipy as sp
from trimesh import Trimesh
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from learn2assemble.stability import compute_rbe_static_attribs, num_vars, compute_rbe_dynamic_attribs, \
    compute_indices_mapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def from_scipy_to_torch_sparse(A: sp.sparse.coo_matrix,
                               floatType=torch.float32):
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((A.row, A.col))),
                                   torch.tensor(A.data, dtype=floatType),
                                   torch.Size(A.shape)).to(device)


def init_parameters(settings: dict):
    default = {
        "nt": 8,
        "mu": 0.55,
        "Ccp": 1E6,
        "density": 1e4,
        "boundary_part_ids": [],
        "sigma": 1E-6,
        "r": 0.1,
        "alpha": 1.6,
        "max_iter": 2000,
        "evaluate_iter": 100,
        "velocity_tol": 1e-3,
        "verbose": False,
        "float_type": torch.float32,
    }

    for name, value in default.items():
        if name not in settings:
            settings[name] = value

    return settings


def init_admm(parts: list[Trimesh],
              contacts: list[dict],
              settings: dict):
    settings = init_parameters(settings)

    nf, nλn, nλt = num_vars(parts, contacts, settings["nt"])
    L, A, g, Jn, Jt, invM = compute_rbe_static_attribs(parts,
                                                       contacts,
                                                       nt=settings["nt"],
                                                       mu=settings["mu"],
                                                       density=settings["density"],
                                                       boundary_part_ids=settings["boundary_part_ids"])

    prefactorize = {}
    prefactorize["g"] = g
    prefactorize["Jn"] = Jn
    prefactorize["Jt"] = Jt
    prefactorize["invM"] = invM
    prefactorize["nλn"] = nλn
    prefactorize["nλt"] = nλt
    prefactorize["nf"] = nf

    nx = A.shape[1]
    Inx = sp.sparse.coo_matrix(sp.sparse.eye_array(nx, dtype=np.float64))
    Ah = sp.sparse.block_array([[A],
                                [Inx]])
    AhT = Ah.transpose()

    float_type = settings["float_type"]
    prefactorize["Ah"] = from_scipy_to_torch_sparse(Ah, floatType=float_type)
    prefactorize["AhT"] = from_scipy_to_torch_sparse(AhT, floatType=float_type)
    prefactorize["nx"] = nx
    prefactorize["nA"] = A.shape[0]
    prefactorize["nAh"] = Ah.shape[0]
    prefactorize["contacts"] = contacts

    lhs = (L.T @ L + settings["sigma"] * Inx + settings["r"] * AhT @ Ah)
    lhs = torch.tensor(lhs.todense(), dtype=torch.float64, device=device)
    cholesky_lhs = torch.linalg.cholesky(lhs)
    prefactorize["inv_lhs"] = torch.cholesky_inverse(cholesky_lhs).type(float_type)
    settings["prefactorize"] = prefactorize
    return settings


def simulate_admm(batch_part_states: list[dict],
                  settings: dict):
    Ccp = settings["Ccp"]
    mu = settings["mu"]
    max_iter = settings["max_iter"]
    evaluate_iter = settings["evaluate_iter"]
    alpha = settings["alpha"]
    sigma = settings["sigma"]
    r = settings["r"]
    velocity_tol = settings["velocity_tol"]
    floatType = settings["float_type"]

    n_batch = batch_part_states.shape[0]
    nf = settings["prefactorize"]["nf"]
    nλn = settings["prefactorize"]["nλn"]
    nλt = settings["prefactorize"]["nλt"]
    nx = settings["prefactorize"]["nx"]
    nAh = settings["prefactorize"]["nAh"]
    nA = settings["prefactorize"]["nA"]
    g = settings["prefactorize"]["g"]
    Ah = settings["prefactorize"]["Ah"]
    AhT = settings["prefactorize"]["AhT"]
    Jn = settings["prefactorize"]["Jn"]
    Jt = settings["prefactorize"]["Jt"]
    invM = settings["prefactorize"]["invM"]
    inv_lhs = settings["prefactorize"]["inv_lhs"]

    ps, cs = compute_indices_mapping(nf, nλn, settings["prefactorize"]["contacts"], batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(g, ps, cs, Jn, Jt, invM, mu, Ccp)

    q = torch.tensor(q, dtype=floatType, device=device)
    xl = torch.tensor(xl, dtype=floatType, device=device)
    xu = torch.tensor(xu, dtype=floatType, device=device)
    Al = torch.tensor(Al, dtype=floatType, device=device)
    Au = torch.tensor(Au, dtype=floatType, device=device)
    zl = torch.vstack([Al, xl])
    zu = torch.vstack([Au, xu])

    x = torch.zeros(nx, n_batch, device=device, dtype=floatType)
    y = torch.zeros(nAh, n_batch, device=device, dtype=floatType)
    z = torch.zeros(nAh, n_batch, device=device, dtype=floatType)
    batch_inds = torch.arange(n_batch, device=device, dtype=torch.long)
    stable_flags = torch.zeros(n_batch, device=device, dtype=torch.bool)

    g = torch.tensor(g, dtype=floatType, device=device)
    JnT = from_scipy_to_torch_sparse(Jn.transpose(), floatType=floatType)
    JtT = from_scipy_to_torch_sparse(Jt.transpose(), floatType=floatType)
    invM = from_scipy_to_torch_sparse(invM, floatType=floatType)
    ps = torch.tensor(ps, dtype=floatType, device=device)
    # admm iterations
    for iter in np.arange(stop=max_iter, step=evaluate_iter):

        xk, yk, zk = x[:, batch_inds].clone(), y[:, batch_inds].clone(), z[:, batch_inds].clone()
        qk, zlk, zuk = q[:, batch_inds].clone(), zl[:, batch_inds].clone(), zu[:, batch_inds].clone()

        # admm
        with torch.no_grad():
            for it in range(evaluate_iter):
                rhs = sigma * xk - qk + torch.sparse.mm(AhT, r * zk - yk)
                xh_k1 = (inv_lhs @ rhs)
                x_k1 = (xh_k1 * alpha + xk * (1 - alpha))
                zh_k1 = torch.sparse.mm(Ah, xh_k1)
                z_alpha = alpha * zh_k1 + (1 - alpha) * zk
                z_k1 = z_alpha + yk / r
                z_k1 = torch.clip(z_k1, zlk, zuk)
                y_k1 = yk + r * (z_alpha - z_k1)
                xk, yk, zk = x_k1.clone(), y_k1.clone(), z_k1.clone()

        x[:, batch_inds], y[:, batch_inds], z[:, batch_inds] = xk.clone(), yk.clone(), zk.clone()

        # evualuation
        xclip = torch.clip(xk, zlk[nA:, :], zuk[nA:, :])
        λn, λt = xclip[:nλn, :], xclip[nλn: nλn + nλt, :]
        residual = (torch.sparse.mm(JnT, λn) + torch.sparse.mm(JtT, λt) + g[:, None]) * ps[:, batch_inds]
        velocity = torch.sparse.mm(invM, residual)
        velocity_inf_nrm = torch.max(torch.abs(velocity), dim=0).values

        # termination
        stable_flags[batch_inds] = (velocity_inf_nrm < velocity_tol)
        not_terminated = ~stable_flags[batch_inds]
        batch_inds = batch_inds[not_terminated]
        if batch_inds.shape[0] == 0:
            break

    xclip = torch.clip(x, zl[nA:, :], zu[nA:, :])
    return xclip.cpu().numpy(), stable_flags.cpu().numpy()


def init_gurobi(settings: dict):
    settings = init_parameters(settings)

    env = gp.Env()
    env.setParam('OptimalityTol', 1E-6)
    env.setParam('OutputFlag', settings["verbose"])
    env.setParam('Method', -1)
    settings["env"] = env

    nf, nλn, nλt = num_vars(parts, contacts, settings["nt"])
    L, A, g, Jn, Jt, invM = compute_rbe_static_attribs(parts,
                                                       contacts,
                                                       nt=settings["nt"],
                                                       mu=settings["mu"],
                                                       density=settings["density"],
                                                       boundary_part_ids=settings["boundary_part_ids"])

    prefactorize = {}
    prefactorize["L"] = L
    prefactorize["A"] = A
    prefactorize["g"] = g
    prefactorize["Jn"] = Jn
    prefactorize["Jt"] = Jt
    prefactorize["invM"] = invM
    prefactorize["nλn"] = nλn
    prefactorize["nλt"] = nλt
    prefactorize["nf"] = nf
    prefactorize["contacts"] = contacts
    settings["prefactorize"] = prefactorize
    return settings


def simulate_gurobi(batch_part_states: list[dict],
                    settings: dict):
    env = settings["env"]
    nf = settings["prefactorize"]["nf"]
    nλn = settings["prefactorize"]["nλn"]
    nλt = settings["prefactorize"]["nλt"]
    L = settings["prefactorize"]["L"]
    A = settings["prefactorize"]["A"]
    g = settings["prefactorize"]["g"]
    Jn = settings["prefactorize"]["Jn"]
    Jt = settings["prefactorize"]["Jt"]
    invM = settings["prefactorize"]["invM"]
    mu = settings["mu"]
    Ccp = settings["Ccp"]
    velocity_tol = settings["velocity_tol"]

    ps, cs = compute_indices_mapping(nf, nλn, settings["prefactorize"]["contacts"], batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(g, ps, cs, Jn, Jt, invM, mu, Ccp)

    flags, xs = [], []
    nx, nA = q.shape[0], Al.shape[0]

    for id in range(batch_part_states.shape[0]):
        xli, xui, Ali, Aui, qi = xl[:, id], xu[:, id], Al[:, id], Au[:, id], q[:, id]
        m = gp.Model(env=env)
        x = m.addMVar(nx, lb=xli, ub=xui)

        y = m.addMVar(L.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.setObjective(0.5 * y @ y + qi @ x, gp.GRB.MINIMIZE)
        m.addConstr(L @ x[:L.shape[1]] == y)
        m.addConstr(A @ x <= Aui)
        m.addConstr(A @ x >= Ali)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            xclip = np.clip(x.X, xli, xui)
            λn, λt = xclip[:nλn], xclip[nλn: nλn + nλt]
            residual = (Jn.T @ λn + Jt.T @ λt + g) * ps[:, id]
            velocity = invM @ residual
            velocity_inf_nrm = np.max(np.abs(velocity), axis=0)
            if velocity_inf_nrm < velocity_tol:
                flags.append(True)
            else:
                flags.append(False)
            xs.append(xclip)
        else:
            flags.append(False)
            xs.append(np.zeros(nx))

    xs = np.array(xs).T
    xs = xs.reshape(nx, -1)
    return xs, np.array(flags)


def init(parts: list[Trimesh], contacts: list[dict], settings: dict):
    if "solver" in settings and settings["solver"] == "gurobi":
        return init_gurobi(settings)
    else:
        return init_admm(parts, contacts, settings)


def simulate(batch_part_states: list[dict],
             settings: dict):
    if "solver" in settings and settings["solver"] == "gurobi":
        return simulate_gurobi(batch_part_states, settings)
    else:
        return simulate_admm(batch_part_states, settings)


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.render import draw_assembly, init_polyscope, draw_contacts
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import polyscope as ps
    import time

    init_polyscope()
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/dome")
    np.random.seed(0)
    n_batch = 512
    n_remove = 5
    part_states = np.ones((n_batch, len(parts)), dtype=np.int32)
    for batch_id in range(n_batch):
        remove_part_ids = np.random.choice(len(parts), n_remove, replace=False)
        part_states[batch_id, remove_part_ids] = 0
    part_states[:, -1] = 2
    contacts = compute_assembly_contacts(parts)
    settings = {"density": 1E3,
                "mu": 0.55,
                "evaluate_it": 200,
                "max_iter": 3000,
                "float_type": torch.float32,
                "solver": "admm",
                "velocity_tol": 1e-2,
                "boundary_part_ids": [len(parts) - 1]}

    settings = init(parts, contacts, settings)
    timer = time.perf_counter()
    force_fp32, stable_fp32 = simulate(part_states, settings)
    torch.cuda.synchronize()
    print("fp32 time", (time.perf_counter() - timer) / n_batch)
    print("stable_fp32", stable_fp32)

    # settings["float_type"] = torch.float64
    # settings = init(parts, contacts, settings)
    # timer = time.perf_counter()
    # force_fp64, stable_fp64 = simulate(parts, contacts, part_states, settings)
    # torch.cuda.synchronize()
    # print("fp64 time", (time.perf_counter() - timer) / n_batch)
    # print("stable_fp64", stable_fp64)

    settings["solver"] = "gurobi"
    settings = init(parts, contacts, settings)
    timer = time.perf_counter()
    force_gurobi, stable_gurobi = simulate(part_states, settings)
    torch.cuda.synchronize()
    print("gurobi time", (time.perf_counter() - timer) / n_batch)
    print("stable_gurobi", stable_gurobi)

    # print("error (fp32 vs fp64)", np.sum(np.abs(stable_fp64.astype(np.float32) - stable_fp32.astype(np.float32))) / n_batch * 100, "%")
    print("error (fp32 vs gurobi)",
          np.sum(np.abs(stable_gurobi.astype(np.float32) - stable_fp32.astype(np.float32))) / n_batch * 100, "%")

    draw_contacts(contacts, part_states[0], enable=True)
    draw_assembly(parts, part_states[0])
    ps.show()
