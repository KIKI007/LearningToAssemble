import math
import scipy as sp
from trimesh import Trimesh
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from learn2assemble.RBE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def from_scipy_to_torch_sparse(A: sp.sparse.coo_matrix,
                               floatType=torch.float32):
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((A.row, A.col))),
                                   torch.tensor(A.data, dtype=floatType),
                                   torch.Size(A.shape)).to(device)

def init_admm(parts: list[Trimesh],
              contacts: list[dict],
              settings: dict):
    settings = init_rbe(parts, contacts, settings)

    # set admm parameters
    admm = settings.get("admm", {})
    default = {
        "sigma": 1E-6,
        "r": 0.1,
        "alpha": 1.6,
        "max_iter": 2000,
        "evaluate_iter": 100,
        "float_type": torch.float32,
    }

    for name, value in default.items():
        if name not in admm:
            admm[name] = value

    rbe = settings['rbe']
    A = rbe["A"]
    L = rbe["L"]
    float_type = admm["float_type"]

    nx = A.shape[1]
    Inx = sp.sparse.coo_matrix(sp.sparse.eye_array(nx, dtype=np.float64))
    Ah = sp.sparse.block_array([[A],
                                [Inx]])
    AhT = Ah.transpose()

    admm["Ah"] = from_scipy_to_torch_sparse(Ah, floatType=float_type)
    admm["AhT"] = from_scipy_to_torch_sparse(AhT, floatType=float_type)
    admm["nx"] = nx
    admm["nA"] = A.shape[0]
    admm["nAh"] = Ah.shape[0]

    lhs = (L.T @ L + admm["sigma"] * Inx + admm["r"] * AhT @ Ah)
    lhs = torch.tensor(lhs.todense(), dtype=torch.float64, device=device)
    cholesky_lhs = torch.linalg.cholesky(lhs)
    admm["inv_lhs"] = torch.cholesky_inverse(cholesky_lhs).type(float_type)
    settings["admm"] = admm
    return settings

def simulate_admm(batch_part_states: list[dict],
                  settings: dict):
    n_batch = batch_part_states.shape[0]

    # rbe
    rbe = settings["rbe"]
    Ccp = rbe["Ccp"]
    mu = rbe["mu"]
    velocity_tol = rbe["velocity_tol"]
    nf = rbe["nf"]
    nλn = rbe["nλn"]
    nλt = rbe["nλt"]
    g = rbe["g"]
    Jn = rbe["Jn"]
    Jt = rbe["Jt"]
    invM = rbe["invM"]
    contacts = rbe["contacts"]

    # admm
    admm = settings["admm"]
    floatType = admm["float_type"]
    max_iter = admm["max_iter"]
    evaluate_iter = admm["evaluate_iter"]
    alpha = admm["alpha"]
    sigma = admm["sigma"]
    r = admm["r"]
    nx = admm["nx"]
    nAh = admm["nAh"]
    nA = admm["nA"]
    Ah = admm["Ah"]
    AhT = admm["AhT"]
    inv_lhs = admm["inv_lhs"]

    ps, cs = compute_indices_mapping(nf, nλn, contacts, batch_part_states)
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

    v = torch.zeros(nf, n_batch, device=device, dtype=floatType)

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
        v[:, batch_inds] = velocity
        if rbe["verbose"]:
            print(velocity_inf_nrm)

        # termination
        stable_flags[batch_inds] = (velocity_inf_nrm < velocity_tol)
        not_terminated = ~stable_flags[batch_inds]
        batch_inds = batch_inds[not_terminated]
        if batch_inds.shape[0] == 0:
            break

    return v.cpu().numpy(), stable_flags.cpu().numpy()

def init_gurobi(parts, contacts, settings: dict):
    settings = init_rbe(parts, contacts, settings)
    env = gp.Env()
    env.setParam('OptimalityTol', 1E-6)
    env.setParam('OutputFlag', settings["rbe"]["verbose"])
    env.setParam('Method', -1)
    settings["gurobi"] = {
        "env": env
    }
    return settings

def simulate_gurobi(batch_part_states: list[dict],
                    settings: dict):
    env = settings["gurobi"]["env"]

    # rbe
    rbe = settings["rbe"]
    Ccp = rbe["Ccp"]
    mu = rbe["mu"]
    velocity_tol = rbe["velocity_tol"]
    nf = rbe["nf"]
    nλn = rbe["nλn"]
    nλt = rbe["nλt"]
    L = rbe["L"]
    A = rbe["A"]
    g = rbe["g"]
    Jn = rbe["Jn"]
    Jt = rbe["Jt"]
    invM = rbe["invM"]
    contacts = rbe["contacts"]

    ps, cs = compute_indices_mapping(nf, nλn, contacts, batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(g, ps, cs, Jn, Jt, invM, mu, Ccp)

    flags, vs = [], []
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
            vs.append(velocity)
        else:
            flags.append(False)
            vs.append(np.zeros(nf))
    vs = np.vstack(vs)
    vs = vs.T
    return vs, np.array(flags)


def init(parts: list[Trimesh], contacts: list[dict], settings: dict):
    if "admm" in settings:
        return init_admm(parts, contacts, settings)
    else:
        return init_gurobi(parts, contacts, settings)


def simulate(batch_part_states: list[dict],
             settings: dict):
    if "admm" in settings:
        return simulate_admm(batch_part_states, settings)
    else:
        return simulate_gurobi(batch_part_states, settings)


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import polyscope as ps

    init_polyscope()
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    part_states = np.ones((1, len(parts)))
    part_states[0, 0] = 2
    part_states[0, 1] = 0
    contacts = compute_assembly_contacts(parts)
    settings = {
        "rbe": {
            "density": 1E3,
            "mu": 0.55,
            "velocity_tol": 1e-3,
            "boundary_part_ids": [0],
            "verbose": True,
        },
        "admm": {
            "evaluate_it": 200,
            "max_iter": 3000,
            "float_type": torch.float64,
        }
    }
    settings = init(parts, contacts, settings)
    v_fp32, stable_fp32 = simulate(part_states, settings)
    t = 0
    def callback():
        global t
        changed, t = psim.SliderFloat("time", v = t, v_min=0, v_max=1)
        if changed:
            draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    ps.set_user_callback(callback)
    ps.show()
