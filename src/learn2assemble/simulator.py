import math
import scipy as sp
from trimesh import Trimesh
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from learn2assemble.RBE import *
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def from_scipy_to_torch_sparse(A: sp.sparse.coo_matrix,
                               floatType=torch.float32):
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((A.row, A.col))),
                                   torch.tensor(A.data, dtype=floatType),
                                   torch.Size(A.shape)).to(device)


def init_admm(parts: list[Trimesh],
              contacts: list[dict],
              settings: dict):

    admm = set_default(settings,
                       "admm",
                       {
                           "sigma": 1E-6,
                           "r": 0.1,
                           "alpha": 1.6,
                           "max_iter": 2000,
                           "evaluate_iter": 100,
                           "float_type": torch.float32,
                       })

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
    admm["pre-computed"] = True
    settings["admm"] = admm


def admm_step(xk, yk, zk, qk, zlk, zuk, Ah, AhT, inv_lhs, r, sigma, alpha, evaluate_iter):
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
    return xk, yk, zk


def simulate_admm(batch_part_states: list[dict],
                  settings: dict):
    n_batch = batch_part_states.shape[0]

    # name space
    rbe = SimpleNamespace(**settings["rbe"])
    admm = SimpleNamespace(**settings["admm"])
    floatType = admm.float_type

    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)

    q = torch.tensor(q, dtype=floatType, device=device)
    xl = torch.tensor(xl, dtype=floatType, device=device)
    xu = torch.tensor(xu, dtype=floatType, device=device)
    Al = torch.tensor(Al, dtype=floatType, device=device)
    Au = torch.tensor(Au, dtype=floatType, device=device)
    zl = torch.vstack([Al, xl])
    zu = torch.vstack([Au, xu])

    x = torch.zeros(admm.nx, n_batch, device=device, dtype=floatType)
    y = torch.zeros(admm.nAh, n_batch, device=device, dtype=floatType)
    z = torch.zeros(admm.nAh, n_batch, device=device, dtype=floatType)
    batch_inds = torch.arange(n_batch, device=device, dtype=torch.long)
    stable_flags = torch.zeros(n_batch, device=device, dtype=torch.bool)

    g = torch.tensor(rbe.g, dtype=floatType, device=device)
    JnT = from_scipy_to_torch_sparse(rbe.Jn.transpose(), floatType=floatType)
    JtT = from_scipy_to_torch_sparse(rbe.Jt.transpose(), floatType=floatType)
    invM = from_scipy_to_torch_sparse(rbe.invM, floatType=floatType)
    ps = torch.tensor(ps, dtype=floatType, device=device)

    v = torch.zeros(rbe.nf, n_batch, device=device, dtype=floatType)

    # admm iterations
    for iter in np.arange(stop=admm.max_iter, step=admm.evaluate_iter):

        xk, yk, zk = x[:, batch_inds].clone(), y[:, batch_inds].clone(), z[:, batch_inds].clone()
        qk, zlk, zuk = q[:, batch_inds].clone(), zl[:, batch_inds].clone(), zu[:, batch_inds].clone()

        # admm
        x[:, batch_inds], y[:, batch_inds], z[:, batch_inds] = admm_step(xk, yk, zk, qk, zlk, zuk,
                                                                         admm.Ah, admm.AhT, admm.inv_lhs, admm.r,
                                                                         admm.sigma, admm.alpha, admm.evaluate_iter)

        # evualuation
        xclip = torch.clip(xk, zlk[admm.nA:, :], zuk[admm.nA:, :])
        λn, λt = xclip[:rbe.nλn, :], xclip[rbe.nλn: rbe.nλn + rbe.nλt, :]
        residual = (torch.sparse.mm(JnT, λn) + torch.sparse.mm(JtT, λt) + g[:, None]) * ps[:, batch_inds]
        velocity = torch.sparse.mm(invM, residual)
        velocity_inf_nrm = torch.max(torch.abs(velocity), dim=0).values
        v[:, batch_inds] = velocity
        if rbe.verbose:
            print(velocity_inf_nrm)

        # termination
        stable_flags[batch_inds] = (velocity_inf_nrm < rbe.velocity_tol)
        not_terminated = ~stable_flags[batch_inds]
        batch_inds = batch_inds[not_terminated]
        if batch_inds.shape[0] == 0:
            break

    return v.cpu().numpy(), stable_flags.cpu().numpy()

def init_gurobi(parts, contacts, settings: dict):
    env = gp.Env()
    env.setParam('OptimalityTol', 1E-6)
    env.setParam('OutputFlag', settings["rbe"]["verbose"])
    env.setParam('Method', -1)
    settings["gurobi"] = {
        "env": env,
        "pre-computed": True
    }

def simulate_gurobi(batch_part_states: list[dict],
                    settings: dict):
    env = settings["gurobi"]["env"]

    # rbe
    rbe = SimpleNamespace(**settings["rbe"])

    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)

    flags, vs = [], []
    nx, nA = q.shape[0], Al.shape[0]

    for id in range(batch_part_states.shape[0]):
        xli, xui, Ali, Aui, qi = xl[:, id], xu[:, id], Al[:, id], Au[:, id], q[:, id]
        m = gp.Model(env=env)
        x = m.addMVar(nx, lb=xli, ub=xui)
        y = m.addMVar(rbe.L.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY)

        m.setObjective(0.5 * y @ y + qi @ x, gp.GRB.MINIMIZE)
        m.addConstr(rbe.L @ x[:rbe.L.shape[1]] == y)
        m.addConstr(rbe.A @ x <= Aui)
        m.addConstr(rbe.A @ x >= Ali)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            xclip = np.clip(x.X, xli, xui)
            λn, λt = xclip[:rbe.nλn], xclip[rbe.nλn: rbe.nλn + rbe.nλt]
            residual = (rbe.Jn.T @ λn + rbe.Jt.T @ λt + rbe.g) * ps[:, id]
            velocity = rbe.invM @ residual
            velocity_inf_nrm = np.max(np.abs(velocity), axis=0)
            if velocity_inf_nrm < rbe.velocity_tol:
                flags.append(True)
            else:
                flags.append(False)
            vs.append(velocity)
        else:
            flags.append(False)
            vs.append(np.zeros(rbe.nf))
    vs = np.vstack(vs)
    vs = vs.T
    return vs, np.array(flags)

def simulate(parts: list[Trimesh],
             contacts: list[dict],
             batch_part_states: list[dict],
             settings: dict):

    rbe_pre_computed = settings.get("rbe", {"pre-computed": False}).get("pre-computed", False)
    if not rbe_pre_computed:
        init_rbe(parts, contacts, settings)

    if "gurobi" in settings:
        gurobi_pre_computed = settings["gurobi"].get("pre-computed", False)
        if not gurobi_pre_computed:
            init_gurobi(parts, contacts, settings)
        return simulate_gurobi(batch_part_states, settings)
    else:
        # by default running admm
        admm_computed = settings.get("admm", {"pre-computed": False}).get("pre-computed", False)
        if not admm_computed:
            init_admm(parts, contacts, settings)
        return simulate_admm(batch_part_states, settings)



if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import polyscope as ps

    init_polyscope()
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    part_states = np.zeros((1, len(parts)))
    part_states[0, 3] = 1
    part_states[0, 5] = 1
    b = 0
    part_states[0, b] = 2

    settings = {
        "contact_settings": {
            "shrink_ratio": 0.1,
        },
        "rbe": {
            "density": 1E2,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "verbose": False,
        },
        "admm": {
            "Ccp": 1E6,
            "evaluate_it": 200,
            "max_iter": 3000,
            "float_type": torch.float32,
        },
        "env": {
            "boundary_part_ids": [],
        }
    }

    contacts = compute_assembly_contacts(parts, settings)
    v_fp32, stable_fp32 = simulate(parts, contacts, part_states, settings)
    t = 0
    def callback():
        global t
        changed, t = psim.SliderFloat("time", v=t, v_min=0, v_max=1)
        if changed:
            draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)


    draw_contacts(contacts, part_states[0])
    draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    ps.set_user_callback(callback)
    ps.show()
