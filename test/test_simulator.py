from learn2assemble import ASSEMBLY_RESOURCE_DIR
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.simulator import *
import time

def test_simulate():
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/dome")
    np.random.seed(0)
    n_batch = 10
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
    fn_fp32, ft_fp32, stable_fp32 = simulate(part_states, settings)
    torch.cuda.synchronize()
    print("fp32 time", (time.perf_counter() - timer) / n_batch)
    print("stable_fp32", stable_fp32)

    settings["float_type"] = torch.float64
    settings = init(parts, contacts, settings)
    timer = time.perf_counter()
    force_fp64, stable_fp64 = simulate(parts, contacts, part_states, settings)
    torch.cuda.synchronize()
    print("fp64 time", (time.perf_counter() - timer) / n_batch)
    print("stable_fp64", stable_fp64)
    print("error (fp32 vs fp64)",
          np.sum(np.abs(stable_fp64.astype(np.float32) - stable_fp32.astype(np.float32))) / n_batch * 100, "%")

    settings["solver"] = "gurobi"
    settings = init(parts, contacts, settings)
    timer = time.perf_counter()
    fn_gurobi, ft_gurobi, stable_gurobi = simulate(part_states, settings)
    torch.cuda.synchronize()
    print("gurobi time", (time.perf_counter() - timer) / n_batch)
    print("stable_gurobi", stable_gurobi)
    print("error (fp32 vs gurobi)",
          np.sum(np.abs(stable_gurobi.astype(np.float32) - stable_fp32.astype(np.float32))) / n_batch * 100, "%")