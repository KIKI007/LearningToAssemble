from time import perf_counter
from learn2assemble.simulator import *
from learn2assemble.assembly import *
from learn2assemble import ASSEMBLY_RESOURCE_DIR


def init_dome_env(n_batch, n_remove):
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/dome")
    settings = {
        "rbe": {
            "density": 1E3,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "verbose": False,
        },
        "admm": {
            "evaluate_it": 200,
            "max_iter": 3000,
            "float_type": torch.float32,
        },
        "env": {
            "boundary_part_ids": [len(parts) - 1]
        }
    }

    np.random.seed(0)
    part_states = np.ones((n_batch, len(parts)), dtype=np.int32)
    for batch_id in range(n_batch):
        remove_part_ids = np.random.choice(len(parts), n_remove, replace=False)
        part_states[batch_id, remove_part_ids] = 0
    part_states[:, -1] = 2
    contacts = compute_assembly_contacts(parts, settings)
    return parts, contacts, part_states, settings

def test_speed_simulate():
    parts, contacts, part_states, settings = init_dome_env(512, 5)
    timer = perf_counter()
    v_fp32, stable_fp32 = simulate(parts, contacts, part_states, settings)
    batch_sim_time = (perf_counter() - timer) / part_states.shape[0]
    assert batch_sim_time < 0.02

def test_simulate():
    parts, contacts, part_states, settings = init_dome_env(64, 5)

    settings["admm"]["float_type"] = torch.float32
    settings["admm"]["pre-computed"] = False
    contacts = compute_assembly_contacts(parts, settings)
    v_fp32, stable_fp32 = simulate(parts, contacts, part_states, settings)

    settings["admm"]["float_type"] = torch.float64
    settings["admm"]["pre-computed"] = False
    v_fp64, stable_fp64 = simulate(parts, contacts, part_states, settings)
    assert (stable_fp64 == stable_fp32).all()

    settings.pop("admm")
    settings["gurobi"] = {}
    v_gurobi, stable_gurobi = simulate(parts, contacts, part_states, settings)
    assert (stable_gurobi == stable_fp32).all()
