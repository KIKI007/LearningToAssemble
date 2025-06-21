from time import perf_counter
from learn2assemble.simulator import *
from learn2assemble.assembly import *
from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings


def init_dome_env(n_batch, n_remove):
    name = "dome"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    default_settings["rbe"]["density"] = 1E3
    default_settings["admm"]["max_iter"] = 3000
    default_settings["env"]["boundary_part_ids"] = [len(parts) - 1]

    np.random.seed(0)
    part_states = np.ones((n_batch, len(parts)), dtype=np.int32)
    for batch_id in range(n_batch):
        remove_part_ids = np.random.choice(len(parts), n_remove, replace=False)
        part_states[batch_id, remove_part_ids] = 0
    if name == "dome":
        part_states[:, -1] = 2
    else:
        part_states[:, 0] = 2
    contacts = compute_assembly_contacts(parts, default_settings)
    return parts, contacts, part_states, default_settings

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
