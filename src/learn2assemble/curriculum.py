import numpy as np
import warnings
from scipy.cluster.vq import kmeans2
from trimesh import Trimesh
import learn2assemble.simulator
import time

def cluster(part_states: np.ndarray,
            prev_inds: np.ndarray,
            ncluster: int):
    nbacth = part_states.shape[0]
    if ncluster >= nbacth:
        return part_states, prev_inds
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # shuffle
            arr = np.arange(nbacth)
            np.random.shuffle(arr)

            # bin_states = (part_states > 0).astype(float)
            bin_states = (part_states.copy()).astype(float)
            part_states = part_states[arr, :]
            bin_states, prev_inds = bin_states[arr, :], prev_inds[arr]

            # cluster
            centroids, labels = kmeans2(bin_states.astype(float), ncluster, minit='points')
            centroids = centroids.repeat(nbacth, axis=0)
            points = np.tile(bin_states, (ncluster, 1))
            dist = np.linalg.norm(points - centroids, axis=1).reshape(ncluster, nbacth)
            min = dist.argmin(axis=1)

            # return result
            return part_states[min, :], prev_inds[min]


def forward_install_actions(part_states: np.ndarray,
                            n_robot: int,
                            boundary_part_ids: list = []):
    # nbatch x naction
    nbatch = part_states.shape[0]
    naction = part_states.shape[1]
    prev_inds = np.arange(0, nbatch)
    prev_inds = np.tile(prev_inds, (naction, 1))
    prev_inds = prev_inds.swapaxes(0, 1)

    # nbatch x naction x npart
    new_states = part_states[:, None, :]
    new_states = np.tile(new_states, (1, naction, 1))

    # nbatch x naction x npart
    action = np.identity(naction, dtype=np.int32)
    action = np.tile(action, (nbatch, 1, 1))

    # not install an already installed parts
    flag = np.einsum("ijk, ijk->ij", new_states, action) == 0

    # not install on boundary
    flag[:, boundary_part_ids] = False

    # not use robots more than a given number
    flag_robot = (np.sum(new_states == 2, axis=2) + 1) <= n_robot + len(boundary_part_ids)

    flag = np.logical_and(flag, flag_robot)

    new_states = (new_states + action * 2)[flag, :]
    prev_inds = prev_inds[flag]

    return new_states, prev_inds


def forward_release_actions(part_states: np.ndarray,
                            boundary_part_ids: list = []):
    # nbatch x naction
    nbatch = part_states.shape[0]
    naction = part_states.shape[1]
    prev_inds = np.arange(0, nbatch)
    prev_inds = np.tile(prev_inds, (naction, 1))
    prev_inds = prev_inds.swapaxes(0, 1)

    # nbatch x naction x npart
    new_states = part_states[:, None, :]
    new_states = np.tile(new_states, (1, naction, 1))

    # nbatch x naction x npart
    action = np.identity(naction, dtype=np.int32)
    action = np.tile(action, (nbatch, 1, 1))

    # not install an already installed parts
    flag = np.einsum("ijk, ijk->ij", new_states == 2, action) == 1

    # not install on boundary
    flag[:, boundary_part_ids] = False

    new_states = (new_states - action)[flag, :]
    prev_inds = prev_inds[flag]

    return new_states, prev_inds


def forward_actions(part_states: np.ndarray,
                    n_robot: int,
                    boundary_part_ids: list = []):

    install_states, install_prev_inds = forward_install_actions(part_states, n_robot, boundary_part_ids)
    release_states, release_prev_inds = forward_release_actions(part_states, boundary_part_ids)

    new_states = np.vstack([install_states, release_states])
    prev_inds = np.hstack([install_prev_inds, release_prev_inds])
    stability_verifying_flag = np.hstack([np.zeros(install_states.shape[0]), np.ones(release_states.shape[0])]).astype(
        np.bool_)

    new_states, unique_indices = np.unique(new_states, axis=0, return_index=True)
    prev_inds = prev_inds[unique_indices]
    stability_verifying_flag = stability_verifying_flag[unique_indices]

    return new_states, prev_inds, stability_verifying_flag


def terminate(part_states: np.ndarray,
              boundary_part_ids: list = []):
    npart = part_states.shape[1]
    fixed_states = np.ones(npart)
    fixed_states[boundary_part_ids] = 2
    dist = np.sum(np.abs(part_states - fixed_states[None, :]), axis=1)
    return dist == 0

def compute_solution(records):
    nstep = len(records['part_states'])
    ind = 0
    part_states = []
    for istep in np.arange(start=nstep - 1, stop=-1, step=-1):
        state = records["part_states"][istep][ind, :]
        part_states.append(state)
        if istep > 0:
            ind = records["prev_inds"][istep][ind]
    part_states.reverse()
    return np.array(part_states, dtype=np.int32)


def forward_curriculum(parts: list[Trimesh],
                   contacts: list[dict],
                   settings: dict = {}):
    # create simulator
    learn2assemble.simulator.init(parts, contacts, settings)

    # parameters
    boundary_part_ids = settings["rbe"]["boundary_part_ids"]
    search = set_default(settings,
                         "curriculum",
                         {
                             "n_robot": 2,
                             "n_beam": 64
                         })

    n_robot = search["n_robot"]
    n_beam = search["n_beam"]

    # init states
    part_states = np.zeros((1, len(parts)), dtype=np.int32)
    part_states[:, boundary_part_ids] = 2
    iter = -1

    # beam search
    curriculum = []
    records = {
        "part_states": [],
        "prev_inds": [],
    }

    while (part_states.shape[0] > 0 and not terminate(part_states, boundary_part_ids).any()):
        iter += 1

        part_states, prev_inds, verifying_flag = forward_actions(part_states, n_robot, boundary_part_ids)

        verifying_states = part_states[verifying_flag, :]
        verifying_prev_inds = prev_inds[verifying_flag]

        nflag = np.logical_not(verifying_flag)
        part_states = part_states[nflag, :]
        prev_inds = prev_inds[nflag]

        if verifying_flag.sum() > 0:
            timer = time.perf_counter()
            _, stability_flag = learn2assemble.simulator.simulate(verifying_states, settings)
            n_sim = verifying_states.shape[0]
            print("step:\t", iter,
                  ",\t sim:\t", n_sim,
                  ",\t time:\t", round((time.perf_counter() - timer) / n_sim, 3))

            if stability_flag.sum() > 0:
                part_states = np.vstack([verifying_states[stability_flag, :], part_states])
                prev_inds = np.hstack([verifying_prev_inds[stability_flag], prev_inds])

        if part_states.shape[0] > 0:
            part_states, prev_inds = cluster(part_states, prev_inds, n_beam)
            records["part_states"].append(part_states.copy())
            records["prev_inds"].append(prev_inds.copy())
            curriculum.append(part_states)
        else:
            return False, compute_solution(records), np.vstack(curriculum)

    flag = terminate(part_states, boundary_part_ids)
    records["part_states"][-1] = records["part_states"][-1][flag, :]
    records["prev_inds"][-1] = records["prev_inds"][-1][flag]
    return True, compute_solution(records), np.vstack(curriculum)

if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import torch
    from learn2assemble.render import draw_assembly, init_polyscope, draw_contacts

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/dome")

    settings = {
        "contact_settings": {
            "shrink_ratio": 0.0,
        },
        "rbe": {
            "density": 1E3,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "boundary_part_ids": [len(parts) - 1],
            "verbose": False,
        },
        "admm": {
            "evaluate_it": 200,
            "max_iter": 4000,
            "float_type": torch.float32,
        },
        "curriculum": {
            "n_beam": 128,
            "n_robot": 2,
        }
    }

    contacts = compute_assembly_contacts(parts, settings)
    succeed, solution, curriculum = forward_curriculum(parts, contacts, settings)

    # import polyscope as ps
    # import polyscope.imgui as psim
    # step_id = 0
    # init_polyscope()
    # draw_assembly(parts, solution[-1, :])
    #
    # def interact():
    #     global step_id
    #     changed, step_id = psim.SliderInt("step", v=step_id, v_min=0, v_max=solution.shape[0] - 1)
    #     if changed:
    #         ps.remove_all_structures()
    #         ps.remove_all_groups()
    #         draw_assembly(parts, solution[step_id, :])
    #
    # ps.set_user_callback(interact)
    # ps.show()
