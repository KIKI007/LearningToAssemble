import numpy as np
import learn2assemble.simulator
from time import perf_counter
from trimesh import Trimesh


class Timer:
    def __init__(self):
        self.timer_dict = {}
        self.accumulator = {}

    def start(self, name):
        self.timer_dict[name] = perf_counter()

    def stop(self, name, accumulate=False):
        if name not in self.timer_dict:
            self.timer_dict[name] = perf_counter()
        elapse = perf_counter() - self.timer_dict[name]
        if accumulate and name in self.accumulator:
            self.accumulator[name] += elapse
        else:
            self.accumulator[name] = elapse
        return elapse, self.accumulator[name]

class DisassemblyEnv:
    def __init__(self,
                 parts: list[Trimesh],
                 contacts: dict,
                 n_robot: int,
                 sim_buffer_size: int,
                 settings: dict):

        self.parts = parts
        self.contacts = contacts
        self.settings = settings
        self.n_robot = n_robot
        self.n_part = len(parts)

        self.stability_history = {}
        self.timer = Timer()

        self.boundary_part_ids = settings["rbe"]["boundary_part_ids"]
        self.n_max_held_part = n_robot + len(self.boundary_part_ids)
        self.curriculum = np.ones((1, len(parts)), dtype=np.int32)
        self.curriculum[0, self.boundary_part_ids] = 2

        self.add_stability_history(self.curriculum, np.ones(len(self.curriculum), dtype=np.bool_))

        self.sim_buffer = []
        self.buffer_size = sim_buffer_size

    @property
    def action_dim(self):
        return self.n_part * 2

    @property
    def state_dim(self):
        return self.n_part

    def check_terminate(self,
                        part_states: np.ndarray):
        fixed_states = np.zeros(self.n_part)
        fixed_states[self.boundary_part_ids] = 2
        dist = np.sum(np.abs(part_states - fixed_states[None, :]), axis=1)
        return dist == 0

    def reset(self, num_envs: int, weights=None, replacement=True):
        nc = len(self.curriculum)
        if weights is None:
            weights = np.ones(nc) / nc
        inds = np.random.choice(a=nc, size=num_envs, replace=replacement, p=weights)
        part_states = self.curriculum[inds, :]

        self.trajectories = [
            [{
                "state": part_state.copy(),
                "reward": 0,
                "stability": 1,
            }] for part_state in part_states]
        self.update_buffer = False
        return part_states

    def simulate_buffer(self, simulate_remain=False):
        self.timer.start("simulation")
        while len(self.sim_buffer):
            num_to_sim = min(self.buffer_size, len(self.sim_buffer))
            if num_to_sim < self.buffer_size and not simulate_remain:
                break
            part_states = np.vstack(self.sim_buffer[:num_to_sim])
            _, stable_flag = learn2assemble.simulator.simulate(self.parts, self.contacts, part_states, self.settings)
            stable_flag = stable_flag.astype(np.int32) * 2 - 1
            self.add_stability_history(part_states, stable_flag)
            self.sim_buffer = self.sim_buffer[num_to_sim:]
            self.update_buffer = True
        self.timer.stop("simulation", True)

    def sample_actions(self, part_states: np.ndarray):
        masks = self.action_masks(part_states)
        actions = []
        for id, mask in enumerate(masks):
            if sum(mask) > 0:
                weights = np.ones(self.action_dim, dtype=np.float32)
                weights[~mask] = 0
                weights /= np.sum(weights)
                action = np.random.choice(a=self.action_dim, size=1, p=weights, replace=True)
            else:
                print(part_states[id, :])
                action = 0
            actions.append(action)
        return np.array(actions).reshape(-1)

    def add_stability_history(self, part_states, stable_flag):
        self.timer.start("history")
        for ind, part_state in enumerate(part_states):
            state_encode = tuple(part_state.tolist())
            self.stability_history[state_encode] = stable_flag[ind]
        self.timer.stop("history", True)

    def get_stability_history(self, part_states, actions):

        self.timer.start("history")
        for ind, part_state in enumerate(part_states):
            state_encode = tuple(part_state.tolist())
            if state_encode not in self.stability_history:
                if actions[ind] < self.n_part:
                    self.stability_history[state_encode] = 1
                else:
                    self.stability_history[state_encode] = 0
                self.sim_buffer.append(part_state)
        self.timer.stop("history", True)

        self.simulate_buffer()

        self.timer.start("history")
        results = []
        for ind, part_state in enumerate(part_states):
            state_encode = tuple(part_state.tolist())
            results.append(self.stability_history[state_encode])
        self.timer.stop("history", True)

        return np.array(results, dtype=np.int32)

    def action_masks(self, part_states: np.ndarray):
        part_installed_states = part_states >= 1
        part_held_states = part_states >= 2

        # installed & not fixed parts can be held
        hold_mask = np.logical_and(part_installed_states == 1, part_held_states == 0)
        # the total held parts should be smaller the nrobot
        robot_flag = (np.sum(part_held_states, axis=1) + 1) <= self.n_max_held_part
        hold_mask = hold_mask * robot_flag[:, None]

        # installed & fixed & removable parts can be removed
        remove_mask = np.logical_and(part_installed_states == 1, part_held_states == 1)
        remove_mask[:, self.boundary_part_ids] = 0
        return np.hstack([hold_mask, remove_mask])

    def next_states(self, part_states: np.ndarray, actions: np.ndarray):
        # copy
        new_part_states = part_states.copy()
        inds = np.arange(part_states.shape[0])

        # hold action
        hold_action_flag = actions < self.n_part
        new_part_states[inds[hold_action_flag], actions[hold_action_flag]] = 2

        # remove action
        remove_action_flag = actions >= self.n_part
        new_part_states[inds[remove_action_flag], actions[remove_action_flag] - self.n_part] = 0
        return new_part_states

    def step(self,
             part_states: np.ndarray,
             actions: np.ndarray):

        # batch size
        n_batch = part_states.shape[0]
        inds = np.arange(n_batch)

        # new bin states
        new_part_states = self.next_states(part_states, actions)

        # mask
        masks = self.action_masks(part_states)

        # reward
        valid_action_flag = masks[inds, actions]
        rewards = np.where(valid_action_flag, 0, -1).astype(np.int32)  # 2 means not know

        if not np.all(valid_action_flag):
            # all action must be valid
            print("Illegal action taken")

        # success termination
        success_flag = np.logical_and(self.check_terminate(new_part_states), valid_action_flag)
        rewards[success_flag] = 1

        stability = self.get_stability_history(new_part_states, actions)
        rewards = np.where(stability == -1, -1, rewards)

        return new_part_states, rewards, stability

    def update_trajectories(self):
        self.timer.start("history")
        # update buffer
        if self.update_buffer:
            for env_id in range(len(self.trajectories)):
                for jd, step in enumerate(self.trajectories[env_id]):
                    if step["stability"] == 0:
                        state_encode = tuple(step["state"].tolist())
                        step["stability"] = self.stability_history[state_encode]
                    if step["stability"] == -1:
                        step["reward"] = -1
                        self.trajectories[env_id] = self.trajectories[env_id][: jd + 1]
                        break
            self.update_buffer = False

        # update enviroment inds
        env_inds = []
        for env_id in range(len(self.trajectories)):
            if self.trajectories[env_id][-1]["reward"] == 0:
                env_inds.append(env_id)
        self.timer.stop("history", True)
        return  np.array(env_inds)

if __name__ == "__main__":
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import torch

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/rulin")
    settings = {
        "contact_settings": {
            "shrink_ratio": 0.0,
        },
        "rbe": {
            "density": 1E4,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "boundary_part_ids": [0],
            "verbose": False,
        },
        "admm": {
            "Ccp": 1E6,
            "evaluate_it": 200,
            "max_iter": 2000,
            "float_type": torch.float32,
        }
    }

    num_envs = 100000
    contacts = compute_assembly_contacts(parts, settings)
    env = DisassemblyEnv(parts, contacts, n_robot=2, sim_buffer_size=100000, settings=settings)

    part_states = env.reset(num_envs=num_envs)
    env_inds = np.arange(num_envs)
    n_step = 0
    while True:
        current_part_states = part_states[env_inds, :]
        current_actions = env.sample_actions(current_part_states)
        new_states, rewards, stability = env.step(current_part_states, current_actions)
        part_states[env_inds, :] = new_states
        for id, env_id in enumerate(env_inds):
            env.trajectories[env_id].append({"state": new_states[id].copy(),
                                              "reward": rewards[id],
                                              "stability": stability[id]
                                              })
        env_inds = env.update_trajectories()
        print(f"n_step {n_step},\t rollout {len(env_inds)}", end=",\t ")
        print(f"sim {env.timer.accumulator['simulation']: .2f},\t history {env.timer.accumulator['history']: .2f}")
        n_step += 1
        if env_inds.shape[0] == 0:
            break

    env.simulate_buffer(simulate_remain=True)
    env.update_trajectories()

    import polyscope as ps
    env_id = 0
    step_id = 0
    init_polyscope()

    solutions = []
    max_length = 0
    for traj in env.trajectories:
        max_length = max(max_length, len(traj))
    for traj in env.trajectories:
        if len(traj) == max_length:
            solutions.append(traj)

    def interact():
        global step_id, env_id
        render = False
        changed, env_id = psim.SliderInt("env", v=env_id, v_min=0, v_max=len(solutions) - 1)
        if changed:
            step_id = 0
        render = render or changed
        changed, step_id = psim.SliderInt("step", v=step_id, v_min=0, v_max=len(solutions[env_id]) - 1)
        render = render or changed
        if render:
            ps.remove_all_structures()
            ps.remove_all_groups()
            draw_assembly(parts, solutions[env_id][step_id]["state"])

    draw_assembly(parts, env.curriculum[0])
    ps.set_user_callback(interact)
    ps.show()