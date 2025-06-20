from types import SimpleNamespace

import numpy as np
import learn2assemble.simulator
from time import perf_counter

from trimesh import Trimesh
from learn2assemble import update_default_settings
from learn2assemble.buffer import RolloutBuffer
from multiprocessing import Process, Queue



class Timer:
    def __init__(self):
        self.timer_dict = {}
        self.accumulator = {}

    def __call__(self, name):
        if name in self.accumulator:
            return self.accumulator[name]
        else:
            return 0.0

    def start(self, name):
        self.timer_dict[name] = perf_counter()

    def stop(self, name):
        if name not in self.timer_dict:
            self.timer_dict[name] = perf_counter()
        elapse = perf_counter() - self.timer_dict[name]
        if name in self.accumulator:
            self.accumulator[name] += elapse
        else:
            self.accumulator[name] = elapse
        return elapse, self.accumulator[name]


class DisassemblyEnv:
    def __init__(self,
                 parts: list[Trimesh],
                 contacts: dict,
                 settings: dict = {}):

        env_settings = update_default_settings(settings,
                                   "env",
                                               {
                                       "n_robot": 2,
                                       "boundary_part_ids": [0],
                                       "sim_buffer_size": 64,
                                       "num_rollouts": 64,
                                       "seed": 0,
                                       "verbose": False,
                                   })

        env_settings = SimpleNamespace(**env_settings)
        self.settings = settings

        self.parts = parts
        self.contacts = contacts
        self.verbose = env_settings.verbose

        self.n_robot = env_settings.n_robot
        self.n_part = len(parts)
        self.num_rollouts = env_settings.num_rollouts
        self.boundary_part_ids = env_settings.boundary_part_ids
        self.n_max_held_part = self.n_robot + len(self.boundary_part_ids)
        self.sim_buffer_size = env_settings.sim_buffer_size

        self.timer = Timer()
        self.sim_buffer = []
        self.stability_history = {}
        self.updated_simulation = False

        # set the complete assembly state as curriculum
        curriculum = np.ones((1, len(parts)), dtype=np.int32)
        curriculum[0, self.boundary_part_ids] = 2
        self.set_curriculum(curriculum)

    @property
    def action_dim(self):
        return self.n_part * 2

    @property
    def state_dim(self):
        return self.n_part

    def set_curriculum(self, curriculum: np.ndarray):
        self.curriculum = curriculum.copy()
        stable_flag = np.ones(len(self.curriculum), dtype=np.int32)
        self.add_stability_history(self.curriculum, stable_flag)

    def check_terminate(self,
                        part_states: np.ndarray):
        fixed_states = np.zeros(self.n_part)
        fixed_states[self.boundary_part_ids] = 2
        dist = np.sum(np.abs(part_states - fixed_states[None, :]), axis=1)
        return dist == 0

    def reset(self, weights=None, replacement=True):
        nc = len(self.curriculum)
        if weights is None:
            weights = np.ones(nc) / nc
        inds = np.random.choice(a=nc, size=self.num_rollouts, replace=replacement, p=weights)
        part_states = self.curriculum[inds, :]
        return part_states, np.arange(self.num_rollouts)

    def simulate_buffer(self, simulate_remain=False):

        while len(self.sim_buffer):
            # check whether the buffer is filled
            num_to_sim = min(self.sim_buffer_size, len(self.sim_buffer))
            if num_to_sim < self.sim_buffer_size and not simulate_remain:
                break

            self.timer.start("simulation")
            part_states = np.vstack(self.sim_buffer[:num_to_sim])
            _, stable_flag = learn2assemble.simulator.simulate(self.parts, self.contacts, part_states, self.settings)
            elapse, _ = self.timer.stop("simulation")
            if self.verbose:
                print(f"num_sim {part_states.shape[0]}, \t avg_sim {round(elapse / part_states.shape[0], 4)}")

            stable_flag = stable_flag.astype(np.int32) * 2 - 1
            self.add_stability_history(part_states, stable_flag)
            self.sim_buffer = self.sim_buffer[num_to_sim:]
            self.updated_simulation = True

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
        self.timer.stop("history")

    def get_history(self, part_state):
        state_encode = tuple(part_state.tolist())
        if state_encode in self.stability_history:
            return self.stability_history[state_encode]
        return 0

    def simulate(self, part_states, actions):

        self.timer.start("history")
        for ind, part_state in enumerate(part_states):
            state_encode = tuple(part_state.tolist())
            if actions[ind] >= self.n_part and state_encode not in self.stability_history:
                self.sim_buffer.append(part_state)  # add state into simulation buffer
        self.timer.stop("history")

        self.simulate_buffer()

        self.timer.start("history")
        results = []
        for ind, part_state in enumerate(part_states):
            if actions[ind] < self.n_part:
                results.append(1)  # stability = 1 for held actions
            else:
                results.append(self.get_history(part_state))
        self.timer.stop("history")
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

        stability = self.simulate(new_part_states, actions)
        rewards = np.where(stability == -1, -1, rewards)

        return new_part_states, rewards, stability

def env_debug(parts:list[Trimesh], settings:dict, queue: Queue):
    from learn2assemble.assembly import compute_assembly_contacts
    contacts = compute_assembly_contacts(parts, settings)
    env = DisassemblyEnv(parts, contacts, settings=settings)

    part_states, env_inds = env.reset()
    buffer = RolloutBuffer(n_robot=env.n_robot,
                           gamma=0,
                           base_entropy_weight=0,
                           entropy_weight_increase=0,
                           max_entropy_weight=0,
                           per_alpha=0.8,
                           per_beta=0.1,
                           per_num_anneal=1)
    buffer.clear_replay_buffer(env.num_rollouts)
    buffer.reset_curriculum(1)
    n_step = 0
    while True:
        current_part_states = part_states[env_inds, :]
        current_actions = env.sample_actions(current_part_states)
        next_states, rewards, next_stability = env.step(current_part_states, current_actions)
        part_states[env_inds, :] = next_states

        buffer.add("part_states", current_part_states, env_inds)
        buffer.add("next_states", next_states, env_inds)
        buffer.add("rewards_per_step", rewards, env_inds)
        buffer.add("next_stability", next_stability, env_inds)

        if queue is not None:
            data = buffer.get_current_states_for_rendering(settings["training"]["num_render_debug"])
            queue.put(data)

        env_inds, _ = buffer.get_valid_env_inds(env)

        print(f"n_step {n_step},\t rollout {len(env_inds)}")

        n_step += 1
        if env_inds.shape[0] == 0:
            break

    env.simulate_buffer(True)
    _, rewards = buffer.get_valid_env_inds(env)

    if queue is not None:
        data = buffer.get_current_states_for_rendering(settings["training"]["num_render_debug"])
        queue.put(data)


if __name__ == "__main__":
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, update_default_settings, default_settings
    import torch
    from learn2assemble.render import render_batch_simulation
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/rulin")
    default_settings["rbe"]["density"] = 1E4
    queue = Queue()
    p1 = Process(target=env_debug, args=(parts, default_settings, queue))
    p2 = Process(target=render_batch_simulation, args=(parts, default_settings["env"]["boundary_part_ids"], queue))
    p2.start()
    p1.start()
    p1.join()
    p2.join()
