import numpy as np
from learn2assemble.curriculum import check_terminate
import learn2assemble.simulator
from time import perf_counter
from trimesh import Trimesh

class DisassemblyEnv:
    def __init__(self,
                 parts: list[Trimesh],
                 contacts: dict,
                 n_robot:int,
                 settings: dict):

        self.n_robot = n_robot
        self.n_part = len(parts)
        self.boundary = settings["rbe"]["boundary_part_ids"]

        self.record = {}
        self.dict_time = 0
        self.sim_time = 0
        self.num_sim = 0

        learn2assemble.simulator.init(parts, contacts, settings)
        self.settings = settings
        self.curriculum = np.ones((1, len(parts)), dtype=np.int32)

    @property
    def action_dim(self):
        return self.n_part * 2

    @property
    def state_dim(self):
        return self.n_part

    def reset(self, num_envs: int, weights = None, replacement = True):
        nc = len(self.curriculum)
        inds = np.arange(nc)
        if weights is None or (nc < num_envs and not replacement):
            if nc > num_envs:
                inds = inds[np.randperm(nc)]
                inds = inds[:num_envs]
        else:
            inds = np.multinomial(weights, num_envs, replacement=replacement)
        return self.curriculum[inds, :]

    def simulate(self, part_states: np.ndarray, sub_inds: np.ndarray):
        unstable_inds, sim_inds = self.check_record(part_states, sub_inds)

        if sim_inds.shape[0] > 0:
            # simulatuon
            start_timer = perf_counter()
            sim_states = part_states[sim_inds, :]
            _, stable_flag = learn2assemble.simulator.simulate(sim_states, self.settings)
            self.sim_time += perf_counter() - start_timer
            self.num_sim = sim_inds.shape[0]

            # record
            start_timer = perf_counter()
            for id in range(sim_states.shape[0]):
                state = sim_states[id, :]
                state_encode = tuple(state.tolist())
                self.record[state_encode] = stable_flag[id]
                if not stable_flag[id]:
                    unstable_inds.append(sim_inds[id])
            self.dict_time += perf_counter() - start_timer

        return unstable_inds, None

    def check_record(self, part_states, sub_inds):
        unstable_inds = []
        sim_inds = []
        # check record
        start_timer = perf_counter()
        for ind in sub_inds:
            state = part_states[ind, :]
            state_encode = tuple(state.tolist())
            if state_encode not in self.record:
                sim_inds.append(ind)
            else:
                flag = self.record[state_encode]
                if not flag:
                    unstable_inds.append(ind)
        self.dict_time += perf_counter() - start_timer
        return np.array(unstable_inds), np.array(sim_inds)

    def mask(self, part_states):
        installed_states = part_states >= 1
        fixed_states = part_states >= 2

        # installed & not fixed parts can be held
        hold_mask = np.logical_and(installed_states == 1, fixed_states == 0)
        # the total held parts should be smaller the nrobot
        robot_flag = (np.sum(fixed_states, axis=1) + 1) <= self.n_robot + len(self.boundary)
        hold_mask = hold_mask * robot_flag[:, None]

        # installed & fixed & removable parts can be removed
        remove_mask = np.logical_and(installed_states == 1, fixed_states == 1)
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
             actions: np.ndarray,
             use_simulation = True):

        # batch size
        n_batch = part_states.shape[0]
        inds = np.arange(n_batch)

        # new bin states
        new_part_states = self.next_states(part_states, actions)

        # mask
        masks = self.mask(part_states)
        
        # reward
        valid_action_flag = masks[inds, actions]
        if not np.all(valid_action_flag):
            print("Illegal action taken")
        rewards = np.where(valid_action_flag, 0, -1).astype(np.int32)

        # success termination
        success_flag = np.logical_and(check_terminate(part_states, self.boundary), valid_action_flag)
        rewards[success_flag] = 1

        # simulation
        remove_flag = actions >= self.n_part
        sim_flag = np.logical_and(valid_action_flag, remove_flag)
        sim_inds = inds[sim_flag]
        done = np.ones(part_states.shape[0], dtype=np.bool)
        if use_simulation:
            unstable_inds, _ = self.simulate(new_part_states, sim_inds)
        else:
            unstable_inds, sim_inds = self.check_record(new_part_states, sim_inds)
            done[sim_inds] = False

        notdone = np.logical_not(done)
        new_part_states[notdone, :] = new_part_states[notdone, :].clone()
        rewards[notdone] = np.where(rewards[notdone] < 0, -1, 0)

        unstable_inds = np.tensor(unstable_inds)
        if unstable_inds.shape[0] > 0:
            rewards[unstable_inds] = -1

        return new_part_states, rewards, done