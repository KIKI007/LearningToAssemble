import numpy as np
import torch
import torch_geometric
import wandb
from torch.cuda import Stream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


class RolloutDataset(torch_geometric.data.Dataset):
    def __init__(self, transform=None, pre_transform=None, batch_size=100):
        super().__init__(None, transform, pre_transform)
        self._indices = None
        self.batch_size = batch_size
        self.nstates = 0
        self.nstreams = 5
        self.tos = Stream(device)

    def __getitem__(self, idx):
        if idx < len(self) - 1:
            return self.batch_data_from_to(idx * self.batch_size, (idx + 1) * self.batch_size)
        else:
            return self.batch_data_from_to(idx * self.batch_size, len(self.states))

    def randperm(self):
        self.perm_inds = torch.randperm(self.nstates).to('cpu', dtype=intType)

    def __len__(self):
        if hasattr(self, "states"):
            return int(np.ceil(len(self.states) / self.batch_size))
        else:
            return 0

    def add_states(self, states):
        self.states = torch.stack(states)
        self.nstates = len(states)

    def batch_data_from_to(self, idx_start, idx_end):
        if not hasattr(self, "perm_inds"):
            self.perm_inds = torch.arrange(0, self.nstates).to('cpu', dtype=intType)
        sub_inds = self.perm_inds[idx_start:idx_end]
        with torch.cuda.stream(self.tos):
            return (torch.index_select(self.states, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.actions, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.masks, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.logprobs, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.rewards, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.advantages, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.weights, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.entropy_weights, 0, sub_inds).to(device, non_blocking=True),
                    torch.index_select(self.curriculum_id, 0, sub_inds).to(device, non_blocking=True))


class RolloutBuffer:

    def __init__(self,
                 n_robot,
                 gamma,
                 base_entropy_weight,
                 entropy_weight_increase,
                 max_entropy_weight,
                 per_alpha,
                 per_beta,
                 per_num_anneal):

        self.n_robot = n_robot
        self.gamma = gamma
        self.base_entropy_weight = base_entropy_weight
        self.entropy_weight_increase = entropy_weight_increase
        self.max_entropy_weight = max_entropy_weight

        # priortize experience replay
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increase = (1.0 - self.per_beta) / per_num_anneal
        self.num_step_anneal = per_num_anneal

        self.clear_replay_buffer()
        self.reset_curriculum(0)

    def update_per_beta(self):
        if self.num_step_anneal > 0:
            self.num_step_anneal -= 1
            self.per_beta += self.per_beta_increase

    def modify_entropy(self):
        who_failed = self.num_failed > 0
        who_success = self.num_success > 0
        self.entropy_weights[who_failed] = self.entropy_weights[who_failed] + self.entropy_weight_increase
        self.entropy_weights[who_failed] = torch.clamp(self.entropy_weights[who_failed], min=0,
                                                       max=self.max_entropy_weight)
        self.entropy_weights[who_success] = self.base_entropy_weight

    def reset_curriculum(self, n_curriculums):
        self.num_failed = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.num_success = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.curriculum_visited = torch.zeros(n_curriculums, device=device, dtype=torch.bool)
        self.curriculum_cumsurloss = torch.zeros(n_curriculums, device=device, dtype=floatType)
        self.curriculum_rank = torch.arange(n_curriculums, dtype=intType, device=device)
        self.entropy_weights = self.base_entropy_weight * torch.ones((n_curriculums), dtype=floatType, device=device)

    def clear_replay_buffer(self, num_env=0):
        self.actions = [[] for _ in range(num_env)]
        self.states = [[] for _ in range(num_env)]
        self.logprobs = [[] for _ in range(num_env)]
        self.state_values = [[] for _ in range(num_env)]
        self.masks = [[] for _ in range(num_env)]
        self.part_states = [[] for _ in range(num_env)]
        self.next_states = [[] for _ in range(num_env)]
        self.next_stability = [[] for _ in range(num_env)]
        self.rewards_per_step = [[] for _ in range(num_env)]

        self.num_envs = num_env
        self.rewards = None  # one reward for each env
        self.weights = None  # one weight for each env

    def truncate(self, env_id, step_id):
        names = ["actions", "states", "logprobs", "state_values", "masks", "part_states", "next_states",
                 "next_stability", "rewards_per_step"]
        for name in names:
            self.__dict__[name][env_id] = self.__dict__[name][env_id][: step_id]

    def sample_curriculum(self, num_rollouts, complete_assembly_ratio = 0):
        visited = self.curriculum_visited
        rank = self.curriculum_rank
        num_nonvisited = (visited == False).sum()
        sample_new_curriculum = (num_nonvisited > 0)
        if sample_new_curriculum:
            sample_weights = torch.zeros_like(visited).to(floatType)
            sample_weights[visited == False] = 1.0 / (visited == False).sum()
            if num_rollouts > num_nonvisited:
                curriculum_inds = torch.multinomial(sample_weights, num_rollouts, True)
            else:
                curriculum_inds = torch.multinomial(sample_weights, num_rollouts, False)
        else:
            inds = torch.arange(1, rank.shape[0] + 1, device=device, dtype=floatType)
            sample_weights = torch.pow(inds, -self.per_alpha) / torch.pow(inds, -self.per_alpha).sum()
            curriculum_inds = torch.multinomial(sample_weights[rank], num_rollouts, True)
        # add complete assemble
        complete_assembly_inds = torch.arange(num_rollouts - max(int (num_rollouts * complete_assembly_ratio), 1), num_rollouts, device=device, dtype=intType)
        curriculum_inds[complete_assembly_inds] = len(self.curriculum_rank) - 1
        return curriculum_inds.cpu(), sample_new_curriculum, complete_assembly_inds.cpu()

    def get_current_states_for_rendering(self, num_render):
        batch_part_states = []
        batch_stability = []
        num_render = min(num_render, self.num_envs)
        inds = np.arange(start=self.num_envs - num_render, stop=self.num_envs)
        for env_id in inds:
            batch_part_states.append(self.next_states[env_id][-1])
            batch_stability.append(self.next_stability[env_id][-1])
        batch_part_states = np.vstack(batch_part_states)
        batch_stability = np.array(batch_stability)
        return {
            "type": "render",
            "data": [batch_part_states, batch_stability]
        }

    def get_valid_env_inds(self, env):
        if env.updated_simulation:
            for env_id in range(self.num_envs):
                for step_id in range(len(self.next_stability[env_id])):
                    if self.next_stability[env_id][step_id] == 0:
                        stability = env.get_history(self.next_states[env_id][step_id])
                        self.next_stability[env_id][step_id] = stability
                        if stability == -1:
                            self.rewards_per_step[env_id][step_id] = -1
                            self.truncate(env_id, step_id + 1)
                            break
            env.updated_simulation = False

        # update enviroment inds
        env_inds = []
        rewards = []
        for env_id in range(self.num_envs):
            data = self.rewards_per_step[env_id]
            if len(data) == 0:
                rewards.append(0)
            else:
                if data[-1] == 0:
                    env_inds.append(env_id)
                rewards.append(data[-1])

        return np.array(env_inds), np.array(rewards)

    def add(self, name, vals, env_inds):
        for ival, ienv in enumerate(env_inds):
            if torch.is_tensor(vals[ival]):
                val = vals[ival].detach().cpu()
            else:
                val = vals[ival]
            self.__dict__[name][ienv].append(val)

    def build_dataset(self, batch_size):
        dataset = RolloutDataset(batch_size=batch_size)
        actions = []
        logprobs = []
        state_values = []
        masks = []
        states = []
        rewards = []
        curriculum_id = []

        for ienv in range(self.num_envs):
            discounted_reward = self.rewards[ienv]
            for istep in torch.arange(len(self.actions[ienv]) - 1, -1, -1):
                # collect dataset
                rewards.append(discounted_reward.pin_memory())
                states.append(self.states[ienv][istep])
                actions.append(self.actions[ienv][istep].pin_memory())
                logprobs.append(self.logprobs[ienv][istep].pin_memory())
                state_values.append(self.state_values[ienv][istep].pin_memory())
                masks.append(self.masks[ienv][istep].pin_memory())
                curriculum_id.append(self.curriculum_inds[ienv].pin_memory())
                discounted_reward = self.gamma * discounted_reward

        if len(masks) > 0:
            state_values = torch.tensor(state_values, device="cpu", dtype=floatType, pin_memory=True).detach()

            # dataset
            dataset.actions = torch.tensor(actions, device="cpu", dtype=intType, pin_memory=True).detach()
            dataset.masks = torch.vstack(masks)
            dataset.logprobs = torch.tensor(logprobs, device="cpu", dtype=floatType, pin_memory=True).detach()
            dataset.rewards = torch.tensor(rewards, device="cpu", dtype=floatType, pin_memory=True).detach()

            # std adv
            adv = dataset.rewards - state_values
            dataset.advantages = ((adv - adv.mean()) / (adv.std(unbiased=False) + 1e-7)).pin_memory()

            # states
            dataset.add_states(states)
            dataset.curriculum_id = torch.hstack(curriculum_id)

            dataset.weights = torch.pow(1 + self.curriculum_rank[dataset.curriculum_id],
                                        self.per_alpha - self.per_beta).cpu().pin_memory()
            dataset.weights /= dataset.weights.max()
            dataset.entropy_weights = self.entropy_weights[dataset.curriculum_id].cpu().pin_memory()
        return dataset