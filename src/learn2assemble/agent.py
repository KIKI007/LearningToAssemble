import pickle
from types import SimpleNamespace
from warnings import warn

import torch
from torch.distributions import Categorical
from torch import nn
import torch_geometric
import numpy as np

from trimesh import Trimesh
from time import perf_counter

from learn2assemble import set_default
from learn2assemble.graph import GFTFGraphConstructor
from learn2assemble.policy import ActorCriticGATG
from learn2assemble.buffer import RolloutBuffer
from learn2assemble.disassemly_env import DisassemblyEnv
from learn2assemble.curriculum import forward_curriculum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32
from learn2assemble.disassemly_env import Timer


class PPO:
    def __init__(self, parts, contacts, settings):
        self.scheduler = None
        ppo_config = set_default(settings,
                                 "ppo",
                                 {
                                     "gamma": 0.95,
                                     "eps_clip": 0.2,

                                     "base_entropy_weight": 0.005,
                                     "entropy_weight_increase": 0.001,
                                     "max_entropy_weight": 0.01,

                                     "lr_milestones": [100, 300],
                                     "lr_actor": 2e-3,
                                     "betas_actor": [0.95, 0.999],

                                     "per_alpha": 0.8,
                                     "per_beta": 0.1,
                                     "per_num_anneal": 500,
                                 })
        n_robot = settings["env"]["n_robot"]
        ppo_config = SimpleNamespace(**ppo_config)
        self.graph_constructor = GFTFGraphConstructor(parts, contacts)

        self.accuracy_of_sample_curriculum = 0
        self.accuracy_of_entire_curriculum = 0
        self.episode = 0
        self.deterministic = False
        self.eps_clip = ppo_config.eps_clip
        self.timer = Timer()

        self.buffer = RolloutBuffer(n_robot=n_robot,
                                    gamma=ppo_config.gamma,
                                    base_entropy_weight=ppo_config.base_entropy_weight,
                                    entropy_weight_increase=ppo_config.entropy_weight_increase,
                                    max_entropy_weight=ppo_config.max_entropy_weight,
                                    per_alpha=ppo_config.per_alpha,
                                    per_beta=ppo_config.per_beta,
                                    per_num_anneal=ppo_config.per_num_anneal)

        self.policy = ActorCriticGATG(len(parts), settings).to(device)
        self.optimizer_params = {
            'params': self.policy.actor.parameters(),
            'lr': ppo_config.lr_actor,
            'weight_decay': 0,
            'betas': ppo_config.betas_actor
        }
        self.optimizer = torch.optim.Adam([self.optimizer_params])
        self.MseLoss = nn.MSELoss(reduction='none')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=ppo_config.lr_milestones)

        self.policy_old = ActorCriticGATG(len(parts), settings).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self,
                      part_states: np.ndarray,
                      masks: np.ndarray,
                      env_inds: np.ndarray):
        # build graphs
        masks = torch.tensor(masks, dtype=floatType, device=device)
        graphs = self.graph_constructor.graphs(part_states)

        # to batch graphs
        batch_graph = torch_geometric.data.Batch.from_data_list(graphs).to(device)

        # inference
        self.policy_old.eval()
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(batch_graph, masks, self.deterministic)

        # add to buffer
        namelist = ['states', 'actions', 'logprobs', "state_values", "masks"]
        variables = [graphs, action, action_logprob, state_val, masks]
        for id, name in enumerate(namelist):
            self.buffer.add(name, variables[id], env_inds)

        return action.cpu().numpy()

    def update_policy(self, batch_size=None, update_iter=5):

        # extract training dataset from buffer
        dataset = self.buffer.build_dataset(batch_size=batch_size)

        # record
        loss = []
        entropy = []

        self.policy.train(True)
        self.optimizer = torch.optim.Adam([self.optimizer_params])

        for epoch_index in range(update_iter):
            sur_loss = 0
            val_loss = 0
            loss = 0
            entropy = 0
            self.optimizer.zero_grad()
            for i, batch in enumerate(dataset):
                if len(batch) > 0:
                    loss_batch, sur_loss_batch, val_loss_batch, entropy_batch = self.train_one_epoch(*batch,
                                                                                                     dataset.nstates,
                                                                                                     first_batch=(
                                                                                                                 i == 0))
                    sur_loss += sur_loss_batch
                    val_loss += val_loss_batch
                    loss += loss_batch
                    entropy += entropy_batch
            self.optimizer.step()

        sampled, n = torch.unique(dataset.curriculum_id.to(device), return_counts=True)
        self.buffer.curriculum_cumsurloss[sampled] /= (n * update_iter)
        self.buffer.curriculum_rank = torch.argsort(torch.argsort(self.buffer.curriculum_cumsurloss, descending=True))

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # reset optimizer
        self.buffer.clear_replay_buffer()
        self.scheduler.step()
        self.optimizer_params['lr'], *_ = self.scheduler.get_last_lr()

        return loss, val_loss, sur_loss, entropy

    def train_one_epoch(self,
                        old_states,
                        old_actions,
                        old_masks,
                        old_logprobs,
                        rewards,
                        advantages,
                        weights,
                        entropy_weights,
                        curriculum_id,
                        nstates,
                        first_batch):

        # Evaluating old actions and values
        batch_percentage = float(old_actions.shape[0]) / nstates
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks)

        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values).reshape(-1)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        value_loss = self.MseLoss(state_values, rewards)
        surrogate_loss = -torch.min(surr1, surr2)

        # final loss of clipped objective PPO
        loss = (surrogate_loss * weights).mean() + 0.5 * (value_loss * weights).mean() - (
                    entropy_weights * dist_entropy * weights).mean()
        self.buffer.curriculum_visited[curriculum_id] = True
        self.buffer.curriculum_cumsurloss = self.buffer.curriculum_cumsurloss.scatter_reduce(0, curriculum_id,
                                                                                             torch.abs(
                                                                                                 surrogate_loss.detach()),
                                                                                             reduce='sum',
                                                                                             include_self=not first_batch)

        entropy_mean = dist_entropy.mean().item()
        loss *= batch_percentage
        loss_mean = loss.item()
        loss.mean().backward()

        return (loss_mean,
                batch_percentage * surrogate_loss.mean().item(),
                batch_percentage * value_loss.mean().item(),
                batch_percentage * entropy_mean)

    def save(self, checkpoint_path, settings, curriculum, stability_history):
        d = {'settings': settings,
             'state_dict': self.policy_old.state_dict(),
             "curriculum": curriculum,
             "stability_history": stability_history,}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
