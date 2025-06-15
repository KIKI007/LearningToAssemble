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
from learn2assemble.buffer import RolloutBufferGNN
from learn2assemble.disassemly_env import DisassemblyEnv
from learn2assemble.curriculum import forward_curriculum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


class PPO:
    def __init__(self, env, settings):
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
                                     "num_step_anneal": 500,
                                     "lr_actor": 2e-3,
                                     "betas_actor": [0.95, 0.999],
                                 })

        ppo_config = SimpleNamespace(**ppo_config)
        self.graph_constructor = GFTFGraphConstructor(env.parts, env.contacts)

        n_curriculums = env.curriculum.shape[0]
        self.saved_accuracy = 0
        self.num_failed = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.num_success = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.buffer = RolloutBufferGNN(n_robot=env.n_robot,
                                       num_curriculums=n_curriculums,
                                       gamma=ppo_config.gamma,
                                       base_entropy_weight=ppo_config.base_entropy_weight,
                                       entropy_weight_increase=ppo_config.entropy_weight_increase,
                                       max_entropy_weight=ppo_config.max_entropy_weight,
                                       num_step_anneal=ppo_config.num_step_anneal)

        self.policy = ActorCriticGATG(env.n_part, settings).to(device)
        self.optimizer_params = {'params': self.policy.actor.parameters(),
                                 'lr': ppo_config.lr_actor,
                                 'weight_decay': 0,
                                 'betas': ppo_config.betas_actor}
        self.optimizer = torch.optim.Adam([self.optimizer_params])
        self.MseLoss = nn.MSELoss(reduction='none')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=ppo_config.lr_milestones)

        self.policy_old = ActorCriticGATG(env.n_part, settings).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self,
                      part_states: np.ndarray,
                      masks: np.ndarray,
                      env_inds: np.ndarray):
        # build graphs
        masks = torch.tensor(masks, dtype=floatType, device=device)
        part_states = torch.tensor(part_states, dtype=intType, device=device)
        graphs = self.graph_constructor.graphs(part_states)

        # to batch graphs
        batch_graph = torch_geometric.data.Batch.from_data_list(graphs)

        # inference
        self.policy_old.eval()
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(batch_graph, masks)

        # add to buffer
        namelist = ['states', 'actions', 'logprobs', "state_values", "masks"]
        variables = [graphs, action, action_logprob, state_val, masks]
        for id, name in enumerate(namelist):
            self.buffer.add(name, variables[id], env_inds)
        return action

    def compute_policy(self,
                       part_states: np.ndarray,
                       masks: np.ndarray):
        # build graphs
        masks = torch.tensor(masks, dtype=floatType, device=device)
        part_states = torch.tensor(part_states, dtype=intType, device=device)
        graphs = self.graph_constructor.graphs(part_states)

        # to batch graphs
        batch_graph = torch_geometric.data.Batch.from_data_list(graphs)

        # inference
        self.policy_old.eval()
        with torch.no_grad():
            action_probs, state_val = self.policy_old.actor(batch_graph)
            action_probs = masks * action_probs + masks * self.policy_old.mask_prob
        action_probs /= action_probs.sum(1, keepdim=True)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, action_probs, state_val

    def update_policy(self, batch_size=None, update_iter=5):

        # extract training dataset from buffer
        self.buffer.set_policy(self.policy_old)
        dataset = self.buffer.build_dataset(batch_size=batch_size)

        if self.buffer.num_step_anneal > 0:
            self.buffer.num_step_anneal -= 1
            self.buffer.beta += self.buffer.slope

        # record
        loss = []
        entropy = []

        # dl = torch_geometric.utils.data.DataLoader(dataset,pin_memory=True)
        self.policy.train(True)
        if self.scheduler is not None:
            self.scheduler.step()

        # reset optimizer
        self.optimizer_params['lr'], *_ = self.scheduler.get_last_lr()
        self.optimizer = torch.optim.Adam([self.optimizer_params])

        for epoch_index in range(update_iter):
            self.sur_loss = 0
            self.val_loss = 0
            self.optimizer.zero_grad()
            for i, batch in enumerate(dataset):
                if len(batch) > 0:
                    loss_all, sur_loss, val_loss, entropy_batch = self.train_one_epoch(*batch,
                                                                                       nsampletot=dataset.nsamples,
                                                                                       first=(i == 0))
                    self.sur_loss += sur_loss
                    self.val_loss += val_loss
                    loss.append(loss_all)
                    entropy.append(entropy_batch)
            self.optimizer.step()
        sampled, n = torch.unique(dataset.curriculum_id.to(device), return_counts=True)
        self.buffer.curriculum_cumsurloss[sampled] /= (n * update_iter)
        self.buffer.curriculum_rank = torch.argsort(torch.argsort(self.buffer.curriculum_cumsurloss, descending=True))

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        num_training_data = self.buffer.num_training_data
        self.buffer.clear()

        # avoid empty list
        if len(entropy) == 0:
            entropy = [0]
            loss = [0]
        return np.sum(loss), np.mean(entropy), num_training_data

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
                        nsampletot=None,
                        first=False):

        # Evaluating old actions and values
        if nsampletot is not None:
            scale = old_actions.shape[0] / nsampletot
        else:
            scale = 1
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

        self.buffer.curriculum_cumvalloss = self.buffer.curriculum_cumvalloss.scatter_reduce(0, curriculum_id,
                                                                                             value_loss.detach(),
                                                                                             reduce='sum',
                                                                                             include_self=not first)

        self.buffer.curriculum_cumsurloss = self.buffer.curriculum_cumsurloss.scatter_reduce(0, curriculum_id,
                                                                                             torch.abs(
                                                                                                 surrogate_loss.detach()),
                                                                                             reduce='sum',
                                                                                             include_self=not first)

        entropy_mean = dist_entropy.mean().item()
        loss *= scale
        loss_all = loss.item()
        loss.mean().backward()

        return loss_all, scale * surrogate_loss.mean().item(), scale * value_loss.mean().item(), entropy_mean

    def save(self, checkpoint_path, print_info=None):
        if print_info is None:
            print_info = f"Untested policy; Accuracy on the prioritized replay buffer: {self.saved_accuracy}"
        d = {'config': wandb.config.as_dict(),
             'state_dict': self.policy_old.state_dict(),
             'print_info': print_info}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, checkpoint_path):
        warn(
            "Legacy loader for the policy. If the file name ends with .pol, use init_ppo with the path to the file to load it")
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False))
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False))


def init_ppo_agent(parts: list[Trimesh],
                   contacts: dict,
                   settings: dict):

    env = DisassemblyEnv(parts, contacts, settings=settings)
    torch_geometric.seed.seed_everything(settings["env"]["seed"])

    _, _, curriculum = forward_curriculum(parts, contacts, settings=settings)
    env.set_curriculum(curriculum)

    ppo_agent = PPO(env, settings)
    return ppo_agent


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import torch

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    settings = {
        "contact_settings": {
            "shrink_ratio": 0.0,
        },
        "env": {
            "n_robot": 2,
            "boundary_part_ids": [0],
            "sim_buffer_size": 1024,
            "num_rollouts": 512,
        },
        "rbe": {
            "density": 1E2,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "verbose": False,
        },
        "admm": {
            "Ccp": 1E6,
            "evaluate_it": 100,
            "max_iter": 1000,
            "float_type": torch.float32,
        },
        "search": {
            "n_beam": 64,
        },
        "policy": {
            "gat_layers": 8,
            "gat_heads": 1,
            "gat_hidden_dims": 16,
            "gat_dropout": 0.0,
            "centroids": False
        },
        "ppo": {
            "gamma": 0.95,
            "eps_clip": 0.2,

            "base_entropy_weight": 0.005,
            "entropy_weight_increase": 0.001,
            "max_entropy_weight": 0.01,

            "lr_milestones": [100, 300],
            "num_step_anneal": 500,
            "lr_actor": 2e-3,
            "betas_actor": [0.95, 0.999],
        }
    }

    contacts = compute_assembly_contacts(parts)
    ppo_agent = init_ppo_agent(parts, contacts, settings)
