import pickle
from time import perf_counter
from warnings import warn
import torch
from torch.distributions import Categorical
import numpy as np
from torch import nn
import torch_geometric
from learn2assemble.graph import GFTFGraphConstructor
from learn2assemble.policy import ActorCriticGATG,ActorCriticMLP
from learn2assemble.buffer import RolloutBufferGNN, RolloutBufferMLP
from time import perf_counter
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

class PPO:
    def __init__(self, metadata, state_dim, action_dim, config,n_curriculums):
        self.scheduler = None
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.entropy_weight = config.entropy_weight
        self.saved_accuracy = 0
        self.num_failed = torch.zeros(n_curriculums,dtype=intType,device=device)
        self.num_success = torch.zeros(n_curriculums,dtype=intType,device=device)
        self.MseLoss = nn.MSELoss(reduction='none')
        self.init_policy(metadata, state_dim, action_dim, config, n_curriculums)
        if hasattr(config,"lr_milestones"):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones= config.lr_milestones)
        else:
            self.scheduler =None
        # timer
        self.reset_timer()

    def reset_timer(self):
        self.training_time = 0
        self.inference_time = 0
        self.batch_graph_time = 0
        self.build_graph_time = 0
        self.build_ds_time = 0
    def init_policy(self,metadata, state_dim, action_dim, config,n_curriculums):

        self.buffer = RolloutBufferMLP(self.gamma, config.her,n_curriculums,config.entropy_weight,config.entropy_slope,config.max_entropy_weight,num_step_anneal=config.num_step_anneal)
        
        self.policy = ActorCriticMLP(state_dim, action_dim, config).to(device)
        self.optimizer_params = [{'params': self.policy.actor.parameters(),'lr': config.lr_actor,'weight_decay':0, 'betas':config.betas_actor},
                                 {'params': self.policy.critic.parameters(),'lr': config.lr_actor,'weight_decay':0, 'betas':config.betas_actor}]

        self.optimizer = torch.optim.Adam(self.optimizer_params)
        """self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': config.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': config.lr_actor}
        ])"""
        self.policy_old = ActorCriticMLP(state_dim, action_dim, config).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, bin_states, masks, env_inds,curriculum_inds):

        # flatten
        bin_states_flat = bin_states.swapaxes(1, 2)
        bin_states_flat = bin_states_flat.flatten(start_dim=1)

        # inference
        start_timer = perf_counter()
        self.policy_old.eval()
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(bin_states_flat, masks)
        self.inference_time += perf_counter() - start_timer

        # add to buffer
        namelist = ['states', 'actions', 'logprobs', "state_values", "masks", "bin_states"]
        variables = [bin_states_flat, action, action_logprob, state_val, masks, bin_states]
        for id, name in enumerate(namelist):
            self.buffer.add(name, variables[id], env_inds)
        return action

    def reset_assembly(self, parts, contacts):
        pass

    def set_deterministic_policy(self, status = False):
        self.policy_old.deterministic = status
        self.policy.deterministic = status

    def update_policy(self, batch_size=None, shuffle=False, update_iter=5):
        # extract training dataset from buffer
        self.buffer.set_policy(self.policy_old)
        start_timer = perf_counter()
        dataset = self.buffer.build_batch_dataset(batch_size=batch_size, shuffle = shuffle)
        self.build_ds_time += perf_counter()-start_timer
        if self.buffer.num_step_anneal >0:
            self.buffer.num_step_anneal-=1
            self.buffer.beta+=self.buffer.slope

        # record
        loss = []
        entropy = []

        # update policy
        start_timer = perf_counter()
        #dl = torch_geometric.utils.data.DataLoader(dataset,pin_memory=True)
        self.policy.train(True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.reset_optim()
        for epoch_index in range(update_iter):
            self.sur_loss = 0
            self.val_loss = 0
            self.optimizer.zero_grad()
            for i, batch in enumerate(dataset):
                if len(batch) > 0:
                    loss_all, sur_loss,val_loss, entropy_batch = self.train_one_epoch(*batch,nsampletot=dataset.nsamples,first = i==0)
                    self.sur_loss +=sur_loss
                    self.val_loss += val_loss
                    loss.append(loss_all)
                    entropy.append(entropy_batch)
            self.optimizer.step()
        sampled,n = torch.unique(dataset.curriculum_id.to(device),return_counts=True)
        self.buffer.curriculum_cumsurloss[sampled] /= (n*update_iter)
        self.buffer.curriculum_rank = torch.argsort(torch.argsort(self.buffer.curriculum_cumsurloss,descending=True))
        
        self.training_time += perf_counter() - start_timer
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
    def reset_optim(self):
        new_lrs = self.scheduler.get_last_lr()
        self.optimizer_params[0]['lr'] = new_lrs[0]
        self.optimizer_params[1]['lr'] = new_lrs[1]
        self.optimizer = torch.optim.Adam(self.optimizer_params)
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
                        first = False):

        # Evaluating old actions and values
        if nsampletot is not None:
            scale = old_actions.shape[0]/nsampletot
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
        loss = (surrogate_loss*weights).mean() + 0.5 * (value_loss*weights).mean() - (entropy_weights * dist_entropy*weights).mean()
        self.buffer.curriculum_cumvalloss = self.buffer.curriculum_cumvalloss.scatter_reduce(0,curriculum_id,value_loss.detach(),reduce='sum',include_self=not first)
        self.buffer.curriculum_cumsurloss =  self.buffer.curriculum_cumsurloss.scatter_reduce(0,curriculum_id,torch.abs(surrogate_loss.detach()),reduce='sum',include_self=not first)
        
        # take gradient step
        #self.optimizer.zero_grad()
        
        entropy_mean = dist_entropy.mean().item()
        loss*=scale
        loss_all = loss.item()
        loss.mean().backward()
        #self.optimizer.step()
        #print(scale)
        return loss_all, scale*surrogate_loss.mean().item(), scale*value_loss.mean().item(), entropy_mean
    def save(self, checkpoint_path,print_info=None):
        if print_info is None:
            print_info = f"Untested policy; Accuracy on the prioritized replay buffer: {self.saved_accuracy}"
        d =  {'config':wandb.config.as_dict(),
              'state_dict':self.policy_old.state_dict(),
              'print_info': print_info}
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, checkpoint_path):
        warn("Legacy loader for the policy. If the file name ends with .pol, use init_ppo with the path to the file to load it")
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage,weights_only=False))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage,weights_only=False))
    def compute_policy(self, bin_states, masks):
        # flatten
        bin_states_flat = bin_states.swapaxes(1, 2)
        bin_states_flat = bin_states_flat.flatten(start_dim=1)

        # inference
        start_timer = perf_counter()
        self.policy_old.eval()
        with torch.no_grad():
            action_probs = self.policy_old.actor(bin_states_flat)
            state_val = self.policy_old.critic(bin_states_flat)
            action_probs = masks * action_probs + masks * self.policy_old.mask_prob
        self.inference_time += perf_counter() - start_timer
        action_probs /=  action_probs.sum(1,keepdim=True)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, action_probs, state_val

class PPO_GAT(PPO):
    def __init__(self, metadata, n_part, action_dim, config,n_curriculums):
        super().__init__(metadata, n_part, action_dim, config,n_curriculums)
    def init_policy(self, metadata, n_part, action_dim, config,n_curriculums):
        self.buffer = RolloutBufferGNN(self.gamma, config.her,n_curriculums,config.entropy_weight,config.entropy_slope,config.max_entropy_weight,num_step_anneal=config.num_step_anneal)
        if config.get('policy_type',None)=='ziqi':
            raise DeprecationWarning
        else:
            self.policy = ActorCriticGATG(metadata, n_part, action_dim, config).to(device)
            self.optimizer_params = {'params': self.policy.actor.parameters(),'lr': config.lr_actor,'weight_decay':0, 'betas':config.betas_actor}

            self.optimizer = torch.optim.Adam([self.optimizer_params])
            self.policy_old = ActorCriticGATG(metadata, n_part, action_dim, config).to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())

    
    def reset_assembly(self, parts, contacts):

        """if wandb.config.policy_type == 'ziqi':
            self.graph_constructor = FTGraphConstructor(self.assembly)
        elif wandb.config.policy_type == 'gabriel':"""
        self.graph_constructor = GFTFGraphConstructor(parts, contacts)
       
    def select_action(self, bin_states, masks, env_inds,curriculum_inds):
        # build graphs
        start_timer = perf_counter()
        graphs = self.graph_constructor.graphs(bin_states)
        self.build_graph_time += perf_counter() - start_timer

        # to batch graphs
        start_timer = perf_counter()
        batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
        self.batch_graph_time += perf_counter() - start_timer

        # inference
        start_timer = perf_counter()
        self.policy_old.eval()
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(batch_graph, masks)
        self.inference_time += perf_counter() - start_timer

        # add to buffer
        namelist = ['states', 'actions', 'logprobs', "state_values", "masks", "bin_states"]
        variables = [graphs, action, action_logprob, state_val, masks, bin_states]
        for id, name in enumerate(namelist):
            self.buffer.add(name, variables[id], env_inds)
        return action

    def compute_policy(self, bin_states, masks):
        # build graphs
        start_timer = perf_counter()
        graphs = self.graph_constructor.graphs(bin_states)
        self.build_graph_time += perf_counter() - start_timer

        # to batch graphs
        start_timer = perf_counter()
        batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
        self.batch_graph_time += perf_counter() - start_timer

        # inference
        start_timer = perf_counter()
        self.policy_old.eval()
        with torch.no_grad():
            action_probs, state_val = self.policy_old.actor(batch_graph)
            action_probs = masks * action_probs + masks * self.policy_old.mask_prob
        self.inference_time += perf_counter() - start_timer
        action_probs /=  action_probs.sum(1,keepdim=True)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, action_probs, state_val

    def reset_optim(self):
        self.optimizer_params['lr'],*_ = self.scheduler.get_last_lr()
        self.optimizer = torch.optim.Adam([self.optimizer_params])
