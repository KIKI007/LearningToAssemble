import numpy as np
import torch
import torch_geometric
import wandb
from stability.ppo.helper import *
from torch.cuda import Stream
class RolloutDatasetMLP(torch.utils.data.Dataset):
    def __init__(self,batch_size=100):
        super(RolloutDatasetMLP, self).__init__()
        self.batch_size = batch_size
        self.nsamples = 0
    def __len__(self):
        if hasattr(self,"states"):
            return int(np.ceil(len(self.states)/self.batch_size))
        else:
            return 0
    def __getitem__(self, idx):
        if idx < len(self)-1:
            return (self.states[idx*self.batch_size:(idx+1)*self.batch_size, :].to(device),
                self.actions[idx*self.batch_size:(idx+1)*self.batch_size].to(device),
                self.masks[idx*self.batch_size:(idx+1)*self.batch_size, :].to(device),
                self.logprobs[idx*self.batch_size:(idx+1)*self.batch_size].to(device),
                self.rewards[idx*self.batch_size:(idx+1)*self.batch_size].to(device),
                self.advantages[idx*self.batch_size:(idx+1)*self.batch_size].to(device),
                self.weights[idx*self.batch_size:(idx+1)*self.batch_size].to(device),
                self.entropy_weights[idx*self.batch_size:(idx+1)*self.batch_size].to(device),
                self.curriculum_id[idx*self.batch_size:(idx+1)*self.batch_size].to(device))
        elif idx == len(self)-1:
            return (self.states[idx*self.batch_size:, :].to(device),
                    self.actions[idx*self.batch_size:].to(device),
                    self.masks[idx*self.batch_size:, :].to(device),
                    self.logprobs[idx*self.batch_size:].to(device),
                    self.rewards[idx*self.batch_size:].to(device),
                    self.advantages[idx*self.batch_size:].to(device),
                    self.weights[idx*self.batch_size:].to(device),
                    self.entropy_weights[idx*self.batch_size:].to(device),
                    self.curriculum_id[idx*self.batch_size:].to(device))
        else:
            raise IndexError

    def batch_data(self):
        return (self.states,
         self.actions,
         self.masks,
         self.logprobs,
         self.rewards,
         self.advantages,
         self.weights,
         self.entropy_weights,
         self.curriculum_id)

    def add_states(self, states):
        if isinstance(states,list):
            states = torch.vstack(states)
        self.states = states
        self.nsamples = self.states.shape[0]
    def merge(self, dataset):
        self.actions = torch.hstack([self.actions, dataset.actions])
        self.masks = torch.vstack([self.masks, dataset.masks])
        self.rewards = torch.hstack([self.rewards, dataset.rewards])
        self.weights = torch.hstack([self.weights, dataset.weights])
        self.entropy_weights = torch.hstack([self.entropy_weights, dataset.entropy_weights])
        self.logprobs = torch.hstack([self.logprobs, dataset.logprobs])
        self.advantages = torch.hstack([self.advantages, dataset.advantages])
        self.states = torch.vstack([self.states,dataset.states])
        self.curriculum_id = torch.hstack([self.curriculum_id, dataset.curriculum_id])

        self.nsamples = self.rewards.shape[0]
class RolloutDatasetGNN(torch_geometric.data.Dataset):

    def __init__(self, transform=None, pre_transform=None,batch_size=100):
        super().__init__(None, transform, pre_transform)
        self._indices = None
        self.batch_size = batch_size
        self.nsamples = 0
        self.nstreams = 5
        self.tos = Stream(device)
    def get(self, idx):
        return (self.states[idx],
                self.actions[idx],
                self.masks[idx, :],
                self.logprobs[idx],
                self.rewards[idx],
                self.advantages[idx],
                self.weights[idx],
                self.entropy_weights[idx],
                self.curriculum_id[idx])
    def __getitem__(self,idx):
            if idx < len(self)-1:
                return self.batch_data_from_to(idx*self.batch_size,(idx+1)*self.batch_size)
            else:
                return self.batch_data_from_to(idx*self.batch_size,len(self.states))
    def __len__(self):
        if hasattr(self,"states"):
            return int(np.ceil(len(self.states)/self.batch_size))
        else:
            return 0
    def add_states(self, states):
        self.states = states
        self.batch_graphs = torch_geometric.data.Batch.from_data_list(self.states)
        self.nsamples = len(states)
    def merge(self, dataset):
        self.actions = torch.hstack([self.actions, dataset.actions])
        self.masks = torch.vstack([self.masks, dataset.masks])
        self.rewards = torch.hstack([self.rewards, dataset.rewards])
        self.logprobs = torch.hstack([self.logprobs, dataset.logprobs])
        self.advantages = torch.hstack([self.advantages, dataset.advantages])
        self.states.extend(dataset.states)
        self.batch_graphs = torch_geometric.data.Batch.from_data_list(self.states)
        self.weights = torch.hstack([self.weights, dataset.weights])
        self.entropy_weights = torch.hstack([self.entropy_weights, dataset.entropy_weights])
        self.curriculum_id = torch.hstack([self.curriculum_id, dataset.curriculum_id])
        self.nsamples+= dataset.nsamples
    def batch_data(self):
        return (self.batch_graphs.to(device),
         self.actions.to(device),
         self.masks.to(device),
         self.logprobs.to(device),
         self.rewards.to(device),
         self.advantages.to(device),
         self.weights.to(device),
         self.entropy_weights.to(device),
         self.curriculum_id.to(device))
    def batch_data_from_to(self,idx_start,idx_end):
        with torch.cuda.stream(self.tos):
            return (torch_geometric.data.Batch.from_data_list(self.states[idx_start:idx_end]).to(device,non_blocking=True),
            self.actions[idx_start:idx_end].to(device,non_blocking=True),
            self.masks[idx_start:idx_end].to(device,non_blocking=True),
            self.logprobs[idx_start:idx_end].to(device,non_blocking=True),
            self.rewards[idx_start:idx_end].to(device,non_blocking=True),
            self.advantages[idx_start:idx_end].to(device,non_blocking=True),
            self.weights[idx_start:idx_end].to(device,non_blocking=True),
            self.entropy_weights[idx_start:idx_end].to(device,non_blocking=True),
            self.curriculum_id[idx_start:idx_end].to(device,non_blocking=True))
class RolloutBufferMLP:
    def __init__(self, gamma, her,num_curriculums,base_entropy_weight,entropy_weight_increase,max_entropy,max_loss=100,num_step_anneal = 100):
        self.n_robot = wandb.config.n_robot
        self.gamma = gamma
        self.positive_reward = False
        self.her = her
        self.mode = "mlp"
        self.std_adv = True
        self.history = {}
        self.clear()
        self.max_loss = max_loss
        #store the cummulative loss of each episode
        self.curriculum_cumvalloss = max_loss*torch.ones((num_curriculums), dtype=floatType, device=device)
        #note that as this loss can be negative, using it as a priority marker can be missleading
        self.curriculum_cumsurloss = max_loss*torch.ones((num_curriculums), dtype=floatType, device=device)
        self.curriculum_rank = torch.arange(num_curriculums, dtype=intType, device=device)
        self.entropy_weights = base_entropy_weight*torch.ones((num_curriculums), dtype=floatType, device=device)
        self.alpha = 0.8
        self.beta = 0.1
        self.num_step_anneal = num_step_anneal
        self.beta0 = 1
        self.slope = (self.beta0-self.beta)/num_step_anneal

        self.base_entropy_weight = base_entropy_weight
        self.entropy_weight_increase = entropy_weight_increase
        self.max_entropy = max_entropy
    def set_policy(self, policy):
        self.policy = policy
    def modify_entropy(self,failed_inds,success_inds):
        increasable = self.entropy_weights[failed_inds]<self.max_entropy
        self.entropy_weights[failed_inds[increasable]]+=self.entropy_weight_increase
        self.entropy_weights[success_inds]=self.base_entropy_weight
    def from_bin(self, bin_states):
        states = bin_states[:, :, 0] + bin_states[:, :, 1]
        return states

    def clear(self, num_env=0):

        self.actions = [[] for _ in range(num_env)]
        self.states = [[] for _ in range(num_env)]
        self.logprobs = [[] for _ in range(num_env)]
        self.state_values = [[] for _ in range(num_env)]
        self.masks = [[] for _ in range(num_env)]
        self.bin_states = [[] for _ in range(num_env)]
        self.num_envs = num_env
        self.rewards = None # one reward for each env
        self.num_training_data = 0
        self.weights = None # one weight for each env
    def add_curriculums(self,n_new_curriculums):
        #store the cummulative loss of each episode
        self.curriculum_cumvalloss = torch.hstack([self.curriculum_cumvalloss,self.max_loss*torch.ones((n_new_curriculums), dtype=floatType, device=device)])
        #note that as this loss can be negative, using it as a priority marker can be missleading
        self.curriculum_cumsurloss = torch.hstack([self.curriculum_cumsurloss,self.max_loss*torch.ones((n_new_curriculums), dtype=floatType, device=device)])
        self.curriculum_rank = torch.hstack([self.curriculum_rank+n_new_curriculums,torch.arange(n_new_curriculums, dtype=intType, device=device)])
        self.entropy_weights = torch.hstack([self.entropy_weights,self.base_entropy_weight*torch.ones((n_new_curriculums), dtype=floatType, device=device)])
    def add(self, name, vals, env_inds):
        for ival, ienv in enumerate(env_inds):
            val = vals[ival].detach().cpu()
            self.__dict__[name][ienv.item()].append(val)
        if name == "bin_states":
            self.num_training_data += vals.shape[0]

    def create_dataset(self,batch_size):
        return RolloutDatasetMLP(batch_size)

    def build_batch_dataset(self, batch_size, shuffle):
        dataset_base = self.build_dataset(batch_size)
        if self.her:
            dataset_her = self.build_dataset_her(batch_size)
            if len(dataset_her) > 0:
                dataset_base.merge(dataset_her)
            #datasets.append(self.build_dataset_her(env_inds))
        return dataset_base

    def build_dataset(self, batchsize):

        dataset = self.create_dataset(batchsize)
        actions = []
        logprobs = []
        state_values = []
        masks = []
        states = []
        rewards = []
        curriculum_id = []
        env_inds = range(self.num_envs)
        for ienv in env_inds:
            discounted_reward = self.rewards[ienv]
            if not self.positive_reward or discounted_reward > 0:
                for istep in torch.arange(len(self.bin_states[ienv]) - 1, -1, -1):
                    # collect dataset
                    rewards.append(discounted_reward.pin_memory())
                    states.append(self.states[ienv][istep].pin_memory())
                    actions.append(self.actions[ienv][istep].pin_memory())
                    logprobs.append(self.logprobs[ienv][istep].pin_memory())
                    state_values.append(self.state_values[ienv][istep].pin_memory())
                    masks.append(self.masks[ienv][istep].pin_memory())
                    curriculum_id.append(self.curriculum_inds[ienv].pin_memory())
                    # history
                    """bin_state = self.bin_states[ienv][istep].to(device)
                    state = (bin_state[:, 0] + bin_state[:, 1] * 2 + bin_state[:, 2] * 4).cpu().numpy().astype(np.int32)
                    encode_state = tuple(state.tolist())
                    if encode_state in self.history:
                        discounted_reward = self.history[encode_state]
                    elif discounted_reward > 0:
                        self.history[encode_state] = discounted_reward"""
                    discounted_reward = self.gamma * discounted_reward
        if len(masks) > 0:
            state_values = torch.tensor(state_values, device="cpu", dtype=floatType,pin_memory=True).detach()
            # dataset
            dataset.actions = torch.tensor(actions, device="cpu", dtype=intType,pin_memory=True).detach()
            dataset.masks = torch.vstack(masks)
            dataset.logprobs = torch.tensor(logprobs, device="cpu", dtype=floatType,pin_memory=True).detach()
            dataset.rewards = torch.tensor(rewards, device="cpu", dtype=floatType,pin_memory=True).detach()
            if self.std_adv:
                adv = dataset.rewards - state_values
                dataset.advantages = ((adv-adv.mean())/(adv.std(unbiased=False)+1e-7)).pin_memory()

            else:
                dataset.advantages = dataset.rewards - state_values.detach()

            # states
            dataset.add_states(states)
            dataset.curriculum_id = torch.hstack(curriculum_id)

            dataset.weights = torch.pow(1+self.curriculum_rank[curriculum_id],self.alpha-self.beta).cpu().pin_memory()
            dataset.weights /= dataset.weights.max()
            dataset.entropy_weights = self.entropy_weights[curriculum_id].cpu().pin_memory()
        return dataset

    def build_dataset_her(self, batchsize):
        # her
        dataset_her = self.create_dataset(batchsize)
        actions = []
        masks = []
        rewards = []
        states = []
        bin_states = []
        curriculum_id = []
        for ienv in range(self.num_envs):
            ntraj = len(self.bin_states[ienv])
            #if last action is hold, then discard it and the last state
            for ir in range(self.n_robot):
                if ntraj >1 and self.actions[ienv][-ir-2]<self.bin_states[ienv][0].shape[0]:
                    ntraj-= 1
                    #print("HER mask error avoided")
                else:
                    break
            
            if  ntraj >= 2 and self.rewards[ienv] < 0:
                if self.actions[ienv][ntraj-2]<self.bin_states[ienv][0].shape[0]:
                    print("Error in HER")
                    print()
                    print([a for a in self.actions[ienv]])
                discounted_reward = 1.0
                nonremove = self.bin_states[ienv][ntraj-1][:, 0].clone()
                for istep in torch.arange(ntraj - 2, -1, -1):

                    discounted_reward = self.gamma * discounted_reward
                    rewards.append(discounted_reward)
                    actions.append(self.actions[ienv][istep])

                    # bin state
                    bin_state = self.bin_states[ienv][istep].clone()
                    bin_state[:, 2] = nonremove
                    bin_states.append(bin_state)

                    # graph
                    if self.mode == "gnn":
                        graph = self.states[ienv][istep].clone()
                        #if self.graph_type == 'gabriel':
                        graph["part"].part_state[:, 1] = nonremove[bin_state[:, 0] > 0]
                        states.append(graph)

                    # mask
                    mask = self.masks[ienv][istep].clone()
                    npart = int(mask.shape[0] / 2)
                    mask[npart:] = mask[npart:] * (1 - nonremove)
                    masks.append(mask)
                    if not torch.any(mask):
                        print("mask error in her")
                    curriculum_id.append(self.curriculum_inds[ienv])
        if len(masks) > 0:
            dataset_her.actions = torch.tensor(actions, device="cpu", dtype=intType,pin_memory=True).detach()
            dataset_her.masks = torch.vstack(masks)
            dataset_her.rewards = torch.tensor(rewards, device="cpu", dtype=floatType,pin_memory=True)
            dataset_her.curriculum_id = torch.hstack(curriculum_id).cpu().pin_memory()

            dataset_her.weights = torch.pow(1+self.curriculum_rank[dataset_her.curriculum_id],self.alpha-self.beta).cpu().pin_memory()
            dataset_her.weights /= dataset_her.weights.max()

            dataset_her.entropy_weights = self.entropy_weights[dataset_her.curriculum_id].cpu().pin_memory()
            # evaluate network
            self.policy.train(False)
            with torch.no_grad():
                if self.mode == "gnn":
                    dataset_her.add_states(states)
                    dataset_her.logprobs = torch.zeros(len(masks),pin_memory=True)
                    dataset_her.state_values = torch.zeros(len(masks),pin_memory=True)
                    dataset_her.advantages = torch.zeros(len(masks),pin_memory=True)

                    state_values = []
                    logprobs = []
                    #dl = torch.utils.data.DataLoader(dataset_her,pin_memory=True)
                    batches = []
                    for i,batch in enumerate(dataset_her):
                        state_valuesb, logprobsb, _ = self.policy.evaluate(batch[0].to(device),
                                                                           batch[1].to(device),
                                                                           batch[2].to(device))
                        state_values.append(state_valuesb.detach().cpu())
                        logprobs.append(logprobsb.detach().cpu())
                    dataset_her.logprobs = torch.vstack(logprobs).reshape(-1).pin_memory()
                    dataset_her.state_values = torch.hstack(state_values).pin_memory()
                else:
                    bin_states = torch.stack(bin_states)
                    new_states = bin_states.swapaxes(1,2)
                    new_states = new_states.flatten(start_dim=1)
                    dataset_her.add_states(new_states)
                    state_values, logprobs, _ = self.policy.evaluate(dataset_her.states.to(device),
                                                                    dataset_her.actions.to(device),
                                                                    dataset_her.masks.to(device))
                    dataset_her.logprobs = logprobs.cpu().flatten()
                    dataset_her.state_values = state_values.cpu()
                
                if self.std_adv:
                    adv = dataset_her.rewards.detach() - dataset_her.state_values
                    dataset_her.advantages = ((adv-adv.mean())/(adv.std(unbiased=False)+1e-7)).pin_memory()
                else:
                    dataset_her.advantages = dataset_her.rewards.detach() - state_values.detach()

            
        return dataset_her

class RolloutBufferGNN(RolloutBufferMLP):
    def __init__(self, gamma, her,n_curriculums,base_entropy_weight,entropy_weight_increase,max_entropy,num_step_anneal = 100):
        super(RolloutBufferGNN, self).__init__(gamma, her,n_curriculums,base_entropy_weight,entropy_weight_increase,max_entropy,num_step_anneal =num_step_anneal)
        self.mode = "gnn"
        #self.graph_type = wandb.config.policy_type
    def create_dataset(self,batch_size):
        return RolloutDatasetGNN(batch_size=batch_size)
