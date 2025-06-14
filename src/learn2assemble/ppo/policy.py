import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.nn import GAT, to_hetero
from torch_geometric.nn import aggr
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from stability.ppo.env import DisassemblyGymEnv
from stability.ppo.graph import *

class GATGFTFSharedEncoder(nn.Module):
    def __init__(self, metadata, n_part, hidden_channels=8, heads=3, num_layers=4, out_channels = 2, dropout = 0.,centroids=False):
        super().__init__()
        n_layers_full = 1
        n_neurons_full = hidden_channels
        self.centroids = centroids
        if self.centroids:
            n_inpart = 4
        else:
            n_inpart = 1
        self.n_part = n_part
        self.embedding_parts = torch.nn.Linear(n_inpart,hidden_channels//2,device=device,bias=False)

        self.embedding_state = torch.nn.Embedding(4,hidden_channels//2,device=device)

        #self.bnin = torch.nn.BatchNorm1d(config['n_neurons'],device=self.device)
        self.convs = torch_geometric.nn.models.GAT(in_channels=(-1, -1),hidden_channels=hidden_channels,out_channels = hidden_channels, heads=1,dropout=dropout,num_layers=num_layers,add_self_loops=False)
        self.convs= torch_geometric.nn.to_hetero(self.convs,metadata,aggr='sum').to(device)

        self.convActor = torch_geometric.nn.models.GAT(in_channels=hidden_channels,
                                                       heads=1,
                                                       hidden_channels=hidden_channels,
                                                       #out_channels = 2,
                                                       dropout=dropout,num_layers=1,add_self_loops=False)
        self.convActor= torch_geometric.nn.to_hetero(self.convActor,metadata,aggr='sum').to(device)
        k=3
        self.aggr = aggr.MultiAggregation([aggr.MaxAggregation(),aggr.MinAggregation(),aggr.MeanAggregation()])#aggr.MedianAggregation().to(self.device)
        self.bn = torch.nn.BatchNorm1d(hidden_channels*k*3,device=device)
        self.innet = torch.nn.Linear(3*k*hidden_channels,n_neurons_full,device=device)
        self.full_net = torch.nn.ModuleList([nn.Linear(n_neurons_full,n_neurons_full,device = device) for i in range(n_layers_full)])
        self.outnet = torch.nn.Linear(n_neurons_full,1,device=device)
        self.ln = torch_geometric.nn.LayerNorm(hidden_channels,mode='node')
        self.out_a = torch.nn.Linear(hidden_channels,out_channels)
    def forward(self,inputs):
        self.convs.to(device)
        if self.centroids:
            reppart = torch.cat([self.embedding_parts(torch.cat([inputs['part'].mass,inputs['part'].centroid],axis=-1)),
                                self.embedding_state((inputs['part'].part_state[:,0] +2*inputs['part'].part_state[:,1]).to(torch.int64))],dim=-1)
        else:
            reppart = torch.cat([self.embedding_parts(inputs['part'].mass),#torch.cat([inputs['part'].mass,inputs['part'].centroid],axis=-1)),
                                self.embedding_state((inputs['part'].part_state[:,0] +2*inputs['part'].part_state[:,1]).to(torch.int64))],dim=-1)
        rep = self.convs({'part':reppart,
                          'torque':inputs['torque'].x,
                          'force':inputs['force'].x},inputs.edge_index_dict)
        #concat all output
        repA = self.convActor(rep,inputs.edge_index_dict)
        #rep_actions = repA['part']
        rep_actions = self.ln(repA['part'])
        rep_actions = self.out_a(rep_actions)
        actions = torch.zeros((inputs['part'].batch.max()+1, 2, self.n_part), device=device, dtype=floatType)
        actions[inputs['part'].batch, :, inputs['part'].part_id] = torch_geometric.utils.softmax(rep_actions,ptr=inputs['part'].ptr)
        actions = actions.flatten(start_dim=1)

        rep = torch.cat([self.aggr(rep['part'],ptr=inputs.ptr_dict['part']),
                         self.aggr(rep['torque'],ptr=inputs.ptr_dict['torque']),
                         self.aggr(rep['force'],ptr = inputs.ptr_dict['force']),
                        ],axis = 1)
        #batchnorm work well in general, but as the batches are not shuffled it can cause problems
        #rep  = self.bn(rep)
        repV = F.gelu(self.innet(rep))
        for layer in self.full_net:
            repV = F.gelu(layer(repV))
        V = F.tanh(self.outnet(repV))
        
        return actions, V


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = None
        self.critic = None
        self.mask_prob = 1E-9
        self.deterministic = False

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask):
        state = state.to(floatType)
        action_probs = self.actor(state)
        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)
        if not self.deterministic:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs)
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, mask):
        state = state.to(floatType)
        action_probs = self.actor(state)
        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class ActorCriticGATG(ActorCritic):
    def __init__(self, metadata, n_part, action_dim, config):
        super(ActorCriticGATG, self).__init__()
        hidden_dim = config.gat_hidden_dims
        heads = config.gat_heads
        layers = config.gat_layers
        self.actor = GATGFTFSharedEncoder(metadata=metadata,
                                  hidden_channels=hidden_dim,
                                  heads=heads,
                                  num_layers=layers,
                                  n_part=n_part,
                                  dropout=0,
                                  centroids=config['centroids'])#config.gat_dropout)
    def act(self, state, mask):
        action_probs, state_val = self.actor(state)
        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)
        if not self.deterministic:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1).reshape(-1)
        action_logprob = dist.log_prob(action)
        #print(f"{dist.probs[0,mask[0]]=}")
        #print(f"{mask[0].sum()=}")
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, mask):
        action_probs, state_values = self.actor(state)

        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy


class ActorCriticMLP(ActorCritic):
    def __init__(self, state_dim, action_dim, config):
        super(ActorCriticMLP, self).__init__()

        hidden_dims = config.mlp_hidden_dims

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, 1),
            nn.Tanh()
        )