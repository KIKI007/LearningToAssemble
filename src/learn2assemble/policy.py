from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.nn import aggr
from learn2assemble import set_default

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


class GATGFTFSharedEncoder(nn.Module):
    def __init__(self, metadata,
                 n_part,
                 hidden_channels=8,
                 heads=3,
                 num_layers=4,
                 dropout=0,
                 centroids=False):
        super().__init__()
        self.centroids = centroids
        self.n_part = n_part
        self.embedding_part_geom = torch.nn.Linear(1, hidden_channels // 2, device=device, bias=False)
        self.embedding_part_state = torch.nn.Embedding(4, hidden_channels // 2, device=device)

        # shared gat
        self.convs = torch_geometric.nn.models.GAT(in_channels=(-1, -1), hidden_channels=hidden_channels,
                                                   out_channels=hidden_channels, heads=1, dropout=dropout,
                                                   num_layers=num_layers, add_self_loops=False)

        self.convs = torch_geometric.nn.to_hetero(self.convs, metadata, aggr='sum').to(device)

        # actor
        self.actor_convs = torch_geometric.nn.models.GAT(in_channels=hidden_channels,
                                                         heads=1,
                                                         hidden_channels=hidden_channels,
                                                         dropout=dropout, num_layers=1, add_self_loops=False)
        self.actor_convs = torch_geometric.nn.to_hetero(self.actor_convs, metadata, aggr='sum').to(device)
        self.actor_layernrm = torch_geometric.nn.LayerNorm(hidden_channels, mode='node')
        self.actor_out = torch.nn.Linear(hidden_channels, 2)

        # critic
        aggr_dim = 3
        node_types = 3
        num_mlp_critic = 1
        self.aggr = aggr.MultiAggregation([aggr.MaxAggregation(),
                                           aggr.MinAggregation(),
                                           aggr.MeanAggregation()])
        self.critic_concat = torch.nn.Linear(node_types * aggr_dim * hidden_channels, hidden_channels, device=device)
        mlp = [nn.Linear(hidden_channels, hidden_channels, device=device) for i in range(num_mlp_critic)]
        self.critic_mlp = torch.nn.Sequential(*mlp)
        self.critic_out = torch.nn.Linear(hidden_channels, 1, device=device)

    def forward(self, inputs):
        self.convs.to(device)
        emb_geom = self.embedding_part_geom(inputs['part'].mass)
        emb_state = self.embedding_part_state(inputs['part'].x)

        h = self.convs({'part': torch.cat([emb_geom, emb_state], dim=-1),
                        'torque': inputs['torque'].x,
                        'force': inputs['force'].x},
                       inputs.edge_index_dict)

        # actor
        actor_h = self.actor_convs(h, inputs.edge_index_dict)
        actor_out = self.actor_layernrm(actor_h['part'])
        actor_out = self.actor_out(actor_out)
        actions = torch.zeros((inputs['part'].batch.max() + 1, 2, self.n_part), device=device, dtype=floatType)
        actions[inputs['part'].batch, :, inputs['part'].part_id] = torch_geometric.utils.softmax(actor_out,
                                                                                                 ptr=inputs['part'].ptr)
        actions = actions.flatten(start_dim=1)

        # critic
        critic_h_cat = torch.cat([self.aggr(h['part'], ptr=inputs.ptr_dict['part']),
                                  self.aggr(h['torque'], ptr=inputs.ptr_dict['torque']),
                                  self.aggr(h['force'], ptr=inputs.ptr_dict['force'])], axis=1)
        critic_h = F.gelu(self.critic_concat(critic_h_cat))
        critic_h = self.critic_mlp(critic_h)
        v = F.tanh(self.critic_out(critic_h))
        return actions, v


class ActorCriticGATG(nn.Module):
    def __init__(self, n_part, settings):
        super(ActorCriticGATG, self).__init__()
        policy_config = set_default(settings,
                                    'policy',
                                    {
                                        "gat_layers": 8,
                                        "gat_heads": 1,
                                        "gat_hidden_dims": 16,
                                        "gat_dropout": 0.0,
                                        "centroids": False
                                    })

        metadata = (['part', 'force', 'torque'],
                    [('part', 'pp', 'part'), ('torque', 'tt', 'torque'), ('part', 'pt', 'torque'),
                     ('torque', 'rev_pt', 'part'),
                     ('part', 'pfplus', 'force'), ('force', 'rev_pfplus', 'part'), ('part', 'pfminus', 'force'),
                     ('force', 'rev_pfminus', 'part'), ('force', 'ft', 'torque'), ('torque', 'rev_ft', 'force'),
                     ('part', 'is', 'part'),
                     ('force', 'is', 'force'), ('torque', 'is', 'torque')])

        policy_config = SimpleNamespace(**policy_config)

        self.actor = GATGFTFSharedEncoder(metadata=metadata,
                                          hidden_channels=policy_config.gat_hidden_dims,
                                          heads=policy_config.gat_heads,
                                          num_layers=policy_config.gat_layers,
                                          n_part=n_part,
                                          dropout=0,
                                          centroids=policy_config.centroids)

        self.mask_prob = 1E-9

    def act(self, state, mask):
        action_probs, state_val = self.actor(state)
        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, mask):
        action_probs, state_values = self.actor(state)
        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


if __name__ == "__main__":
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
    from learn2assemble.graph import GFTFGraphConstructor
    import numpy as np
    import time

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/rulin")
    settings = {
        "contact_settings":
        {
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
        },
        "policy": {
            "gat_layers": 8,
            "gat_heads": 1,
            "gat_hidden_dims": 16,
            "gat_dropout": 0.0,
            "centroids": False
        }
    }

    contacts = compute_assembly_contacts(parts)
    graph_constructor = GFTFGraphConstructor(parts, contacts)
    torch.manual_seed(100)

    graphs = []
    nbatch = 1024
    part_states = torch.randint(low=0, high=3, size=(nbatch, len(parts)), dtype=torch.int32, device="cpu")
    start = time.perf_counter()
    graphs = graph_constructor.graphs(part_states)
    print("build graph:\t", time.perf_counter() - start)
    start = time.perf_counter()
    batch_graph = torch_geometric.data.Batch.from_data_list(graphs).to(device)
    print("batch graph:\t", time.perf_counter() - start)

    start = time.perf_counter()
    A2C = ActorCriticGATG(len(parts), settings).to(device)
    mask = torch.ones((nbatch, len(parts) * 2), device=device, dtype=bool)
    a, alp, v = A2C.act(batch_graph, mask)
    print("inference graph:\t", time.perf_counter() - start)
