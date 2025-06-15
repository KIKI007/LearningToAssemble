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
    def __init__(self, metadata, n_part, hidden_channels=8, heads=3, num_layers=4, out_channels=2, dropout=0.,
                 centroids=False):
        super().__init__()
        n_layers_full = 1
        n_neurons_full = hidden_channels
        self.centroids = centroids
        self.n_part = n_part
        self.embedding_parts = torch.nn.Linear(1, hidden_channels // 2, device=device, bias=False)
        self.embedding_state = torch.nn.Embedding(4, hidden_channels // 2, device=device)

        # self.bnin = torch.nn.BatchNorm1d(config['n_neurons'],device=self.device)
        self.convs = torch_geometric.nn.models.GAT(in_channels=(-1, -1), hidden_channels=hidden_channels,
                                                   out_channels=hidden_channels, heads=1, dropout=dropout,
                                                   num_layers=num_layers, add_self_loops=False)

        self.convs = torch_geometric.nn.to_hetero(self.convs, metadata, aggr='sum').to(device)

        self.convActor = torch_geometric.nn.models.GAT(in_channels=hidden_channels,
                                                       heads=1,
                                                       hidden_channels=hidden_channels,
                                                       # out_channels = 2,
                                                       dropout=dropout, num_layers=1, add_self_loops=False)
        self.convActor = torch_geometric.nn.to_hetero(self.convActor, metadata, aggr='sum').to(device)
        k = 3
        self.aggr = aggr.MultiAggregation([aggr.MaxAggregation(), aggr.MinAggregation(),
                                           aggr.MeanAggregation()])  # aggr.MedianAggregation().to(self.device)
        self.bn = torch.nn.BatchNorm1d(hidden_channels * k * 3, device=device)
        self.innet = torch.nn.Linear(3 * k * hidden_channels, n_neurons_full, device=device)
        self.full_net = torch.nn.ModuleList(
            [nn.Linear(n_neurons_full, n_neurons_full, device=device) for i in range(n_layers_full)])
        self.outnet = torch.nn.Linear(n_neurons_full, 1, device=device)
        self.ln = torch_geometric.nn.LayerNorm(hidden_channels, mode='node')
        self.out_a = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, inputs):
        self.convs.to(device)
        emb_part = self.embedding_parts(inputs['part'].mass)
        emb_state = self.embedding_state(
            (inputs['part'].part_state[:, 0] + 2 * inputs['part'].part_state[:, 1]).to(torch.int64))
        reppart = torch.cat([emb_part, emb_state], dim=-1)

        rep = self.convs({'part': reppart,
                          'torque': inputs['torque'].x,
                          'force': inputs['force'].x},
                         inputs.edge_index_dict)

        # concat all output
        repA = self.convActor(rep, inputs.edge_index_dict)
        # rep_actions = repA['part']
        rep_actions = self.ln(repA['part'])
        rep_actions = self.out_a(rep_actions)
        actions = torch.zeros((inputs['part'].batch.max() + 1, 2, self.n_part), device=device, dtype=floatType)
        actions[inputs['part'].batch, :, inputs['part'].part_id] = torch_geometric.utils.softmax(rep_actions,
                                                                                                 ptr=inputs['part'].ptr)
        actions = actions.flatten(start_dim=1)

        rep = torch.cat([self.aggr(rep['part'], ptr=inputs.ptr_dict['part']),
                         self.aggr(rep['torque'], ptr=inputs.ptr_dict['torque']),
                         self.aggr(rep['force'], ptr=inputs.ptr_dict['force']),
                         ], axis=1)

        # batchnorm work well in general, but as the batches are not shuffled it can cause problems
        # rep  = self.bn(rep)
        repV = F.gelu(self.innet(rep))
        for layer in self.full_net:
            repV = F.gelu(layer(repV))
        V = F.tanh(self.outnet(repV))
        return actions, V


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
        self.deterministic = False

    def act(self, state, mask):
        action_probs, state_val = self.actor(state)
        action_probs = mask * action_probs + mask * self.mask_prob
        dist = Categorical(action_probs)
        if not self.deterministic:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1).reshape(-1)
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

    graphs = []
    nbatch = 3
    part_states = torch.randint(low=0, high=3, size=(nbatch, len(parts)), dtype=torch.int32, device=device)
    graphs = graph_constructor.graphs(part_states)
    batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
    torch.manual_seed(100)
    A2C = ActorCriticGATG(len(parts), settings).to(device)
    print(batch_graph)
    mask = torch.ones((nbatch, len(parts) * 2), device=device, dtype=bool)
    mask[0, 15] = False
    mask[1, 75] = False
    mask[2, 0] = False
    a, alp, v = A2C.act(batch_graph, mask)
    print(a)
    print(alp)
    print(v)
    alp, v, ent = A2C.evaluate(batch_graph, a, mask)
    print(alp)
    print(v)
    print(ent)
