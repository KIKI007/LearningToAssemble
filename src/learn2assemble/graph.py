import time
from types import SimpleNamespace
import os
import networkx as nx
import numpy as np
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import torch
from multiprocessing import Pool

device = "cpu"
floatType = torch.float32
intType = torch.int32

class GFTFGraphConstructor:
    def __init__(self, parts, contacts):
        self.parts = parts
        self.contacts = contacts
        self.n_part = len(parts)
        self.init_graph()

    def _int(self, array):
        return torch.tensor(array, device=device, dtype=torch.long)

    def _float(self, array):
        return torch.tensor(array, device=device, dtype=floatType)

    def _arange(self, n):
        return torch.arange(start = 0, end = n, dtype=torch.long, device=device)

    def init_graph(self):
        self.part_part_edge = []

        self.mass = torch.stack([torch.tensor([self.parts[part].volume], device=device, dtype=floatType) for part in range(self.n_part)])

        self.force_x = []
        self.force_part_ids = []

        self.torque_x = []
        self.torque_part_ids = []
        self.torque_force_ids = []

        for contact in self.contacts:
            iA = contact["iA"]
            iB = contact["iB"]

            # normal
            n = contact["plane"][:3, 2]
            fid = len(self.force_x)
            self.force_x.append(self._float(n))
            self.force_part_ids.append(self._int([iA, iB]))

            # points
            pts = contact["mesh"].vertices
            for pt in pts:
                tauA = self._float(pt - self.parts[iA].center_mass)
                tauB = self._float(pt - self.parts[iB].center_mass)
                self.torque_x.append(tauA)
                self.torque_part_ids.append(self._int([iA, iB]))
                self.torque_x.append(tauB)
                self.torque_part_ids.append(self._int([iB, iA]))
                self.torque_force_ids += [fid, fid]

        for iA in range(self.n_part):
            for iB in range(self.n_part):
                self.part_part_edge.append(self._int([iA, iB]))

        self.torque_x = torch.stack(self.torque_x)
        self.torque_part_ids = torch.stack(self.torque_part_ids)
        self.force_x = torch.stack(self.force_x)
        self.force_part_ids = torch.stack(self.force_part_ids)
        self.part_part_edge = torch.stack(self.part_part_edge)
        self.torque_force_ids = self._int(self.torque_force_ids)

    def compute_graph(self, install_state, x):
        data = HeteroData()

        data["part"].part_id = self._arange(self.n_part)[install_state]
        data["part"].mass = self.mass[install_state]
        data["part"].x = x[install_state]

        # force nodes
        force_state = install_state[self.force_part_ids[:, 0]] * install_state[self.force_part_ids[:, 1]]
        data["force"].x = self.force_x[force_state, :]

        # torque nodes
        torque_state = install_state[self.torque_part_ids[:, 0]] * install_state[self.torque_part_ids[:, 1]]
        data["torque"].x = self.torque_x[torque_state, :]

        # part part edge
        part_part_state = install_state[self.part_part_edge[:, 0]] * install_state[self.part_part_edge[:, 1]]
        edge = self.part_part_edge[part_part_state, :]
        map_part = torch.zeros(self.n_part, dtype=torch.long, device=device)
        map_part[data["part"].part_id] = self._arange(data["part"].x.shape[0])
        edge = map_part[edge]
        data["part", "pp", "part"].edge_index = edge.t().contiguous()

        # torque torque edge
        tA = self._arange(data["torque"].x.shape[0] // 2) * 2
        tB = tA + 1
        edge = torch.hstack([torch.vstack([tA, tB]), torch.vstack([tB, tA])])
        data["torque", "tt", "torque"].edge_index = edge.contiguous()

        # part torque edge
        tid = self._arange(data["torque"].x.shape[0])
        pid = map_part[self.torque_part_ids[torque_state, 0]]
        edge = torch.vstack([pid, tid])
        data["part", "pt", "torque"].edge_index = edge.contiguous()
        data["torque", "rev_pt", "part"].edge_index = edge[[1, 0], :].contiguous()

        # part force plus edge
        fid = self._arange(data["force"].x.shape[0])
        pid = map_part[self.force_part_ids[force_state, 0]]
        edge = torch.vstack([pid, fid])
        data["part", "pfplus", "force"].edge_index = edge.contiguous()
        data["force", "rev_pfplus", "part"].edge_index = edge[[1, 0], :].contiguous()

        # part force minus edge
        pid = map_part[self.force_part_ids[force_state, 1]]
        edge = torch.vstack([pid, fid])
        data["part", "pfminus", "force"].edge_index = edge.contiguous()
        data["force", "rev_pfminus", "part"].edge_index = edge[[1, 0], :].contiguous()

        # force torque edge
        fid = self.torque_force_ids[torque_state]
        force_id = self._arange(self.force_x.shape[0])[force_state]
        map_force = torch.zeros(self.force_x.shape[0], dtype=torch.long, device=device)
        map_force[force_id] = self._arange(data["force"].x.shape[0])
        fid = map_force[fid]
        tid = self._arange(data["torque"].x.shape[0])
        edge = torch.vstack([fid, tid])
        data["force", "ft", "torque"].edge_index = edge.contiguous()
        data["torque", "rev_ft", "force"].edge_index = edge[[1, 0], :].contiguous()

        # nodes' self-loop edges
        data["part", "is", "part"].edge_index = torch.stack(
            [torch.arange(data["part"].x.shape[0], device=device),
             torch.arange(data["part"].x.shape[0], device=device)])
        data["force", "is", "force"].edge_index = torch.stack(
            [torch.arange(data["force"].x.shape[0], device=device),
             torch.arange(data["force"].x.shape[0], device=device)])
        data["torque", "is", "torque"].edge_index = torch.stack(
            [torch.arange(data["torque"].x.shape[0], device=device),
             torch.arange(data["torque"].x.shape[0], device=device)])
        return data

    def graphs(self, part_states: np.ndarray, target_states: np.ndarray = None):
        n_batch = part_states.shape[0]
        install_states = torch.tensor(part_states >= 1, device=device, dtype=torch.bool)
        held_states =  torch.tensor(part_states == 2, device=device, dtype=torch.bool)
        if target_states is None:
            nonremove_states = torch.zeros(part_states.shape, device=device, dtype=torch.bool)
        else:
            nonremove_states = torch.tensor(target_states >= 1, device=device, dtype=torch.bool)
        x = held_states.to(torch.int64) + nonremove_states.to(torch.int64) * 2
        datalist = []
        for i in range(n_batch):
            datalist.append(self.compute_graph(install_states[i, :], x[i, :]))
        return datalist

def draw_pygdata(PyGdata: HeteroData, k = 1, iter = 10000):
    graph = to_networkx(PyGdata, to_undirected=False)

    node_shapes = {}
    for node, attrs in graph.nodes(data=True):
        if attrs["type"] == "part":
            node_shapes[node] = "o"
        elif attrs["type"] == "force":
            node_shapes[node] = "s"
        elif attrs["type"] == "torque":
            node_shapes[node] = "^"

    # Define colors for the edges
    edge_type_colors = {
        ("part", "pfplus", "force"): "#FF7777",
        ("force", "rev_pfplus", "part"): "#FF7777",
    }

    edge_colors = []
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        if edge_type in edge_type_colors:
            edge_colors.append(edge_type_colors[edge_type])
        else:
            edge_colors.append("#000000")

    # pos
    pos = nx.spring_layout(graph, k=k, iterations=iter)

    for node, shape in node_shapes.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_shape=shape, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True)

    plt.show()

def draw_pygdata(PyGdata: HeteroData, k = 1, iter = 10000):
    graph = to_networkx(PyGdata, to_undirected=False)

    node_shapes = {}
    for node, attrs in graph.nodes(data=True):
        if attrs["type"] == "part":
            node_shapes[node] = "o"
        elif attrs["type"] == "force":
            node_shapes[node] = "s"
        elif attrs["type"] == "torque":
            node_shapes[node] = "^"

    # Define colors for the edges
    edge_type_colors = {
        ("part", "pfplus", "force"): "#FF7777",
        ("force", "rev_pfplus", "part"): "#FF7777",
    }

    edge_colors = []
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        if edge_type in edge_type_colors:
            edge_colors.append(edge_type_colors[edge_type])
        else:
            edge_colors.append("#000000")

    # pos
    pos = nx.spring_layout(graph, k=k, iterations=iter, seed = 10)

    for node, shape in node_shapes.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_shape=shape, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True)

    plt.show()

if __name__ == "__main__":
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    from learn2assemble import ASSEMBLY_RESOURCE_DIR
    from learn2assemble.render import draw_assembly_batches, init_polyscope
    import polyscope as ps
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    contacts = compute_assembly_contacts(parts)

    #graph_constructor = GFTFGraphConstructor(parts, contacts)

    init_polyscope()
    batch_part_states = np.random.randint(low = 0, high=3, size = (512, len(parts)), dtype=np.int32)
    batch_stability = np.random.randint(low = -1, high=2, size = (512), dtype=np.int32)
    draw_assembly_batches(parts, batch_part_states, batch_stability, scale=2)
    batch_part_states = torch.tensor(batch_part_states, dtype=torch.int32, device=device)
    #start = time.perf_counter()
    #graphs = graph_constructor.graphs(part_states)
    #end = time.perf_counter()
    #print(end - start)
    ps.show()
    #draw_pygdata(graphs[0], iter = 500)