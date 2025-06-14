import torch
from trimesh import Trimesh
from torch_geometric.data import HeteroData
from types import SimpleNamespace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def init_graph(parts: list[Trimesh],
               contacts: list[dict],
               settings: dict):
    n_part = len(parts)
    part_part_edge = []
    part_x = torch.stack([torch.tensor([0,  # fixed
                                        0,  # nonremove
                                        parts[part_id].volume]  # mass
                                       + parts[part_id].centroid.tolist(),  # centroid
                                       device=device, dtype=floatType)
                          for part_id in range(n_part)])

    centroid = torch.stack([torch.tensor(parts[part_id].centroid,  # centroid
                                         device=device, dtype=floatType)
                            for part_id in range(n_part)])

    mass = torch.stack([torch.tensor([parts[part_id].volume],
                                     device=device, dtype=floatType)
                        for part_id in range(n_part)])

    force_x = []
    force_part_ids = []

    torque_x = []
    torque_part_ids = []
    torque_force_ids = []

    def iAB(iA, iB):
        return torch.tensor([iA, iB], dtype=torch.long, device=device)

    for contact in contacts:
        iA = contact["iA"]
        iB = contact["iB"]

        # normal
        n = contact["plane"][:3, 2]
        fid = len(force_x)
        force_x.append(torch.tensor(n, device=device, dtype=floatType))

        force_part_ids.append(iAB(iA, iB))

        # points
        pts = contact["mesh"].vertices
        pts = torch.tensor(pts, device=device, dtype=floatType)

        for pt in pts:
            tauA = pt - centroid[iA]
            tauB = pt - centroid[iB]
            torque_x.append(tauA)
            torque_part_ids.append(iAB(iA, iB))
            torque_x.append(tauB)
            torque_part_ids.append(iAB(iB, iA))
            torque_force_ids += [fid, fid]

    for iA in range(n_part):
        for iB in range(n_part):
            part_part_edge.append(iAB(iA, iB))

    graph = {}
    graph["n_part"] = n_part
    graph["part_x"] = part_x
    graph["centroid"] = centroid
    graph["mass"] = mass
    graph["torque_x"] = torch.stack(torque_x)
    graph["torque_part_ids"] = torch.stack(torque_part_ids)
    graph["force_x"] = torch.stack(force_x)
    graph["force_part_ids"] = torch.stack(force_part_ids)
    graph["part_part_edge"] = torch.stack(part_part_edge)
    graph["torque_force_ids"] = torch.tensor(torque_force_ids, device=device, dtype=torch.long)
    settings["graph"] = graph

def compute_graphs(part_states: torch.Tensor,
                   settings: dict):
    datalist = []
    graph = SimpleNamespace(**settings["graph"])
    n_part = graph.n_part

    for id in range(part_states.shape[0]):
        data = HeteroData()
        part_id = torch.arange(0, n_part, dtype=torch.long, device=device)
        part_state = part_states[id, :]
        install_state = part_states[id, :, 0].to(bool)
        # part nodes
        data["part"].x = graph.part_x.detach().clone()
        data["part"].x[:, 0] = (part_state[:, 1]).type(floatType)
        data["part"].x[:, 1] = (part_state[:, 2]).type(floatType)
        data["part"].x = data["part"].x[install_state, :]
        data["part"].part_id = part_id[install_state]
        data["part"].centroid = graph.centroid[install_state]
        data["part"].mass = graph.mass[install_state]
        data["part"].part_state = part_state[install_state, 1:]
        # force nodes
        force_state = install_state[graph.force_part_ids[:, 0]] * install_state[graph.force_part_ids[:, 1]]
        data["force"].x = graph.force_x.detach().clone()
        data["force"].x = data["force"].x[force_state, :]

        # torque nodes
        torque_state = install_state[graph.torque_part_ids[:, 0]] * install_state[graph.torque_part_ids[:, 1]]
        data["torque"].x = graph.torque_x.detach().clone()
        data["torque"].x = data["torque"].x[torque_state, :]

        # part part edge
        part_part_state = install_state[graph.part_part_edge[:, 0]] * install_state[graph.part_part_edge[:, 1]]
        edge = graph.part_part_edge[part_part_state, :].detach().clone()
        map_part = torch.zeros(graph.n_part, dtype=torch.long, device=device)
        map_part[data["part"].part_id] = torch.arange(0, data["part"].x.shape[0], device=device, dtype=torch.long)
        edge = map_part[edge]
        data["part", "pp", "part"].edge_index = edge.detach().clone().t().contiguous()

        # torque torque edge
        tA = torch.arange(0, data["torque"].x.shape[0] // 2, device=device, dtype=torch.long) * 2
        tB = tA + 1
        edge = torch.hstack([torch.vstack([tA, tB]), torch.vstack([tB, tA])])
        data["torque", "tt", "torque"].edge_index = edge.contiguous()

        # part torque edge
        tid = torch.arange(0, data["torque"].x.shape[0], device=device, dtype=torch.long)
        pid = map_part[graph.torque_part_ids[torque_state, 0]]
        edge = torch.vstack([pid, tid])
        data["part", "pt", "torque"].edge_index = edge.contiguous()
        data["torque", "rev_pt", "part"].edge_index = edge[[1, 0], :].contiguous()

        # part force plus edge
        fid = torch.arange(0, data["force"].x.shape[0], device=device, dtype=torch.long)
        pid = map_part[graph.force_part_ids[force_state, 0]]
        edge = torch.vstack([pid, fid])
        data["part", "pfplus", "force"].edge_index = edge.contiguous()
        data["force", "rev_pfplus", "part"].edge_index = edge[[1, 0], :].contiguous()

        # part force minus edge
        pid = map_part[graph.force_part_ids[force_state, 1]]
        edge = torch.vstack([pid, fid])
        data["part", "pfminus", "force"].edge_index = edge.contiguous()
        data["force", "rev_pfminus", "part"].edge_index = edge[[1, 0], :].contiguous()

        # force torque edge
        fid = graph.torque_force_ids[torque_state]
        force_id = torch.arange(0, graph.force_x.shape[0], device=device, dtype=torch.long)
        force_id = force_id[force_state]
        map_force = torch.zeros(graph.force_x.shape[0], dtype=torch.long, device=device)
        map_force[force_id] = torch.arange(0, data["force"].x.shape[0], device=device, dtype=torch.long)
        fid = map_force[fid]
        tid = torch.arange(0, data["torque"].x.shape[0], device=device, dtype=torch.long)
        edge = torch.vstack([fid, tid])
        data["force", "ft", "torque"].edge_index = edge.contiguous()
        data["torque", "rev_ft", "force"].edge_index = edge[[1, 0], :].contiguous()

        data["part", "is", "part"].edge_index = torch.stack([torch.arange(data["part"].x.shape[0], device=device),
                                                             torch.arange(data["part"].x.shape[0], device=device)])
        data["force", "is", "force"].edge_index = torch.stack(
            [torch.arange(data["force"].x.shape[0], device=device),
             torch.arange(data["force"].x.shape[0], device=device)])
        data["torque", "is", "torque"].edge_index = torch.stack(
            [torch.arange(data["torque"].x.shape[0], device=device),
             torch.arange(data["torque"].x.shape[0], device=device)])
        datalist.append(data)
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
    from learn2assemble import RESOURCE_DIR
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    parts = load_assembly_from_files(RESOURCE_DIR + "/assembly/arch")
    settings = {}
    contacts = compute_assembly_contacts(parts, settings)
    init_graph(parts, contacts, settings)

    bin_states = torch.zeros((1, len(parts), 3), device=device, dtype=torch.long)
    bin_states[0, 0, 0] = 1
    bin_states[0, 1, 0] = 1
    bin_states[0, 1, 1] = 1
    bin_states[0, 1, 2] = 2
    graphs = compute_graphs(bin_states, settings)
    draw_pygdata(graphs[0], iter = 500)