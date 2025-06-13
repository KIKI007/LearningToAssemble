import torch
from trimesh import Trimesh
from torch_geometric.data import HeteroData
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32

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
                          for part_id in range(n_part())])

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
    return settings

def compute_graphs(part_states: torch.Tensor,
                   settings: dict):
    datalist = []
    graph = settings["graph"]
    n_part = graph["n_part"]

    for id in range(part_states.shape[0]):
        data = HeteroData()

        part_state = part_states[id, :]
        install_state = part_states[id, :, 0].to(bool)
        # part nodes
        data["part"].x = graph["part_x"].detach().clone()
        data["part"].x[:, 0] = (part_state[:, 1]).type(floatType)
        data["part"].x[:, 1] = (part_state[:, 2]).type(floatType)
        data["part"].x = data["part"].x[install_state, :]
        data["part"].part_id = self._arange(n_part)[install_state]
        data["part"].centroid = self.centroid[install_state]
        data["part"].mass = self.mass[install_state]
        data["part"].part_state = part_state[install_state, 1:]
        # force nodes
        force_state = install_state[self.force_part_ids[:, 0]] * install_state[self.force_part_ids[:, 1]]
        data["force"].x = self.force_x.detach().clone()
        data["force"].x = data["force"].x[force_state, :]

        # torque nodes
        torque_state = install_state[self.torque_part_ids[:, 0]] * install_state[self.torque_part_ids[:, 1]]
        data["torque"].x = self.torque_x.detach().clone()
        data["torque"].x = data["torque"].x[torque_state, :]

        # part part edge
        part_part_state = install_state[self.part_part_edge[:, 0]] * install_state[self.part_part_edge[:, 1]]
        edge = self.part_part_edge[part_part_state, :].detach().clone()
        map_part = torch.zeros(self.assembly.n_part(), dtype=torch.long, device=device)
        map_part[data["part"].part_id] = self._arange(data["part"].x.shape[0])
        edge = map_part[edge]
        data["part", "pp", "part"].edge_index = edge.detach().clone().t().contiguous()

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
