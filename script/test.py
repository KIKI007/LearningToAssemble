# gui
from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.training import train, evaluation
from learn2assemble.render import render_batch_simulation
import argparse
from torch.multiprocessing import Queue, Process

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learning to assemble with alternative plans')
    parser.add_argument('name', type=str, help='The name of the assembly model')
    parser.add_argument('--policy', type=str, default=None, help='The name of the policy')
    parser.add_argument('--robot', type=int, default=2, help='The number of robots')
    parser.add_argument('--gui', action="store_true", help='Activate GUI')

    args = parser.parse_args()
    name = args.name
    gui = args.gui
    policy = args.policy
    if policy is None:
        policy = name
    default_settings["n_robot"] = args.robot

    if "tetris" in name:
        default_settings["training"]["num_render_debug"] = 16 * 16
        default_settings["rbe"]["denisty"] = 1E2
        default_settings["rbe"]["max_iter"] = 1000
        default_settings["env"]["boundary_part_ids"] = [0]
    elif "rulin" in name:
        default_settings["rbe"]["denisty"] = 1E4
        default_settings["rbe"]["max_iter"] = 1000
        default_settings["env"]["boundary_part_ids"] = [0]

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    contacts = compute_assembly_contacts(parts, default_settings)
    if not gui:
        evaluation(parts, contacts, policy, 0, None)
    else:
        queue = Queue()
        p1 = Process(target=evaluation, args=(parts, contacts, policy, default_settings["training"]["num_render_debug"], queue))
        p2 = Process(target=render_batch_simulation, args=(parts, default_settings["env"]["boundary_part_ids"], queue))
        p2.start()
        p1.start()
        p1.join()
        p2.join()

