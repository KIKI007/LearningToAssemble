from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.training import train, evaluation
import argparse
import torch
import wandb
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learning to assemble with alternative plans')
    parser.add_argument('name', type=str, help='The name of the assembly model')
    parser.add_argument('--robot', type=int, default=2, help='The number of robots')
    parser.add_argument('--output', type=str, default=None, help='The name of policy output')
    parser.add_argument('--curriculum', type=int, default=64, help='The beam search width for generating curriculum')
    parser.add_argument('--wandb', type=str, default=None, help='Activate WanDB')

    args = parser.parse_args()
    name = args.name
    default_settings["n_robot"] = args.robot
    default_settings["curriculum"]["n_beam"] = args.curriculum
    default_settings["training"]["policy_name"] = name
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")

    if args.output is not None:
        default_settings["training"]["policy_name"] = args.output

    if "rulin" in name:
        default_settings["rbe"]["density"] = 1E4
    elif "mario" in name:
        default_settings["rbe"]["density"] = 1E4
        default_settings["rbe"]["max_iter"] = 3000
        default_settings["env"]["boundary_part_ids"] = [0, 12, 59, 14, 2, 41, 5, 6, 7, 10, 17, 1, 13, 58, 9, 15, 38]
    elif "dome" in name:
        default_settings["rbe"]["density"] = 1E3
        default_settings["rbe"]["max_iter"] = 3000
        default_settings["env"]["boundary_part_ids"] = [len(parts) - 1]

    memory_GB = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    if memory_GB < 24:
        default_settings["training"]["policy_update_batch_size"] = 1024
        default_settings["env"]["num_rollouts"] = 1024
    elif memory_GB < 48:
        default_settings["training"]["policy_update_batch_size"] = 2048
        default_settings["env"]["num_rollouts"] = 1024
    elif memory_GB > 75:
        default_settings["training"]["policy_update_batch_size"] = 4096
        default_settings["env"]["num_rollouts"] = 1024
    default_settings["curriculum"]["verbose"] = True
    if args.wandb:
        wandb.login(key = args.wandb, relogin = True, force = True)
        run = wandb.init(project="Disassembly_train", name = name)
    else:
        run = None

    contacts = compute_assembly_contacts(parts, default_settings)
    train(parts, contacts, default_settings, run)
    evaluation(parts, contacts, default_settings["training"]["policy_name"], 0, None)