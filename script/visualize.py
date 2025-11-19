# gui
from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.training import train, evaluation
from learn2assemble.render import render_batch_simulation
import argparse
from torch.multiprocessing import Queue, Process
from learn2assemble.ppo import PPO
from learn2assemble.env import DisassemblyEnv
from learn2assemble.grasp import compute_grasp_table
from learn2assemble.insertion import compute_insertion_table
import pickle
import torch_geometric
from learn2assemble.training import training_rollout
from types import SimpleNamespace
import numpy as np
from learn2assemble.render import render_sequence, init_polyscope
import polyscope as ps
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learning to assemble with alternative plans')
    parser.add_argument('name', type=str, help='The name of the assembly model')
    parser.add_argument('--policy', type=str, default=None, help='The name of the policy')
    parser.add_argument('--robot', type=int, default=2, help='The number of robots')
    parser.add_argument('--grasp', action="store_true", default=False, help='Check grasp')
    parser.add_argument('--insertion', action="store_true", default=False, help='Check insertion')

    args = parser.parse_args()
    check_grasp = int(args.grasp)
    check_insertion = int(args.insertion)

    name = args.name
    policy = args.policy
    if policy is None:
        policy = name
    default_settings["n_robot"] = args.robot

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    contacts = compute_assembly_contacts(parts, default_settings)
    pretrained_file = f"../models/{policy}.pol"
    with open(pretrained_file, 'rb') as handle:
        agent = pickle.load(handle)
        state_dict = agent['state_dict']
        default_settings = agent['settings']

    default_settings["rbe"]["pre-computed"] = False
    default_settings["admm"]["pre-computed"] = False

    env = DisassemblyEnv(parts, contacts, settings=default_settings)
    if args.grasp:
        env.table_grasp, env.grasp_frames, env.scaled_parts = compute_grasp_table(parts, default_settings)
    if args.insertion:
        env.table_insertion, env.drts = compute_insertion_table(parts, default_settings)

    # single forward curriculum (training)
    torch_geometric.seed.seed_everything(default_settings["env"]["seed"])
    env.num_rollouts = env.curriculum.shape[0]

    ppo_agent = PPO(env.parts, env.contacts, default_settings)
    ppo_agent.policy_old.load_state_dict(state_dict)
    ppo_agent.policy.load_state_dict(state_dict)
    ppo_agent.deterministic = True

    part_states = env.curriculum
    ppo_agent.buffer.clear_replay_buffer(part_states.shape[0])
    n_env = part_states.shape[0]
    env_inds = np.arange(n_env)
    env.updated_simulation = False

    n_step = 0
    while True:

        current_states = part_states[env_inds, :]
        masks = env.action_masks(current_states)
        current_actions = ppo_agent.select_action(current_states, masks, env_inds)
        next_states, rewards, next_stability = env.step(current_states, current_actions)
        part_states[env_inds, :] = next_states

        ppo_agent.buffer.add("next_states", next_states, env_inds)
        ppo_agent.buffer.add("rewards_per_step", rewards, env_inds)
        ppo_agent.buffer.add("next_stability", next_stability, env_inds)

        env_inds, _ = ppo_agent.buffer.get_valid_env_inds(env)
        n_step = n_step + 1

        if env_inds.shape[0] == 0:
            break

    env.simulate_buffer(simulate_remain=True)
    _, rewards = ppo_agent.buffer.get_valid_env_inds(env)

    sequence = np.vstack(ppo_agent.buffer.next_states[0][::-1])

    init_polyscope()
    render_sequence(parts, sequence, default_settings)
    ps.show()