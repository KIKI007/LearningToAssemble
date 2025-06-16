import math
import os
import pickle
from types import SimpleNamespace

import torch
from pathlib import Path
import polyscope as ps
import torch_geometric

from learn2assemble import set_default
from learn2assemble.disassemly_env import DisassemblyEnv
from learn2assemble.curriculum import forward_curriculum
from learn2assemble.agent import PPO

from trimesh import Trimesh
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


def rollout_asyn(ppo_agent, env, curriculum_inds):
    part_states = env.curriculum[curriculum_inds, :]
    ppo_agent.buffer.clear(part_states.shape[0])
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

        # ppo_agent.buffer.add("part_states", current_states, env_inds)
        ppo_agent.buffer.add("next_states", next_states, env_inds)
        ppo_agent.buffer.add("rewards_per_step", rewards, env_inds)
        ppo_agent.buffer.add("next_stability", next_stability, env_inds)

        env_inds, _ = ppo_agent.buffer.get_valid_env_inds(env)
        #print(f"n_step {n_step},\t rollout {len(env_inds)}")
        n_step = n_step + 1

        if env_inds.shape[0] == 0:
            break

    env.simulate_buffer(simulate_remain=True)
    _, rewards = ppo_agent.buffer.get_valid_env_inds(env)
    return torch.tensor(rewards, dtype=floatType)


def save_policy(ppo_agent, env, accuracy, training_settings):
    delta_accuracy = training_settings.save_delta_accuracy
    if accuracy > ppo_agent.saved_accuracy + delta_accuracy:
        ppo_agent.saved_accuracy = accuracy
        folder_path = "./models/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create the folder
        model_path = f"{folder_path}/policy.pol"
        ppo_agent.save(model_path)


def update_stats(ppo_agent, curriculum_inds: np.ndarray, training_settings, accuracy=None):
    # update number of fails and successes
    rewards = ppo_agent.buffer.rewards
    avg_reward = rewards.mean().item()

    inds = curriculum_inds.to(device)
    failed_inds = inds[(rewards < 0)]
    success_inds = inds[(rewards > 0)]

    ppo_agent.num_success[success_inds] += 1
    ppo_agent.num_failed[success_inds] = 0
    ppo_agent.num_failed[failed_inds] = ppo_agent.num_failed[failed_inds] + 1
    ppo_agent.buffer.modify_entropy(failed_inds, success_inds)

    # print status
    ppo_agent.print_iter = ppo_agent.print_iter + 1
    print_epoch = training_settings.print_epochs
    if ppo_agent.print_iter % print_epoch == print_epoch - 1:
        print()
        print("Episode : {} \t\t Accuracy : {} ".format(ppo_agent.episode, round(accuracy, 2)))
        #print("Failed Curriculum: {}".format(curriculum_inds[ppo_agent.buffer.rewards < 0]))


def policy_check(ppo_agent, curriculum_inds):
    num_success = torch.sum(ppo_agent.buffer.rewards > 0)
    num_envs = curriculum_inds.shape[0]
    # compute accuracy
    accuracy = (num_success / num_envs)
    return accuracy.item()


def sample_curriculum(ppo_agent, num_envs):
    sample_weights = None
    if (ppo_agent.buffer.curriculum_cumvalloss == 100).sum() > num_envs:
        print("New envs only")
        sample_weights = torch.zeros_like(ppo_agent.buffer.curriculum_cumvalloss)
        sample_weights[ppo_agent.buffer.curriculum_cumvalloss == 100] = 1 / (
                ppo_agent.buffer.curriculum_cumvalloss == 100).sum()
        curriculum_inds = torch.multinomial(sample_weights,
                                            num_envs,
                                            False)
    else:
        sample_weights = (torch.pow(
            torch.arange(1, ppo_agent.buffer.curriculum_rank.shape[0] + 1, device=device, dtype=floatType),
            -ppo_agent.buffer.alpha) /
                          torch.pow(torch.arange(1, ppo_agent.buffer.curriculum_rank.shape[0] + 1, device=device,
                                                 dtype=floatType), -ppo_agent.buffer.alpha).sum())
        curriculum_inds = torch.multinomial(sample_weights[ppo_agent.buffer.curriculum_rank],
                                            num_envs,
                                            True)
    return curriculum_inds.cpu()


def policy_update(ppo_agent, training_settings):
    loss, entropy = ppo_agent.update_policy(
        batch_size=training_settings.policy_update_batch_size,
        update_iter=training_settings.K_epochs)


def train(parts, contacts, settings):
    training_settings = set_default(settings,
                                   "training", {
                                       "max_train_epochs": 50000,
                                       "save_delta_accuracy": 0.01,
                                       "print_epochs": 1,
                                       "policy_update_batch_size": 2048,
                                       "K_epochs": 5
                                   })
    training_settings = SimpleNamespace(**training_settings)

    # create ppo agent
    env = DisassemblyEnv(parts, contacts, settings=settings)
    torch_geometric.seed.seed_everything(settings["env"]["seed"])
    _, _, curriculum = forward_curriculum(parts, contacts, settings=settings)
    env.set_curriculum(curriculum)
    ppo_agent = PPO(env, settings)

    print(f"Start training, curriculum: {env.curriculum.shape[0]}")
    # update variables
    ppo_agent.print_iter = 0

    assert ppo_agent.buffer.curriculum_rank.shape[0] == env.curriculum.shape[0]

    # training
    while ppo_agent.episode <= training_settings.max_train_epochs and ppo_agent.saved_accuracy < 1:
        curriculum_inds = sample_curriculum(ppo_agent, env.num_rollouts)
        ppo_agent.buffer.curriculum_inds = curriculum_inds
        ppo_agent.buffer.rewards = rollout_asyn(ppo_agent, env, curriculum_inds)

        # compute accuracy
        accuracy = policy_check(ppo_agent, curriculum_inds)

        # update number of fails
        update_stats(ppo_agent, curriculum_inds, training_settings, accuracy)

        # save model
        save_policy(ppo_agent, env, accuracy, training_settings)

        # policy update
        policy_update(ppo_agent, training_settings)
        ppo_agent.episode = ppo_agent.episode + 1


if __name__ == "__main__":
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import torch

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    settings = {
        "contact_settings": {
            "shrink_ratio": 0.0,
        },
        "env": {
            "n_robot": 2,
            "boundary_part_ids": [0],
            "sim_buffer_size": 512,
            "num_rollouts": 512,
            "verbose": False,
        },
        "rbe": {
            "density": 1E2,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "verbose": False,
        },
        "admm": {
            "Ccp": 1E6,
            "evaluate_it": 100,
            "max_iter": 1000,
            "float_type": torch.float32,
        },
        "search": {
            "n_beam": 64,
        },
        "policy": {
            "gat_layers": 8,
            "gat_heads": 1,
            "gat_hidden_dims": 16,
            "gat_dropout": 0.0,
            "centroids": False
        },
        "ppo": {
            "gamma": 0.95,
            "eps_clip": 0.2,

            "base_entropy_weight": 0.005,
            "entropy_weight_increase": 0.001,
            "max_entropy_weight": 0.01,

            "lr_milestones": [100, 300],
            "num_step_anneal": 500,
            "lr_actor": 2e-3,
            "betas_actor": [0.95, 0.999],
        },
        "training": {
            "max_train_epochs": 50000,
            "save_delta_accuracy": 0.01,
            "print_epochs": 1,
            "policy_update_batch_size": 2048,
            "K_epochs": 5
        }
    }
    contacts = compute_assembly_contacts(parts, settings)
    train(parts, contacts, settings)
