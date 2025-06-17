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
import time
from multiprocessing import Queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

def training_rollout(ppo_agent: PPO,
                     env: DisassemblyEnv,
                     curriculum_inds: np.ndarray,
                     training_settings,
                     queue: Queue):
    part_states = env.curriculum[curriculum_inds, :]
    ppo_agent.buffer.clear(part_states.shape[0])
    n_env = part_states.shape[0]
    env_inds = np.arange(n_env)
    env.updated_simulation = False

    n_step = 0
    while True:
        queue.put(part_states)

        current_states = part_states[env_inds, :]
        masks = env.action_masks(current_states)
        current_actions = ppo_agent.select_action(current_states, masks, env_inds)
        next_states, rewards, next_stability = env.step(current_states, current_actions)
        part_states[env_inds, :] = next_states

        ppo_agent.buffer.add("next_states", next_states, env_inds)
        ppo_agent.buffer.add("rewards_per_step", rewards, env_inds)
        ppo_agent.buffer.add("next_stability", next_stability, env_inds)

        env_inds, _ = ppo_agent.buffer.get_valid_env_inds(env)
        if queue is not None:
            data = ppo_agent.buffer.get_current_states_for_rendering(training_settings.num_render_debug)
            queue.put(data)
        n_step = n_step + 1

        if env_inds.shape[0] == 0:
            break

    env.simulate_buffer(simulate_remain=True)
    _, rewards = ppo_agent.buffer.get_valid_env_inds(env)
    if queue is not None:
        data = ppo_agent.buffer.get_current_states_for_rendering(training_settings.num_render_debug)
        queue.put(data)
    return torch.tensor(rewards, dtype=floatType)

def train(parts: list[Trimesh],
          contacts: dict,
          settings:dict,
          queue: Queue = None):
    training_settings = set_default(settings,
                                   "training", {
                                       "max_train_epochs": 50000,
                                       "save_delta_accuracy": 0.01,
                                       "print_epochs": 1,
                                       "policy_update_batch_size": 2048,
                                       "K_epochs": 5,
                                       "output_name": "example",
                                        "num_render_debug": 8 * 8,
    })
    training_settings = SimpleNamespace(**training_settings)

    # policy output folder
    folder_path = f"./models/{training_settings.output_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder
    saved_model_path = f"{folder_path}/policy.pol"

    # create env
    env = DisassemblyEnv(parts, contacts, settings=settings)
    torch_geometric.seed.seed_everything(settings["env"]["seed"])

    # forward curriculum
    _, _, curriculum = forward_curriculum(parts, contacts, settings=settings)
    env.set_curriculum(curriculum)

    # create ppo agent
    ppo_agent = PPO(env, settings)

    print(f"Start training, curriculum: {env.curriculum.shape[0]}")
    ppo_agent.print_iter = 0
    print_epoch = training_settings.print_epochs
    ppo_agent.episode_timer = time.perf_counter()

    # training
    while ppo_agent.episode <= training_settings.max_train_epochs and ppo_agent.saved_accuracy < 1:
        curriculum_inds = ppo_agent.buffer.sample_curriculum(env.num_rollouts)
        ppo_agent.buffer.curriculum_inds = curriculum_inds
        ppo_agent.buffer.rewards = training_rollout(ppo_agent, env, curriculum_inds, training_settings, queue)

        # update entropy weights
        rewards = ppo_agent.buffer.rewards
        inds = curriculum_inds.to(device)
        failed_inds = inds[(rewards < 0)]
        success_inds = inds[(rewards > 0)]
        ppo_agent.num_success[success_inds] += 1
        ppo_agent.num_failed[success_inds] = 0
        ppo_agent.num_failed[failed_inds] = ppo_agent.num_failed[failed_inds] + 1
        ppo_agent.buffer.modify_entropy(failed_inds, success_inds)

        # print status
        ppo_agent.print_iter = ppo_agent.print_iter + 1
        if ppo_agent.print_iter % print_epoch == print_epoch - 1:
            elaspe = time.perf_counter() - ppo_agent.episode_timer
            accuracy = (torch.sum(rewards > 0) / env.num_rollouts).item()
            print("Episode : {} \t\t Accuracy : {:.2f}\t\t Time : {:.2f}".format(ppo_agent.episode,
                                                                             accuracy,
                                                                             elaspe))
            ppo_agent.episode_timer = time.perf_counter()

        # save policy
        delta_accuracy = training_settings.save_delta_accuracy
        if accuracy > ppo_agent.saved_accuracy + delta_accuracy:
            ppo_agent.saved_accuracy = accuracy
            ppo_agent.save(saved_model_path, settings)

        # policy update
        loss, sur_loss, val_loss, entropy = ppo_agent.update_policy(batch_size=training_settings.policy_update_batch_size,
                                                                    update_iter=training_settings.K_epochs)
        print("Loss : {:.2f} \t\t Sur: {:.2f} \t\t Val: {:.2f} \t\t Entropy : {:.2f}".format(loss, sur_loss, val_loss, entropy))
        ppo_agent.episode = ppo_agent.episode + 1