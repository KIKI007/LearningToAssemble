import math
import os
import pickle
from types import SimpleNamespace

import torch
from pathlib import Path
import torch_geometric

from learn2assemble import update_default_settings
from learn2assemble.env import DisassemblyEnv
from learn2assemble.curriculum import forward_curriculum
from learn2assemble.ppo import PPO

from trimesh import Trimesh
import numpy as np
import time
from multiprocessing import Queue

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

def training_rollout(ppo_agent: PPO,
                     env: DisassemblyEnv,
                     curriculum_inds: np.ndarray,
                     training_settings,
                     queue: Queue):
    part_states = env.curriculum[curriculum_inds, :]
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


def compute_accuracy(env, state_dict, settings, queue):

    ppo_agent = PPO(env.parts, env.contacts, settings)
    ppo_agent.policy_old.load_state_dict(state_dict)
    ppo_agent.policy.load_state_dict(state_dict)
    ppo_agent.deterministic = True
    ppo_agent.buffer.reset_curriculum(env.curriculum.shape[0])
    num_of_sample = min(env.num_rollouts, env.curriculum.shape[0])
    ppo_agent.buffer.curriculum_inds = np.random.choice(env.curriculum.shape[0], size=num_of_sample, replace=False)
    ppo_agent.buffer.rewards = training_rollout(ppo_agent,
                                                env,
                                                ppo_agent.buffer.curriculum_inds,
                                                SimpleNamespace(**settings["training"]),
                                                queue)
    return torch.sum(ppo_agent.buffer.rewards > 0).item() / ppo_agent.buffer.curriculum_inds.shape[0]


def evaluation(parts: list[Trimesh],
               contacts: dict,
               policy_name: str,
               num_render_debug: int = 4 * 4,
               queue: Queue = None):
    pretrained_file = f"./models/{policy_name}.pol"
    with open(pretrained_file, 'rb') as handle:
        agent = pickle.load(handle)
        state_dict = agent['state_dict']
        settings = agent['settings']
        curriculum = agent.get('curriculum', None)
        stability_history = agent.get('stability_history', {})

    settings['training']['num_render_debug'] = num_render_debug

    env = DisassemblyEnv(parts, contacts, settings=settings)

    # single forward curriculum (training)
    torch_geometric.seed.seed_everything(settings["env"]["seed"])
    if curriculum is None:
        _, _, curriculum = forward_curriculum(parts, contacts, settings=settings)
    env.set_curriculum(curriculum)
    accuracy = compute_accuracy(env, state_dict, settings, queue)
    print(f"Train Size:\t {env.curriculum.shape[0]}", "\t\t", "Accuracy:\t", accuracy)

    # double forward curriculum (testing)
    settings["curriculum"]["n_beam"] *= 2
    torch_geometric.seed.seed_everything(settings["env"]["seed"])
    _, _, curriculum = forward_curriculum(parts, contacts, settings=settings)
    new_curriculum = []
    for part_state in curriculum:
        if env.get_history(part_state) == 0:
            new_curriculum.append(part_state)
    new_curriculum = np.vstack(new_curriculum)
    env.set_curriculum(curriculum)
    env.stability_history = stability_history
    accuracy = compute_accuracy(env, state_dict, settings, queue)
    print(f"Test Size:\t {env.curriculum.shape[0]}", "\t\t", "Accuracy:\t", accuracy)


def train(parts: list[Trimesh],
          contacts: dict,
          settings: dict,
          wandb = None):
    training_settings = update_default_settings(settings,
                                    "training", {
                                        "max_train_epochs": 50000,
                                        "save_delta_accuracy": 0.01,
                                        "print_epochs": 1,
                                        "policy_update_batch_size": 2048,
                                        "K_epochs": 5,
                                        "policy_name": "example",
                                        "num_render_debug": 8 * 8,
                                        "accuracy_terminate_threshold": 0.98,
                                    })
    training_settings = SimpleNamespace(**training_settings)

    # policy output folder
    folder_path = f"./models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder
    saved_model_path = f"{folder_path}/{training_settings.policy_name}.pol"

    # create env and ppo
    env = DisassemblyEnv(parts, contacts, settings=settings)
    ppo_agent = PPO(parts, contacts, settings)
    torch_geometric.seed.seed_everything(settings["env"]["seed"])

    # forward curriculum
    _, _, curriculum = forward_curriculum(parts, contacts, settings=settings)
    env.set_curriculum(curriculum)
    ppo_agent.buffer.reset_curriculum(env.curriculum.shape[0])

    print(f"Start training, curriculum: {env.curriculum.shape[0]}")
    save_epochs = training_settings.save_epochs

    # training
    while ppo_agent.episode <= training_settings.max_train_epochs:
        ppo_agent.episode_timer = time.perf_counter()

        # the first several run is to visit all curriculum
        curriculum_inds, sample_new_curriculum = ppo_agent.buffer.sample_curriculum(env.num_rollouts)
        ppo_agent.buffer.curriculum_inds = curriculum_inds
        rewards = ppo_agent.buffer.rewards = training_rollout(ppo_agent, env, curriculum_inds, training_settings, None)

        # update entropy weights
        inds = curriculum_inds.to(device)
        success_inds, failed_inds = inds[(rewards > 0)], inds[(rewards < 0)]
        ppo_agent.buffer.num_success[success_inds] += 1
        ppo_agent.buffer.num_failed[success_inds] = 0
        ppo_agent.buffer.num_failed[failed_inds] = ppo_agent.buffer.num_failed[failed_inds] + 1
        ppo_agent.buffer.modify_entropy()

        # save policy
        accuracy_of_sample_curriculum = (torch.sum(rewards > 0) / rewards.shape[0]).item()
        accuracy_of_entire_curriculum = None
        if (ppo_agent.episode + 1) % save_epochs == 0:
            accuracy_of_entire_curriculum = compute_accuracy(env, ppo_agent.policy_old.state_dict(), settings, None)
            if accuracy_of_entire_curriculum > ppo_agent.accuracy_of_entire_curriculum:
                ppo_agent.accuracy_of_entire_curriculum = accuracy_of_entire_curriculum
                ppo_agent.save(saved_model_path, settings, env.curriculum, env.stability_history)

        # policy update
        loss, sur_loss, val_loss, entropy = ppo_agent.update_policy(
            batch_size=training_settings.policy_update_batch_size,
            update_iter=training_settings.K_epochs)

        # print status
        episode_time = time.perf_counter() - ppo_agent.episode_timer
        print("")
        print("Episode : {} \t\t Accuracy : {:.2f}\t\t Time : {:.2f}".format(ppo_agent.episode,
                                                                             accuracy_of_sample_curriculum, episode_time))
        print("Loss : {:.2e} \t\t Sur: {:.2e} \t\t Val: {:.2e} \t\t Entropy : {:.2e}".format(loss, sur_loss, val_loss,
                                                                                             entropy))

        if wandb is not None:
            wandb.log({"episode_time": episode_time,
                       "update_time": ppo_agent.timer("update"),
                       "inference_time": ppo_agent.timer("inference"),
                       "graph_time": ppo_agent.timer("graph"),
                       "sim_time": env.timer("simulation"),
                       "history_time": env.timer("history"),
                       "accuracy_sampled": accuracy_of_sample_curriculum,
                       "loss": loss,
                       "entropy": entropy,
                       "value loss": val_loss,
                       "surrogate loss": sur_loss, }, step=ppo_agent.episode)

        if accuracy_of_entire_curriculum is not None:
            print(f"Saved! accuracy of entire curriculum:\t", accuracy_of_entire_curriculum)
            if wandb is not None:
                wandb.log({"accuracy_curriculum": accuracy_of_entire_curriculum}, step=ppo_agent.episode)
            if accuracy_of_entire_curriculum > training_settings.accuracy_terminate_threshold:
                break
        ppo_agent.episode = ppo_agent.episode + 1
