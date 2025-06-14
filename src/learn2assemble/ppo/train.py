import math
import os
import pickle

import numpy as np
import torch
import wandb
from pathlib import Path
import polyscope as ps
from stability.ppo.helper import intType,floatType, device
from copy import deepcopy

def rollout_asyn(ppo_agent, env, curriculum_inds, ui = False):

    states = env.curriculum.bin_states[curriculum_inds, :]
    rewards = torch.zeros(states.shape[0], dtype=intType, device=device)
    actions = torch.ones(states.shape[0], dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(states.shape[0], dtype=intType, device=device)
    ppo_agent.buffer.clear(states.shape[0])
    n_env = states.shape[0]

    while (torch.min(num_actions) < env.state_dim + 1):
        if ui:
            env.render(states)

        # 1. masks
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        sub_masks = env.action_mask(states[env_inds, :])
        non_actions = env_inds[torch.sum(sub_masks, dim=1) == 0]
        rewards[non_actions] = -1

        # 2. inference
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        if env_inds.shape[0] > 0:
            num_actions[env_inds] = num_actions[env_inds] + 1
            sub_states = states[env_inds, :]
            sub_masks = env.action_mask(sub_states)
            if not torch.all(torch.any(sub_masks,1)):
                print("filter nonaction failed")
            actions[env_inds] = ppo_agent.select_action(sub_states, sub_masks, env_inds,curriculum_inds)

            wandb.log({"build_graph_time": ppo_agent.build_graph_time,
                       "batch_graph_time": ppo_agent.batch_graph_time,
                       "inference_time": ppo_agent.inference_time},
                      step=ppo_agent.time_step)
        # 3. check record
        env_inds = torch.logical_and(rewards == 0, actions >= 0).nonzero().reshape(-1)
        states[env_inds, :], rewards[env_inds], done = env.step(states[env_inds, :], actions[env_inds], False)
        actions[env_inds[done]] = -1
        # 4. simulation
        num_inference = float(torch.sum(actions == -1) - torch.sum(rewards != 0))
        if num_inference < 0.1 * n_env:
            env_inds = torch.logical_and(rewards == 0, actions >= 0).nonzero().reshape(-1)
            states[env_inds, :], rewards[env_inds], done = env.step(states[env_inds, :], actions[env_inds], True)
            actions[env_inds[done]] = -1

            wandb.log({"sim_time": env.sim_time,
                       "num_sim": env.num_sim,
                       "dict_time": env.dict_time}, step=ppo_agent.time_step)

        if torch.sum(rewards != 0) == n_env:
            if ui:
                env.render(states)
            break
        ppo_agent.time_step = ppo_agent.time_step + 1

    rewards = rewards.type(floatType)
    wandb.log({"epoch": ppo_agent.episode,
               "Reward": rewards.mean().item()}, step=ppo_agent.time_step)
    return rewards.cpu()
def save_policy(ppo_agent, env, accuracy):
    delta_accuracy = wandb.config.save_delta_accuracy
    if accuracy > ppo_agent.saved_accuracy + delta_accuracy:
        ppo_agent.saved_accuracy = accuracy
        name = f"{wandb.config.model_name}_{int(ppo_agent.saved_accuracy * 100)}"
        folder_path = "./models/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create the folder
        model_path = f"{folder_path}/{name}.pol"
        ppo_agent.save(model_path)
        if wandb.config.online:
            policy_artifact = wandb.Artifact(f"{wandb.config.model_name}_{wandb.config.architecture}_pol",
                                             type='model')
            policy_artifact.add_file(local_path=model_path, name=name)
            wandb.log_artifact(policy_artifact)
def update_stats(ppo_agent, curriculum_inds, accuracy = None):
    # update number of fails and successes
    rewards = ppo_agent.buffer.rewards
    avg_reward = rewards.mean().item()
    failed_inds = curriculum_inds[(rewards < 0)]
    success_inds = curriculum_inds[(rewards > 0)]

    ppo_agent.num_success[success_inds] += 1
    ppo_agent.num_failed[success_inds] = 0
    ppo_agent.num_failed[failed_inds] = ppo_agent.num_failed[failed_inds] + 1
    ppo_agent.buffer.modify_entropy(failed_inds,success_inds)
    if accuracy is None:
        accuracy = (avg_reward+1)/2

    # print status
    ppo_agent.print_iter = ppo_agent.print_iter + 1
    print_epoch = wandb.config.print_epochs
    if ppo_agent.print_iter % print_epoch == print_epoch - 1:
        print()
        print("Episode : {} \t\t Accuracy : {} \t\t Reward : {}".format( ppo_agent.episode,
                                                                          round(accuracy, 2),
                                                                          round(avg_reward, 2)))
        print("Failed Curriculum: {}".format(curriculum_inds[ppo_agent.buffer.rewards < 0]))

def policy_check(ppo_agent, curriculum_inds):

    num_success = torch.sum(ppo_agent.buffer.rewards > 0)
    num_envs = curriculum_inds.shape[0]
    # compute accuracy
    accuracy = (num_success / num_envs).cpu().item()

    wandb.log({"training accuracy": accuracy, "min_success": torch.min(ppo_agent.num_success)}, step=ppo_agent.time_step)
    return accuracy
      
def sample_curriculum(ppo_agent):
    num_envs = wandb.config.num_train_rollouts
    sample_weights = None
    if (ppo_agent.buffer.curriculum_cumvalloss==100).sum()>num_envs:
        print("New envs only")
        sample_weights = torch.zeros_like(ppo_agent.buffer.curriculum_cumvalloss)
        sample_weights[ppo_agent.buffer.curriculum_cumvalloss==100] = 1/(ppo_agent.buffer.curriculum_cumvalloss==100).sum()
        curriculum_inds = torch.multinomial(sample_weights,
                                            num_envs,
                                            False)
    else:
        sample_weights = (torch.pow(torch.arange(1,ppo_agent.buffer.curriculum_rank.shape[0]+1,device=device,dtype=floatType),-ppo_agent.buffer.alpha)/
                            torch.pow(torch.arange(1,ppo_agent.buffer.curriculum_rank.shape[0]+1,device=device,dtype=floatType),-ppo_agent.buffer.alpha).sum())
        curriculum_inds = torch.multinomial(sample_weights[ppo_agent.buffer.curriculum_rank],
                                            num_envs,
                                            True)
    return curriculum_inds

def policy_update(ppo_agent):
    loss, entropy, num_training_data = ppo_agent.update_policy(
        batch_size=wandb.config.policy_update_batch_size,
        shuffle=wandb.config.policy_update_shuffle,
        update_iter=wandb.config.K_epochs)

    ppo_agent.uniform_training_data = num_training_data
    ppo_agent.uniform_entropy = entropy
    wandb.log({"train time": ppo_agent.training_time,
               "datasetbuild time":ppo_agent.build_ds_time,
                "loss": loss,
                "entropy": entropy,
                "train data": num_training_data,
                "value loss":ppo_agent.val_loss,
                "surrogate loss":ppo_agent.sur_loss,}, step=ppo_agent.time_step)
def train(ppo_agent, env, ui = False):    
    print(f"Start training, curriculum: {env.num_curriculum()}")
    # update variables
    ppo_agent.print_iter = 0
    
    assert ppo_agent.buffer.curriculum_rank.shape[0]==len(env.curriculum)

    if ui:
        env.init_render()
        env.render(env.curriculum.bin_states)
        ps.show()

    # training
    while ppo_agent.episode <= wandb.config.max_train_epochs and ppo_agent.saved_accuracy < 1:

        curriculum_inds = sample_curriculum(ppo_agent)
        ppo_agent.buffer.curriculum_inds = curriculum_inds.cpu()
        ppo_agent.buffer.rewards = rollout_asyn(ppo_agent, env, curriculum_inds, ui = ui).cpu()
        
        # compute accuracy
        accuracy = policy_check(ppo_agent, curriculum_inds)

        # update number of fails
        update_stats(ppo_agent, curriculum_inds, accuracy)

        # save model
        save_policy(ppo_agent, env, accuracy)

        
        # policy update
        policy_update(ppo_agent)
        ppo_agent.episode = ppo_agent.episode + 1