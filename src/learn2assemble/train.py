import math
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import wandb
from pathlib import Path
import polyscope as ps
import torch_geometric
from copy import deepcopy
from learn2assemble.disassemly_env import DisassemblyEnv

from trimesh import Trimesh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

def init_training(parts: list[Trimesh],
                  contacts: dict,
                  n_robot: int,
                  settings: dict):
    print("model:\t", wandb.config.model_name)
    config = SimpleNamespace(settings["policy"])
    torch_geometric.seed.seed_everything(config.random_seed)


    env = DisassemblyEnv(parts, contacts, n_robot=n_robot, settings=settings)
    problem_admm, problem_gurobi, admm_threshold = rbe_problem(assembly)

    env.init_solver(problem_admm, problem_gurobi, admm_threshold)

    if wandb.config.directions:
        env.assembly.sample_removing_directions(wandb.config.n_remove_dir,
                                                tol=wandb.config.remove_dir_tol,
                                                front_axis=wandb.config.remove_dir_front_axis,
                                                positive_z=wandb.config.remove_dir_positive_z)
        env.assembly.save_removing_directions_by_neighbours("drts.json", [0], 10)

    # load previous record
    if isfile(f'./records/{wandb.config.model_name}.rec'):
        with open(f'./records/{wandb.config.model_name}.rec', 'rb') as handle:
            env.record = pickle.load(handle)
        print(f"{len(env.record)} sampled presimulated")

    # load curriculum
    if curriculum:
        if n_beam is None:
            n_beam = wandb.config.n_beam_curriculum
        # create the curriculum if needed
        forward_folder, backward_folder = None, None
        ending = ""
        if wandb.config.directions:
            ending += '_drts'
        if wandb.config.n_robot != 2:
            ending += f"_{wandb.config.n_robot}robots"

        forward_folder = build_curriculum(assembly, forward_learning=True, nbeam=n_beam, solver="admm",
                                          directed=wandb.config.directions, ending=ending)
        if wandb.config.backward:
            backward_folder = build_curriculum(assembly, forward_learning=False, nbeam=n_beam, solver="admm",
                                               directed=wandb.config.directions, ending=ending)
        # load the curriculum
        if forward_folder:
            env.load_curriculum(filepath=forward_folder + "/total.textbook", forward_learning=True)
            forward_curriculum = copy.deepcopy(env.curriculum)

        if backward_folder:
            env.load_curriculum(filepath=backward_folder + "/total.textbook", forward_learning=False)
            if forward_folder:
                env.curriculum.append_dataset(forward_curriculum)
    return env

def load_admm_setting(file=None):
    if file and os.path.exists(file):
        with open(file, "r") as f:
            print(f"load parameter from {file}")
            setting = json.load(f)
        return setting
    return None


def rbe_problem(assembly):
    config = wandb.config
    callback = Callback(assembly, False)
    solver = ADMMSolver(eps=config.qtol,
                        max_iter=config.max_admm_iter,
                        eps_conv=config.eps_conv,
                        verbose=False,
                        rs=config.rs,
                        evaluate_iter=config.evaluate_iter,
                        alpha=config.alpha)

    solver.set_callback(callback)

    problem_admm = RBEProblem2(assembly,
                               solver,
                               mu=config.mu,
                               Ccp=config.Ccp,
                               verbose=False,
                               lb=config.qtol,
                               ub=config.ub,
                               max_iter=config.max_qp_iter)

    if wandb.config.admm_batch > 0:
        problem_gurobi = RBEProblem2(assembly,
                                     GUROBISolver(),
                                     mu=config.mu,
                                     Ccp=config.Ccp,
                                     verbose=False,
                                     lb=config.qtol,
                                     max_iter=config.max_qp_iter)
    else:
        problem_gurobi = None

    return problem_admm, problem_gurobi, config.admm_batch


def build_curriculum(assembly, forward_learning=True, nbeam=64, solver="gurobi", directed=False, ending=""):
    curriculum_folder = f"{DATA_DIR}/curriculum/{wandb.config.model_name}_{'forward' if forward_learning else 'backward'}_{nbeam}{ending}"

    if os.path.exists(curriculum_folder):
        curriculum_files_dict = {}
        for filename in os.listdir(curriculum_folder):
            file_path = os.path.join(curriculum_folder, filename)
            if isfile(file_path) and Path(file_path).stem.isnumeric():
                curriculum_files_dict[int(Path(file_path).stem)] = file_path
        curriculum_files = sorted(curriculum_files_dict.items())
        curriculum_files = [x[1] for x in curriculum_files]
        return curriculum_folder
    else:
        print('creating new curriculum')
    # env
    if forward_learning:
        env = AssemblySearchEnv(assembly.n_part(), wandb.config.n_robot, wandb.config.boundary)
    else:
        env = DisassemblySearchEnv(assembly.n_part(), wandb.config.n_robot, wandb.config.boundary)
    env.set_assembly(assembly)

    # solver
    problem_admm, problem_gurobi, _ = rbe_problem(assembly)

    if solver == "gurobi":
        searcher = BeamSearch(env, problem_gurobi, render=False)
    else:
        searcher = BeamSearch(env, problem_admm, render=False)
    # search
    searcher.verbose = False
    searcher.solve(nbeam=nbeam)
    curriculum_files = searcher.save_curriculum(curriculum_folder)
    return curriculum_folder


def init_env(static_random_seed=True, curriculum=False, n_beam=None):



def init_ppo(env, pretrained_file=None):
    if pretrained_file is not None:
        # Use the same config as before
        with open(pretrained_file, 'rb') as handle:
            agent = pickle.load(handle)
            config = agent['config']
            state_dict = agent['state_dict']
            print_info = agent['print_info']
            print(print_info)

    # initialize a PPO agent
    if wandb.config.architecture == "PPO":
        ppo_agent = PPO(None, env.state_dim, env.action_dim, wandb.config, len(env.curriculum))
    elif wandb.config.architecture == "PPO_GAT":
        metadata = (['part', 'force', 'torque'],
                    [('part', 'pp', 'part'), ('torque', 'tt', 'torque'), ('part', 'pt', 'torque'),
                     ('torque', 'rev_pt', 'part'),
                     ('part', 'pfplus', 'force'), ('force', 'rev_pfplus', 'part'), ('part', 'pfminus', 'force'),
                     ('force', 'rev_pfminus', 'part'), ('force', 'ft', 'torque'), ('torque', 'rev_ft', 'force'),
                     ('part', 'is', 'part'),
                     ('force', 'is', 'force'), ('torque', 'is', 'torque')])
        ppo_agent = PPO_GAT(metadata, env.n_part, env.action_dim, wandb.config, len(env.curriculum))
        ppo_agent.reset_assembly(env.assembly)
    if pretrained_file is not None:
        ppo_agent.policy_old.load_state_dict(state_dict)
        ppo_agent.policy.load_state_dict(state_dict)
    ppo_agent.time_step = 0
    ppo_agent.episode = 0
    return ppo_agent


def load_forward_curriculum(env, problem_level, solver="gurobi"):
    curriculum_folder = f"{DATA_DIR}/curriculum/{wandb.config.model_name}_forward_{solver}"
    env.load_curriculum(filepath=f"{curriculum_folder}/{problem_level}.textbook")
    return curriculum_folder


def load_backward_curriculum(env, problem_level, solver="gurobi"):
    curriculum_folder = f"{DATA_DIR}/curriculum/{wandb.config.model_name}_backward_{solver}"
    env.load_curriculum(filepath=f"{curriculum_folder}/{problem_level}.textbook")
    return curriculum_folder



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
        sub_masks = env.mask(states[env_inds, :])
        non_actions = env_inds[torch.sum(sub_masks, dim=1) == 0]
        rewards[non_actions] = -1

        # 2. inference
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        if env_inds.shape[0] > 0:
            num_actions[env_inds] = num_actions[env_inds] + 1
            sub_states = states[env_inds, :]
            sub_masks = env.mask(sub_states)
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