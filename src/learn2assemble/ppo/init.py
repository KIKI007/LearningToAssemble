import copy
import os
from warnings import warn
import torch
from torch import nn
import torch_geometric
import numpy as np
from stability.geometry.assembly_stability import AssemblyStability
import pickle
import wandb
from stability.search.env import AssemblySearchEnv, DisassemblySearchEnv
from stability.search.search import BeamSearch
from stability.solver.qp_solver import Callback
from stability import ADMMSolver, GUROBISolver, RBEProblem2
import numpy as np
from stability import DATA_DIR
from stability.ppo.env import DisassemblyGymEnv
from stability.ppo.agent import PPO, PPO_GAT
import json
from os.path import isfile
from pathlib import Path

import platform
if platform.processor() != "aarch64" or platform.system() != "Linux":
    import polyscope as ps
    import polyscope.imgui as psim


def config_init(model_name="arch",
                method="PPO_GAT",
                train = True,
                online = False,
                name=None,
                config_file=None,
                pretrained_file=None,
                artifact_name = None,
                n_beam_test=None
                ):
    if pretrained_file is not None:
        assert config_file is None,"Only use one of the methods to initialize the config"
        with open(pretrained_file, 'rb') as handle:
            agent = pickle.load(handle)
            config = agent['config']
    else:
        with open(config_file) as json_file:
            print(f"load parameter from {config_file}")
            config = json.load(json_file)
        if config['boundary'].get(model_name,None) is None:
            warn("""Please update the parameter file to include the boundary
                    of your structure, now using [0] as default""")
            config['boundary']=[0]
        else:
            config['boundary']=config['boundary'][model_name]
        config.update({key:val for key,val in config['default_admm_params'].items()})
        if config['admm_params'].get(model_name,None) is None:
            warn("""Please update the parameter file to include the admm parameters of your structure.
                    Now using the default. To supress warning, add an empty dict""")
        config.update({key:val for key,val in config['admm_params'][model_name].items()})
    name = name or f"{method}_{model_name}"
    config.update({'model_name':model_name,
                   "online":online,
                   "architecture":method})
    config['n_beam_test']=n_beam_test
    wandb.init(
        mode="online" if online else "offline",
        project=f"Disassembly_train" if train else "Disassembly_test",
        name=name,
        config=config
    )
    if artifact_name is not None:
        artifact = wandb.use_artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        pretrained_file = artifact_dir+"/"+os.listdir(artifact_dir)[0]
        with open(pretrained_file, 'rb') as handle:
            agent = pickle.load(handle)
            config = agent['config']
            if n_beam_test is not None:
                config['n_beam_test']=n_beam_test
        wandb.config.update(config,allow_val_change=True)
        return pretrained_file
    return None
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

def build_curriculum(assembly, forward_learning = True, nbeam = 64, solver = "gurobi", directed=False, ending = ""):

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
    searcher.solve(nbeam = nbeam)
    curriculum_files = searcher.save_curriculum(curriculum_folder)
    return curriculum_folder

def init_env(static_random_seed=True,curriculum=False,n_beam=None):
    
    print("model:\t", wandb.config.model_name)
    if wandb.config.random_seed and static_random_seed:
        torch_geometric.seed.seed_everything(wandb.config.random_seed)

    assembly = AssemblyStability(rho=wandb.config.rho, nt=wandb.config.nt)
    assembly.from_objs(DATA_DIR + f"/scene/{wandb.config.model_name}")

    env = DisassemblyGymEnv(assembly, n_robot=wandb.config.n_robot, boundary = wandb.config.boundary)
    problem_admm, problem_gurobi, admm_threshold = rbe_problem(assembly)

    env.init_solver(problem_admm, problem_gurobi, admm_threshold)

    if wandb.config.directions:
        env.assembly.sample_removing_directions(wandb.config.n_remove_dir,
                                                tol = wandb.config.remove_dir_tol, front_axis=wandb.config.remove_dir_front_axis, positive_z=wandb.config.remove_dir_positive_z)
        env.assembly.save_removing_directions_by_neighbours("drts.json", [0], 10)

    #load previous record
    if isfile(f'./records/{wandb.config.model_name}.rec'):
        with open(f'./records/{wandb.config.model_name}.rec', 'rb') as handle:
            env.stability_history = pickle.load(handle)
        print(f"{len(env.stability_history)} sampled presimulated")


    #load curriculum
    if curriculum:
        if n_beam is None:
            n_beam = wandb.config.n_beam_curriculum
        #create the curriculum if needed
        forward_folder, backward_folder = None, None
        ending = ""
        if wandb.config.directions:
            ending+='_drts'
        if wandb.config.n_robot!=2:
            ending+=f"_{wandb.config.n_robot}robots"

        forward_folder = build_curriculum(assembly, forward_learning = True, nbeam =n_beam, solver = "admm", directed=wandb.config.directions, ending = ending)
        if wandb.config.backward:
            backward_folder = build_curriculum(assembly, forward_learning = False, nbeam = n_beam, solver = "admm", directed=wandb.config.directions, ending = ending)
        #load the curriculum
        if forward_folder:
            env.load_curriculum(filepath=forward_folder + "/total.textbook", forward_learning = True)
            forward_curriculum = copy.deepcopy(env.curriculum)

        if backward_folder:
            env.load_curriculum(filepath=backward_folder + "/total.textbook", forward_learning = False)
            if forward_folder:
                env.curriculum.append_dataset(forward_curriculum)
    return env

def init_ppo(env,pretrained_file=None):
    if pretrained_file is not None:
        #Use the same config as before
        with open(pretrained_file, 'rb') as handle:
            agent = pickle.load(handle)
            config = agent['config']
            state_dict = agent['state_dict']
            print_info = agent['print_info']
            print(print_info)
            
    # initialize a PPO agent
    if wandb.config.architecture == "PPO":
        ppo_agent = PPO(None,env.state_dim, env.action_dim, wandb.config,len(env.curriculum))
    elif wandb.config.architecture == "PPO_GAT":
        metadata = (['part', 'force', 'torque'], 
                        [('part', 'pp', 'part'),('torque', 'tt', 'torque'),('part', 'pt', 'torque'),('torque', 'rev_pt', 'part'),
                         ('part', 'pfplus', 'force'),('force', 'rev_pfplus', 'part'),('part', 'pfminus', 'force'),
                         ('force', 'rev_pfminus', 'part'),('force', 'ft', 'torque'),('torque', 'rev_ft', 'force'),('part', 'is', 'part'),
                         ('force', 'is', 'force'),('torque', 'is', 'torque')])
        ppo_agent = PPO_GAT(metadata, env.n_part, env.action_dim, wandb.config,len(env.curriculum))
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