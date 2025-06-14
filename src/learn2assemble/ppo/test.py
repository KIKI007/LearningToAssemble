from matplotlib import pyplot as plt
from stability.ppo.init import *
from stability.ppo.helper import *




def test_trained_policy(file):
    config_init(model_name="mario",
                method="PPO_GAT",
                online = False,
                name='test_init',
                config_file=None,
                pretrained_file=file)
    # init env

    wandb.config.update({"admm_batch":np.inf},allow_val_change=True)
    env = init_env(static_random_seed=True)
    ppo_agent = init_ppo(env,file)
    wandb.config.update({"admm_batch":np.inf},allow_val_change=True
                        )
    state = env.curriculum.bin_states[[0]]
    #state[0,8]=1
    #state[0,29]=1
    test_greedy_policy(env,ppo_agent,state)
def test_greedy_policy(env,ppo_agent,states,ui = True):
    traj = -torch.ones((states.shape[0],env.state_dim + 1,states.shape[1],states.shape[2]),device=device)
    traj_act = -torch.ones((states.shape[0],env.state_dim + 1),device=device,dtype=intType)
    traj[:,0]=states.to(device)
    rewards = torch.zeros(states.shape[0], dtype=intType, device=device)
    actions = torch.ones(states.shape[0], dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(states.shape[0], dtype=intType, device=device)
    ppo_agent.buffer.clear(states.shape[0])
    n_env = states.shape[0]
    if ui:
        env.init_render()
    while (torch.min(num_actions) < env.state_dim + 1):
        if ui:
            env.render(states, use_simulate = False, n_col = 8, offset_dx = 0, offset_dy =0)

        # 1. compute action masks
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        sub_masks = env.action_mask(states[env_inds, :])
        non_actions = env_inds[torch.sum(sub_masks, dim=1) == 0]
        if len(non_actions)>0:
            pass
        rewards[non_actions] = -1

        # 2. sample action
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        if env_inds.shape[0] > 0:
            num_actions[env_inds] = num_actions[env_inds] + 1
            sub_states = states[env_inds, :]
            sub_masks = env.action_mask(sub_states)
            action_stoch,logprobs,values = ppo_agent.compute_policy(sub_states, sub_masks)
            actions[env_inds] = torch.argmax(logprobs,1)

            wandb.log({"build_graph_time": ppo_agent.build_graph_time,
                       "batch_graph_time": ppo_agent.batch_graph_time,
                       "inference_time": ppo_agent.inference_time},
                      step=ppo_agent.time_step)


        # 3. env step without simulation
        env_inds = torch.logical_and(rewards == 0, actions >= 0).nonzero().reshape(-1)
        states[env_inds, :], rewards[env_inds], done = env.step(states[env_inds, :], actions[env_inds], False)
        traj[env_inds,num_actions[env_inds]]=states[env_inds]
        traj_act[env_inds,num_actions[env_inds]-1]=actions[env_inds]
        actions[env_inds[done]] = -1

        # 4. env step with simulation
        # to have as many simulation runs as possible at the same time
        num_inference = float(torch.sum(actions == -1) - torch.sum(rewards != 0))
        if num_inference < 0.1 * n_env:
            env_inds = torch.logical_and(rewards == 0, actions >= 0).nonzero().reshape(-1)
            states[env_inds, :], rewards[env_inds], done = env.step(states[env_inds, :], actions[env_inds], True)
            traj[env_inds,num_actions[env_inds]]=states[env_inds]
            traj_act[env_inds,num_actions[env_inds]-1]=actions[env_inds]
            actions[env_inds[done]] = -1

            wandb.log({"sim_time": env.sim_time,
                       "num_sim": env.num_sim,
                       "dict_time": env.dict_time}, step=ppo_agent.time_step)
        
        if torch.sum(rewards != 0) == n_env:
            if ui:
                env.render(states, use_simulate = False, n_col = 8, offset_dx = 0, offset_dy = 0)
            break
        ppo_agent.time_step = ppo_agent.time_step + 1

    rewards = rewards.type(floatType)
    wandb.log({"epoch": ppo_agent.episode}, step=ppo_agent.time_step)
    return traj,traj_act,rewards
def test_stochastic_policy(env,ppo_agent,states,ui = True):
    traj = -torch.ones((states.shape[0],env.state_dim + 1,states.shape[1],states.shape[2]),device=device)
    traj_act = -torch.ones((states.shape[0],env.state_dim + 1),device=device,dtype=intType)
    traj[:,0]=states.to(device)
    rewards = torch.zeros(states.shape[0], dtype=intType, device=device)
    actions = torch.ones(states.shape[0], dtype=intType, device=device) * (-1)
    num_actions = torch.zeros(states.shape[0], dtype=intType, device=device)
    ppo_agent.buffer.clear(states.shape[0])
    n_env = states.shape[0]
    if ui:
        env.init_render()
    while (torch.min(num_actions) < env.state_dim + 1):
        if ui:
            env.render(states, use_simulate = False, n_col = 8, offset_dx = 0, offset_dy =0)

        # 1. compute action masks
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        sub_masks = env.action_mask(states[env_inds, :])
        non_actions = env_inds[torch.sum(sub_masks, dim=1) == 0]
        if len(non_actions)>0:
            pass
        rewards[non_actions] = -1

        # 2. sample action
        env_inds = torch.logical_and(actions == -1, rewards == 0).nonzero().reshape(-1)
        if env_inds.shape[0] > 0:
            num_actions[env_inds] = num_actions[env_inds] + 1
            sub_states = states[env_inds, :]
            sub_masks = env.action_mask(sub_states)
            action_stoch,logprobs,values = ppo_agent.compute_policy(sub_states, sub_masks)
            actions[env_inds] = action_stoch
            wandb.log({"build_graph_time": ppo_agent.build_graph_time,
                       "batch_graph_time": ppo_agent.batch_graph_time,
                       "inference_time": ppo_agent.inference_time},
                      step=ppo_agent.time_step)


        # 3. env step without simulation
        env_inds = torch.logical_and(rewards == 0, actions >= 0).nonzero().reshape(-1)
        states[env_inds, :], rewards[env_inds], done = env.step(states[env_inds, :], actions[env_inds], False)
        traj[env_inds,num_actions[env_inds]]=states[env_inds]
        traj_act[env_inds,num_actions[env_inds]-1]=actions[env_inds]
        actions[env_inds[done]] = -1

        # 4. env step with simulation
        # to have as many simulation runs as possible at the same time
        num_inference = float(torch.sum(actions == -1) - torch.sum(rewards != 0))
        if num_inference < 0.1 * n_env:
            env_inds = torch.logical_and(rewards == 0, actions >= 0).nonzero().reshape(-1)
            states[env_inds, :], rewards[env_inds], done = env.step(states[env_inds, :], actions[env_inds], True)
            traj[env_inds,num_actions[env_inds]]=states[env_inds]
            traj_act[env_inds,num_actions[env_inds]-1]=actions[env_inds]
            actions[env_inds[done]] = -1

            wandb.log({"sim_time": env.sim_time,
                       "num_sim": env.num_sim,
                       "dict_time": env.dict_time}, step=ppo_agent.time_step)
        
        if torch.sum(rewards != 0) == n_env:
            if ui:
                env.render(states, use_simulate = False, n_col = 8, offset_dx = 0, offset_dy = 0)
            break
        ppo_agent.time_step = ppo_agent.time_step + 1

    rewards = rewards.type(floatType)
    wandb.log({"epoch": ppo_agent.episode}, step=ppo_agent.time_step)
    return traj,traj_act,rewards
def check_trajectory(file,traj):
    #config_init(model_name="mario",
    #            method="PPO_GAT",
    #            online = False,
    #            name='test_init',
    #            pretrained_file=file)
    # init env
    #wandb.config.update({"admm_batch":np.inf,"boundary":[0, 12, 59, 14, 2, 41, 5, 6, 7, 10, 17, 1, 13, 58, 9, 15, 38,8,29]},allow_val_change=True)
    env = init_env(static_random_seed=True)
    ppo_agent = init_ppo(env,file)
    for i,frame in enumerate(traj[:-1]):
        bin_states = torch.tensor(np.stack([frame>0,frame>1,frame>2],axis=-1).astype(float),device=device)
        bin_states[:,env.boundary,2]=1
        mask = env.action_mask(bin_states)
        diff = frame-traj[i+1]
        action = np.array(np.nonzero(diff))
        if np.max(diff,axis=1)==2:
            action[1]+=env.action_dim//2
        action_stoch,probs,values = ppo_agent.compute_policy(bin_states, mask)
        print(f"Action at time {i} is {'in' if mask[0,action[1]] else 'out of'} the mask")
        if not mask[0,action[1]]:
            mask = env.action_mask(bin_states)
        print(f"probability of taking action at time {i} :{probs[action[0],action[1]]}")
        actions = torch.argmax(probs,1)
def complete_test_policy(pretrained_file):
    eps=1e-5
    #test the policy on the final state
    env = init_env(static_random_seed=True)
    ppo_agent = init_ppo(env,pretrained_file)
    init_state = torch.zeros((1,env.action_dim//2,3),device=device,dtype=floatType)
    init_state[:,:,0]=1
    init_state[:,env.boundary,1:]=1
    print("Testing the greedy policy on the whole structure")
    
    traj,traj_act,rewards = test_greedy_policy(env,ppo_agent,init_state,ui=False)
    wandb.run.summary.update({"success":(rewards.cpu().numpy()>0).all()})
    print("Policy successful" if (rewards.cpu().numpy()>0).all() else "Policy failed")
    #test the policy on the training curriculum
    env = init_env(static_random_seed=True,curriculum=True)
    batch_size = wandb.config.policy_update_batch_size

    training_states = env.curriculum.bin_states
    rewards = torch.zeros(training_states.shape[0], dtype=floatType, device=device)
    dl = torch.utils.data.DataLoader(torch.arange(training_states.shape[0]), batch_size=wandb.config.policy_update_batch_size)
    suc_level = torch.zeros(training_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    tot_level = torch.zeros(training_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    print("Testing the greedy policy on the train dataset")
    for env_inds in dl:
        traj,traj_act,rewards_inds = test_greedy_policy(env,ppo_agent,training_states[env_inds].to(floatType),ui=False)
        rewards[env_inds]=rewards_inds
        levels = (training_states[env_inds].sum(-1)>0).sum(1)-env.boundary.shape[0]-1
        level_batch, num = torch.unique(levels,return_counts=True)
        tot_level[level_batch]+=num
        suc_batch,num_suc = torch.unique(levels[rewards_inds>0],return_counts=True)
        suc_level[suc_batch] += num_suc
    print(suc_level/tot_level)
    wandb.run.summary.update({'greedy_training_perlevel':(suc_level/(tot_level+eps)).cpu().numpy(),
               'tot_training_level':tot_level,
               'greedy_training_accuracy':((rewards+1)/2).mean().cpu().numpy()})
    print("Testing the stochastic policy on the train dataset")
    rewards = torch.zeros(training_states.shape[0], dtype=floatType, device=device)
    suc_level = torch.zeros(training_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    tot_level = torch.zeros(training_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    for n in range(wandb.config.num_test_rollouts):
        for env_inds in dl:
            traj,traj_act,rewards_inds = test_stochastic_policy(env,ppo_agent,training_states[env_inds].to(floatType),ui=False)
            rewards[env_inds]=rewards_inds
            levels = (training_states[env_inds].sum(-1)>0).sum(1)-env.boundary.shape[0]-1
            level_batch, num = torch.unique(levels,return_counts=True)
            tot_level[level_batch]+=num
            suc_batch,num_suc = torch.unique(levels[rewards_inds>0],return_counts=True)
            suc_level[suc_batch] += num_suc
    print(suc_level/tot_level)
    wandb.run.summary.update({'stochastic_training_perlevel':(suc_level/(tot_level+eps)).cpu().numpy(),
               'stochastic_training_accuracy':(rewards.mean()/wandb.config.num_test_rollouts).cpu().numpy()})
    

    #create a new curriculum on which the agent has not been trained specifically
    env = init_env(static_random_seed=True,curriculum=True,n_beam=wandb.config.n_beam_test)
    intraining = (env.curriculum.bin_states[:,None]==training_states[None]).all(-1).all(-1).any(-1)
    test_states = env.curriculum.bin_states[~intraining]
    print(f"{intraining.sum()} test states removed")
    print(f"{test_states.shape[0]} test states")
    
    rewards = torch.zeros(test_states.shape[0], dtype=floatType, device=device)
    dl = torch.utils.data.DataLoader(torch.arange(test_states.shape[0]), batch_size=wandb.config.policy_update_batch_size)
    suc_level = torch.zeros(test_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    tot_level = torch.zeros(test_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    print("Testing the greedy policy on the test dataset")
    for env_inds in dl:
        traj,traj_act,rewards_inds = test_greedy_policy(env,ppo_agent,test_states[env_inds].to(floatType),ui=False)
        rewards[env_inds]=rewards_inds
        levels = (test_states[env_inds].sum(-1)>0).sum(1)-env.boundary.shape[0]-1
        level_batch, num = torch.unique(levels,return_counts=True)
        tot_level[level_batch]+=num
        suc_batch,num_suc = torch.unique(levels[rewards_inds>0],return_counts=True)
        suc_level[suc_batch] += num_suc
    greedy_test_accuracy = ((rewards+1)/2).mean().cpu()
    print(torch.nan_to_num(suc_level/tot_level).cpu())
    wandb.run.summary.update({'greedy_test_perlevel':(suc_level/(tot_level+eps)).cpu().numpy(),
               'tot_test_level':tot_level.cpu().numpy(),
               'greedy_test_accuracy':greedy_test_accuracy})
    rewards = torch.zeros(test_states.shape[0], dtype=floatType, device=device)
    suc_level = torch.zeros(test_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    tot_level = torch.zeros(test_states.shape[1]-env.boundary.shape[0], dtype=floatType, device=device)
    print("Testing the stochastic policy on the test dataset")
    for n in range(wandb.config.num_test_rollouts):
        for env_inds in dl:
            traj,traj_act,rewards_inds = test_stochastic_policy(env,ppo_agent,test_states[env_inds].to(floatType),ui=False)
            rewards[env_inds]+=rewards_inds>0
            levels = (test_states[env_inds].sum(-1)>0).sum(1)-env.boundary.shape[0]-1
            level_batch, num = torch.unique(levels,return_counts=True)
            tot_level[level_batch]+=num
            suc_batch,num_suc = torch.unique(levels[rewards_inds>0],return_counts=True)
            suc_level[suc_batch] += num_suc

    print(torch.nan_to_num(suc_level/tot_level).cpu())
    wandb.run.summary.update({'stochastic_test_perlevel':(suc_level/(tot_level+eps)).cpu().numpy(),
               'stochastic_test_accuracy':(rewards.mean()/wandb.config.num_test_rollouts).cpu().numpy()})
    print("Saving results")
    name = f"{wandb.config.model_name}_{int(greedy_test_accuracy*100)}"
    folder_path = "./models/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder
    model_path = f"{folder_path}/{name}.polt"
    ppo_agent.save(model_path,
                   f"Tested policy; Greedy accuracy on the test dataset: {greedy_test_accuracy}")
    if wandb.config.online:
        policy_artifact = wandb.Artifact(f"{wandb.config.model_name}_{wandb.config.architecture}_polt",
                                             type='tested_model')
        policy_artifact.add_file(local_path=model_path, name=name)
        wandb.log_artifact(policy_artifact)
def draw_Markov_chain(file,width=10,depth=1000, target_width = 20):
    config_init(model_name="mario",
                method="PPO_GAT",
                online = False,
                name='test_init',
                pretrained_file=file)
    wandb.config.update({"admm_batch":np.inf},allow_val_change=True)
    env = init_env(static_random_seed=True)
    ppo_agent = init_ppo(env,file)
    init_state = torch.zeros((1,env.action_dim//2,3),device=device,dtype=floatType)
    init_state[:,:,0]=1
    init_state[:,env.boundary,1:]=1
    fig,ax= plt.subplots(1)
    #ax.set_ylim([0,init_state.shape[1]])
    states = init_state[0].unsqueeze(0)
    mask = env.action_mask(init_state)
    action_stoch,probs,values = ppo_agent.compute_policy(init_state, mask)
    values_all = values
    policies = probs
    Q = [values[0]*torch.ones_like(probs)]
    N = [1]
    parents = [[]]
    n_actions_left = np.array([mask.sum().cpu().numpy()])
    level = torch.tensor([0])
    level_idx = torch.tensor([0])
    best_state = torch.tensor([0])
    terminated = False
    passage = 0
    for i in range(depth):
        #select nodes to expand
        if not terminated:
            states_idx = np.repeat(best_state[-1].numpy(),width)
            level_idx_s = torch.zeros(width,dtype=intType)
        else:
            unfinished, = np.nonzero(n_actions_left>0)
            #tries to reach target width
            states_idx = np.random.choice(unfinished,width,replace=True)
            levels, levels_idx, n_level = torch.unique(level,return_counts=True,return_inverse=True)
            level_idx_s = n_level[level[states_idx]]
            
            #level_idx_s = n_level.sort()[0][2]
            #unfinished = unfinished[levels_idx==current_level]
            
        #chose an action in each node
        action_idx = []
        state_j = []
        keep_state = np.ones(width,dtype=bool)
        for j in range(width):
            if n_actions_left[states_idx[j]] >0:
                action_i = torch.argmax(policies[states_idx[j]])
                policies[states_idx[j]][action_i]=0
                n_actions_left[states_idx[j]]-=1
                action_idx.append(action_i)
                state_j.append(j)
            else:
                keep_state[j]=False
        level_idx_s = level_idx_s[state_j]
        current_idx = states_idx[state_j]
        current_states = states[current_idx]
        pot_new_states = env.next_states(current_states,torch.stack(action_idx))
        #pot_new_states = pot_new_states#[done]
        #current_idx = current_idx#[done.cpu()]

        isnew = torch.ones(pot_new_states.shape[0],dtype=bool)

        ids = torch.nonzero(torch.all(pot_new_states[None]==states[:,None,...],axis=(2,3)))
        
        idsnew =  torch.nonzero(torch.all(pot_new_states[None]==pot_new_states[:,None,...],axis=(2,3)))
        idsnew = idsnew[(idsnew[:,0]<idsnew[:,1])]
        idsnew = idsnew[~torch.isin(idsnew[:,0],ids[:,1])]
        idsnew[:,0]+=states.shape[0]
        ids = torch.cat([ids,idsnew])
        isnew[ids[:,1]]=False
        
        actual_new_states = pot_new_states[isnew]
        _,_,stable = env.problem_gurobi.simulate(env.from_bin(actual_new_states).cpu())
        stable &= ~env.check_terminate(actual_new_states).cpu().numpy()
        states = torch.cat([states,actual_new_states])        
        
        current_level = level[current_idx[isnew.numpy()]]+1
        level = torch.cat([level,current_level])
        
        offset = torch.zeros(isnew.sum(),dtype= intType)
        new_level_offset = torch.nonzero(current_level[None]==current_level[:,None])
        new_level_offset = new_level_offset[new_level_offset[:,0]<new_level_offset[:,1]]
        idx, count = torch.unique(new_level_offset[:,1],return_counts=True)
        offset[idx]+=count
        new_level_idx = level_idx_s[isnew]+offset

        level_idx = torch.cat([level_idx,new_level_idx])
        probs = torch.zeros((stable.shape[0],env.action_dim),device=device)
        values = -torch.ones((stable.shape[0],1),device=device)
        masks = env.action_mask(actual_new_states)
        if actual_new_states[stable].shape[0]>0 and masks[stable].any():
            action_stoch,probs[stable],values[stable] = ppo_agent.compute_policy(actual_new_states[stable], masks[stable])
        
        masks[~stable]=False
        if env.check_terminate(actual_new_states).any():
            terminated = True
            values[env.check_terminate(actual_new_states)]=1
        policies = torch.cat([policies,probs])
        values_all = torch.cat([values_all,values])
        n_actions_left = np.concatenate([n_actions_left,masks.sum(1).cpu().numpy()])
        firstnew_idx = len(parents)
        best_state = torch.cat([best_state,torch.tensor([firstnew_idx])])
        [parents.append([j]) for j in current_idx[isnew.numpy()]]

        for old_state,new_state in ids:
            parents[old_state].append(current_idx[new_state])
        #draw all
        """ v = values_all.cpu()
            for c,p in enumerate(parents):
                x = np.stack([p,[c]*len(p)])
                y= np.stack([[level[c]-1]*len(p),[level[c]]*len(p)])
                ax.plot(x,y,color=plt.colormaps['turbo']((v[c]+1)/2))
                plt.show(block=False)
                plt.pause(0.1)"""
        #draw new 
        v = values.cpu()
        for c in range(firstnew_idx,firstnew_idx+isnew.sum()):
            p = parents[c]
            x = np.stack([level_idx[p],np.repeat(level_idx[c],len(p))])
            """x = np.stack([p,[c]*len(p)])
            x[0,:] -= best_state[level[c]-1].numpy()
            x[1,:] -= best_state[level[c]].numpy()
            """
            y= np.stack([[level[c]-1]*len(p),[level[c]]*len(p)])
            ax.plot(x,y,color=plt.colormaps['turbo']((v[c-firstnew_idx]+1)/2))
            plt.show(block=False)
            plt.pause(0.1)
def draw_Markov_chain_npass(file,n_passages=1000):
    config_init(model_name="mario",
                method="PPO_GAT",
                online = False,
                name='test_init',
                pretrained_file=file)
    wandb.config.update({"admm_batch":np.inf},allow_val_change=True)
    env = init_env(static_random_seed=True)
    ppo_agent = init_ppo(env,file)
    init_state = torch.zeros((1,env.action_dim//2,3),device=device,dtype=floatType)
    init_state[:,:,0]=1
    init_state[:,env.boundary,1:]=1
    fig,ax= plt.subplots(1)
    #ax.set_ylim([0,init_state.shape[1]])
    states = init_state[0].unsqueeze(0)
    mask = env.action_mask(init_state)
    action_stoch,probs,values = ppo_agent.compute_policy(init_state, mask)
    values_all = values
    policies = probs
    Q = [values[0]*torch.ones_like(probs)]
    N = [1]
    parents = [[]]
    n_actions_left = np.array([mask.sum().cpu().numpy()])
    level = torch.tensor([0])
    level_idx = torch.tensor([0])
    n_level = torch.zeros(91)
    n_level[0]=1
    ax.grid(True)
    for i in range(n_passages):
        terminated = False
        last_state_idx = (n_actions_left>0).argmax()
        current_level = level[last_state_idx]
        while not terminated:
            current_idx = last_state_idx
            if n_actions_left[current_idx] >0:
                action_i = torch.argmax(policies[current_idx])
                policies[current_idx][action_i]=0
                n_actions_left[current_idx]-=1
                action_idx = action_i.unsqueeze(0)
            else:
                break
            level_idx_s = n_level[current_level+1].unsqueeze(0)
            current_states = states[current_idx].unsqueeze(0)
            pot_new_state = env.next_states(current_states,action_idx)

            
            ids = torch.nonzero(torch.all(pot_new_state[None]==states[:,None,...],axis=(2,3)))
            
            if ids.shape[0]==0:
                _,_,stable = env.problem_gurobi.simulate(env.from_bin(pot_new_state).cpu())
                stable &= ~env.check_terminate(pot_new_state).cpu().numpy()

                states = torch.cat([states,pot_new_state])        
                level = torch.cat([level,torch.tensor([current_level+1])])
                level_idx = torch.cat([level_idx,level_idx_s])

                probs = torch.zeros((stable.shape[0],env.action_dim),device=device)
                values = -torch.ones((stable.shape[0],1),device=device)
                masks = env.action_mask(pot_new_state)
                stable = stable & masks.any().cpu().numpy()
                if stable:
                    action_stoch,probs,values = ppo_agent.compute_policy(pot_new_state, masks)
                else:
                    masks[:]=False
                if env.check_terminate(pot_new_state).any():
                    terminated = True
                    masks[:]=False
                    values[env.check_terminate(pot_new_state)]=1
                
                policies = torch.cat([policies,probs])
                values_all = torch.cat([values_all,values])
                n_actions_left = np.concatenate([n_actions_left,masks.sum(1).cpu().numpy()])
                parents.append([current_idx])
                last_state_idx = len(parents)-1
                converged = False
            else:
                parents[ids[0,0]].append(current_idx)
                last_state_idx = int(ids[0,0].cpu().numpy())
                converged = True
            #draw new 
            v = values_all[last_state_idx].cpu()
            p = parents[last_state_idx]
            x = np.stack([level_idx[p],np.repeat(level_idx[last_state_idx],len(p))])
            """x = np.stack([p,[c]*len(p)])
            x[0,:] -= best_state[level[c]-1].numpy()
            x[1,:] -= best_state[level[c]].numpy()
            """
            y= np.stack([[current_level]*len(p),[current_level+1]*len(p)])
            ax.plot(x,y,color=plt.colormaps['turbo']((v+1)/2))
            plt.show(block=False)
            plt.pause(0.1)

            current_level +=1
            n_level[current_level]+=1
            if converged:
                break
def forward_compat(file):
    config_init(model_name="mario",
                method="PPO_GAT",
                online = False,
                boundary=[0, 12, 59, 14, 2, 41, 5, 6, 7, 10, 17, 1, 13, 58, 9, 15, 38],
                name='test_init',
                config_file=DATA_DIR+"/parameters/default.json")
    # init env
    #wandb.config.update({"n_robot":3},allow_val_change=True)
    env = init_env(static_random_seed=True)
    ppo_agent = init_ppo(env)
    ppo_agent.load(file)
    ppo_agent.saved_accuracy=95
    ppo_agent.save(file+".pol")

def test_new_starting_states(ppo_agent, env, train_dict,batch_size=1):
    num_success = 0
    num_attempts = 0

    env.problem_level = -1
    env.curriculum_rollouts = [wandb.config.num_test_rollouts]
    #level_up(ppo_agent, env)

    ppo_agent.buffer = RolloutBufferGNN(ppo_agent.gamma, False,len(env.curriculum),
                                        wandb.config.entropy_weight,
                                        wandb.config.entropy_slope,
                                        wandb.config.max_entropy_weight,
                                        num_step_anneal=wandb.config.num_step_anneal)
    ppo_agent.test_epoch = 0

    for id in range(0,env.curriculum.bin_states.shape[0]-batch_size,batch_size):
        curriculum_inds = []
        for idb in range(id,batch_size+id):
            bin_state = env.curriculum.bin_states[idb, :, :]
            if encode(bin_state) not in train_dict:
                curriculum_inds.append(idb)
            else:
                pass
        curriculum_inds = torch.tensor(curriculum_inds,dtype=intType,device=device)
        curriculum_inds = curriculum_inds.repeat(wandb.config.num_test_rollouts)
        if curriculum_inds.shape[0]>0:
            rewards = rollout_asyn(ppo_agent, env, curriculum_inds, ui=False)
            num_attempts += curriculum_inds.shape[0]
            num_success += torch.sum(rewards > 0)
            accuracy = (torch.sum(rewards > 0) / curriculum_inds.shape[0]).item()
            wandb.log({"epoch": ppo_agent.test_epoch, "accuracy": accuracy, "envs": num_attempts}, step=ppo_agent.time_step)
            print(f"epoch {ppo_agent.test_epoch}, accuracy {round(accuracy, 2)}")
            ppo_agent.test_epoch = ppo_agent.test_epoch + curriculum_inds.shape[0]/wandb.config.num_test_rollouts
        
    #takes care of the last batch
    curriculum_inds = []
    for idb in range(id+batch_size,env.curriculum.bin_states.shape[0]):
        bin_state = env.curriculum.bin_states[idb, :, :]
        if encode(bin_state) not in train_dict:
            curriculum_inds.append(idb)
    curriculum_inds = torch.tensor(curriculum_inds,dtype=intType,device=device)
    curriculum_inds = curriculum_inds.repeat(wandb.config.num_test_rollouts)
    if curriculum_inds.shape[0]>0:
        rewards = rollout_asyn(ppo_agent, env, curriculum_inds, ui=False)
        num_attempts += curriculum_inds.shape[0]
        num_success += torch.sum(rewards > 0)
        accuracy = (torch.sum(rewards > 0) / curriculum_inds.shape[0]).item()
        wandb.log({"epoch": ppo_agent.test_epoch, "accuracy": accuracy, "envs": num_attempts}, step=ppo_agent.time_step)
        print(f"epoch {ppo_agent.test_epoch}, accuracy {round(accuracy, 2)}")
        ppo_agent.test_epoch = ppo_agent.test_epoch + curriculum_inds.shape[0]

    print("total", num_success / num_attempts)
    print(f"epoch {ppo_agent.test_epoch}, envs {num_attempts}")
if __name__ == '__main__':
    #forward_compat(f"{DATA_DIR}/pretrained/mario/mario_test2")
    #test_trained_policy(f"{DATA_DIR}/pretrained/mario/mario_3_robots_95acc.pol")
    """with open('./script/test/Setup2_keyframe.pkl', 'rb') as handle:
        traj = pickle.load(handle)
    check_trajectory(f"{DATA_DIR}/pretrained/mario/mario_3_robots_95acc.pol",traj)"""
    draw_Markov_chain_npass(f"{DATA_DIR}/pretrained/mario/mario_3_robots_95acc.pol")