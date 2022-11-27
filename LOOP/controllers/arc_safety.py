
# Safe controllers which do a black box optimization incorporating the constraint costs.

import copy
import numpy as np
import scipy.stats as stats
import torch
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")



class safeARC(object):
    def __init__(self, env, models, critic, termination_function):

        ###########
        # params
        ###########
        self.horizon = 5 # Hyperparam search [5,8]
        self.reward_horizon = 8 
        self.N = 50 # Hyperparam search [100,400]
        self.models = models
        self.env = copy.deepcopy(env)
        self.critic = critic
        self.mixture_coefficient = 0.05
        self.max_iters = 5
        self.actor_traj = int(self.mixture_coefficient*self.N)
        self.num_elites = 5 
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.sol_dim = self.env.action_space.shape[0] * self.horizon
        self.ub = np.repeat(self.env.action_space.high,self.horizon,axis=0)
        self.lb = np.repeat(self.env.action_space.low,self.horizon,axis=0)
        self.alpha = 0.1
        self.mean = np.zeros((self.sol_dim,))
        self.termination_function=termination_function
        self.particles = 4
        self.safety_threshold = 0.2
        self.minimal_elites = 10
        self.kappa = 1

    def reset(self):
        self.mean = np.zeros((self.sol_dim,))

    def get_action(self, curr_state,env=None):
        
        actor_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * (self.actor_traj))# Size [actor_traj,state_dim]
        curr_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * ((self.N+self.actor_traj)*self.particles))
        curr_state = np.expand_dims(curr_state, axis=0)
        curr_state = np.repeat(
            curr_state,
            self.models.model.network_size,
            0)  # [numEnsemble, N+actor_traj,state_dim]

        # initial mean and var of the sampling normal dist
        self.mean[:-self.action_dim] = self.mean[self.action_dim:]
        self.mean[-self.action_dim:] = self.mean[-2*self.action_dim:-self.action_dim]
        mean=self.mean
        var = np.tile(np.square(self.env.action_space.high[0]-self.env.action_space.low[0]) / 16, [self.sol_dim]) #/16
        

        # Add trajectories using actions suggested by actors
        actor_trajectories = np.zeros((self.actor_traj,self.sol_dim))
        actor_state = torch.FloatTensor(actor_state).to(device)
        actor_state_m = actor_state[0,:].reshape(1,-1)
        actor_state_m2 = actor_state[1,:].reshape(1,-1)
        for h in range(self.horizon):
            actor_actions_m = self.critic.ac.act_batch(actor_state_m.reshape(1,-1)[:,1:],deterministic=True)
            actor_state_m = self.models.get_forward_prediction_random_ensemble_t(actor_state_m[:,1:],actor_actions_m)
            actor_trajectories[0,h*self.action_dim:(h+1)*self.action_dim]=actor_actions_m.detach().cpu().numpy()

            actor_actions = self.critic.ac.act_batch(actor_state_m2[:,1:])
            actor_state_m2 = self.models.get_forward_prediction_random_ensemble_t(actor_state_m2[:,1:],actor_actions)
            actor_trajectories[1:,h*self.action_dim:(h+1)*self.action_dim]=actor_actions.detach().cpu().numpy()

        

        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))

        t = 0
        while (t < self.max_iters):
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            action_traj = (X.rvs(size=(self.N, self.sol_dim)) * np.sqrt(constrained_var) + mean).astype(np.float32)
            action_traj = np.concatenate((action_traj,actor_trajectories),axis=0)
            # Multiple particles go through the same action sequence
            action_traj = np.repeat(action_traj,self.particles,axis=0)

            # actions clipped between -1 and 1
            action_traj = np.clip(action_traj, -1, 1)

            states = torch.from_numpy(np.expand_dims(curr_state.copy(),axis=0)).float().to(device)
            actions = np.repeat(np.expand_dims(action_traj,axis=0),self.models.model.network_size,axis=0)
            actions = torch.FloatTensor(actions).to(device)
            

            for h in range(self.horizon):
                states_h = states[h,:,:,1:]
                next_states = self.models.get_forward_prediction_t(states_h, actions[:,:, h * self.action_dim:(h + 1) * self.action_dim])
                states = torch.cat((states,next_states.unsqueeze(0)), axis=0)
                # print(f"States shape: {states.shape}")
            
            
            states = states.cpu().detach().numpy()
            
            done = np.zeros((states.shape[1],states.shape[2],1)) # Shape [Ensembles, (actor_traj+N)*particles,1]
            # Set the reward of terminated states to zero
            for h in range(1,self.horizon+1):
                for ens in range(states.shape[1]):
                    done[ens,:,:] = np.logical_or(done[ens,:,:],self.termination_function(None,None,states[h,ens,:,1:]))
                    not_done = 1-done[ens,:,:]
                    states[h,ens,:,0]*=not_done.astype(np.float32).reshape(-1)
            

            # Find average cost of each trajectory
            returns = np.zeros((self.N+self.actor_traj,))
            safety_costs = np.zeros((self.N+self.actor_traj,))
            
            
            actions_H = torch.from_numpy(action_traj[:, (self.reward_horizon - 1) * self.action_dim:(
                self.reward_horizon) * self.action_dim].reshape((self.N+self.actor_traj)*self.particles, -1)).float().to(device)
            actions_H = actions_H.repeat(self.models.model.network_size, 1)
            # actions_H = actions_H.repeat_interleave(repeats=states.shape[1],dim=0)
            states_H = torch.from_numpy(
                    states[self.reward_horizon-1, :, :, 1:].reshape((self.N+self.actor_traj)*self.particles*states.shape[1], -1)).float().to(device)
            
            
            terminal_Q_rewards = self.critic.ac.q1(
                    states_H, actions_H).cpu().detach().numpy() 
            terminal_Q_rewards = terminal_Q_rewards.reshape(states.shape[1],-1)
            
            states_flatten = states[:,:,:,1:].reshape(-1,self.obs_dim)
            # print(states_flatten.shape)
            all_safety_costs = np.zeros((states_flatten.shape[0],))
            fc1 = (states_flatten[:,10]**2 + states_flatten[:,11]**2)**0.5 - 0.25
            fc2 = (states_flatten[:,12]**2 + states_flatten[:,13]**2)**0.5 - 0.25
            fc3 = (states_flatten[:,14]**2 + states_flatten[:,15]**2)**0.5 - 0.25
            fc4 = (states_flatten[:,16]**2 + states_flatten[:,17]**2)**0.5 - 0.25
            fc5 = (states_flatten[:,18]**2 + states_flatten[:,19]**2)**0.5 - 0.25
            fc6 = (states_flatten[:,20]**2 + states_flatten[:,21]**2)**0.5 - 0.25
            fc7 = (states_flatten[:,22]**2 + states_flatten[:,23]**2)**0.5 - 0.25
            fc8 = (states_flatten[:,24]**2 + states_flatten[:,25]**2)**0.5 - 0.25
            fc9 = (states_flatten[:,8]**2 + states_flatten[:,9]**2)**0.5 - 0.3
            fc10 = states_flatten[:,6]**2 + states_flatten[:,7]**2
            fc = np.min(np.array([fc1, fc2, fc3, fc4,fc5,fc6,fc7,fc8,fc9]).T, 1)
            
            
            # all_safety_costs = env.get_observation_cost(states_flatten)
            all_safety_costs = 1/(fc + 1e-5) + fc10
            # print(all_safety_costs.shape)
            all_safety_costs = all_safety_costs.reshape(states.shape[0],states.shape[1],states.shape[2],1)
            # print(all_safety_costs.shape)
            # print(vdfsv)  
            
            
            
            

            for ensemble in self.models.model.elite_model_idxes:
                done[ensemble,:,:] = np.logical_or(done[ensemble,:,:],self.termination_function(None,None,states[self.horizon-1,ensemble,:,1:]))
                not_done = 1-done[ensemble,:,:]
                q_rews = terminal_Q_rewards[ensemble,:]*not_done.reshape(-1)
                n = np.arange(0,self.N+self.actor_traj,1).astype(int)
                # print(n)
                for particle in range(self.particles):
                    returns[n]+= np.sum(states[:self.reward_horizon, ensemble, n*self.particles+particle, 0],axis=0) + q_rews.reshape(-1)[n*self.particles+particle]
                    safety_costs[n] = np.maximum(safety_costs,np.sum(all_safety_costs[0:self.horizon, ensemble, n*self.particles+particle, 0],axis=0))

            # print(self.particles)
            # print(Dvv)
            returns /= (states.shape[1]*self.particles)
            costs = -returns
            
            
            if((safety_costs<self.safety_threshold).sum()<self.minimal_elites):
                safety_rewards = -safety_costs
                max_safety_reward = np.max(safety_rewards)
                score = np.exp(self.kappa*(safety_rewards-max_safety_reward))
                # print(f"Score is {score.shape}")
                indices = np.argsort(safety_costs)
                mean = np.sum(action_traj[np.arange(0,self.N+self.actor_traj,1).astype(int)*self.particles,:]*score.reshape(-1,1),axis=0)/(np.sum(score)+1e-10)
                new_var = np.average((action_traj[np.arange(0,self.N+self.actor_traj,1).astype(int)*self.particles,:]-mean)**2, weights=score.reshape(-1),axis=0)
            else:                 
                costs = (safety_costs<self.safety_threshold)*costs + (safety_costs>=self.safety_threshold)*1e4
                indices = np.arange(costs.shape[0])
                indices = np.array([idx for idx in indices if costs[idx]<1e3])
                safe_action_traj =  action_traj[np.arange(0,self.N+self.actor_traj,1).astype(int)*self.particles,:][indices,:]
                rewards = -costs[indices]
                max_reward = np.max(rewards)
                score = np.exp(self.kappa*(rewards-max_reward))
                # print(f"Score is {score.shape}")
                mean = np.sum(safe_action_traj*score.reshape(-1,1),axis=0)/(np.sum(score)+1e-10)
                new_var = np.average((safe_action_traj-mean)**2, weights=score.reshape(-1),axis=0)


            var =  (self.alpha) * var + (1 - self.alpha) * new_var
            t += 1
            if((t+1)%6==0):
                var = np.tile(np.square(self.env.action_space.high[0]-self.env.action_space.low[0]) / 16.0, [self.sol_dim])*(1.5**((t+1)//6))
            if (((safety_costs<self.safety_threshold).sum()>=self.minimal_elites) and t>5) or t>25:
                break


        self.mean = mean
        # print(mean.shape, mean[:self.action_dim])

        
        return mean[:self.action_dim]





class safeCEM(object):
    def __init__(self, env, models, critic, termination_function):
        ###########
        # params
        ###########
        self.horizon = 5 # Hyperparam search [5,8]
        self.reward_horizon = 8 
        self.N = 100 # Hyperparam search [100,400]
        self.models = models
        self.env = copy.deepcopy(env)
        self.critic = critic
        self.mixture_coefficient = 0.05
        self.max_iters = 5
        self.actor_traj = int(self.mixture_coefficient*self.N)
        self.num_elites = 20 
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.sol_dim = self.env.action_space.shape[0] * self.horizon
        self.ub = np.repeat(self.env.action_space.high,self.horizon,axis=0)
        self.lb = np.repeat(self.env.action_space.low,self.horizon,axis=0)
        self.alpha = 0.1
        self.mean = np.zeros((self.sol_dim,))
        self.termination_function=termination_function
        self.particles = 4
        self.safety_threshold = 0.0
        self.minimal_elites = 5

    def reset(self):
        self.mean = np.zeros((self.sol_dim,))

    def get_action(self, curr_state,env=None):
        print("Hello")
        actor_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * (self.actor_traj))# Size [actor_traj,state_dim]
        curr_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * ((self.N+self.actor_traj)*self.particles))
        curr_state = np.expand_dims(curr_state, axis=0)
        curr_state = np.repeat(
            curr_state,
            self.models.model.network_size,
            0)  # [numEnsemble, N+actor_traj,state_dim]

        # initial mean and var of the sampling normal dist
        self.mean[:-self.action_dim] = self.mean[self.action_dim:]
        self.mean[-self.action_dim:] = self.mean[-2*self.action_dim:-self.action_dim]
        mean=self.mean

        var = np.tile(np.square(self.env.action_space.high[0]-self.env.action_space.low[0]) / 16, [self.sol_dim]) #/16


        # Add trajectories using actions suggested by actors
        actor_trajectories = np.zeros((self.actor_traj,self.sol_dim))
        actor_state = torch.FloatTensor(actor_state).to(device)
        actor_state_m = actor_state[0,:].reshape(1,-1)
        actor_state_m2 = actor_state[1,:].reshape(1,-1)

        for h in range(self.horizon):
            actor_actions_m = self.critic.ac.act_batch(actor_state_m.reshape(1,-1)[:,1:],deterministic=True)
            actor_state_m = self.models.get_forward_prediction_random_ensemble_t(actor_state_m[:,1:],actor_actions_m)
            actor_trajectories[0,h*self.action_dim:(h+1)*self.action_dim]=actor_actions_m.detach().cpu().numpy()

            actor_actions = self.critic.ac.act_batch(actor_state_m2[:,1:])
            actor_state_m2 = self.models.get_forward_prediction_random_ensemble_t(actor_state_m2[:,1:],actor_actions)
            actor_trajectories[1:,h*self.action_dim:(h+1)*self.action_dim]=actor_actions.detach().cpu().numpy()


        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))
        t = 0
        # CEM
        while (1):
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            action_traj = (X.rvs(size=(self.N, self.sol_dim)) * np.sqrt(constrained_var) + mean).astype(np.float32)
            action_traj = np.concatenate((action_traj,actor_trajectories),axis=0)
            # Multiple particles go through the same action sequence
            action_traj = np.repeat(action_traj,self.particles,axis=0)

            # actions clipped between -1 and 1
            action_traj = np.clip(action_traj, -1, 1)

            states = torch.from_numpy(np.expand_dims(curr_state.copy(),axis=0)).float().to(device)
            actions = np.repeat(np.expand_dims(action_traj,axis=0),self.models.model.network_size,axis=0)
            actions = torch.FloatTensor(actions).to(device)
            

            for h in range(self.horizon):
                states_h = states[h,:,:,1:]
                next_states = self.models.get_forward_prediction_t(states_h, actions[:,:, h * self.action_dim:(h + 1) * self.action_dim])
                states = torch.cat((states,next_states.unsqueeze(0)), axis=0)
            states = states.cpu().detach().numpy()

            done = np.zeros((states.shape[1],states.shape[2],1)) # Shape [Ensembles, (actor_traj+N)*particles,1]
            # Set the reward of terminated states to zero
            for h in range(1,self.horizon+1):
                for ens in range(states.shape[1]):
                    done[ens,:,:] = np.logical_or(done[ens,:,:],self.termination_function(None,None,states[h,ens,:,1:]))
                    not_done = 1-done[ens,:,:]
                    states[h,ens,:,0]*=not_done.astype(np.float32).reshape(-1)
            
 
            # Find average cost of each trajectory
            returns = np.zeros((self.N+self.actor_traj,))
            safety_costs = np.zeros((self.N+self.actor_traj,))
            
            
            actions_H = torch.from_numpy(action_traj[:, (self.reward_horizon - 1) * self.action_dim:(
                self.reward_horizon) * self.action_dim].reshape((self.N+self.actor_traj)*self.particles, -1)).float().to(device)

            actions_H = actions_H.repeat_interleave(repeats=states.shape[1],dim=0)
            states_H = torch.from_numpy(
                    states[self.reward_horizon-1, :, :, 1:].reshape((self.N+self.actor_traj)*self.particles*states.shape[1], -1)).float().to(device)
            print(f"*************\n{h}, {states_H.shape}\n*************************")
            print(fsfvvdvdvvdvsd)
            terminal_Q_rewards = self.critic.ac.q1(
                    states_H, actions_H).cpu().detach().numpy() 
            terminal_Q_rewards = terminal_Q_rewards.reshape(states.shape[1],-1)
            
            states_flatten = states[:,:,:,1:].reshape(-1,self.obs_dim)
            all_safety_costs = np.zeros((states_flatten.shape[0],))
            all_safety_costs = env.get_observation_cost(states_flatten)
            all_safety_costs = all_safety_costs.reshape(states.shape[0],states.shape[1],states.shape[2],1)
            
            for ensemble in self.models.model.elite_model_idxes:
                done[ensemble,:,:] = np.logical_or(done[ensemble,:,:],self.termination_function(None,None,states[self.horizon-1,ensemble,:,1:]))
                not_done = 1-done[ensemble,:,:]
                q_rews = terminal_Q_rewards[ensemble,:]*not_done

                n = np.arange(0,self.N+self.actor_traj,1).astype(int)       
                for particle in range(self.particles):
                    returns[n]+= np.sum(states[:self.reward_horizon, ensemble, n*self.particles+particle, 0],axis=0) + q_rews.reshape(-1)[n*self.particles+particle]


            returns /= (states.shape[1]*self.particles)
            costs = -returns
            if((safety_costs<self.safety_threshold).sum()<self.minimal_elites):
                indices = np.argsort(safety_costs)
                indices*=self.particles
                elites = action_traj[indices][:self.num_elites]
            else:                 
                costs = (safety_costs<self.safety_threshold)*costs + (safety_costs>=self.safety_threshold)*1e4
                indices = np.argsort(costs)
                indices = np.array([idx for idx in indices if costs[idx]<1e3])
                indices*=self.particles
                elites = action_traj[indices][:min(self.num_elites,indices.shape[0])]

            mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            var =  (self.alpha) * var + (1 - self.alpha) * new_var
            # Diagonalize
            t += 1
            if((t+1)%6==0):
                var = np.tile(np.square(self.env.action_space.high[0]-self.env.action_space.low[0]) / 16.0, [self.sol_dim])*(1.5**((t+1)//6))
            if (((safety_costs<self.safety_threshold).sum()>=1) and t>5) or t>25:
                break

        self.mean = mean

        return mean[:self.action_dim]



