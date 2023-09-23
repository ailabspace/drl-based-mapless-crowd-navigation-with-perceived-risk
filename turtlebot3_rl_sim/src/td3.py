#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TD3 is based on https://github.com/higgsfield/RL-Adventure-2 #

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# USE CUDA GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# Experience Replay memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Ournstein-Uhlenbeck noise
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.2, decay_period=100000):
        self.mu = mu * np.ones(action_space)
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self, step=0):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        # Decay
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step / self.decay_period)
        return self.state


# Gaussian noise
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/gaussian_strategy.py
class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):  # 1000000
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space

    def sample(self, action, step=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step * 1.0 / self.decay_period)
        noise = np.random.normal(loc=0, size=len(action[0]), scale=sigma)  # * sigma

        return noise


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, max_lin_vel, max_ang_vel, init_w=3e-4):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # Was commented out before we had object tracking implemented
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # action = F.tanh(self.linear3(x))
        action = self.linear3(x)

        # Give two action, linear vel: {0,1}, angular vel: {-1,1}
        action[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel

        return action


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Was commented out before we had object tracking implemented
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value


class Agent:
    """Main DDPG agent that extracts experiences and learns from them"""

    def __init__(self, state_size, action_size, hidden_size, actor_learning_rate, critic_learning_rate, batch_size,
                 buffer_size, discount_factor, softupdate_coefficient, max_lin_vel, max_ang_vel, noise_std, noise_clip,
                 policy_update):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.loss_function = nn.MSELoss()
        self.tau = softupdate_coefficient
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_update = policy_update

        # Actor network
        self.actor_local = Actor(self.state_size, self.action_size, self.hidden_size, self.max_lin_vel,
                                 self.max_ang_vel).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.hidden_size, self.max_lin_vel,
                                  self.max_ang_vel).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic network
        self.critic_local1 = Critic(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_target1 = Critic(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=self.critic_learning_rate)

        self.critic_local2 = Critic(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_target2 = Critic(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=self.critic_learning_rate)

        # Noise process
        self.noise = GaussianExploration(self.action_size)
        # self.noise = OUNoise(self.action_size)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size)

        # Update target network with hard updates
        self.hard_update(self.critic_target1, self.critic_local1)
        self.hard_update(self.critic_target2, self.critic_local2)
        self.hard_update(self.actor_target, self.actor_local)

    def step(self, state, action, reward, next_state, done):
        """
        Saves an experience in the replay memory to learn from using random sampling.
        @Param:
        1. state: current state, S.
        2. action: action taken based on current state.
        3. reward: immediate reward from state, action.
        4. next_state: next state, S', from action, a.
        5. done: (bool) has the episode terminated?
        Exracted version for trajectory used in calculating the value for an action, a."""

        self.memory.add(state, action, reward, next_state, done)

    # def reset(self):
    #     """Resets the noise process to mean"""
    #     self.noise.reset()

    def act(self, state, decay_step, add_noise=True):
        """
        Returns a deterministic action given current state.
        @Param:
        1. state: current state, S.
        2. add_noise: (bool) add bias to agent, default = True (training mode)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # typecast to torch.Tensor
        self.actor_local.eval()  # set in evaluation mode
        with torch.no_grad():  # reset gradients
            action = self.actor_local(state).cpu().data.numpy()  # deterministic action based on Actor's forward pass.
        self.actor_local.train()  # set training mode

        if add_noise:
            noise = self.noise.sample(action, decay_step)
            action += noise

        # Set upper and lower bound of action spaces
        action[0, 0] = np.clip(action[0, 0], 0.0, self.max_lin_vel)  # 0.22
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)

        # Real robot
        # action[0, 0] = np.clip(action[0, 0], 0.11, 0.22)
        # action[0, 1] = np.clip(action[0, 1], -1.0, 1.0)

        # print(action)

        return action

    def learn(self, step):
        """
        Learn from a set of experiences picked up from a random sampling of even frequency (not prioritized)
        of experiences when buffer_size = MINI_BATCH.
        Updates policy and value parameters accordingly
        @Param:
        1. experiences: (Tuple[torch.Tensor]) set of experiences, trajectory, tau. tuple of (s, a, r, s', done)
        2. gamma: immediate reward hyper-parameter, 0.99 by default.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(states).to(device)
        next_state = torch.FloatTensor(next_states).to(device)
        action = torch.FloatTensor(actions).to(device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)

        next_action = self.actor_target(next_state)
        # noise = torch.normal(torch.zeros(next_action.size()), self.noise_std).to(device)
        noise = torch.randn_like(torch.zeros(next_action.size())).to(device) * self.noise_std
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action += noise

        target_Q1 = self.critic_target1(next_state, next_action)
        target_Q2 = self.critic_target2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        expected_Q = reward + (1.0 - done) * self.discount_factor * target_Q

        # Remedy: Change action shape from (X, 1, 2) to (1, 2) for concat
        # Cause of action changing from (1, 2) after converting to FloatTensor
        # becoming (X, 1, 2) unknown
        action = torch.squeeze(action, 1)

        Q1 = self.critic_local1(state, action)
        Q2 = self.critic_local2(state, action)

        critic_loss1 = self.loss_function(Q1, expected_Q.detach())
        critic_loss2 = self.loss_function(Q2, expected_Q.detach())

        # Update Critic network
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if step % self.policy_update == 0:
            actor_loss = self.critic_local1(state, self.actor_local(state))
            actor_loss = -actor_loss.mean()

            # Update Actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network with soft updates
            self.soft_update(self.critic_local1, self.critic_target1)
            self.soft_update(self.critic_local2, self.critic_target2)
            self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters. Copies model τ every experience.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + local_param.data * self.tau
            )

    def hard_update(self, target_model, local_param):
        for target_param, param in zip(target_model.parameters(), local_param.parameters()):
            target_param.data.copy_(param.data)

    def save_actor_model(self, outdir, name):
        torch.save(self.actor_target.state_dict(), outdir + '/' + str(name))

    def save_critic1_model(self, outdir, name):
        torch.save(self.critic_target1.state_dict(), outdir + '/' + str(name))

    def save_critic2_model(self, outdir, name):
        torch.save(self.critic_target2.state_dict(), outdir + '/' + str(name))

    def load_models(self, actor_outdir, critic1_outdir, critic2_outdir):
        self.actor_local.load_state_dict(torch.load(actor_outdir))
        self.critic_local1.load_state_dict(torch.load(critic1_outdir))
        self.critic_local2.load_state_dict(torch.load(critic2_outdir))
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target1, self.critic_local1)
        self.hard_update(self.critic_target2, self.critic_local2)
