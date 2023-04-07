#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SAC is based on https://github.com/higgsfield/RL-Adventure-2 #

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import time

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


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, max_lin_vel, max_ang_vel, init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, num_actions)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        action = self.get_action(mean, log_std)

        return action, mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        action, mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        # Give two action, linear vel: {0,1}, angular vel: {-1,1}
        action[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel

        return action, log_prob, z, mean, log_std

    def get_action(self, mean, log_std):
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        # Give two action, linear vel: {0,1}, angular vel: {-1,1}
        action[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel

        return action


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Agent:
    """Main DDPG agent that extracts experiences and learns from them"""

    def __init__(self, state_size, action_size, hidden_size, actor_learning_rate, critic_v_learning_rate,
                 critic_soft_q_learning_rate, batch_size, discount_factor, buffer_size, softupdate_coefficient,
                 max_lin_vel, max_ang_vel, mean_lambda, std_lambda, z_lambda):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_v_learning_rate = critic_v_learning_rate
        self.critic_q_learning_rate = critic_soft_q_learning_rate
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.tau = softupdate_coefficient
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.mean_lambda = mean_lambda
        self.std_lambda = std_lambda
        self.z_lambda = z_lambda

        # Actor network
        self.actor = Actor(self.state_size, self.action_size, self.hidden_size, self.max_lin_vel,
                           self.max_ang_vel).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        # Critic network
        # Value
        self.critic_v_net = ValueNetwork(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_target_v_net = ValueNetwork(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_v_optimizer = optim.Adam(self.critic_v_net.parameters(), lr=self.critic_v_learning_rate)

        # Soft Q
        self.critic_soft_q_net = SoftQNetwork(self.state_size, self.action_size, self.hidden_size).to(device)
        self.critic_soft_q_net_optimizer = optim.Adam(self.critic_soft_q_net.parameters(),
                                                      lr=self.critic_q_learning_rate)

        self.value_loss_function = nn.MSELoss()
        self.soft_q_loss_function = nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size)

        # Update target network with hard updates
        self.hard_update(self.critic_target_v_net, self.critic_v_net)

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

    def act(self, state):
        """
        Returns a deterministic action given current state.
        @Param:
        1. state: current state, S.
        2. add_noise: (bool) add bias to agent, default = True (training mode)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # typecast to torch.Tensor
        self.actor.eval()  # set in evaluation mode
        with torch.no_grad():  # reset gradients
            action, mean, log_std = self.actor(state)  # deterministic action based on Actor's forward pass.
            action = action.cpu().data.numpy()
        self.actor.train()  # set training mode

        print("ACTION", str(action))

        # Set upper and lower bound of action spaces
        action[0, 0] = np.clip(action[0, 0], 0., self.max_lin_vel)
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)

        print("CLIPPED ACTION", str(action))
        # print(action)

        return action

    def learn(self):
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

        # Remedy: Change action shape from (X, 1, 2) to (1, 2) for concat
        # Cause of action changing from (1, 2) after converting to FloatTensor
        # becoming (X, 1, 2) unknown
        action = torch.squeeze(action, 1)

        expected_q_value = self.critic_soft_q_net(state, action)
        expected_value = self.critic_v_net(state)
        new_action, log_prob, z, mean, log_std = self.actor.evaluate(state)

        target_value = self.critic_target_v_net(next_state)
        next_q_value = reward + (1 - done) * self.discount_factor * target_value
        q_value_loss = self.soft_q_loss_function(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.critic_soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_loss_function(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss = self.std_lambda * log_std.pow(2).mean()
        z_loss = self.z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        # Update critic soft Q network
        self.critic_soft_q_net_optimizer.zero_grad()
        q_value_loss.backward()
        self.critic_soft_q_net_optimizer.step()

        # Update critic value network
        self.critic_v_optimizer.zero_grad()
        value_loss.backward()
        self.critic_v_optimizer.step()

        # Update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update target network with soft updates
        self.soft_update(self.critic_target_v_net, self.critic_v_net)

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
        torch.save(self.actor.state_dict(), outdir + '/' + str(name))

    def save_critic_soft_q_model(self, outdir, name):
        torch.save(self.critic_soft_q_net.state_dict(), outdir + '/' + str(name))

    def save_critic_v_model(self, outdir, name):
        torch.save(self.critic_target_v_net.state_dict(), outdir + '/' + str(name))

    def load_models(self, actor_outdir, critic_soft_q_outdir, critic_v_outdir):
        self.actor.load_state_dict(torch.load(actor_outdir))
        self.critic_soft_q_net.load_state_dict(torch.load(critic_soft_q_outdir))
        self.critic_v_net.load_state_dict(torch.load(critic_v_outdir))
        self.hard_update(self.actor, self.actor)
        self.hard_update(self.critic_soft_q_net, self.critic_soft_q_net)
        self.hard_update(self.critic_target_v_net, self.critic_v_net)
