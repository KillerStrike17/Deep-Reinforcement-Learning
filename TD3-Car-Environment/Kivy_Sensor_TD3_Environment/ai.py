# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd


# Creating the architecture of the Neural Network

class Actor(nn.Module):
	def __init__(self, state_dims, action_dim, max_action):
		super(Actor, self).__init__()
		self.layer_1 = nn.Linear(state_dims, 400)
		self.layer_2 = nn.Linear(400,300)
		self.layer_3 = nn.Linear(300,action_dim)
		self.max_action = max_action

	def forward(self, x):
		x = F.relu(self.layer_1(x))
		x = F.relu(self.layer_2(x))
		x= self.max_action*torch.tanh(self.layer_3(x))
		return x

class Critic(nn.Module):
	def __init__(self, state_dims, action_dim):
		super(Critic, self).__init__()

		self.layer_1 = nn.Linear(state_dims + action_dim,400)
		self.layer_2 = nn.Linear(400,300)
		self.layer_3 = nn.Linear(300,action_dim)

		self.layer_4 = nn.Linear(state_dims + action_dim, 400)
		self.layer_5 = nn.Linear(400,300)
		self.layer_6 = nn.Linear(300,action_dim)

	def forward(self,x,u):
		# print(x.shape,type(x))
		# print(u.shape,type(u))
		xu  =torch.cat([x,u],1)
		x1 = F.relu(self.layer_1(xu))
		x1 = F.relu(self.layer_2(x1))
		x1 = self.layer_3(x1)

		x2 = F.relu(self.layer_4(xu))
		x2 = F.relu(self.layer_5(x2))
		x2 = self.layer_6(x2)

		return x1, x2


	def Q1(self, x, u):
		xu = torch.cat([x,u],1)
		x1 = F.relu(self.layer_1(xu))
		x1 = F.relu(self.layer_2(x1))
		x1 = self.layer_3(x1)
		return x1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Implementing Experience Replay

class ReplayBuffer(object):
	def __init__(self, max_size = 1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self,transition):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = transition
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(transition)

	def sample(self, batch_size):
		ind = np.random.randint(1,len(self.storage),batch_size)
		batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
		# print("Random list:",ind)
		# print("Storage Length:",len(self.storage))
		# print("Storage indexes: ",self.storage.index)
		# print(ind)
		for i in ind:
			state, next_state, action, reward, done = self.storage[i]
			batch_states.append(np.array(state,copy = False))
			batch_next_states.append(np.array(next_state,copy = False))
			batch_actions.append(np.array(action, copy = False))
			batch_rewards.append(np.array(reward,copy= False))
			batch_dones .append(np.array(done, copy= False))

		return np.array(batch_states),np.array(batch_next_states),np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

# Implementing Deep Q Learning

class T3D(object):
	def __init__(self, state_dims,action_dim,max_action):
		print(state_dims,action_dim)
		self.actor = Actor(state_dims,action_dim,max_action).to(device)
		self.actor_target = Actor(state_dims, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self. actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.reward_window = []
		self.critic = Critic(state_dims, action_dim).to(device)
		self.critic_target = Critic(state_dims,action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.max_action = max_action
		self.memory = ReplayBuffer()
		self.last_state = torch.Tensor(state_dims).unsqueeze(0)
		self.last_action = 0
		self.last_reward = 0

	def select_action(self, state):
		state = torch.Tensor(state.reshape(1,-1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	def update(self, reward,new_signal,done_bool,episode_timesteps):
		new_state = np.asarray(list(new_signal))
		# new_state = torch.Tensor(new_signal).float().unsqueeze(0)
		# print("New state:",new_state.dtype)
		# print("Last State:",self.last_state) 
		# print("Last Action:",np.asarray(self.last_action).dtype)
		# print("Last Reward:",self.last_reward) 
		print(type(done_bool))
		self.last_action = np.asarray(self.last_action)
		# self.last_action = torch.Tensor(new_signal).float().unsqueeze(0)
		self.memory.add((self.last_state, new_state, self.last_action, self.last_reward,done_bool))
		action = self.select_action(new_state)
		if len(self.memory.storage) > 100:
			# batch_state, batch_next_state, batch_action, batch_reward,batch_dones = self.memory.sample(100)
			# self.train(batch_state, batch_next_state, batch_reward, batch_action)
			# self.train(self.memory, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
			self.train(self.memory, episode_timesteps)
		self.last_action = action
		self.last_state = new_state
		self.last_reward = reward
		self.reward_window.append(reward)
		if len(self.reward_window) > 1000:
			del self.reward_window[0]
		return action

	def train(self, replay_buffer, iterations, batch_size=10,discount = 0.99,tau = 0.005,policy_noise = 0.2,noise_clip = 0.5,policy_freq = 2):
		for it in range(iterations):
			
			batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
			# print("batch states:\n",batch_states.shape, batch_states.dtype)
			state = torch.Tensor(batch_states).to(device)
			# print("batch next states:\n",batch_next_states.shape, batch_next_states.dtype)
			next_state = torch.Tensor(batch_next_states).to(device)
			# print(next_state)
			# print("batch next actions:\n",batch_actions.shape, batch_actions.dtype)
			action = torch.Tensor(batch_actions).to(device)
			# print(action)
			reward = torch.Tensor(batch_rewards).to(device)
			done = torch.Tensor(batch_dones).to(device)
			next_action = self. actor_target.forward(next_state)
			noise  = torch.Tensor(batch_actions).data.normal_(0,policy_noise).to(device)
			noise = noise.clamp(-noise_clip,noise_clip)
			next_action = (next_action + noise).clamp(-self.max_action,self.max_action)
			# print("next_state: ",next_state.shape)
			# print("next_action: ",next_action.shape)
			# print("next_state: ",next_state.dtype)
			# print("next_action: ",next_action.dtype)
			# print("next_state: ",next_state)
			# print("next_action: ",next_action)
			# print("Init Training")
			target_Q1,target_Q2 = self.critic_target.forward(next_state,next_action)
			target_Q = torch.min(target_Q1,target_Q2)
			target_Q = reward  + ((1-done)*discount*target_Q).detach() 

			current_Q1,current_Q2 = self.critic.forward(state, action)
			
			critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
			

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			if it % policy_freq == 0:
				# print("Target update")
				actor_loss = -(self.critic.Q1(state, self. actor(state)).mean())
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
					target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)

				for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
					target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
	
	def score(self):
		return sum(self.reward_window)/(len(self.reward_window)+1.)