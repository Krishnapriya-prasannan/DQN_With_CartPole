
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Environment Setup
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
episodes = 500
memory = deque(maxlen=10000)

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Helper functions
def get_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    state = torch.FloatTensor(state).unsqueeze(0)
    q_values = model(state)
    return torch.argmax(q_values).item()

def train():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    curr_Q = model(states).gather(1, actions)
    next_Q = target_model(next_states).max(1)[0].detach().unsqueeze(1)
    target_Q = rewards + gamma * next_Q * (1 - dones)

    loss = criterion(curr_Q, target_Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize models
model = DQN(state_dim, action_dim)
target_model = DQN(state_dim, action_dim)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training Loop
rewards_list = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):
        action = get_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_list.append(total_reward)

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {episode+1}: Reward = {total_reward}, Epsilon = {epsilon:.2f}")

# Plot the results
plt.plot(rewards_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN - CartPole")
plt.show()
