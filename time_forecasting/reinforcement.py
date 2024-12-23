import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt


# Environment for Portfolio Management
class PortfolioEnv:
    def __init__(self, prices, initial_cash=10000):
        self.prices = prices
        self.n_assets = prices.shape[1]
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.portfolio = np.zeros(self.n_assets)
        self.current_step = 0
        return self._get_state()

    def step(self, actions):
        actions = np.clip(actions, 1e-8, 1)  # Prevent zero actions
        allocation = actions / (actions.sum() + 1e-8)  # Avoid division by zero
        total_value = self.cash + np.sum(
            self.portfolio * self.prices[self.current_step]
        )
        self.portfolio = allocation * total_value / self.prices[self.current_step]
        self.cash = 0

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        next_state = self._get_state()
        reward = self._compute_reward()
        return next_state, reward, done

    def _get_state(self):
        return np.concatenate((self.prices[self.current_step], self.portfolio))

    def _compute_reward(self):
        total_value = max(
            1e-8, self.cash + np.sum(self.portfolio * self.prices[self.current_step])
        )
        return total_value / self.initial_cash - 1


# Neural Network for Q-Learning
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Deep Q-Learning Agent
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.memory = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Training Loop
def train_dqn(env, agent, episodes, batch_size):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay(batch_size)

        agent.update_target_network()
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    return rewards


# Visualization
def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.show()


# Example usage
prices = np.random.rand(200, 3) + 1  # Simulated prices for 3 assets
env = PortfolioEnv(prices)
agent = DQNAgent(state_dim=env.n_assets * 2, action_dim=env.n_assets)

rewards = train_dqn(env, agent, episodes=100, batch_size=32)
plot_rewards(rewards)
