import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

from environment import MultiAgentGridEnv
import json

class QTRANQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QTRANQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)


class QTRANMixer(nn.Module):
    def __init__(self, state_dim, n_agents, embed_dim=32):
        super(QTRANMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.Q = nn.Sequential(
            nn.Linear(state_dim + n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, states, agent_qs):
        bs = states.size(0)
        
        # Reshape states if necessary
        if len(states.shape) == 3:
            states = states.view(bs, -1)
        
        # Ensure agent_qs is the right shape
        agent_qs = agent_qs.view(bs, self.n_agents)
        
        # Concatenate states and agent_qs
        inputs = torch.cat([states, agent_qs], dim=1)
        
        q = self.Q(inputs)
        v = self.V(states)
        
        return q, v




class QTRANAgent:
    def __init__(self, state_size, action_size, num_agents, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.q_networks = [QTRANQNetwork(state_size, action_size) for _ in range(num_agents)]
        self.target_networks = [QTRANQNetwork(state_size, action_size) for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.mixer = QTRANMixer(state_size * num_agents, num_agents)
        self.target_mixer = QTRANMixer(state_size * num_agents, num_agents)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = optim.Adam(list(self.mixer.parameters()) + 
                                    [p for net in self.q_networks for p in net.parameters()], 
                                    lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=10000)


    def act(self, states, sensor_readings):
        actions = []
        for i in range(self.num_agents):
            if random.random() < self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(states[i]).unsqueeze(0)
                    action_values = self.q_networks[i](state).squeeze(0)
                    mask = np.zeros(self.action_size, dtype=float)
                    for j, reading in enumerate(sensor_readings[i]):
                        if reading == 1:
                            mask[j] = float('-inf')
                    masked_action_values = action_values.cpu().numpy() + mask
                    valid_action_indices = np.where(mask == 0)[0]
                    if len(valid_action_indices) == 0:
                        actions.append(self.action_size - 1)
                    else:
                        best_action_index = valid_action_indices[np.argmax(masked_action_values[valid_action_indices])]
                        actions.append(best_action_index)
        return actions

    def remember(self, states, actions, reward, next_states, done):
        self.memory.append((states, actions, reward, next_states, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = [torch.FloatTensor(np.array(state)) for state in zip(*states)]
        next_states = [torch.FloatTensor(np.array(next_state)) for next_state in zip(*next_states)]
        actions = [torch.LongTensor(action) for action in zip(*actions)]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = torch.stack([self.q_networks[i](states[i]).gather(1, actions[i].unsqueeze(1)).squeeze(1) 
                            for i in range(self.num_agents)], dim=1)
        
        state_batch = torch.stack(states, dim=1)
        next_state_batch = torch.stack(next_states, dim=1)

        q_tot, v = self.mixer(state_batch, current_q_values)

        with torch.no_grad():
            next_q_values = torch.stack([self.target_networks[i](next_states[i]).max(1)[0] for i in range(self.num_agents)], dim=1)
            next_q_tot, next_v = self.target_mixer(next_state_batch, next_q_values)

        target_q_tot = rewards + (1 - dones) * self.gamma * next_q_tot

        td_error = nn.MSELoss()(q_tot, target_q_tot)
        opt_loss = nn.MSELoss()(q_tot, current_q_values.sum(dim=1, keepdim=True) - v)
        nopt_loss = torch.mean(torch.clamp(q_tot - v - current_q_values.sum(dim=1, keepdim=True), min=0))

        loss = td_error + opt_loss + nopt_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def update_target_network(self):
        for i in range(self.num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path):
        torch.save({
            'q_networks_state_dict': [net.state_dict() for net in self.q_networks],
            'target_networks_state_dict': [net.state_dict() for net in self.target_networks],
            'mixer_state_dict': self.mixer.state_dict(),
            'target_mixer_state_dict': self.target_mixer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(checkpoint['q_networks_state_dict'][i])
        for i, net in enumerate(self.target_networks):
            net.load_state_dict(checkpoint['target_networks_state_dict'][i])
        self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_qtran(num_episodes=600, batch_size=32, update_freq=50, save_freq=100, epsilon_start=1.0, epsilon_min=0.00, epsilon_decay=0.005):
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )
    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    qtran_agent = QTRANAgent(state_size, action_size, env.num_agents, epsilon=epsilon_start)

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episode_rewards = []
    best_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        while not done:
            sensor_readings = env.get_sensor_readings()
            actions = qtran_agent.act(state, sensor_readings)
            next_state, reward, done, actual_actions = env.step(actions)
            episode_actions.append(actual_actions)

            qtran_agent.remember(state, actual_actions, reward, next_state, done)
            qtran_agent.replay(batch_size)

            state = next_state
            total_reward += reward

        if episode % update_freq == 0:
            qtran_agent.update_target_network()

        qtran_agent.epsilon = max(epsilon_min, epsilon_start * np.exp(-epsilon_decay * episode))

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {qtran_agent.epsilon}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode

        if episode % save_freq == 0:
            qtran_agent.save(f'models/qtran_agent_episode_{episode}.pth')

        with open('logs/rewards.txt', 'a') as f:
            f.write(f"{episode},{total_reward}\n")

    qtran_agent.save('models/best_qtran_agent.pth')

    save_best_episode(env.initial_positions, best_episode_actions, best_episode_number, filename='qtran_best_strategy.json')
    save_final_positions(env, best_episode_actions, filename='qtran_final_positions.png')
    visualize_and_record_best_strategy(env, best_episode_actions, filename='qtran_best_episode.mp4')
    return qtran_agent, best_episode_actions, best_episode_number
  

# The helper functions save_best_episode, save_final_positions, and visualize_and_record_best_strategy 


def save_best_episode(initial_positions, best_episode_actions, best_episode_number, filename='vdn_best_strategy.json'):
    action_map = ['forward', 'backward', 'left', 'right', 'stay']
    
    best_episode = {
        "episode_number": best_episode_number
    }
    
    for i in range(len(initial_positions)):
        best_episode[f'agent_{i}'] = {
            'actions': [action_map[action[i]] for action in best_episode_actions],
            'initial_position': initial_positions[i]
        }
    
    with open(filename, 'w') as f:
        json.dump(best_episode, f, indent=4)

    print(f"Best episode actions and initial positions saved to {filename}")



def save_final_positions(env, best_episode_actions, filename='vdn_final_positions.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    for actions in best_episode_actions:
        env.step(actions)
    
    env.render(ax, actions=best_episode_actions[-1], step=len(best_episode_actions)-1)
    plt.title("Final Positions")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final positions saved as {filename}")



def visualize_and_record_best_strategy(env, best_episode_actions, filename='vdn_best_episode.mp4'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    # Set up the video writer
    writer = FFMpegWriter(fps=2)
    
    with writer.saving(fig, filename, dpi=100):
        for step, actions in enumerate(best_episode_actions):
            env.step(actions)
            ax.clear()
            env.render(ax, actions=actions, step=step)
            writer.grab_frame()
            plt.pause(0.1)
    
    plt.close(fig)
    print(f"Best episode visualization saved as {filename}")


if __name__ == "__main__":
    trained_qtran_agent, best_episode_actions, best_episode_number = train_qtran()
    print(f"Best episode: {best_episode_number}")
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )
