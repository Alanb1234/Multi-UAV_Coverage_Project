import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import math

from environment import MultiAgentGridEnv
import json

class QMIXQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QMIXQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)




class QMIXMixer(nn.Module):
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32, hypernet_embed=64):
        super(QMIXMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * num_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim)
        )

        # State-dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)
        
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

    




class QMIXAgent:
    def __init__(self, state_size, action_size, num_agents, hidden_size=64, learning_rate=0.001, gamma=0.99, 
             epsilon_start=1.0, epsilon_min=0.0, epsilon_decay=0.995, decay_method='exponential'):

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_method = decay_method
        self.total_steps = 0
        
        self.q_networks = nn.ModuleList([QMIXQNetwork(state_size, hidden_size, action_size) for _ in range(num_agents)])
        self.target_q_networks = nn.ModuleList([QMIXQNetwork(state_size, hidden_size, action_size) for _ in range(num_agents)])
        
        self.mixer = QMIXMixer(num_agents, state_size * num_agents)
        self.target_mixer = QMIXMixer(num_agents, state_size * num_agents)

        self.optimizer = optim.Adam(list(self.q_networks.parameters()) + list(self.mixer.parameters()), lr=learning_rate)
        self.memory = deque(maxlen=10000)


        self.update_target_network()




    def update_epsilon(self, episode=None):
        if self.decay_method == 'exponential':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_method == 'linear':
            self.epsilon = max(self.epsilon_min, self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (self.total_steps / 1000000))
        elif self.decay_method == 'cosine':
            self.epsilon = self.epsilon_min + 0.5 * (self.epsilon_start - self.epsilon_min) * (1 + math.cos(math.pi * self.total_steps / 1000000))
        elif self.decay_method == 'step':
            if episode is not None and episode % 100 == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)
        
        self.total_steps += 1


    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)





    def act(self, states, sensor_readings):
        actions = []
        for i, state in enumerate(states):
            if random.random() < self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0)
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
        
        batch = self.sample_memory(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)


        # Convert to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Then convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Rest of the method remains the same
        current_q_values = [self.q_networks[i](states[:, i]) for i in range(self.num_agents)]
        current_q_values = torch.stack([values.gather(1, actions[:, i].unsqueeze(1)) for i, values in enumerate(current_q_values)], dim=1)

        with torch.no_grad():
            next_q_values = [self.target_q_networks[i](next_states[:, i]) for i in range(self.num_agents)]
            next_q_values = torch.stack([values.max(1)[0] for values in next_q_values], dim=1)

        current_q_total = self.mixer(current_q_values, states.view(batch_size, -1)).squeeze(-1)
        next_q_total = self.target_mixer(next_q_values.unsqueeze(2), next_states.view(batch_size, -1)).squeeze(-1)

        expected_q_total = rewards + self.gamma * next_q_total * (1 - dones)

        loss = F.mse_loss(current_q_total, expected_q_total)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()

        return loss.item()



    def update_target_network(self):
        for q_network, target_q_network in zip(self.q_networks, self.target_q_networks):
            target_q_network.load_state_dict(q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def parameters(self):
        return list(self.q_networks.parameters()) + list(self.mixer.parameters())

    
    def save(self, path):
        torch.save({
            'q_networks_state_dict': [net.state_dict() for net in self.q_networks],
            'target_q_networks_state_dict': [net.state_dict() for net in self.target_q_networks],
            'mixer_state_dict': self.mixer.state_dict(),
            'target_mixer_state_dict': self.target_mixer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(checkpoint['q_networks_state_dict'][i])
        for i, net in enumerate(self.target_q_networks):
            net.load_state_dict(checkpoint['target_q_networks_state_dict'][i])
        self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']



def train_qmix(num_episodes=2000, batch_size=32, update_freq=50, save_freq=100, 
               epsilon_start=1.0, epsilon_min=0.0, epsilon_decay=0.992, 
               decay_method='exponential'):
    
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=2,
        max_steps_per_episode=20,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )

    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    num_agents = env.num_agents

    agent = QMIXAgent(state_size, action_size, num_agents, 
                      epsilon_start=epsilon_start, epsilon_min=epsilon_min, 
                      epsilon_decay=epsilon_decay, decay_method=decay_method)
    

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episode_rewards = []
    best_episode_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None  

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []


        while not done:
            sensor_readings = env.get_sensor_readings()
            actions = agent.act(state, sensor_readings)
            next_state, reward, done, actual_actions = env.step(actions)
            agent.remember(state, actual_actions, reward, next_state, done)
            loss = agent.replay(batch_size)
            state = next_state
            total_reward += reward
            episode_actions.append(actual_actions)

        if episode % update_freq == 0:
            agent.update_target_network()


        episode_rewards.append(total_reward)

        print(f"Episode {episode}, Score: {total_reward}, Epsilon: {agent.epsilon}")

        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode

        if episode % save_freq == 0:
            agent.save(f'models/qmix_agent_episode_{episode}.pth')

        with open('logs/rewards.txt', 'a') as f:
            f.write(f"{episode},{total_reward}\n")
        
        agent.update_epsilon(episode)


    agent.save('models/best_qmix_agent.pth')

    save_best_episode(env.initial_positions, best_episode_actions, best_episode_number, best_episode_reward)  
    save_final_positions(env, best_episode_actions)
    visualize_and_record_best_strategy(env, best_episode_actions)
    return agent, best_episode_actions, best_episode_number


# The helper functions save_best_episode, save_final_positions, and visualize_and_record_best_strategy 


def save_best_episode(initial_positions, best_episode_actions, best_episode_number,best_episode_reward , filename='qmix_best_strategy.json'):
    action_map = ['forward', 'backward', 'left', 'right', 'stay']
    
    best_episode = {
        "episode_number": int(best_episode_number),  # Convert to int if it's np.int64
        "episode_reward": float(best_episode_reward)  # Convert to float if it's np.float64

    }
    
    for i in range(len(initial_positions)):
        best_episode[f'agent_{i}'] = {
            'actions': [action_map[action[i]] for action in best_episode_actions],
            'initial_position': initial_positions[i]
        }
    
    with open(filename, 'w') as f:
        json.dump(best_episode, f, indent=4)

    print(f"Best episode actions and initial positions saved to {filename}")



def save_final_positions(env, best_episode_actions, filename='qmix_final_positions.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    for actions in best_episode_actions:
        env.step(actions)
    
    env.render(ax, actions=best_episode_actions[-1], step=len(best_episode_actions)-1)
    plt.title("Final Positions")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final positions saved as {filename}")



def visualize_and_record_best_strategy(env, best_episode_actions, filename='qmix_best_episode.mp4'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    # Set up the video writer
    writer = FFMpegWriter(fps=2)
    
    with writer.saving(fig, filename, dpi=100):
        # Capture the initial state
        ax.clear()
        env.render(ax, actions=None, step=0)
        writer.grab_frame()
        plt.pause(0.1)
        
        for step, actions in enumerate(best_episode_actions, start=1):
            env.step(actions)
            ax.clear()
            env.render(ax, actions=actions, step=step)
            writer.grab_frame()
            plt.pause(0.1)
    
    plt.close(fig)
    print(f"Best episode visualization saved as {filename}")



if __name__ == "__main__":
    agent, best_episode_actions, best_episode_number = train_qmix(
        decay_method='exponential'  # or 'linear', 'cosine', 'step'
    )
    print(f"Best episode: {best_episode_number}")

    