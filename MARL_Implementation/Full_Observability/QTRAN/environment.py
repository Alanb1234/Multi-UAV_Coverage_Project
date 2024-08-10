# environment.py
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class MultiAgentGridEnv:
    def __init__(self, grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions, reward_type='global'):
        self.grid = self.load_grid(grid_file)
        self.grid_size = self.grid.shape[1]
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = num_agents
        self.initial_positions = initial_positions
        self.reward_type = reward_type
        self.obs_size = self.grid_size * self.grid_size * 3 + 4  # Calculate obs_size in init
        self.reset()

    def load_grid(self, filename):
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def reset(self):
        self.agent_positions = list(self.initial_positions)
        self.coverage_grid = np.zeros_like(self.grid)
        self.current_step = 0
        self.update_coverage()
        return self.get_observations()

    def step(self, actions):
        self.current_step += 1
        new_positions = []
        actual_actions = []
        sensor_readings = self.get_sensor_readings()

        # First, calculate all new positions
        for i, action in enumerate(actions):
            new_pos = self.get_new_position(self.agent_positions[i], action)
            new_positions.append(new_pos)
            actual_actions.append(action)

        # Then, validate moves and update positions
        for i, new_pos in enumerate(new_positions):
            if not self.is_valid_move(new_pos, sensor_readings[i], actual_actions[i], new_positions[:i] + new_positions[i+1:]):
                new_positions[i] = self.agent_positions[i]
                actual_actions[i] = 4  # Stay action

        self.agent_positions = new_positions
        self.update_coverage()
        global_reward = self.calculate_global_reward()
        done = self.current_step >= self.max_steps_per_episode
        return self.get_observations(), global_reward, done, actual_actions


    def is_valid_move(self, new_pos, sensor_reading, action, other_new_positions):
        x, y = new_pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        if self.grid[y, x] == 1:  # Check for obstacles
            return False
        if new_pos in self.agent_positions or new_pos in other_new_positions:  # Check for other agents
            return False
        # Check sensor readings for specific direction
        if action == 0 and sensor_reading[0] == 1:  # forward
            return False
        elif action == 1 and sensor_reading[1] == 1:  # backward
            return False
        elif action == 2 and sensor_reading[2] == 1:  # left
            return False
        elif action == 3 and sensor_reading[3] == 1:  # right
            return False
        return True


    def update_coverage(self):
        self.coverage_grid = np.zeros_like(self.grid)
        for pos in self.agent_positions:
            self.cover_area(pos)

    def get_new_position(self, position, action):
        x, y = position
        if action == 0:  # forward (positive x)
            return (min(x + 1, self.grid_size - 1), y)
        elif action == 1:  # backward (negative x)
            return (max(x - 1, 0), y)
        elif action == 2:  # left (positive y)
            return (x, min(y + 1, self.grid_size - 1))
        elif action == 3:  # right (negative y)
            return (x, max(y - 1, 0))
        else:  # stay
            return (x, y)


    def cover_area(self, state):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1

    def calculate_global_reward(self):
        total_area = np.sum(self.coverage_grid > 0)  # Count cells covered by at least one agent
        overlap_penalty = self.calculate_overlap()
        num_components = self.count_connected_components()
        penalty = num_components if num_components == self.num_agents else num_components - 1
        penalty_score = 2 * penalty * (total_area / self.num_agents)

        reward = total_area - overlap_penalty - penalty_score
        return reward


    
    def calculate_overlap(self):
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid
        
        # Calculate weighted overlap
        overlap_counts = overlap_grid[overlap_grid > 1] - 1  # Subtract 1 to get the number of extra agents in each cell
        weighted_overlap = np.sum(overlap_counts)

        return weighted_overlap



    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1  # Increment instead of setting to 1



    def count_connected_components(self):
        graph = self.build_graph()
        visited = set()
        components = 0
        for node in range(self.num_agents):
            if node not in visited:
                components += 1
                self.bfs(node, graph, visited)
        return components

    
    def build_graph(self):
        graph = defaultdict(list)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.areas_overlap(self.agent_positions[i], self.agent_positions[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        return graph

    def areas_overlap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 2 * self.coverage_radius and abs(y1 - y2) <= 2 * self.coverage_radius

    def bfs(self, start, graph, visited):
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)


    def get_observations(self):
        observations = []
        sensor_readings = self.get_sensor_readings()
        for i, pos in enumerate(self.agent_positions):
            obs = np.zeros((self.grid_size, self.grid_size, 3))
            obs[:,:,0] = self.grid
            obs[:,:,1] = self.coverage_grid
            x, y = pos
            obs[y, x, 2] = 1
            flat_obs = obs.flatten()
            flat_obs = np.concatenate([flat_obs, sensor_readings[i]])  # Add sensor readings to observation
            observations.append(flat_obs)
        return observations

    def get_obs_size(self):
        return self.obs_size

    def get_total_actions(self):
        return 5  # forward, backward, left, right, stay

    def get_sensor_readings(self):
        readings = []
        for pos in self.agent_positions:
            x, y = pos
            reading = [
                1 if x == self.grid_size - 1 or self.grid[y, x + 1] == 1 or (x + 1, y) in self.agent_positions else 0,  # forward
                1 if x == 0 or self.grid[y, x - 1] == 1 or (x - 1, y) in self.agent_positions else 0,  # backward
                1 if y == self.grid_size - 1 or self.grid[y + 1, x] == 1 or (x, y + 1) in self.agent_positions else 0,  # left
                1 if y == 0 or self.grid[y - 1, x] == 1 or (x, y - 1) in self.agent_positions else 0  # right
            ]
            readings.append(reading)
        return readings



    def render(self, ax=None, actions=None, step=None, return_rgb=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure
        
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        # Draw the grid and obstacles
        for (j, i) in np.ndindex(self.grid.shape):
            if self.grid[i, j] == 1:  # Obstacles are black
                rect = plt.Rectangle((j, i), 1, 1, color='black')
                ax.add_patch(rect)
        
        # Define a color map for agents
        agent_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        
        # Draw the coverage area and agents
        for idx, pos in enumerate(self.agent_positions):
            x, y = pos
            agent_color = agent_colors[idx % len(agent_colors)]
            
            # Draw coverage area
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                        rect = plt.Rectangle((nx, ny), 1, 1, color=agent_color, alpha=0.3)
                        ax.add_patch(rect)
            
            # Draw the agent
            rect = plt.Rectangle((x, y), 1, 1, color=agent_color)
            ax.add_patch(rect)
            
            # Add agent number
            ax.text(x + 0.5, y + 0.5, str(idx + 1), color='black', ha='center', va='center', fontweight='bold')
        
        # Display sensor readings
        sensor_readings = self.get_sensor_readings()
        for agent_idx, pos in enumerate(self.agent_positions):
            readings = sensor_readings[agent_idx]
            ax.text(pos[0] + 0.5, pos[1] - 0.3, f'{readings}', color='red', ha='center', va='center', fontsize=8)

        ax.grid(True)
        if actions is not None:
            action_texts = ['forward', 'backward', 'left', 'right', 'stay']
            action_display = ' | '.join([f"Agent {i+1}: {action_texts[action]}" for i, action in enumerate(actions)])
            title = f'{action_display}'
            if step is not None:
                title += f' || Step: {step}'
            ax.set_title(title, fontsize=10)
        
        if return_rgb:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        else:
            plt.draw()
            plt.pause(0.001)









