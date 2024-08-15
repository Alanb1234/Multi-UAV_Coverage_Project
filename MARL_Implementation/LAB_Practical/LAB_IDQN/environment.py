# environment.py
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx

class MultiAgentGridEnv:
    def __init__(self, grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions, reward_type='global'):
        self.grid = self.load_grid(grid_file)
        self.grid_height, self.grid_width = self.grid.shape  # Use height and width instead of grid_size
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = num_agents
        self.initial_positions = initial_positions
        self.reward_type = reward_type

        
        # Calculate new obs_size for local rich observations
        self.obs_size = (
            2 +  # Agent's own position (x, y)
            4 +  # Sensor readings
            1 +  # Current time step
            (2*coverage_radius + 1)**2 * 2 +  # Local view of coverage and obstacles
            (num_agents - 1) * 2  # Relative positions of other agents
        )
        
        self.nx = nx
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
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):  # Use grid_width and grid_height
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
            return (min(x + 1, self.grid_width - 1), y)
        elif action == 1:  # backward (negative x)
            return (max(x - 1, 0), y)
        elif action == 2:  # left (positive y)
            return (x, min(y + 1, self.grid_height - 1))
        elif action == 3:  # right (negative y)
            return (x, max(y - 1, 0))
        else:  # stay
            return (x, y)



    def cover_area(self, state):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1

    ### ***********
    ### Reward Calculation
    ### ***********

    def calculate_global_reward(self):
        self.total_area = np.sum(self.coverage_grid > 0)
        self.overlap_penalty = self.calculate_overlap()
        
        graph = self.build_graph()
        self.num_components = nx.number_connected_components(graph)
        
        if self.num_components == 1:
            self.connectivity_penalty = 0
        else:
            self.connectivity_penalty = (self.num_agents) * (self.num_components - 1) * ((1 + 2*self.coverage_radius)**2)

        self.hole_penalty = self.calculate_hole_penalty(graph)


        self.sensor_1s = self.calculate_sensor_penalty()
        self.sensor_penalty = self.sensor_1s* ((1 + 2*self.coverage_radius)**2)

        reward = (
            self.total_area 
            - (0.5) * self.overlap_penalty 
            - self.connectivity_penalty 
            - self.hole_penalty 
            - self.sensor_penalty  # Adjust the weight as needed
        )
        return reward

    def calculate_sensor_penalty(self):
        sensor_readings = self.get_sensor_readings()
        total_penalty = 0
        for readings in sensor_readings:
            # Sum up the number of 'blocked' directions (1's in the sensor reading)
            penalty = sum(readings)
            if penalty > 0:
                total_penalty += 1
            
        return total_penalty


    
    ### Hole penalty Implementation, using chordless cycles
    def calculate_hole_penalty(self, graph):
        chordless_cycles = self.find_chordless_cycles(graph)
        num_holes = len(chordless_cycles)
        return num_holes * (self.num_agents * (1 + 2*self.coverage_radius)**2)


    def find_chordless_cycles(self, graph):
        chordless_cycles = []
        visited_cycles = set()
        for node in graph.nodes():
            self._find_cycles_from_node(graph, node, [node], set([node]), chordless_cycles, visited_cycles)
        return chordless_cycles

    def _find_cycles_from_node(self, graph, start, path, visited, chordless_cycles, visited_cycles):
        neighbors = set(graph.neighbors(path[-1])) - set(path[1:])
        for neighbor in neighbors:
            if neighbor == start and len(path) > 3:
                cycle = path[:]
                if self._is_chordless(graph, cycle):
                    cycle_key = tuple(sorted(cycle))
                    if cycle_key not in visited_cycles:
                        chordless_cycles.append(cycle)
                        visited_cycles.add(cycle_key)
            elif neighbor not in visited:
                self._find_cycles_from_node(graph, start, path + [neighbor], visited | {neighbor}, chordless_cycles, visited_cycles)

    def _is_chordless(self, graph, cycle):
        for i in range(len(cycle)):
            for j in range(i+2, len(cycle)):
                if (i != 0 or j != len(cycle)-1) and graph.has_edge(cycle[i], cycle[j]):
                    return False
        return True

    def build_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))
        for i, pos1 in enumerate(self.agent_positions):
            for j, pos2 in enumerate(self.agent_positions[i+1:], i+1):
                if self.areas_overlap(pos1, pos2):
                    G.add_edge(i, j)
        return G
    

    def areas_overlap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 2 * self.coverage_radius and abs(y1 - y2) <= 2 * self.coverage_radius


  

    ### End of hole penalty implementation

    
    def calculate_overlap(self):
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid
        
        overlap_counts = overlap_grid[overlap_grid > 1] - 1
        weighted_overlap = np.sum(overlap_counts)
        return weighted_overlap




    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1  # Increment instead of setting to 1



    
    ### ***********
    ### Reward Calculation end
    ### ***********

    def get_observations(self):
        observations = []
        sensor_readings = self.get_sensor_readings()
        
        for i, pos in enumerate(self.agent_positions):
            x, y = pos
            obs = [
                x, y,  # Agent's own position (x, y)
                *sensor_readings[i],  # Sensor readings
                self.current_step,  # Current time step
            ]

            # Local view of coverage and obstacles
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        obs.extend([
                            self.coverage_grid[ny, nx],
                            self.grid[ny, nx]
                        ])
                    else:
                        obs.extend([0, 1])  # Treat out-of-bounds as uncovered and obstacle            

            # Relative positions of nearby agents
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    ox, oy = other_pos
                    if abs(x - ox) <= self.coverage_radius and abs(y - oy) <= self.coverage_radius:
                        obs.extend([ox - x, oy - y])
                    else:
                        obs.extend([self.coverage_radius + 1, self.coverage_radius + 1])  # Indicate agent is out of local view
            
            observations.append(np.array(obs, dtype=np.float32))
        
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
                1 if x == self.grid_width - 1 or self.grid[y, x + 1] == 1 or (x + 1, y) in self.agent_positions else 0,  # forward
                1 if x == 0 or self.grid[y, x - 1] == 1 or (x - 1, y) in self.agent_positions else 0,  # backward
                1 if y == self.grid_height - 1 or self.grid[y + 1, x] == 1 or (x, y + 1) in self.agent_positions else 0,  # left
                1 if y == 0 or self.grid[y - 1, x] == 1 or (x, y - 1) in self.agent_positions else 0  # right
            ]
            readings.append(reading)
        return readings

    
    ### Can be useful for debugging
    def get_metrics(self):
        return {
            "Total Area": self.total_area,
            "Overlap Penalty": self.overlap_penalty,
            "Connectivity Penalty": self.connectivity_penalty,
            "Hole Penalty": self.hole_penalty,
            "Number of Components": self.num_components,
            "Number of Holes": len(self.find_chordless_cycles(self.build_graph())),
            "Reward": self.total_area - self.overlap_penalty - self.connectivity_penalty - self.hole_penalty
        }



    def render(self, ax=None, actions=None, step=None, return_rgb=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure
        
        ax.clear()
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        
        # Draw the grid and obstacles
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] == 1:  # Obstacles are black
                    rect = plt.Rectangle((j, i), 1, 1, color='black')
                    ax.add_patch(rect)
        
        # Define consistent colors for 10 agents
        agent_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
        
        # Draw the coverage area and agents
        for idx, pos in enumerate(self.agent_positions):
            x, y = pos
            agent_color = agent_colors[idx % len(agent_colors)]
            
            # Draw coverage area
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.grid[ny, nx] == 0:
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





