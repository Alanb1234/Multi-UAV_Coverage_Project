import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import networkx as nx

class RewardExplorer:
    def __init__(self, grid_file, coverage_radius):
        self.grid = self.load_grid(grid_file)
        self.grid_size = self.grid.shape[1]
        self.coverage_radius = coverage_radius
        self.num_agents = 0
        self.nx = nx
        self.reset_metrics()

    def reset_metrics(self):
        self.total_area = 0
        self.overlap_penalty = 0
        self.connectivity_penalty = 0
        self.hole_penalty = 0
        self.num_components = 0
        self.num_holes = 0
        self.reward = 0


    def load_grid(self, filename):
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def calculate_reward(self, agent_positions):
        self.reset_metrics()
        self.agent_positions = agent_positions
        self.num_agents = len(agent_positions)
        self.coverage_grid = np.zeros_like(self.grid)
        self.update_coverage()
        self.calculate_global_reward()
        return self.reward


    def calculate_global_reward(self):
        self.total_area = np.sum(self.coverage_grid > 0)
        self.overlap_penalty = self.calculate_overlap()
        self.num_components = self.count_connected_components()
        
        if self.num_components == 1:
            self.connectivity_penalty = 0
        else:
            self.connectivity_penalty = (self.num_agents)*(self.num_components - 1) * ((1+2*self.coverage_radius)**2)

        self.hole_penalty = self.calculate_hole_penalty()

        self.reward = 1.5*self.total_area - 0.75*self.overlap_penalty - self.connectivity_penalty - self.hole_penalty


    ### Helper penalty functions

    def calculate_hole_penalty(self):
        graph = self.build_graph()
        chordless_cycles = self.find_chordless_cycles(graph)
        self.num_holes = len(chordless_cycles)
        return self.num_holes * (self.num_agents*(1+2*self.coverage_radius)**2)  # Penalize for each hole



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

    ### End of hole penalty Implementation





    def update_coverage(self):
        self.coverage_grid = np.zeros_like(self.grid)
        for pos in self.agent_positions:
            self.cover_area(pos)

    def cover_area(self, state):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1

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
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1

    def count_connected_components(self):
        graph = self.build_graph()
        if not graph.nodes():
            return self.num_agents  # If no edges, each agent is its own component
        return nx.number_connected_components(graph)

    def build_graph(self):
        G = self.nx.Graph()
        G.add_nodes_from(range(self.num_agents))  # Ensure all agents are represented as nodes
        for i, pos1 in enumerate(self.agent_positions):
            for j, pos2 in enumerate(self.agent_positions[i+1:], i+1):
                if self.areas_overlap(pos1, pos2):
                    G.add_edge(i, j)
        return G



    def areas_overlap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 2 * self.coverage_radius and abs(y1 - y2) <= 2 * self.coverage_radius

    def visualize(self, agent_positions):
        self.agent_positions = agent_positions
        self.coverage_grid = np.zeros_like(self.grid)
        self.update_coverage()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        self._draw_grid(ax1)
        self._draw_graph(ax2)

        graph = self.build_graph()
        chordless_cycles = self.find_chordless_cycles(graph)
        self._draw_chordless_cycles(ax2, chordless_cycles)
        
        plt.suptitle(f"Reward: {self.reward:.2f}, Holes: {len(chordless_cycles)}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def _draw_chordless_cycles(self, ax, chordless_cycles):
        pos = {i: self.agent_positions[i] for i in range(self.num_agents)}
        for cycle in chordless_cycles:
            cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            nx.draw_networkx_edges(self.build_graph(), pos, edgelist=cycle_edges, 
                                   edge_color='r', width=2, ax=ax)



    def _draw_grid(self, ax):
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        for (j, i) in np.ndindex(self.grid.shape):
            if self.grid[i, j] == 1:
                rect = plt.Rectangle((j, i), 1, 1, color='black')
                ax.add_patch(rect)
        
        agent_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
        
        for idx, pos in enumerate(self.agent_positions):
            x, y = pos
            agent_color = agent_colors[idx % len(agent_colors)]
            
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                        rect = plt.Rectangle((nx, ny), 1, 1, color=agent_color, alpha=0.3)
                        ax.add_patch(rect)
            
            rect = plt.Rectangle((x, y), 1, 1, color=agent_color)
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.5, str(idx + 1), color='black', ha='center', va='center', fontweight='bold')

        ax.grid(True)
        ax.set_title("Agent Positions and Coverage")

    def _draw_graph(self, ax):
        graph = self.build_graph()
        pos = {i: self.agent_positions[i] for i in range(self.num_agents)}
        self.nx.draw(graph, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=500, font_size=16, font_weight='bold')
        ax.set_title("Agent Connectivity Graph")


    def get_metrics(self):
        return {
            "Total Area": self.total_area,
            "Overlap Penalty": self.overlap_penalty,
            "Connectivity Penalty": self.connectivity_penalty,
            "Hole Penalty": self.hole_penalty,
            "Number of Components": self.num_components,
            "Number of Holes": self.num_holes,
            "Reward": self.reward
        }


def main():
    explorer = RewardExplorer('grid_world.json', coverage_radius=3)
    
    configurations = [
        ## Holes
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(5, 14), (11, 9), (11, 19), (15, 14)],
        [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (3, 2), (4, 1)],
        [(5, 14), (11, 9), (11, 19), (15, 14), (21, 9), (21, 19), (26, 14)],
        [(10, 4), (4, 6), (4, 12), (10, 17),(16, 6), (16, 12)],

        ## Coonnections
        [(5, 14), (11, 9), (11, 19), (11, 14)],
        [(5, 14), (11, 9), (11, 19), (24, 14)],
        [(5, 14), (11, 9), (11, 24), (24, 14)],
        [(5, 14), (16, 9), (11, 24), (24, 14)],


        ### Lines
        [(5, 14), (11, 14), (17, 14), (23, 14)],






    ]

    for i, positions in enumerate(configurations, 1):
        reward = explorer.calculate_reward(positions)
        metrics = explorer.get_metrics()
        
        print(f"\nConfiguration {i}:")
        print(f"Positions: {positions}")
        print(f"Total Area: {metrics['Total Area']}")
        print(f"Overlap Penalty: {metrics['Overlap Penalty']:.2f}")
        print(f"Connectivity Penalty: {metrics['Connectivity Penalty']:.2f}")
        print(f"Hole Penalty: {metrics['Hole Penalty']:.2f}")
        print(f"Number of Components: {metrics['Number of Components']}")
        print(f"Number of Holes: {metrics['Number of Holes']}")
        print(f"Reward: {metrics['Reward']:.2f}")
        
        explorer.visualize(positions)

if __name__ == "__main__":
    main()

