import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import networkx as nx
from scipy.spatial import Delaunay, ConvexHull

class RewardExplorer:
    def __init__(self, grid_file, coverage_radius):
        self.grid = self.load_grid(grid_file)
        self.grid_size = self.grid.shape[1]
        self.coverage_radius = coverage_radius
        self.num_agents = 0  # This will be set when calculate_reward is called


    def load_grid(self, filename):
        """Load the grid world from a JSON file."""
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def calculate_reward(self, agent_positions):
        """Calculate the overall reward for given agent positions."""
        self.agent_positions = agent_positions
        self.num_agents = len(agent_positions)
        self.coverage_grid = np.zeros_like(self.grid)
        self.update_coverage()
        return self.calculate_global_reward()


    def calculate_global_reward(self):
        total_coverage = np.sum(self.coverage_grid > 0)
        overlap_penalty = self.calculate_overlap_penalty()
        connectivity_score = self.calculate_connectivity_score()
        compactness_reward = self.calculate_coverage_completeness_reward()

        print(f"Total Coverage: {total_coverage}, Overlap Penalty: {overlap_penalty}, Connectivity Score: {connectivity_score}, Compactness Reward: {compactness_reward}")
        
        reward = (total_coverage  # Increase weight of coverage
                  + connectivity_score  # Keep connectivity bonus as is
                  + compactness_reward   # Reduce weight of compactness
                  - overlap_penalty)  # Increase overlap penalty

        return reward
    

    def calculate_overlap_penalty(self):
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid
        
        # Calculate overlap penalty with reduced severity
        overlap_counts = overlap_grid[overlap_grid > 1] - 1
        penalty = np.sum(overlap_counts)  # Linear penalty instead of exponential
        return penalty  # Reduce the scale of the penalty
    
    # Connectivity-related methods
    def calculate_connectivity_score(self):
        graph = self.build_graph()
        if not graph:
            return -1000  # Severe penalty for no connections

        num_components = self.count_connected_components(graph)
        num_agents = len(self.agent_positions)

        if num_components == 1:
            return 100  # Full bonus for complete connectivity
        else:
            # Calculate penalty based on the number of components
            penalty = num_components if num_components == num_agents else num_components - 1
            penalty_score = penalty * (1000 / num_agents)
            return -penalty_score


    

    def update_coverage(self):
        self.coverage_grid = np.zeros_like(self.grid)
        for pos in self.agent_positions:
            self.cover_area_on_grid(pos, self.coverage_grid)

    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    grid[ny, nx] += 1



    def count_connected_components(self, graph):
        visited = set()
        components = 0
        for node in range(self.num_agents):
            if node not in visited:
                components += 1
                self.bfs(node, graph, visited)
        return components
    
    def bfs(self, start, graph, visited):
        """Perform BFS to mark all nodes in a connected component."""
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)


    def calculate_overlap(self):
        """Calculate the overlap with reduced penalty for necessary overlaps."""
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid
        
        # Calculate overlap with reduced penalty
        overlap_counts = overlap_grid[overlap_grid > 1] - 1
        weighted_overlap = np.sum(np.power(overlap_counts, 0.75))  # Use power of 0.75 to reduce penalty
        return weighted_overlap
    





    


    def build_graph(self):
        graph = defaultdict(list)
        for i in range(len(self.agent_positions)):
            for j in range(i + 1, len(self.agent_positions)):
                if self.areas_overlap(self.agent_positions[i], self.agent_positions[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        return graph


    def areas_overlap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 2 * self.coverage_radius and abs(y1 - y2) <= 2 * self.coverage_radius

    def calculate_coverage_completeness_reward(self):
        positions = np.array(self.agent_positions)
        
        # Create a convex hull of the agent positions
        hull = ConvexHull(positions)
        
        # Get the points that make up the convex hull
        hull_points = positions[hull.vertices]
        
        # Create a mask of the area within the convex hull
        mask = np.zeros_like(self.coverage_grid, dtype=bool)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.point_in_hull((i, j), hull_points):
                    mask[i, j] = True
        
        # Calculate the total area within the convex hull
        total_area = np.sum(mask)
        
        # Calculate the covered area within the convex hull
        covered_area = np.sum(self.coverage_grid[mask] > 0)
        
        # Calculate the completeness ratio
        completeness_ratio = covered_area / total_area if total_area > 0 else 0
        
        # The reward is higher when the completeness ratio is closer to 1
        reward = completeness_ratio * 1000  # Scale as needed
        
        return reward
    
    def point_in_hull(self, point, hull_points):
        """Check if a point is inside the convex hull."""
        def side(p1, p2, p):
            return (p2[0] - p1[0])*(p[1] - p1[1]) - (p2[1] - p1[1])*(p[0] - p1[0])
        
        n = len(hull_points)
        inside = True
        for i in range(n):
            if side(hull_points[i], hull_points[(i+1)%n], point) < 0:
                inside = False
                break
        return inside


    

    # Internal coverage penalty methods
    def calculate_internal_uncovered_penalty(self):
        """Calculate a penalty for uncovered areas within the agent formation."""
        if len(self.agent_positions) < 3:
            return 0

        positions = np.array(self.agent_positions)
        tri = Delaunay(positions)
        
        total_penalty = 0
        for simplex in tri.simplices:
            p1, p2, p3 = positions[simplex]
            triangle_area = 0.5 * abs(np.cross(p2-p1, p3-p1))
            covered_area = self.calculate_covered_area_in_triangle(p1, p2, p3)
            uncovered_area = max(0, triangle_area - covered_area)
            total_penalty += uncovered_area

        return total_penalty

    def calculate_covered_area_in_triangle(self, p1, p2, p3):
        """Calculate the covered area within a triangle formed by three points."""
        min_x, max_x = int(min(p1[0], p2[0], p3[0])), int(max(p1[0], p2[0], p3[0])) + 1
        min_y, max_y = int(min(p1[1], p2[1], p3[1])), int(max(p1[1], p2[1], p3[1])) + 1

        covered_area = 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if self.is_point_in_triangle((x, y), p1, p2, p3) and self.coverage_grid[y, x] > 0:
                    covered_area += 1

        return covered_area

    def is_point_in_triangle(self, p, p1, p2, p3):
        """Check if a point is inside a triangle."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, p1, p2)
        d2 = sign(p, p2, p3)
        d3 = sign(p, p3, p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def visualize_with_graph(self, agent_positions):
        """Visualize the agent positions, coverage, connectivity, and compactness."""
        self.agent_positions = agent_positions
        self.coverage_grid = np.zeros_like(self.grid)
        self.update_coverage()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40, 10))
        
        self._draw_grid(ax1)
        self._draw_graph(ax2)
        self._draw_convex_hull(ax3)
        self._draw_overlap(ax4)

        reward = self.calculate_global_reward()
        overlap_penalty = self.calculate_overlap_penalty()
        compactness_reward = self.calculate_coverage_completeness_reward()
        plt.suptitle(f"Reward: {reward:.2f}, Overlap Penalty: {overlap_penalty:.2f}, Compactness Reward: {compactness_reward:.2f}", fontsize=16)
        plt.tight_layout()
        plt.show()


    
    def _draw_convex_hull(self, ax):
        """Draw the convex hull of agent positions."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        positions = np.array(self.agent_positions)
        ax.scatter(positions[:, 0], positions[:, 1], c='red')
        
        if len(self.agent_positions) >= 3:
            hull = ConvexHull(positions)
            for simplex in hull.simplices:
                ax.plot(positions[simplex, 0], positions[simplex, 1], 'r-')
        
        ax.set_title("Agent Positions and Convex Hull")
        ax.grid(True)




    def _draw_overlap(self, ax):
        """Draw the overlap areas."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        overlap_grid = self.coverage_grid.copy()
        overlap_grid[overlap_grid <= 1] = 0
        
        im = ax.imshow(overlap_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        
        ax.set_title("Overlap Areas")
        ax.grid(True)






    def _draw_grid(self, ax):
        """Draw the grid world with agent positions and coverage areas."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        for (j, i) in np.ndindex(self.grid.shape):
            if self.grid[i, j] == 1:
                rect = plt.Rectangle((j, i), 1, 1, color='black')
                ax.add_patch(rect)
        
        agent_colors = ['red', 'blue', 'green', 'yellow']
        
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
        """Draw the connectivity graph of the agents."""
        graph = self.build_graph()
        G = nx.Graph(graph)
        
        # Create a complete set of nodes, including disconnected ones
        all_nodes = set(range(len(self.agent_positions)))
        
        # Identify disconnected nodes
        disconnected_nodes = all_nodes - set(G.nodes())
        
        # Add disconnected nodes to the graph
        G.add_nodes_from(disconnected_nodes)
        
        # Generate layout for all nodes
        pos = nx.spring_layout(G)
        
        # Draw connected nodes
        nx.draw(G, pos, ax=ax, with_labels=False, node_color='lightblue', 
                node_size=500, font_size=16, font_weight='bold')
        
        # Draw disconnected nodes
        if disconnected_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=disconnected_nodes, node_color='red', 
                                   node_size=500, ax=ax)
        
        # Draw labels for all nodes
        labels = {i: f"Agent {i+1}" for i in range(len(self.agent_positions))}
        nx.draw_networkx_labels(G, pos, labels, ax=ax)
        
        ax.set_title("Agent Overlap Graph")

    def _draw_delaunay(self, ax):
        """Draw the Delaunay triangulation of agent positions."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        positions = np.array(self.agent_positions)
        ax.scatter(positions[:, 0], positions[:, 1], c='red')
        
        if len(self.agent_positions) >= 3:
            tri = Delaunay(positions)
            ax.triplot(positions[:, 0], positions[:, 1], tri.simplices)
        
        ax.set_title("Agent Positions and Delaunay Triangulation")
        ax.grid(True)



def main():
    explorer = RewardExplorer('grid_world.json', coverage_radius=7)
    
    # Initial 
    positions1 = [(1, 1), (2, 1), (1, 2), (2, 2)]
    reward1 = explorer.calculate_reward(positions1)
    print(f"Reward for positions {positions1}: {reward1:.2f}")
    explorer.visualize_with_graph(positions1)

    # Best permituation
    positions2 = [(7, 7), (7, 21), (21, 7), (21, 21)]
    reward2 = explorer.calculate_reward(positions2)
    print(f"Reward for positions {positions2}: {reward2:.2f}")
    explorer.visualize_with_graph(positions2)

    


    positions3 = [(15, 5), (2, 12), (21, 12), (15, 22)]
    reward3 = explorer.calculate_reward(positions3)
    print(f"Reward for positions {positions3}: {reward3:.2f}")
    explorer.visualize_with_graph(positions3)
    
    positions4 = [(15, 5), (5, 12), (21, 12), (15, 22)]
    reward4 = explorer.calculate_reward(positions4)
    print(f"Reward for positions {positions4}: {reward4:.2f}")
    explorer.visualize_with_graph(positions4)


    positions5 = [(15, 5), (8, 12), (21, 12), (15, 22)]
    reward5 = explorer.calculate_reward(positions5)
    print(f"Reward for positions {positions5}: {reward5:.2f}")
    explorer.visualize_with_graph(positions5)


    positions6 = [(15, 8), (8, 12), (21, 12), (15, 22)]
    reward6 = explorer.calculate_reward(positions6)
    print(f"Reward for positions {positions6}: {reward6:.2f}")
    explorer.visualize_with_graph(positions6)


    positions7 = [(1, 1), (21, 1), (1, 21), (28, 28)]
    reward7 = explorer.calculate_reward(positions7)
    print(f"Reward for positions {positions7}: {reward7:.2f}")
    explorer.visualize_with_graph(positions7)


     # Best permituation
    positions8 = [(7, 7), (7, 21), (21, 7), (25, 25)]
    reward8 = explorer.calculate_reward(positions8)
    print(f"Reward for positions {positions8}: {reward8:.2f}")
    explorer.visualize_with_graph(positions8)


    






if __name__ == "__main__":
    main()