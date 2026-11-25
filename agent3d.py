"""agent.py's Agent on the 3D column graph (node_manager3d.py). The observation is built exactly as in 2D:
(x, y) of each column relative to the drone's column, utility, visited."""
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from node_manager3d import NodeManager3D
from parameter import *


class Agent3D(Agent):
    def __init__(self, policy_net, device='cpu', plot=False):
        super().__init__(policy_net, device, plot)
        self.node_manager = NodeManager3D(plot=plot)

    def update_planning_state(self, map_info, location):
        """map_info: the belief (env3d.Map3D, with the drone's 3D position); location: the drone's column (x, y)."""
        self.map_info = map_info
        self.location = np.asarray(location)
        self.node_manager.update_graph(map_info)
        self.frontier = map_info.frontier_pts
        self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
            self.update_observation()

    def route_to(self, next_location):
        position, dist, _ = self.node_manager.route_to(next_location)
        return position, dist

    def plot_env(self):
        plt.switch_backend('agg')
        plt.figure(figsize=(18, 5.5))
        plt.subplot(1, 3, 2)
        nm, res = self.node_manager, self.map_info.res
        known = self.map_info.map
        img = np.full(known.shape[1:], 0.5)
        img[(known == 2).any(axis=0)] = 0.0
        img[(known == 1).any(axis=0)] = 1.0
        ny, nx = img.shape
        plt.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=(0, nx * res, 0, ny * res))
        if len(self.frontier):
            plt.scatter(self.frontier[:, 0], self.frontier[:, 1], c='r', s=1.5, zorder=2)
        start = self.map_info.start[:2]
        xy = self.node_coords + start
        for coords in self.node_coords:
            node = nm.nodes_dict.find(coords.tolist()).data
            for nb in node.neighbor_set:
                end = (np.array(nb) - coords) / 2 + coords + start
                plt.plot([coords[0] + start[0], end[0]], [coords[1] + start[1], end[1]], 'tan', lw=0.6, zorder=1)
        heights = np.array([nm.lat.pos[nm.nodes_dict.find(c.tolist()).data.kept, 2] for c in self.node_coords])
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=heights, cmap='viridis', s=14 + 2 * np.minimum(self.utility, 60),
                         edgecolors=np.where(self.utility > 0, 'r', 'none'), zorder=3)
        plt.colorbar(sc, fraction=0.04, label='kept height, m')
        plt.plot(*self.map_info.robot_position[:2], 'mo', markersize=10, zorder=5)
        plt.axis('off')
        plt.title('columns (size: utility, red rim: > 0)')
