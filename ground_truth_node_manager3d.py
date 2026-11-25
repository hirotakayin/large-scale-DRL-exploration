"""The ground-truth critic's graph in 3D: the column graph of node_manager3d.py built once on the true map.

Every column with a lattice node reachable from the start is a node; two columns are neighbours when a lattice edge
on the true free configuration space joins them. The belief's columns and edges are subsets of these (the belief's
free space is a subset of the true one), so the critic sees the actor's graph plus the unexplored rest, as in 2D.
The observation itself is ground_truth_node_manager.py's.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ground_truth_node_manager import GroundTruthNodeManager, Node
from node_manager3d import ColumnDict, Lattice
from parameter import *

_GRAPHS = {}


def ground_truth_columns(map_info):
    """(columns, column pairs) of the true map, cached per world (a worker keeps its process across episodes)."""
    key = map_info.name
    if key in _GRAPHS:
        return _GRAPHS[key]
    lat = Lattice.get(map_info.map.shape, map_info.res, map_info.start)
    valid = lat.valid(map_info.cfree)
    a, b = lat.pairs[:, 0], lat.pairs[:, 1]
    cand = np.nonzero(valid[a] & valid[b])[0]
    ok = np.zeros(len(a), bool)
    ok[cand] = lat.segments_free(cand, map_info.cfree)
    ea, eb, w = a[ok], b[ok], lat.length[ok]
    n = len(lat.vox)
    graph = csr_matrix((np.concatenate([w, w]), (np.concatenate([ea, eb]), np.concatenate([eb, ea]))), shape=(n, n))
    reach = np.isfinite(dijkstra(graph, indices=lat.index_of(map_info.start)))
    cols = np.unique(lat.col[reach])
    rm = reach[ea] & reach[eb]
    ca, cb = lat.col[ea[rm]], lat.col[eb[rm]]
    x = ca != cb
    cp = np.unique(np.sort(np.stack([ca[x], cb[x]], 1), axis=1), axis=0)
    _GRAPHS.clear()
    _GRAPHS[key] = (lat, cols, cp)
    return _GRAPHS[key]


class GroundTruthNodeManager3D(GroundTruthNodeManager):
    def initialize_graph(self):
        self.nodes_dict = ColumnDict()
        lat, cols, cp = ground_truth_columns(self.ground_truth_map_info)
        self.lat = lat
        gaps, free_height = lat.column_profile(self.ground_truth_map_info.cfree)
        self.true_profile = {tuple(map(float, lat.col_xy[c])): (int(gaps[c]), float(free_height[c])) for c in cols}
        for c in cols:
            self.nodes_dict.insert(lat.col_xy[c], Node(lat.col_xy[c].copy()))
        for c1, c2 in cp:
            k1, k2 = tuple(map(float, lat.col_xy[c1])), tuple(map(float, lat.col_xy[c2]))
            self.nodes_dict.find(k1).data.neighbor_set.add(k2)
            self.nodes_dict.find(k2).data.neighbor_set.add(k1)

    def extra_node_inputs(self, node_coords, explored_sign):
        """The actor's 3 stage-1 inputs for explored columns; for the rest, no height offset and the true gaps and
        free height (the critic may know the true map)."""
        nm = self.node_manager
        z = nm.lat.pos[nm.here, 2]
        out = []
        for c in node_coords:
            key = (float(c[0]), float(c[1]))
            item = nm.nodes_dict.find(key)
            if item is not None:
                n = item.data
                out.append(((n.height - z) / HEIGHT_NORM_3D, n.gaps / GAPS_NORM_3D, n.free_height / HEIGHT_NORM_3D))
            else:
                g, h = self.true_profile[key]
                out.append((0.0, g / GAPS_NORM_3D, h / HEIGHT_NORM_3D))
        return np.array(out).reshape(-1, NODE_INPUTS_3D)

    def plot_ground_truth_env(self, robot_location):
        plt.subplot(1, 3, 3)
        info = self.ground_truth_map_info
        res, start = info.res, info.start[:2]
        img = info.cfree.any(axis=0).astype(float)
        ny, nx = img.shape
        plt.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=(0, nx * res, 0, ny * res))
        xy = self.ground_truth_node_coords + start
        plt.scatter(xy[:, 0], xy[:, 1], c=self.explored_sign, vmin=0, vmax=1, s=10, zorder=2)
        plt.plot(*(np.asarray(robot_location) + start), 'mo', markersize=10, zorder=5)
        plt.axis('off')
        plt.title('ground truth columns (yellow: explored)')
