"""The 3D node graph: one node per column ("collapse", voxel_world/docs/column_graph.md), node_manager.py's
counterpart for env3d.py.

Lattice    nodes every NODE_RESOLUTION in x and y and every Z_RESOLUTION in z, aligned with the start, on free
           configuration space (Map3D.cfree). Two lattice nodes are joined when they are at most 2 columns apart
           and Z_NEIGHBOR heights apart and the straight segment between them is free. Routes are planned on it.
Column     the lattice nodes with the same (x, y). The policy sees one node per column, keyed by (x, y) relative to
           the start, exactly like a 2D node.
Kept node  per column, after each scan: the reachable lattice node with the most utility; if none has any, the
           one nearest the drone's height (transit). The drone's own column keeps the drone's node.
Edges      two columns are neighbours when a lattice edge joins one's reachable node to the other's. With the
           2-column reach that is at most 24 neighbours, so K_SIZE = 25 holds as in 2D.
Utility    of a lattice node: downsampled frontiers within UTILITY_RANGE, inside ELEVATION_BAND as seen from the
           node, with line of sight through known free voxels (2D's check_collision: unknown blocks). At most
           MIN_UTILITY (after UTILITY_SCALE_3D) counts as 0; a visited lattice node keeps 0. Frontier voxels the
           sensor failed to resolve no longer count (env3d.Map3D.looked_from).

nodes_dict behaves like the 2D quad tree for what the agent, the worker and the ground-truth critic use:
find((x, y)) -> item with .data (coords, utility, visited, neighbor_set), iteration, len().
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from env3d import segments_blocked, FREE3
from parameter import *


class ColumnDict:
    class Item:
        __slots__ = ('data',)

        def __init__(self, data):
            self.data = data

    def __init__(self):
        self.items = {}

    def find(self, key):
        return self.items.get((float(key[0]), float(key[1])))

    def insert(self, point, data):
        self.items[(float(point[0]), float(point[1]))] = self.Item(data)

    def __iter__(self):
        return iter(self.items.values())

    def __len__(self):
        return len(self.items)


class ColumnNode:
    def __init__(self, coords, column):
        self.coords = coords  # (x, y) relative to the start
        self.column = column
        self.utility = 0  # on the 2D scale (UTILITY_SCALE_3D applied)
        self.raw_utility = 0  # 3D frontier count of the kept node
        self.visited = 0
        self.kept = -1  # lattice index of the kept node
        self.height = 0.0  # metres, the kept node's z
        self.gaps = 0  # separate runs of free configuration space in the column (belief)
        self.free_height = 0.0  # metres of free configuration space in the column (belief)
        self.neighbor_set = {(coords[0], coords[1])}


_LATTICES = {}


class Lattice:
    """The geometry of the lattice for one grid shape and start: positions, columns and candidate edges."""

    @classmethod
    def get(cls, shape_zyx, res, start):
        key = (tuple(shape_zyx), res, tuple(np.round(start, 3)))
        if key not in _LATTICES:
            _LATTICES.clear()  # one world at a time per worker process
            _LATTICES[key] = cls(shape_zyx, res, start)
        return _LATTICES[key]

    def __init__(self, shape_zyx, res, start):
        nz, ny, nx = shape_zyx
        self.res = res
        sx, sz = int(round(NODE_RESOLUTION / res)), int(round(Z_RESOLUTION / res))
        s = np.floor(np.asarray(start) / res).astype(int)
        xs, ys, zs = np.arange(s[0] % sx, nx, sx), np.arange(s[1] % sx, ny, sx), np.arange(s[2] % sz, nz, sz)
        self.step = np.array([sx, sx, sz])
        self.first = np.array([xs[0], ys[0], zs[0]])
        self.dims = np.array([len(xs), len(ys), len(zs)])
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
        self.vox = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        self.pos = (self.vox + 0.5) * res
        ncol = len(xs) * len(ys)
        self.col = np.arange(len(self.vox)) % ncol
        self.ncol = ncol
        self.col_xy = np.round(self.pos[:ncol, :2] - np.asarray(start)[:2], 1)
        self.col_key = {(float(x), float(y)): c for c, (x, y) in enumerate(self.col_xy)}
        # candidate edges: half of the offsets (|dx|, |dy| <= 2 columns, |dz| <= Z_NEIGHBOR heights)
        nxl, nyl, nzl = self.dims
        ids = np.arange(len(self.vox)).reshape(nzl, nyl, nxl)
        pairs = []
        for dz in range(-Z_NEIGHBOR, Z_NEIGHBOR + 1):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if (dz, dy, dx) <= (0, 0, 0):
                        continue
                    a = ids[max(0, -dz):nzl - max(0, dz), max(0, -dy):nyl - max(0, dy), max(0, -dx):nxl - max(0, dx)]
                    b = ids[max(0, dz):nzl + min(0, dz), max(0, dy):nyl + min(0, dy), max(0, dx):nxl + min(0, dx)]
                    pairs.append(np.stack([a.ravel(), b.ravel()], 1))
        self.pairs = np.concatenate(pairs).astype(np.int32)
        self.length = np.linalg.norm(self.pos[self.pairs[:, 0]] - self.pos[self.pairs[:, 1]], axis=1)
        self.reach = float(self.length.max())

    def index_of(self, p):
        v = np.floor(np.asarray(p) / self.res).astype(int)
        q, r = np.divmod(v - self.first, self.step)
        assert not r.any() and np.all((q >= 0) & (q < self.dims)), 'not a lattice node: %s' % (p,)
        return int((q[2] * self.dims[1] + q[1]) * self.dims[0] + q[0])

    def valid(self, cfree):
        return cfree[self.vox[:, 2], self.vox[:, 1], self.vox[:, 0]]

    def column_profile(self, cfree):
        """Per column (through its centre voxel): separate runs of cfree, and their total height in metres."""
        c = cfree[:, self.vox[:self.ncol, 1], self.vox[:self.ncol, 0]]  # (nz, ncol)
        starts = c.copy()
        starts[1:] &= ~c[:-1]
        return starts.sum(axis=0), c.sum(axis=0) * self.res

    def segments_free(self, idx, cfree):
        a, b = self.pairs[idx, 0], self.pairs[idx, 1]
        return ~segments_blocked(self.pos[a], self.pos[b], cfree, self.res)


def utility(lat, nodes, map_info, util_range=UTILITY_RANGE):
    """Live 3D frontier cells (Map3D.frontier_pts) each lattice node in `nodes` can see, before the MIN_UTILITY cut."""
    out = np.zeros(len(nodes), np.int64)
    pts = map_info.frontier_pts
    if len(pts) == 0 or len(nodes) == 0:
        return out
    tree = cKDTree(pts)
    lists = tree.query_ball_point(lat.pos[nodes], util_range)
    counts = np.array([len(l) for l in lists])
    if counts.sum() == 0:
        return out
    ni = np.repeat(np.arange(len(nodes)), counts)
    fi = np.concatenate([np.asarray(l, np.int64) for l in lists if len(l)])
    v = pts[fi] - lat.pos[nodes][ni]
    d = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
    el = np.degrees(np.arcsin(np.clip(v[:, 2] / d, -1, 1)))
    inb = (el >= ELEVATION_BAND[0]) & (el <= ELEVATION_BAND[1])
    ni, fi = ni[inb], fi[inb]
    blocked = segments_blocked(lat.pos[nodes][ni], pts[fi], map_info.map == FREE3, lat.res, skip_end=True)
    return np.bincount(ni[~blocked], minlength=len(nodes))


class NodeManager3D:
    def __init__(self, plot=False):
        self.nodes_dict = ColumnDict()
        self.plot = plot
        self.lat = None
        self.frontier = None

    def _init(self, map_info):
        self.lat = Lattice.get(map_info.map.shape, map_info.res, map_info.start)
        n = len(self.lat.vox)
        self.edge_ok = np.zeros(len(self.lat.pairs), bool)
        self.util = np.zeros(n, np.int64)
        self.visited = np.zeros(n, bool)
        self.was_reachable = np.zeros(n, bool)
        self.kept = np.full(self.lat.ncol, -1)

    def update_graph(self, map_info):
        if self.lat is None:
            self._init(map_info)
        lat, p = self.lat, map_info.robot_position
        here = lat.index_of(p)
        self.here = here
        self.visited[here] = True
        valid = lat.valid(map_info.cfree)
        assert valid[here], 'the drone is not in free configuration space'
        self.valid = valid

        # lattice edges: free space only grows, so an edge once free stays free; recheck the rest near the scan
        a, b = lat.pairs[:, 0], lat.pairs[:, 1]
        near = np.linalg.norm(lat.pos - p, axis=1) <= SENSOR_RANGE + lat.reach + 2 * map_info.res
        todo = np.nonzero(~self.edge_ok & valid[a] & valid[b] & (near[a] | near[b]))[0]
        if len(todo):
            self.edge_ok[todo] = lat.segments_free(todo, map_info.cfree)
        m = self.edge_ok & valid[a] & valid[b]
        ea, eb, w = a[m], b[m], lat.length[m]
        n = len(lat.vox)
        graph = csr_matrix((np.concatenate([w, w]), (np.concatenate([ea, eb]), np.concatenate([eb, ea]))), shape=(n, n))
        self.dist, self.pred = dijkstra(graph, indices=here, return_predecessors=True)
        reach = np.isfinite(self.dist)
        self.reachable = reach

        # utility, recomputed where a scan here can have changed it (farther nodes keep theirs)
        # (and wherever a node just became reachable)
        near_u = np.linalg.norm(lat.pos - p, axis=1) <= UTILITY_RANGE + SENSOR_RANGE + 2 * map_info.res
        upd = np.nonzero(reach & (near_u | ~self.was_reachable))[0]
        self.was_reachable = reach
        self.util[~reach] = 0
        self.util[upd] = utility(lat, upd, map_info)
        self.util[self.util * UTILITY_SCALE_3D <= MIN_UTILITY] = 0  # on the 2D scale, as in 2D
        self.util[self.visited] = 0

        # kept node per column: most utility, then nearest the drone's height
        r = np.nonzero(reach)[0]
        cols = lat.col[r]
        order = np.lexsort((np.abs(lat.pos[r, 2] - p[2]), -self.util[r], cols))
        r, cols = r[order], cols[order]
        first = np.ones(len(r), bool)
        first[1:] = cols[1:] != cols[:-1]
        self.kept[:] = -1
        self.kept[cols[first]] = r[first]
        self.kept[lat.col[here]] = here

        # columns and their neighbours
        rm = reach[ea] & reach[eb]
        ca, cb = lat.col[ea[rm]], lat.col[eb[rm]]
        x = ca != cb
        cp = np.unique(np.sort(np.stack([ca[x], cb[x]], 1), axis=1), axis=0)
        gaps, free_height = lat.column_profile(map_info.cfree)
        for c in cols[first]:
            key = (float(lat.col_xy[c, 0]), float(lat.col_xy[c, 1]))
            item = self.nodes_dict.find(key)
            if item is None:
                self.nodes_dict.insert(key, ColumnNode(lat.col_xy[c].copy(), int(c)))
                item = self.nodes_dict.find(key)
            node = item.data
            node.kept = int(self.kept[c])
            node.raw_utility = int(self.util[node.kept])
            node.utility = node.raw_utility * UTILITY_SCALE_3D
            node.height = float(lat.pos[node.kept, 2])
            node.gaps, node.free_height = int(gaps[c]), float(free_height[c])
            node.neighbor_set = {key}
        self.nodes_dict.find(tuple(lat.col_xy[lat.col[here]])).data.visited = 1
        for c1, c2 in cp:
            n1 = self.nodes_dict.find(tuple(lat.col_xy[c1])).data
            n2 = self.nodes_dict.find(tuple(lat.col_xy[c2])).data
            n1.neighbor_set.add((float(n2.coords[0]), float(n2.coords[1])))
            n2.neighbor_set.add((float(n1.coords[0]), float(n1.coords[1])))

    def route_to(self, xy):
        """The kept node of column (x, y): its position (metres), the route length and the route (lattice indices)."""
        c = self.lat.col_key[(float(xy[0]), float(xy[1]))]
        k = int(self.kept[c])
        assert k >= 0 and np.isfinite(self.dist[k])
        path = [k]
        while path[-1] != self.here:
            path.append(int(self.pred[path[-1]]))
        return self.lat.pos[k].copy(), float(self.dist[k]), path[::-1]

    def column_heights(self):
        """Kept node height (metres) per column, for plots."""
        return {c: self.lat.pos[k, 2] for c, k in enumerate(self.kept) if k >= 0}
