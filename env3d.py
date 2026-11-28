"""The 3D training environment (docs/PLAN_3D.md, stage 0): env.py's counterpart for ari3d worlds.

A world is a folder made by voxel_world/scripts/make_ari3d.py (world.npz: occupied/explorable [z, y, x], start,
resolution). The drone teleports between lattice nodes, as the 2D robot does between nodes, and scans at each one
with the tilted Mid-360's band (ELEVATION_BAND, every heading, SENSOR_RANGE). The belief is a voxel grid: unknown,
free or occupied.

Frames: voxel arrays are [z, y, x]; positions are metres in the world's voxel frame (the grid corner is 0). The
(x, y) coordinates handed to the policy are relative to the start, as in env.py, so the start column is (0, 0).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from voxel_world.fast import traverse
from parameter import *

UNKNOWN3, FREE3, OCC3 = 0, 1, 2


def sensor_dirs(step_deg=SENSOR_STEP_3D, band=ELEVATION_BAND):
    """Unit ray directions over the band, every heading; fewer azimuths at steep elevations (same spacing)."""
    out = []
    for e in np.radians(np.arange(band[0], band[1] + 1e-9, step_deg)):
        n = max(8, int(np.ceil(360.0 * np.cos(e) / step_deg)))
        a = np.arange(n) * 2 * np.pi / n
        out.append(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.full(n, np.sin(e))], 1))
    return np.concatenate(out)


def segments_blocked(pa, pb, ok, res, skip_end=False, chunk=20000):
    """True for each segment pa[i] -> pb[i] (metres) that passes a voxel where `ok` [z, y, x] is False.

    Sampled every res / 2 (the endpoints' voxels included, the end voxel left out with skip_end). With free space
    grown by a voxel of clearance this cannot cut through an obstacle's corner.
    """
    out = np.zeros(len(pa), bool)
    shape = np.array(ok.shape[::-1])
    for s in range(0, len(pa), chunk):
        a, b = pa[s:s + chunk], pb[s:s + chunk]
        d = np.linalg.norm(b - a, axis=1)
        n = np.maximum(2, np.ceil(d / (0.5 * res)).astype(np.int64) + 1)
        seg = np.repeat(np.arange(len(a)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        t = (k / (n - 1)[seg])[:, None]
        q = np.floor((a[seg] + t * (b - a)[seg]) / res).astype(np.int64)
        inside = np.all((q >= 0) & (q < shape), axis=1)
        good = np.zeros(len(q), bool)
        good[inside] = ok[q[inside, 2], q[inside, 1], q[inside, 0]]
        if skip_end:
            e = np.floor(b / res).astype(np.int64)[seg]
            good |= np.all(q == e, axis=1)
        out[s:s + chunk] = np.bincount(seg[~good], minlength=len(a)) > 0
    return out


class Map3D:
    """A voxel map ([z, y, x] int8: UNKNOWN3/FREE3/OCC3) and what the node managers read from it.

    cfree       free configuration space: known free, at least CLEARANCE_3D voxels from anything not known free
    frontier    known free voxels next to unknown ones (see refresh)
    dead        frontier voxels the sensor failed to resolve (see looked_from)
    frontier_keys / frontier_pts
                the live (not dead) frontiers downsampled on FRONTIER_CELL_SIZE_3D: one key and one voxel centre
                per cell; these make utility
    all_frontier_keys
                the same over all frontiers, dead ones included; these make the reward
    robot_position
                the drone (metres), set by the environment
    """

    def __init__(self, known, res, start, name=''):
        self.map = known
        self.res = float(res)
        self.start = np.asarray(start, float)
        self.name = name
        self.shape_xyz = np.array(known.shape[::-1])
        self.robot_position = self.start.copy()
        self.cfree = np.zeros(known.shape, bool)
        self.frontier = np.zeros(known.shape, bool)
        self.looks = np.zeros(known.shape, np.uint8)
        self.dead = np.zeros(known.shape, bool)
        self.refresh()

    def refresh(self, centre=None, radius=None):
        """Recompute cfree and frontier, everywhere or in the box around centre (metres) that a scan could change."""
        if centre is None:
            lo, hi = np.zeros(3, int), self.shape_xyz.copy()
        else:
            lo = np.maximum(0, np.floor((centre - radius) / self.res).astype(int) - 2)
            hi = np.minimum(self.shape_xyz, np.ceil((centre + radius) / self.res).astype(int) + 3)
        m = max(1, CLEARANCE_3D)
        elo, ehi = np.maximum(0, lo - m), np.minimum(self.shape_xyz, hi + m)
        sub = self.map[elo[2]:ehi[2], elo[1]:ehi[1], elo[0]:ehi[0]]
        inner = (slice(lo[2] - elo[2], hi[2] - elo[2]), slice(lo[1] - elo[1], hi[1] - elo[1]),
                 slice(lo[0] - elo[0], hi[0] - elo[0]))
        box = (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        notfree = sub != FREE3
        if CLEARANCE_3D > 0:
            # outside the grid counts as blocked (border_value); inside the margin the true values are used
            notfree = ndimage.binary_dilation(notfree, structure=np.ones((3, 3, 3), bool), iterations=CLEARANCE_3D,
                                              border_value=1)
        self.cfree[box] = ~notfree[inner]
        # a frontier voxel has an unknown face neighbour and, like 2D's 1 < n < 8 of 8, at least
        # FRONTIER_MIN_UNKNOWN_3D unknown voxels among its 26 neighbours (outside the grid: none). A front has 9;
        # an unseen surface voxel or the solid edge where a wall meets the floor or ceiling (only touching free
        # space along an edge, so no ray reaches it) fail one of the two.
        unk = sub == UNKNOWN3
        face = np.zeros_like(unk)
        face[1:] |= unk[:-1]; face[:-1] |= unk[1:]
        face[:, 1:] |= unk[:, :-1]; face[:, :-1] |= unk[:, 1:]
        face[:, :, 1:] |= unk[:, :, :-1]; face[:, :, :-1] |= unk[:, :, 1:]
        n = ndimage.convolve(unk.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode='constant') - unk
        front = (sub == FREE3) & face & (n >= FRONTIER_MIN_UNKNOWN_3D) & (n < 26)
        self.frontier[box] = front[inner]
        self.all_frontier_keys, _ = self._downsample(self.frontier)
        self.frontier_keys, self.frontier_pts = self._downsample(self.frontier & ~self.dead)

    def _downsample(self, mask):
        """One key and one voxel centre (metres) per FRONTIER_CELL_SIZE_3D cell holding a voxel of mask."""
        fz, fy, fx = np.nonzero(mask)
        c = int(round(FRONTIER_CELL_SIZE_3D / self.res))
        keys = (fz // c).astype(np.int64) * 1_000_000 + (fy // c) * 1000 + (fx // c)
        keys, first = np.unique(keys, return_index=True)
        return keys, (np.stack([fx[first], fy[first], fz[first]], 1) + 0.5) * self.res

    def looked_from(self, p):
        """After a scan from p: frontier voxels p had in sight (within UTILITY_RANGE, in the band, line of sight
        through known free) that are still frontiers count one more look. At DEAD_FRONTIER_VIEWS looks a voxel is
        dead: the sensor cannot resolve it (an unseen sliver of wall in a corner, at a grazing angle), so it no
        longer counts for utility and episodes can end. It still counts for the reward if it is ever resolved."""
        fz, fy, fx = np.nonzero(self.frontier & ~self.dead)
        if len(fz) == 0:
            return
        pts = (np.stack([fx, fy, fz], 1) + 0.5) * self.res
        v = pts - p
        d = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        el = np.degrees(np.arcsin(np.clip(v[:, 2] / d, -1, 1)))
        cand = np.nonzero((d <= UTILITY_RANGE) & (el >= ELEVATION_BAND[0]) & (el <= ELEVATION_BAND[1]))[0]
        if len(cand) == 0:
            return
        blocked = segments_blocked(np.broadcast_to(p, (len(cand), 3)), pts[cand], self.map == FREE3, self.res,
                                   skip_end=True)
        seen = cand[~blocked]
        z, y, x = fz[seen], fy[seen], fx[seen]
        self.looks[z, y, x] += 1
        self.dead[z, y, x] |= self.looks[z, y, x] >= DEAD_FRONTIER_VIEWS
        self.frontier_keys, self.frontier_pts = self._downsample(self.frontier & ~self.dead)

    def local_xy(self, pos):
        """(x, y) relative to the start, rounded like the 2D node coordinates."""
        return np.round(np.asarray(pos)[..., :2] - self.start[:2], 1)


class Env3D:
    def __init__(self, episode_index, plot=False, world=None):
        self.episode_index = episode_index
        self.plot = plot
        self.world = world or self.pick_world(episode_index)
        z = np.load(os.path.join(self.world, 'world.npz'))
        self.ground_truth = z['occupied']
        self.explorable = z['explorable']
        self.surface = z['surface']  # occupied voxels next to explorable space
        self.surface_total = max(1, self.surface.sum())
        self.cell_size = float(z['resolution'])
        assert abs(self.cell_size - CELL_SIZE) < 1e-9, 'world resolution %s != CELL_SIZE' % self.cell_size
        self.start = z['start'].astype(float)

        self.ground_truth_info = Map3D(np.where(self.ground_truth, OCC3, FREE3).astype(np.int8), self.cell_size,
                                       self.start, name=self.world)
        self.belief_info = Map3D(np.zeros(self.ground_truth.shape, np.int8), self.cell_size, self.start)
        self.robot_belief = self.belief_info.map
        self.dirs = sensor_dirs()
        self.sensor_range = SENSOR_RANGE
        self.explorable_total = self.explorable.sum()
        self.footprint = self.explorable.any(axis=0)
        self.footprint_total = self.footprint.sum()

        self.robot_position = self.start.copy()  # metres, 3D
        self.robot_location = self.belief_info.local_xy(self.robot_position)  # (x, y) of the column, policy frame
        self.travel_dist = 0
        self.explored_rate = 0
        self.footprint_rate = 0
        self.surface_rate = 0
        self.left = {}                 # left-behind frontier key -> hop it was left at (parameter.py LEAVE_W)
        self.leave_delta = 0
        self.leave_cells = self.late_cells = self.late_hops = self.hops = 0
        self.late_m = 0.0
        self.scan()
        self.global_frontiers = self.belief_info.all_frontier_keys
        self.evaluate_exploration_rate()

        if self.plot:
            self.frame_files = []
            self.trajectory = [self.robot_position.copy()]

    @staticmethod
    def pick_world(episode_index):
        """World episode_index mod n of WORLDS_3D: the sub-folders listed in its worlds.txt (one per line), else
        every sub-folder with a world.npz, sorted."""
        listing = os.path.join(WORLDS_3D, 'worlds.txt')
        if os.path.exists(listing):
            worlds = [l.strip() for l in open(listing) if l.strip() and not l.startswith('#')]
        else:
            worlds = sorted(d for d in os.listdir(WORLDS_3D) if os.path.exists(os.path.join(WORLDS_3D, d, 'world.npz')))
        assert worlds, 'no worlds in %s' % WORLDS_3D
        return os.path.join(WORLDS_3D, worlds[episode_index % len(worlds)])

    def scan(self):
        p = self.robot_position
        known = self.belief_info.map
        ray, x, y, z = traverse(p, p + self.dirs * self.sensor_range, np.zeros(3), self.cell_size,
                                self.belief_info.shape_xyz)
        e = np.arange(len(ray))
        occ = self.ground_truth[z, y, x]
        # traverse lists the voxels step by step, so a ray's first occupied voxel is its first occurrence
        hit = np.full(len(self.dirs), np.iinfo(np.int64).max)
        r, first = np.unique(ray[occ], return_index=True)
        hit[r] = e[occ][first]
        keep = e <= hit[ray]
        known[z[keep & ~occ], y[keep & ~occ], x[keep & ~occ]] = FREE3
        known[z[keep & occ], y[keep & occ], x[keep & occ]] = OCC3
        # the drone's own clearance box is known (the band never sees straight down)
        i = np.floor(p / self.cell_size).astype(int)
        c = max(1, CLEARANCE_3D)
        box = tuple(slice(max(0, i[k] - c), i[k] + c + 1) for k in (2, 1, 0))
        known[box] = np.where(self.ground_truth[box], OCC3, FREE3)
        self.belief_info.robot_position = p.copy()
        self.belief_info.refresh(p, self.sensor_range)
        self.belief_info.looked_from(p)

    def calculate_reward(self, dist):
        reward = -dist / UPDATING_MAP_SIZE * 5 * DIST_W
        new = self.belief_info.all_frontier_keys
        delta_num = len(np.setdiff1d(self.global_frontiers, new, assume_unique=True))
        self.frontier_delta = delta_num
        per_cell = REWARD_SCALE_3D / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        reward += delta_num * per_cell
        if LEAVE_W > 0:
            reward -= LEAVE_W * self.leave_delta * per_cell
        self.global_frontiers = new
        return reward

    def track_left_behind(self, prev, dist):
        """After a hop from prev (metres) of dist metres and its scan: count the frontier left behind (parameter.py
        LEAVE_W), and the earlier left-behind cells this hop resolved (a cell that turned dead is not resolved)."""
        bi = self.belief_info
        if self.left:
            lk = np.fromiter(self.left.keys(), np.int64, len(self.left))
            gone = lk[~np.isin(lk, bi.all_frontier_keys)]
            if len(gone):
                self.late_cells += len(gone)
                self.late_hops += 1
                self.late_m += float(dist)
                for k in gone.tolist():
                    del self.left[k]
        self.leave_delta = 0
        pts = bi.frontier_pts
        if len(pts):
            was = np.linalg.norm(pts - prev, axis=1) <= LEAVE_R
            now = np.linalg.norm(pts - self.robot_position, axis=1) > LEAVE_R
            ab = bi.frontier_keys[was & now]
            self.leave_delta = len(ab)
            self.leave_cells += len(ab)
            for k in ab.tolist():
                self.left.setdefault(k, self.hops)
        self.hops += 1

    def evaluate_exploration_rate(self):
        free = self.belief_info.map == FREE3
        self.explored_rate = float((free & self.explorable).sum() / self.explorable_total)
        self.footprint_rate = float((free.any(axis=0) & self.footprint).sum() / self.footprint_total)
        self.surface_rate = float(((self.belief_info.map == OCC3) & self.surface).sum() / self.surface_total)

    def step(self, next_position, dist):
        """Fly to next_position (metres, a lattice node) along a route of length dist, scan, return the reward."""
        prev = self.robot_position.copy()
        self.robot_position = np.asarray(next_position, float).copy()
        self.robot_location = self.belief_info.local_xy(self.robot_position)
        self.scan()
        self.track_left_behind(prev, dist)
        self.travel_dist += dist
        self.evaluate_exploration_rate()
        if self.plot:
            self.trajectory.append(self.robot_position.copy())
        return self.calculate_reward(dist)

    # -- plotting: top view of the belief (panel 1); panels 2 and 3 come from the agent and the ground truth
    def plot_env(self, step):
        res = self.cell_size
        known = self.belief_info.map
        img = np.full(known.shape[1:], 0.5)
        img[(known == OCC3).any(axis=0)] = 0.0
        img[(known == FREE3).any(axis=0)] = 1.0
        ny, nx = img.shape
        ext = (0, nx * res, 0, ny * res)
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=ext)
        t = np.array(self.trajectory)
        plt.scatter(t[:, 0], t[:, 1], c=t[:, 2], cmap='viridis', s=8, zorder=4)
        plt.plot(t[:, 0], t[:, 1], 'b', linewidth=1.5, zorder=3)
        plt.plot(*self.robot_position[:2], 'mo', markersize=6, zorder=5)
        plt.axis('off')
        plt.title('belief (top), path coloured by height')
        plt.suptitle('Explored volume: {:.4g}  footprint: {:.4g}  Travel distance: {:.4g}  height {:.1f} m'.format(
            self.explored_rate, self.footprint_rate, self.travel_dist, self.robot_position[2]))
        plt.tight_layout()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.savefig(frame, dpi=110)
        plt.close()
        self.frame_files.append(frame)
