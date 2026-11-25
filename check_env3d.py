#!/usr/bin/env python3
"""Stage-0 check of the 3D environment (docs/PLAN_3D.md): run a checkpoint through worker.Worker in the 2D env on
map PNGs and in the 3D env on ari3d worlds, and record what the plan compares.

  ENV_DIM=2 python check_env3d.py --ckpt model/x/checkpoint.pth --maps maps/img_1609.png ... --out c2.json
  ENV_DIM=3 python check_env3d.py --ckpt model/x/checkpoint.pth --worlds worlds_flat/img_1609_s0 ... --out c3.json
  python check_env3d.py --compare c2.json c3.json
  python check_env3d.py --ckpt ... --replay maps/img_1609.png worlds_flat/img_1609_s0 [png world ...]
  python check_env3d.py --greedy --worlds ... --out g.json

--greedy runs no policy: it flies to the column with the most utility per metre of route (any column, not only
neighbours), until no utility is left or --max-steps. It checks that 3D episodes end and how much they cover.

--replay checks the environment without the policy in the loop: it runs the checkpoint in 2D, then flies the same
sequence of columns in the 3D env, and compares them step by step: explored share (2D area vs 3D footprint), number
of nodes, the current node's neighbours, distance.

Per episode: steps, done, distance, explored share (2D: free area; 3D: volume and footprint), the frontier part of
the reward (summed; per step), utility seen by the policy (mean over nodes with any, on the 2D scale), and seconds.
Actions are greedy (argmax) unless --sample, so 2D and 3D runs are comparable. voxel_world/src must be on
PYTHONPATH for ENV_DIM=3.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def run(a):
    import torch
    from parameter import ENV_DIM, UPDATING_MAP_SIZE, NODE_INPUT_DIM, EMBEDDING_DIM
    import worker as W
    from model import PolicyNet

    net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=False)['policy_model'])
    net.eval()
    torch.manual_seed(0)
    np.random.seed(0)

    if not a.sample:
        def greedy(self, observation):
            _, _, _, _, current_edge, _ = observation
            with torch.no_grad():
                logp = self.policy_net(*observation)
            action_index = torch.argmax(logp, dim=1).long()
            return self.node_coords[current_edge[0, action_index.item(), 0].item()], action_index
        W.Agent.select_next_waypoint = greedy

    items = a.worlds if ENV_DIM == 3 else a.maps
    results = []
    for item in items:
        if ENV_DIM == 3:
            W.Env = (lambda path: (lambda idx, plot=False: _Env3D(idx, plot=plot, world=path)))(item)
        else:
            W.Env = (lambda path: (lambda idx, plot=False: _env2d(path, idx, plot)))(item)
        tic = time.time()
        wk = W.Worker(0, net, 0, save_image=a.gif)
        env = wk.env
        fr, steps = [], []
        orig = env.calculate_reward

        curve = [(0.0, float(env.explored_rate), float(getattr(env, 'footprint_rate', env.explored_rate)))]

        def calc(dist, orig=orig):
            r = orig(dist)
            fr.append(r + dist / UPDATING_MAP_SIZE * 5)
            env.evaluate_exploration_rate()
            curve.append((float(env.travel_dist), float(env.explored_rate),
                          float(getattr(env, 'footprint_rate', env.explored_rate))))
            steps.append(time.time())
            return r
        env.calculate_reward = calc
        utils = []
        orig_obs = wk.robot.get_observation

        def obs(orig_obs=orig_obs):
            u = np.asarray(wk.robot.utility, float)
            if (u > 0).any():
                utils.append(float(u[u > 0].mean()))
            return orig_obs()
        wk.robot.get_observation = obs
        wk.run_episode()
        rec = dict(item=os.path.basename(item.rstrip('/')), dim=ENV_DIM, steps=len(fr),
                   done=bool(wk.perf_metrics['success_rate']), distance=round(float(env.travel_dist), 1),
                   explored=round(float(env.explored_rate), 4), frontier_reward=round(float(np.sum(fr)), 3),
                   frontier_reward_per_step=round(float(np.mean(fr)), 4) if fr else 0.0,
                   mean_utility=round(float(np.mean(utils)), 2) if utils else 0.0,
                   seconds=round(time.time() - tic, 1),
                   nodes=len(wk.robot.node_manager.nodes_dict))
        c = np.array(curve)
        for q in (0.9, 0.95, 0.99):  # distance at which the (footprint, in 3D) explored share first reaches q
            hit = np.nonzero(c[:, 2] >= q)[0]
            rec['dist_%d' % round(q * 100)] = round(float(c[hit[0], 0]), 1) if len(hit) else None
        rec['curve'] = [[round(v, 4) for v in row] for row in curve]
        if ENV_DIM == 3:
            rec['footprint'] = round(float(env.footprint_rate), 4)
            rec['height_range'] = [round(float(v), 1) for v in (np.min(env.heights), np.max(env.heights))]
            rec['frontier_cells_seen'] = int(np.sum(env.deltas))
        else:
            rec['frontier_cells_seen'] = None
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    with open(a.out, 'w') as f:
        json.dump(results, f, indent=1)


def _env2d(path, idx, plot):
    from env import Env
    from skimage import io
    from skimage.measure import block_reduce

    class One(Env):
        def import_ground_truth(self, episode_index):
            g = (io.imread(path, 1) * 255).astype(int)
            g = block_reduce(g, 2, np.min)
            rc = np.nonzero(g == 208)
            rc = np.array([np.array(rc)[1, 10], np.array(rc)[0, 10]])
            g = (g > 150) | ((g <= 80) & (g >= 50))
            return g * 254 + 1, rc
    return One(idx, plot=plot)


def _Env3D(idx, plot, world):
    from env3d import Env3D

    class Logged(Env3D):
        def step(self, next_position, dist):
            r = super().step(next_position, dist)
            self.heights.append(self.robot_position[2])
            self.deltas.append(self.frontier_delta)
            return r
    e = Logged.__new__(Logged)
    e.heights, e.deltas = [], []
    Logged.__init__(e, idx, plot=plot, world=world)
    e.heights.append(e.robot_position[2])
    return e


def replay(a):
    import torch
    from parameter import NODE_INPUT_DIM, EMBEDDING_DIM, UPDATING_MAP_SIZE, UTILITY_SCALE_3D, REWARD_SCALE_3D
    from model import PolicyNet
    from agent import Agent
    from agent3d import Agent3D
    from env3d import Env3D

    net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=False)['policy_model'])
    net.eval()
    out = []
    for png, world in zip(a.replay[::2], a.replay[1::2]):
        env, ag = _env2d(png, 0, False), Agent(net)
        env.evaluate_exploration_rate()
        ag.update_planning_state(env.belief_info, env.robot_location)
        path, rec2, fr2, u2 = [], [], [], []
        for _ in range(128):
            nbrs = {tuple(ag.node_coords[i]) for i in ag.neighbor_indices}
            rec2.append((env.travel_dist, env.explored_rate, len(ag.node_coords), nbrs))
            u2.append({tuple(c): float(u) for c, u in zip(ag.node_coords, ag.utility)})
            obs = ag.get_observation()
            with torch.no_grad():
                act = torch.argmax(net(*obs), dim=1).item()
            nxt = ag.node_coords[obs[4][0, act, 0].item()]
            path.append(tuple(nxt))
            d = np.linalg.norm(env.robot_location - nxt)
            fr2.append(env.step(nxt) + d / UPDATING_MAP_SIZE * 5)
            ag.update_planning_state(env.belief_info, env.robot_location)
            if ag.utility.sum() == 0:
                break
        rec2.append((env.travel_dist, env.explored_rate, len(ag.node_coords), set()))
        e3, ag3 = Env3D(0, world=world), Agent3D(net)
        ag3.update_planning_state(e3.belief_info, e3.robot_location)
        rec3, missing, hop, fr3, u3 = [], 0, 0, [], []
        for xy in path:
            nbrs = {tuple(ag3.node_coords[i]) for i in ag3.neighbor_indices}
            rec3.append((e3.travel_dist, e3.footprint_rate, len(ag3.node_coords), nbrs))
            u3.append({tuple(c): float(u) / UTILITY_SCALE_3D for c, u in zip(ag3.node_coords, ag3.utility)})
            if ag3.node_manager.nodes_dict.find(xy) is None:
                missing += 1
                continue
            hop += xy not in nbrs
            pos, d = ag3.route_to(xy)
            fr3.append((e3.step(pos, d) + d / UPDATING_MAP_SIZE * 5) / REWARD_SCALE_3D)
            ag3.update_planning_state(e3.belief_info, e3.robot_location)
        rec3.append((e3.travel_dist, e3.footprint_rate, len(ag3.node_coords), set()))
        n = min(len(rec2), len(rec3))
        jac = [len(rec2[i][3] & rec3[i][3]) / max(1, len(rec2[i][3] | rec3[i][3])) for i in range(n - 1)]
        r = dict(map=os.path.basename(png), steps=len(path), missing_columns=missing, not_neighbours=hop,
                 distance_2d=round(rec2[-1][0], 1), distance_3d=round(rec3[-1][0], 1),
                 explored_2d=round(rec2[-1][1], 4), footprint_3d=round(rec3[-1][1], 4),
                 max_explored_gap=round(max(abs(rec2[i][1] - rec3[i][1]) for i in range(n)), 4),
                 mean_explored_gap=round(float(np.mean([rec3[i][1] - rec2[i][1] for i in range(n)])), 4),
                 nodes_2d=rec2[n - 1][2], nodes_3d=rec3[n - 1][2], neighbour_jaccard=round(float(np.mean(jac)), 3),
                 # unscaled 3D over 2D: the frontier part of the reward (summed), and the utility of the nodes that
                 # have some in both (mean over steps of the mean ratio; also the ratio of the sums)
                 reward_ratio=round(float(np.sum(fr3) / np.sum(fr2)), 3),
                 utility_ratio=round(float(np.mean([np.mean([a3[k] / a2[k] for k in a2 if a2[k] > 0 and a3.get(k, 0) > 0])
                                                    for a2, a3 in zip(u2, u3)
                                                    if any(a2[k] > 0 and a3.get(k, 0) > 0 for k in a2)])), 3),
                 utility_sum_ratio=round(float(sum(sum(a.values()) for a in u3) / sum(sum(a.values()) for a in u2)), 3),
                 utility_nonzero_2d=round(float(np.mean([sum(v > 0 for v in a.values()) for a in u2])), 1),
                 utility_nonzero_3d=round(float(np.mean([sum(v > 0 for v in a.values()) for a in u3])), 1),
                 curve=[[round(rec2[i][0], 1), round(rec2[i][1], 4), round(rec3[i][0], 1), round(rec3[i][1], 4)]
                        for i in range(n)])
        out.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'curve'}), flush=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)


def greedy(a):
    from parameter import NODE_RESOLUTION
    from agent3d import Agent3D
    from env3d import Env3D

    out = []
    for world in a.worlds:
        tic = time.time()
        e, ag = Env3D(0, world=world), Agent3D(None)
        ag.update_planning_state(e.belief_info, e.robot_location)
        nm, steps = ag.node_manager, 0
        while steps < a.max_steps and ag.utility.sum() > 0:
            d = np.array([nm.dist[nm.nodes_dict.find(c.tolist()).data.kept] for c in ag.node_coords])
            score = np.where(ag.utility > 0, ag.utility / (d + NODE_RESOLUTION), -1)
            score[ag.current_index] = -1
            e.step(*ag.route_to(ag.node_coords[int(np.argmax(score))]))
            ag.update_planning_state(e.belief_info, e.robot_location)
            steps += 1
        r = dict(item=os.path.basename(world.rstrip('/')), steps=steps, done=bool(ag.utility.sum() == 0),
                 distance=round(float(e.travel_dist), 1), explored=round(e.explored_rate, 4),
                 footprint=round(e.footprint_rate, 4), seconds=round(time.time() - tic, 1),
                 seconds_per_step=round((time.time() - tic) / max(1, steps), 3), nodes=len(nm.nodes_dict),
                 dead_frontier_voxels=int(e.belief_info.dead.sum()))
        out.append(r)
        print(json.dumps(r), flush=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)


def compare(paths):
    rows = {}
    for p in paths:
        for r in json.load(open(p)):
            rows.setdefault(r['item'].split('_s')[0].replace('.png', ''), {})[r['dim']] = r
    keys = ['steps', 'done', 'distance', 'dist_90', 'dist_95', 'dist_99', 'explored', 'frontier_reward', 'mean_utility',
            'seconds']
    print('%-12s ' % 'map' + ' '.join('%17s' % k for k in keys) + '   footprint(3D)')
    for name, d in sorted(rows.items()):
        cell = lambda k: '/'.join(str(d[dim][k]) if dim in d else '-' for dim in (2, 3))
        print('%-12s ' % name + ' '.join('%17s' % cell(k) for k in keys) + '   %s' % d.get(3, {}).get('footprint', '-'))
    both = [d for d in rows.values() if 2 in d and 3 in d]
    if both:
        for k in ('distance', 'dist_90', 'dist_95', 'dist_99', 'explored', 'frontier_reward', 'mean_utility', 'seconds'):
            r = [d[3][k] / d[2][k] for d in both if d[2][k] and d[3][k]]
            print('3D / 2D %-16s mean %.3f  (%s)' % (k, np.mean(r), ', '.join('%.2f' % v for v in r)))
        r = [d[3]['footprint'] / d[2]['explored'] for d in both]
        print('3D footprint / 2D explored  mean %.3f' % np.mean(r))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt')
    ap.add_argument('--maps', nargs='*', default=[])
    ap.add_argument('--worlds', nargs='*', default=[])
    ap.add_argument('--out', default='check_env3d.json')
    ap.add_argument('--sample', action='store_true', help='sample actions as in training instead of argmax')
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--compare', nargs='+')
    ap.add_argument('--replay', nargs='+', help='png world [png world ...]')
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--max-steps', type=int, default=300)
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    elif a.greedy:
        greedy(a)
    elif a.replay:
        replay(a)
    else:
        run(a)
