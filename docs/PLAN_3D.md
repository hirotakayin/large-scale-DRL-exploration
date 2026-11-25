# Plan: from ARiADNE 2D to a learned 3D exploration planner

Status 2026-11-24: stage 0 done (results below); stage 1 next. Owner: the 3D_learning project
(see the workspace CLAUDE.md). Branch 3d-learning of this repo (training code) and of
hirotakayin/CERLAB-UAV-Autonomy (`voxel_world/`: worlds, simulator, studies).

## What already exists

| Piece | Where | State |
|---|---|---|
| 3D training worlds from ARiADNE's map PNGs | `voxel_world/scripts/make_ari3d.py`, module `ari3d.py` | done. Terraced levels, staggered levels joined by shafts, split corridors (over/under loops), pits, low and lipped islands, hurdle/lintel gates; no single height reaches more than 60% of a world; every opening at least 3.2 m, checked with 0.5 m clearance; `accepted` flag in stats.json. Deterministic from (PNG, seed). Runs in this repo's Docker image. Cloned on cerlab22 at ~/rl/CERLAB-UAV-Autonomy |
| Generating and previewing worlds from the GUI | hirotakayin/cerlab-uav-gui, Training tab | done. Calls make_ari3d.py and make_courses.py, shows previews and stats, and starts/stops training here over SSH |
| Evaluation courses (open rooms, obstacles at every height) | `voxel_world/scripts/make_courses.py --layout room`, test seeds 0–99 | done |
| A 3D exploration simulator (teleport between nodes, tilted Mid-360 band, 3D frontiers, utility, graphs) | `voxel_world/scripts/prune_study.py` (class `Sim`) | prototype, used for the pruning study; ported into this repo in stage 0 |
| The 3D training environment | this repo: `env3d.py`, `agent3d.py`, `node_manager3d.py`, `ground_truth_node_manager3d.py`, `ENV_DIM` in `parameter.py`, `check_env3d.py` | done in stage 0, checked against 2D (see Stage 0 results) |
| The one-node-per-column graph ("collapse") | `voxel_world/docs/column_graph.md` (read this first) | measured: graph at 2D size, about 9% extra distance against the full 3D lattice |
| 2D ARiADNE training, Docker, run script | this repo | done. Stock run backed up in ~/rl/runs/2026-11-23_ariadne_stock |

What is missing is the **training environment in this repo**: the 3D counterparts of `env.py`, `sensor.py`,
`node_manager.py` and `ground_truth_node_manager.py`, so that workers train on ari3d worlds instead of PNGs.
The world builder is not part of this work; it is finished.

## Stage 0: the 3D training environment, checked against 2D (no training)

Build:
- `env3d.py`: loads an ari3d world (`world.npz`), keeps the belief as a 3D voxel grid (unknown/free/occupied),
  scans with the tilted Mid-360 band (−30.5° to +75.5°, every heading) from each node, and finds 3D frontiers.
  Reward has ARiADNE's form: frontier cells seen this step, normalised, minus distance / `UPDATING_MAP_SIZE` × 5,
  plus 20 when done. Port this from `prune_study.Sim`, reusing `voxel_world.fast.traverse`.
- `node_manager3d.py`: the column graph from `voxel_world/docs/column_graph.md`. One kept node per column
  (most frontiers, or a transit node near the drone's height). Columns every 4 m (`NODE_RESOLUTION`). Columns
  linked when neighbours and reachable on the full lattice; edge length = route length.
- `ground_truth_node_manager3d.py`: the same graph on the true map, for the ground-truth critic.
- A parameter that picks the 2D or 3D environment. `voxel_world` goes on `PYTHONPATH` (mount
  ~/rl/CERLAB-UAV-Autonomy/voxel_world into the container).

Check:
- On flat worlds (`make_ari3d.py --flat`: the 2D map extruded), run the existing 2D checkpoint in both the 2D
  and the 3D environment for the same PNGs. Explored share and distance should nearly match. If they don't, the
  bug is in the 3D environment, found before any training time is spent.
- Calibrate the reward normaliser so a 3D step's frontier reward is on the same scale as in 2D.
- Make sure episodes end: frontiers no reachable pose can see (outside the band) must not keep utility alive.
- Measure episodes per minute. Target: close to 2D (about 14 episodes/min with 16 workers). If utility (line
  of sight to frontiers) is too slow, group frontiers on a coarser grid.

Estimate: 1–1.5 days.

## Stage 0 results (2026-11-24)

How to run it: `ENV_DIM=3 WORLDS_3D=~/rl/worlds/<set> ./run_train.sh <run> [limit]` (mounts the worlds and
~/rl/CERLAB-UAV-Autonomy/voxel_world read-only, `voxel_world/src` on PYTHONPATH). ENV_DIM unset = 2D as before.

What was built, and where it differs from the prototype:
- `env3d.py`: `Map3D` (belief voxels, free configuration space, frontiers) and `Env3D` (scan, reward, explored
  share by volume and by footprint, a top-view GIF panel). The scan is 1.25° rays over the band (azimuths thinned
  with elevation), about 0.35 m apart at 16 m. The drone's own clearance box counts as known: the band never sees
  straight down.
- Frontier voxel: known free, an unknown face neighbour, and 4–25 unknown voxels among its 26 neighbours (2D's
  "2–7 of 8"). Without the 26-neighbour count, the solid edges where walls meet floors and ceilings (they touch free
  space only along an edge, no ray reaches them) and single unseen surface voxels kept utility alive forever.
- Dead frontiers: a frontier voxel still there after scans from 2 poses that had it in sight (within
  UTILITY_RANGE, in the band, line of sight) no longer counts for utility (it still counts for the reward if it is
  ever resolved). This is what lets 3D episodes end.
- `node_manager3d.py`: the lattice (columns every 4 m, heights every 1.6 m through the start, edges up to 2 columns
  and 2 heights apart, straight segment in free configuration space), routes by Dijkstra on it, utility per lattice
  node, one kept node per column (most utility, else nearest the drone's height; the drone's column keeps the
  drone's node). Columns are neighbours when a lattice edge joins them: at most 24, so K_SIZE stays 25. Nodes are
  keyed by (x, y) like 2D nodes, so the 2D agent, worker, critic observation and network are used unchanged.
- `ground_truth_node_manager3d.py`: the same column graph on the true map, cached per world.
- Shared 2D code: `agent.py` and `ground_truth_node_manager.py` look up neighbour indices in a dict instead of
  `np.argwhere` per neighbour. 2D results are bit-identical, about 30% faster per step.

Checks (2D stock checkpoint at episode 1376, greedy actions; `check_env3d.py`; flat worlds from `make_ari3d.py
--flat`, 12 PNGs not used for calibration, seeds 100–111):
- Mechanics, without the policy in the loop (`--replay`: the 2D episode's columns flown in the 3D env): with
  clearance 0 the 3D env reproduces 2D exactly: same columns, same neighbours (Jaccard 1.000), coverage within 0.3%
  at every step, distance +1.1% (height changes). With the default clearance of one voxel, 2D columns that touch a
  wall do not exist in 3D, so the replayed path breaks; that is the deliberate difference.
- Calibration (on the same replays): 3D / 2D frontier reward 3.90 (3.73–4.14), utility per node 5.74 (4.98–7.86),
  so `REWARD_SCALE_3D` = 0.26 and `UTILITY_SCALE_3D` = 0.18, set from 4 other maps (img_1609, 627, img_1315,
  img_4537) before these 12 were run. MIN_UTILITY applies to the scaled utility.
- Closed loop (the checkpoint choosing in each env, one-voxel clearance): distance to reach 90 / 95 / 99% of the
  footprint 1.05 / 1.04 / 1.06 × 2D; frontier reward per episode 0.96 × 2D; final coverage 0.995 × 2D. Episodes end
  in 7 of 12 in 3D, 9 of 12 in 2D: once the footprint is covered, 3D still has floor and ceiling strips that the band
  missed (real 3D work), and the 2D policy shuttles instead of flying back for them.
- Episodes end: a greedy explorer (most utility per metre, any column) finishes every world: 12/12 flat (30–64
  steps) and 4/4 ari3d_v4 (72–93 steps, 99.7–100% of the volume). The 2D checkpoint finishes 1 of the 4 ari3d_v4
  worlds within 128 steps (99.3–100% explored in all four); that is for training to fix.
- Graph size on ari3d_v4: up to 206 columns (padding 360). Kept heights used 1–18 m.
- Speed: 12-minute training runs from scratch on cerlab22, 16 workers, same code: 3D about 10.6 episodes/min
  (24 ari3d worlds, `~/rl/worlds/speed_test`), 2D about 12.2, so 3D runs at 0.87 x 2D. A 3D step is 0.2–0.4 s on
  one core (scan about 40%, graph and utility about 30%). Training ran end to end in 3D (critic, buffer, GIFs).

Loose end: `make_ari3d.py` stops the whole batch when one PNG raises "start not in free configuration space"
(seen with `--count 16 --seed 1000`); it should skip that PNG and go on.

## Stage 1: the 2.5D policy, a small training run

- The policy chooses a column, as ARiADNE chooses a node. The height inside the column is the kept node's
  height (most frontiers), by rule.
- Node inputs grow from 4 (x, y, utility, visited) to about 7: add the kept height relative to the drone, the
  number of separate free gaps in the column, and the column's free height.
- Start from the 2D stock checkpoint, with the weights for the new inputs at zero, so training starts from
  ARiADNE's behaviour.
- Worlds: about 200 accepted ari3d worlds pre-generated on cerlab22 (`--pixel 0.2`, ARiADNE's own scale), some flat ones
  mixed in. About 20 held-out worlds from PNGs never used in training.
- Sensor range 16 m (ARiADNE's) for comparability with 2D.
- Run: 8 workers, 4–6 hours.
- Early-look outputs:
  - TensorBoard curves: reward, surface seen, distance;
  - on the held-out worlds, against the greedy collapse explorer and the greedy full-lattice explorer from the
    pruning study;
  - GIFs with a top view and a side cut, to see whether it changes height at gates and pits.

Estimate: 0.5 day to build, the run, then evaluation.

## Stage 1 results (2026-11-25)

Run `stage1_25d`: 6 h, 16 workers, episodes 0 to 3939, from the stock 2D checkpoint, 200 ari3d + 20 flat worlds.
Backup on cerlab22: `~/rl/runs/2026-11-25_stage1_25d/`. Evaluation JSONs: `~/rl/stage1/eval/` (`s1_*`, `stock_*`,
`greedy_*`, `prune/`), made by `check_env3d.py` (policy argmax, one episode per world). Results page:
https://claude.ai/artifact/AT59Y1e3kcgTvhBDtD48XN (private).

Held-out set, 20 worlds:

| Planner | Finished | Median m to 90 % / 95 % / 99 % volume | Mean route m |
|---|---|---|---|
| Stage 1 | 9 / 20 | 280 / 316 / 441 (99 % reached on 18) | 877 |
| Stock 2D checkpoint | 4 / 20 | 278 / 311 / 392 (17) | 970 |
| Greedy column explorer | 20 / 20 | 256 / 283 / 370 (19) | 595 |

- Stage 1 reaches 95 % sooner than stock on 12 of 20 worlds, sooner than greedy on 1 of 20.
- All 11 unfinished Stage 1 episodes end in a two-stop loop (8 of them after seeing 99.5 % or more). img_361_s9000 stalls at 88.6 %.
- Training: success rate 42 -> 91 %, travel per episode 901 -> 678 m, reward flat after about episode 1000.
- The critic diverges: Q loss 2.3 -> 229, critic gradient norm about 400 -> 114,000, log alpha -3.0 -> -6.0, all still
  trending at the end.

Before Stage 2: stabilise the critic (gradient clipping, reward scale, slower critic or fixed alpha) and rerun Stage 1;
look into the finish condition (unfinished episodes at 99.5 %+); beat greedy before learning a height choice.

Follow-up (2026-11-25), why episodes do not finish:
- Every unfinished episode (Stage 1 and stock) ends flipping between two stops until the 128-step limit. At the loop no
  neighbouring column has utility; what is left is 12-120 m away by route. The policy only picks neighbours, and its 6
  encoder layers attend along graph edges (about 6 hops), so far utility barely reaches the neighbours' features; the
  leftover utility is also tiny (2-3 cells). 332_s9006: the one neighbour with utility was the column just left
  (visited, its other height now shows 2.2) and the policy ignores it.
- The leftovers on the "100 % volume" worlds are frontier cells next to never-seen solid (floor, ceiling and corner
  patches missed at grazing angles): no volume to gain, some surface. img_1471: 0 of 402,274 explorable voxels unknown;
  1002 of 72,233 surface voxels unseen.
- check_env3d.py --fallback: when no neighbour has utility or the policy turns back to the previous stop, take the first
  hop of the route to the greedy explorer's target. Held-out, 20 worlds (outputs ~/rl/stage1/eval_fb/):

| Planner | Finished | Median m to 95 % / 99 % volume | Mean route m | Steps overridden |
|---|---|---|---|---|
| Stage 1 + fallback | 20 / 20 | 304 / 390 | 661 | 29 % |
| Stock + fallback | 18 / 20 | 311 / 392 | 698 | 30 % |
| Greedy column explorer | 20 / 20 | 283 / 370 | 595 | - |

  Stage 1 + fallback beats stock + fallback on 14 of 20 routes, greedy on 1.
- Next: rerun Stage 1 with GAMMA 0.99 (GAMMA 1 with episodes cut at 128 steps and not marked done lets the critic's
  value fall without bound in loops: value +5 -> -244). Keep the critic's clip at 20000 so the effect is GAMMA's alone.

## Stage 2: a learned height choice

- The graph stays column-sized. The decoder scores (column, height) pairs for the current node's neighbours:
  about 25 neighbours × up to 4 heights. The heights are the column's free gaps cut into pieces of at most 5 m.
  That is ARiADNE's pointer decoder with more candidates.
- Aim: recover part of collapse's 9%. The most-utility height was near-best in only 60–74% of steps.
- Only after stage 1 looks sane. Estimate: 2–3 days.

## After that

Larger maps (`--pixel 0.4`, double ARiADNE's scale), the 20 m Mid-360 range, evaluation on the room courses and on the Gazebo
excavation worlds, and a 3D version of the ROS planner side (ARiADNE-ROS-Planner, branch 3d-learning).

## Risks

- Speed: line of sight from every candidate height to every frontier in range is the costly part.
- Frontiers the lidar can never see keep episodes from ending unless filtered.
- 3D frontier counts are much larger than 2D ones: the reward needs rescaling (stage 0).
- The pruning study used a greedy explorer and routed on the full lattice; the trained policy may behave differently.

## Decisions (2026-11-24)

- Stepping stone: stage 0, then stage 1, before a learned height choice. Go given; stage 0 started.
- Sensor range 16 m (ARiADNE's) for stages 0–1.
- Stage 1 starts from the 2D stock checkpoint.
- Training sets made with earlier generator versions were deleted; ari3d_v3 and courses/test_v2 are the current
  reference sets.
