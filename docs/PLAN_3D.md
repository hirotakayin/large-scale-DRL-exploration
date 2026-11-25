# Plan: from ARiADNE 2D to a learned 3D exploration planner

Status 2026-11-24: plan agreed; stage 0 in progress. Owner: the 3D_learning project
(see the workspace CLAUDE.md). Branch 3d-learning of this repo (training code) and of
hirotakayin/CERLAB-UAV-Autonomy (`voxel_world/`: worlds, simulator, studies).

## What already exists

| Piece | Where | State |
|---|---|---|
| 3D training worlds from ARiADNE's map PNGs | `voxel_world/scripts/make_ari3d.py`, module `ari3d.py` | done. Terraced levels, staggered levels joined by shafts, split corridors (over/under loops), pits, low and lipped islands, hurdle/lintel gates; no single height reaches more than 60% of a world; every opening at least 3.2 m, checked with 0.5 m clearance; `accepted` flag in stats.json. Deterministic from (PNG, seed). Runs in this repo's Docker image. Cloned on cerlab22 at ~/rl/CERLAB-UAV-Autonomy |
| Generating and previewing worlds from the GUI | hirotakayin/cerlab-uav-gui, Training tab | done. Calls make_ari3d.py and make_courses.py, shows previews and stats, and starts/stops training here over SSH |
| Evaluation courses (open rooms, obstacles at every height) | `voxel_world/scripts/make_courses.py --layout room`, test seeds 0–99 | done |
| A 3D exploration simulator (teleport between nodes, tilted Mid-360 band, 3D frontiers, utility, graphs) | `voxel_world/scripts/prune_study.py` (class `Sim`) | prototype, used for the pruning study, not wired into training |
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
