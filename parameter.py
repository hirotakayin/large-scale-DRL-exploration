import os

# saving path (FOLDER_NAME from the environment lets run_train.sh name each run)
FOLDER_NAME = os.environ.get('FOLDER_NAME', 'ariadne1_ground_truth_critic')
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = int(os.environ.get('NUM_META_AGENT', 16))  # how many threads does your CPU have

# network parameters
NODE_INPUT_DIM = 4  # 3D adds NODE_INPUTS_3D (below)
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 0  # 0 unless you want to collect data using GPUs


# 2D or 3D environment (docs/PLAN_3D.md). ENV_DIM=3 trains on ari3d worlds (voxel_world/scripts/make_ari3d.py):
# env3d.py, agent3d.py, node_manager3d.py, ground_truth_node_manager3d.py. voxel_world/src must be on PYTHONPATH.
ENV_DIM = int(os.environ.get('ENV_DIM', 2))
WORLDS_3D = os.environ.get('WORLDS_3D', 'worlds')  # folder of ari3d worlds, one sub-folder with world.npz each
Z_RESOLUTION = 1.6  # meter, between the lattice heights in a column (CELL_SIZE multiple)
Z_NEIGHBOR = 2  # lattice edges reach this many heights up and down (and 2 columns sideways, as in 2D)
ELEVATION_BAND = (-30.5, 75.5)  # deg from horizontal, the tilted Mid-360, every heading
SENSOR_STEP_3D = 1.25  # deg between rays (azimuth thinned with elevation): about 0.35 m apart at 16 m
CLEARANCE_3D = int(os.environ.get('CLEARANCE_3D', 1))  # voxels: free configuration space is this far from anything not known free
FRONTIER_CELL_SIZE_3D = 4 * CELL_SIZE  # meter, 3D frontiers are downsampled on this grid
FRONTIER_MIN_UNKNOWN_3D = 4  # a frontier voxel has this many unknown voxels among its 26 neighbours (2D: 2 of 8)
DEAD_FRONTIER_VIEWS = 2  # a frontier voxel still there after scans from this many poses that had it in sight is dropped
# 3D frontier counts are larger than 2D ones: these bring them to the 2D scale, so both still divide by
# SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE, and MIN_UTILITY applies to the scaled utility. Calibrated on flat
# worlds (make_ari3d.py --flat), flying the 2D checkpoint's paths in both environments (check_env3d.py --replay):
# 3D / 2D was 5.0-5.8 per node for utility and 3.6-4.0 for the frontier reward, on 4 maps.
UTILITY_SCALE_3D = 0.18  # the policy's utility input (and the critic's)
REWARD_SCALE_3D = 0.26  # frontier cells observed, in the reward

# stage 1 (docs/PLAN_3D.md): 3 more node inputs in 3D, appended after the 2D ones (x, y, utility, visited):
# kept height - drone height, free gaps in the column, the column's free height (heights / HEIGHT_NORM_3D,
# gaps / GAPS_NORM_3D). The critic gets the same 3 after its 2D inputs. A 2D checkpoint loads with zero weights
# for them (model.load_widened), so it starts out acting exactly as in 2D.
NODE_INPUTS_3D = 3
HEIGHT_NORM_3D = 20.0  # meter
GAPS_NORM_3D = 4.0
if ENV_DIM == 3:
    NODE_INPUT_DIM += NODE_INPUTS_3D

# stage 1b: far small leftovers stop counting. A column's utility (UTILITY_SCALE_3D applied) counts only above
# MIN_UTILITY + route / LEFTOVER_M, route = metres from the drone on the lattice: 1 more frontier cell per LEFTOVER_M
# metres (gain against travel cost, as NBVP's gain * exp(-lambda * cost)). It applies wherever utility is read: the
# policy's and the critic's inputs, the finish check, check_env3d's greedy and fallback. 0 turns it off (Stage 1).
LEFTOVER_M = float(os.environ.get('LEFTOVER_M', 0))
# GAMMA 1 with episodes cut at MAX_EPISODE_STEP (not done) let the 3D critic's value fall without bound in loops
if ENV_DIM == 3:
    GAMMA = float(os.environ.get('GAMMA_3D', 0.99))
# start a run from these weights (a checkpoint path; policy, critics and log_alpha; optimizers fresh, episode 0)
INIT_MODEL = os.environ.get('INIT_MODEL', '')
# stage 1c (docs/PLAN_3D.md): shorter paths. Stage 1b flew ~25 % further than the greedy explorer on held-out worlds and
# its training distance stayed flat over 4096 episodes. The frontier reward sums to about the same on any path, so
# under a per-decision discount it mostly pays for clearing big frontiers early.
# DISCOUNT_M > 0: discount each transition by GAMMA ** (route metres / DISCOUNT_M) instead of GAMMA (0 = per decision)
DISCOUNT_M = float(os.environ.get('DISCOUNT_M', 0))
# DIST_W multiplies the per-metre distance penalty in the 3D reward (1 = as before)
DIST_W = float(os.environ.get('DIST_W', 1))
# GREEDY_BONUS > 0: at the finish add GREEDY_BONUS * (greedy m - own m) / greedy m, greedy m read from GREEDY_REF
# ({world folder: metres}, a file in WORLDS_3D) for the episode's world; worlds missing from it get no bonus
GREEDY_BONUS = float(os.environ.get('GREEDY_BONUS', 0))
GREEDY_REF = os.environ.get('GREEDY_REF', 'greedy_L20.json')
# replay buffer slots: 27 as in ARiADNE, plus the transition's discount
BUFFER_SLOTS = 28
