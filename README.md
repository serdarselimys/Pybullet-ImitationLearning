# HexaDog Gait Imitation Learning Pipeline

A PyBullet-based pipeline for training a neural network to imitate a hand-designed
procedural gait controller for a six-legged robot ("HexaDog"), and then driving the
learned policy live with a gamepad.

The pipeline has three stages:

1. **Data generation** (`Imitation_Learning_Datagen.py`) — Runs many parallel PyBullet
   environments, each simulating multiple robots driven by a procedural
   (inverse-kinematics + cycloid gait) controller. Random locomotion commands
   (forward/backward/strafe/spin, body height, "dirt mode") are sampled continuously,
   and the resulting robot state + joint angles are logged to CSV. This produces the
   expert demonstration dataset.
2. **Training** (`Imitation_Learning_MLP_Training.py`) — Trains a small MLP (`GaitNet`) via supervised
   learning / behavioral cloning to map locomotion commands + robot state to joint
   angles, imitating the procedural controller from stage 1.
3. **Live gamepad inference** (`Imitation_Learning_MLP_Controller.py`) — Loads the trained model
   and runs it live in PyBullet, replacing the procedural IK controller with the
   neural network's predictions, controlled interactively via a connected gamepad.

## Download

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Or, if you'd rather not use git, download the repo as a ZIP from GitHub
(**Code → Download ZIP**) and extract it locally.

Make sure `HexaDog_ZBD.urdf`, the `meshes/` folder, and `merged_gaits.csv` end up
in the repo root alongside the three scripts — see [Repository layout](#repository-layout)
below.

## Installation

Requires **Python 3.9+**.

1. (Recommended) create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, for GPU training/inference) install a CUDA-enabled build of
   PyTorch instead of the default CPU wheel — pick the right command for your
   CUDA version from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
   e.g.:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. (Stage 3 only) connect a gamepad/joystick to your machine before launching
   the live controller — `pygame` needs to detect it at startup.

## Usage

Run the three stages in order from the repo root:

```bash
# 1. Generate expert demonstration data (see "Stage 1" below for config knobs)
python Imitation_Learning_Datagen.py

# 2. Train the imitation-learning MLP on the generated dataset
python Imitation_Learning_MLP_Training.py

# 3. Drive the trained policy live with a gamepad
python Imitation_Learning_MLP_Controller.py
```

Each stage consumes the previous stage's output (dataset → model → live policy),
so run them in this order the first time through. See the per-stage sections
below for configuration details and expected outputs.

## Repository layout

This repo expects the following files/folders to sit alongside the scripts:

```
.
├── Imitation_Learning_Datagen.py         # Stage 1: data generation
├── Imitation_Learning_MLP_Training.py    # Stage 2: training
├── Imitation_Learning_MLP_Controller.py  # Stage 3: live inference w/ gamepad
├── HexaDog_ZBD.urdf               # Robot URDF
├── meshes/                        # STL meshes referenced by the URDF
├── merged_gaits.csv               # Reference gait table (speed/frequency/amplitude
│                                   # per direction) used to seed the procedural gait
├── requirements.txt
└── README.md
```

> The scripts reference these files by relative path (e.g. `HexaDog_ZBD.urdf`,
> `merged_gaits.csv`), so run them from the repo root, or update the path
> constants at the top of each script.

## Robot / leg convention

The URDF is expected to expose six legs named `FL`, `ML`, `RL` (left side) and
`FR`, `MR`, `RR` (right side), each with three joints suffixed `1` (hip), `2`
(thigh), `3` (knee) — e.g. `FL1`, `FL2`, `FL3`.

## Stage 1 — Generate the dataset

```bash
python Imitation_Learning_Datagen.py
```

Key config values at the top of the script:

| Variable | Purpose |
|---|---|
| `NUM_ENVS` | Number of parallel PyBullet processes (roughly one per CPU core) |
| `ROBOTS_PER_ENV` | Robots simulated per environment |
| `RUN_HOURS` | Wall-clock time budget for data generation |
| `RENDER_MODE` | `"headless"` (fast, no GUI) or `"windowed"` (opens a PyBullet GUI per env) |
| `SPACING` | Distance between robots in the grid, to avoid inter-robot collisions |

The script spawns `NUM_ENVS` OS processes, each simulating `ROBOTS_PER_ENV` robots
with independently randomized commands. Each process writes to its own
`gait_dataset_pybullet_env_<id>.csv`, and a background thread periodically merges
these into `gait_dataset_pybullet_merged.csv` as a checkpoint. On completion (or
`Ctrl+C`), the final merge runs and the per-env files are cleaned up.

Output columns include the commanded inputs (`joy_fwd`, `joy_side`, `joy_spin`,
`body_height`, `dirt_mode`, gait `phase`), the robot's resulting base linear/angular
velocity, and the actual joint angles for all 18 joints (6 legs × 3 joints) — this
is the supervision target for training.

## Stage 2 — Train the model

```bash
python Imitation_Learning_MLP_Training.py
```

Reads `gait_dataset_pybullet_merged.csv`, standardizes inputs/outputs with
`StandardScaler`, and trains `GaitNet` (a 3-hidden-layer MLP) with MSE loss to
predict all 18 joint angles from a 12-dimensional feature vector:
`joy_fwd, joy_side, joy_spin, phase, body_height, dirt_mode, lin_vel_x/y/z,
ang_vel_x/y/z`.

Artifacts produced:

- `gait_nn.pth` — model `state_dict`
- `gait_nn_full.pth` — full pickled model (used preferentially at inference time)
- `scaler_X.pkl`, `scaler_y.pkl` — input/output scalers
- `feature_columns.pkl` — exact feature/target column order, so inference code can
  reconstruct the input vector correctly

Adjust `EPOCHS`, batch size, or the `GaitNet` architecture directly in the script
if you change the feature/target schema.

## Stage 3 — Live gamepad control with the trained policy

```bash
python Imitation_Learning_MLP_Controller.py
```

Requires a connected gamepad (via `pygame`). Opens a PyBullet GUI window and lets
you drive the robot in real time:

- **Left stick** — forward/back + strafe
- **Right stick X** — spin
- Configurable buttons — body height up/down, "dirt mode" toggle (adjust
  `HEIGHT_UP_BUTTON`, `HEIGHT_DOWN_BUTTON`, `DIRT_MODE_BUTTON` to match your
  controller's mapping)

The script reproduces the same feature vector used in training, feeds it through
the loaded network (with the same scalers), applies exponential smoothing to the
predicted joint targets, and sends them to the robot as PyBullet position control
targets.

## Requirements

See `requirements.txt`. A CUDA-capable GPU is optional but speeds up training and
inference; the scripts fall back to CPU automatically via
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

## Notes / tips

- `RUN_HOURS`, `NUM_ENVS`, and `ROBOTS_PER_ENV` in the data-generation script
  trade off dataset size against generation time — scale down for a quick smoke
  test before committing to a long run.
- `merged_gaits.csv` must contain `direction`, `speed`, `frequency`, and
  `step_amplitude` columns, with `direction` values covering `straight`,
  `sideways`, `diagonal`, and `spin` (rows are used as a lookup table, not a
  strict schedule).
- If a mesh or URDF path error occurs, confirm the `meshes/` folder location
  referenced inside `HexaDog_ZBD.urdf` matches where you've placed it relative to
  the URDF file.
