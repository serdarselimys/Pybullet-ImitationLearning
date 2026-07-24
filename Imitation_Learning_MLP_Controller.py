import pybullet as p
import pybullet_data
import numpy as np
import torch
import torch.nn as nn
import pickle
import math
import time
import pygame

# =================================================================
# CONFIG
# =================================================================
MODEL_PATH   = "gait_nn_full.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
COLS_PATH     = "feature_columns.pkl"

URDF_PATH    = "HexaDog_ZBD.urdf"

PHYSICS_FREQ = 240
DT           = 1.0 / PHYSICS_FREQ

DEFAULT_HEIGHT = 0.20
MIN_HEIGHT     = 0.14
MAX_HEIGHT     = 0.23
HEIGHT_SPEED   = 0.001   

LEG_ORDER  = ["FL", "ML", "RL", "FR", "MR", "RR"]
LEG_PHASES = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])

LATERAL_SCALAR         = 1.0
JOYSTICK_DEADBAND       = 0.15
INPUT_JITTER_THRESHOLD  = 0.05
LERP_SPEED              = 0.15   # smoothing rate for stick inputs -> commands

DIRT_MODE_BUTTON  = 0    # A / Cross button, adjust to your pad's mapping
HEIGHT_UP_BUTTON  = 5    # right trigger/bumper, adjust to your pad
HEIGHT_DOWN_BUTTON = 4   # left trigger/bumper, adjust to your pad


NN_OUTPUT_SMOOTHING = 0.3

GAIT_FREQ = 1.0   # Hz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================================
# MODEL DEFINITION (must match training script)
# =================================================================
class GaitNet(nn.Module):
    def __init__(self, input_size, output_size=18):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)


# =================================================================
# LOAD MODEL + SCALERS + COLUMN ORDER
# =================================================================
with open(COLS_PATH, "rb") as f:
    cols = pickle.load(f)
feature_cols = cols["feature_cols"]
target_cols  = cols["target_cols"]

with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)
with open(SCALER_Y_PATH, "rb") as f:
    scaler_y = pickle.load(f)

try:
    net = torch.load(MODEL_PATH, map_location=device, weights_only=False)
except Exception:
    net = GaitNet(input_size=len(feature_cols), output_size=len(target_cols))
    net.load_state_dict(torch.load("gait_nn.pth", map_location=device))
net.to(device)
net.eval()

print(f"Loaded model. Inputs: {len(feature_cols)}  Outputs: {len(target_cols)}")

HAS_PREV_JOINTS = any(c.startswith("prev_") for c in feature_cols)
HAS_DIRT_MODE   = "dirt_mode" in feature_cols
HAS_LEG_PHASE   = any(c.endswith("_phase_sin") and not c.startswith("prev_") and c != "phase_sin" for c in feature_cols)


# =================================================================
# PYBULLET SETUP
# =================================================================
client_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
p.setGravity(0, 0, -9.81, physicsClientId=client_id)
p.setTimeStep(DT, physicsClientId=client_id)
p.loadURDF("plane.urdf", physicsClientId=client_id)

robot = p.loadURDF(URDF_PATH, [0, 0, DEFAULT_HEIGHT], physicsClientId=client_id)
p.changeDynamics(robot, -1, lateralFriction=1.0, physicsClientId=client_id)
for j in range(p.getNumJoints(robot, physicsClientId=client_id)):
    p.changeDynamics(robot, j, lateralFriction=1.0, physicsClientId=client_id)

joint_map = {}
for j in range(p.getNumJoints(robot, physicsClientId=client_id)):
    name = p.getJointInfo(robot, j, physicsClientId=client_id)[1].decode()
    joint_map[name] = j

joint_ids_flat = []
for leg in LEG_ORDER:
    for suf in ["3", "2", "1"]:
        joint_ids_flat.append(joint_map[f"{leg}{suf}"])

GLOBAL_MAX_TORQUE   = 1.8
GLOBAL_MAX_VELOCITY = 6.0


# =================================================================
# GAMEPAD SETUP
# =================================================================
pygame.init()
if pygame.joystick.get_count() == 0:
    print("No joystick detected! Connect a gamepad and try again.")
    p.disconnect(client_id)
    exit()
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Gamepad connected: {joystick.get_name()}")

def apply_deadband(value, threshold):
    return 0.0 if abs(value) < threshold else value


# =================================================================
# STATE
# =================================================================
phase_accum = 0.0
s_fwd, s_side, s_spin = 0.0, 0.0, 0.0
current_h_trim = 0.0
dirt_mode = 0.0
prev_dirt_button_state = False
pred_smoothed = None


states = p.getJointStates(robot, joint_ids_flat, physicsClientId=client_id)
prev_joints = np.array([s[0] for s in states], dtype=np.float64)

print("Controller Active: Left Stick = Move, Right Stick X = Spin")
print(f"Button {HEIGHT_UP_BUTTON}/{HEIGHT_DOWN_BUTTON} = Height Up/Down, Button {DIRT_MODE_BUTTON} = Toggle Dirt Mode")

try:
    while True:
        pygame.event.pump()

        
        raw_fwd  = apply_deadband(-joystick.get_axis(1), INPUT_JITTER_THRESHOLD)
        raw_side = apply_deadband(-joystick.get_axis(0) * LATERAL_SCALAR, INPUT_JITTER_THRESHOLD)
        raw_spin = apply_deadband(joystick.get_axis(2), INPUT_JITTER_THRESHOLD)

        
        s_fwd  += (raw_fwd - s_fwd) * LERP_SPEED
        s_side += (raw_side - s_side) * LERP_SPEED
        s_spin += (raw_spin - s_spin) * LERP_SPEED

        
        num_buttons = joystick.get_numbuttons()
        btn_up   = joystick.get_button(HEIGHT_UP_BUTTON) if num_buttons > HEIGHT_UP_BUTTON else 0
        btn_down = joystick.get_button(HEIGHT_DOWN_BUTTON) if num_buttons > HEIGHT_DOWN_BUTTON else 0
        if btn_up:
            current_h_trim += HEIGHT_SPEED
        if btn_down:
            current_h_trim -= HEIGHT_SPEED
        cmd_bh = DEFAULT_HEIGHT + np.clip(current_h_trim, MIN_HEIGHT - DEFAULT_HEIGHT, MAX_HEIGHT - DEFAULT_HEIGHT)

        
        dirt_button_state = bool(joystick.get_button(DIRT_MODE_BUTTON)) if num_buttons > DIRT_MODE_BUTTON else False
        if dirt_button_state and not prev_dirt_button_state:
            dirt_mode = 1.0 - dirt_mode
            print(f"[Dirt mode] -> {'ON' if dirt_mode > 0.5 else 'OFF'}")
        prev_dirt_button_state = dirt_button_state

        
        norm_mag = min(1.0, math.sqrt(raw_fwd**2 + raw_side**2))
        active = (norm_mag > JOYSTICK_DEADBAND) or (abs(raw_spin) > JOYSTICK_DEADBAND)
        freq = GAIT_FREQ

        if active:
            phase_accum = (phase_accum + freq * DT) % 1.0
        else:
            
            if phase_accum > 0.01:
                if phase_accum > 0.5:
                    phase_accum = (phase_accum + freq * DT) % 1.0
                else:
                    phase_accum -= phase_accum * LERP_SPEED
            else:
                phase_accum = 0.0

        phase_sin = math.sin(2.0 * math.pi * phase_accum)
        phase_cos = math.cos(2.0 * math.pi * phase_accum)

        p_leg = (phase_accum + LEG_PHASES) % 1.0
        leg_sin = np.sin(2.0 * np.pi * p_leg)
        leg_cos = np.cos(2.0 * np.pi * p_leg)

        lin_vel, ang_vel = p.getBaseVelocity(robot, physicsClientId=client_id)

        
        feat = {
            "joy_fwd": s_fwd, "joy_side": s_side, "joy_spin": s_spin,
            "phase": phase_accum,
            "phase_sin": phase_sin, "phase_cos": phase_cos,
            "body_height": cmd_bh, "dirt_mode": dirt_mode,
            "lin_vel_x": lin_vel[0], "lin_vel_y": lin_vel[1], "lin_vel_z": lin_vel[2],
            "ang_vel_x": ang_vel[0], "ang_vel_y": ang_vel[1], "ang_vel_z": ang_vel[2],
        }
        if HAS_LEG_PHASE:
            for i, leg in enumerate(LEG_ORDER):
                feat[f"{leg}_phase_sin"] = leg_sin[i]
                feat[f"{leg}_phase_cos"] = leg_cos[i]
        if HAS_PREV_JOINTS:
            for i, leg in enumerate(LEG_ORDER):
                feat[f"prev_{leg}3"] = prev_joints[i * 3 + 0]
                feat[f"prev_{leg}2"] = prev_joints[i * 3 + 1]
                feat[f"prev_{leg}1"] = prev_joints[i * 3 + 2]

        x_vec = np.array([[feat[c] for c in feature_cols]], dtype=np.float64)
        x_scaled = scaler_X.transform(x_vec)
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_scaled = net(x_t).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0] 

        
        if pred_smoothed is None:
            pred_smoothed = pred.copy()
        else:
            pred_smoothed = NN_OUTPUT_SMOOTHING * pred_smoothed + (1.0 - NN_OUTPUT_SMOOTHING) * pred

        
        for i, leg in enumerate(LEG_ORDER):
            j3, j2, j1 = pred_smoothed[i * 3 + 0], pred_smoothed[i * 3 + 1], pred_smoothed[i * 3 + 2]
            for suf, val in zip(["3", "2", "1"], [j3, j2, j1]):
                p.setJointMotorControl2(
                    robot, joint_map[f"{leg}{suf}"],
                    p.POSITION_CONTROL, targetPosition=val,
                    force=GLOBAL_MAX_TORQUE, maxVelocity=GLOBAL_MAX_VELOCITY,
                    physicsClientId=client_id
                )

        p.stepSimulation(physicsClientId=client_id)

        
        if HAS_PREV_JOINTS:
            states = p.getJointStates(robot, joint_ids_flat, physicsClientId=client_id)
            prev_joints = np.array([s[0] for s in states], dtype=np.float64)

        time.sleep(DT)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    p.disconnect(physicsClientId=client_id)
    pygame.quit()
    print("Controller stopped.")
