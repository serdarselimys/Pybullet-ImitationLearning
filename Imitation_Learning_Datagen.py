import pybullet as p
import pybullet_data
import pandas as pd
import math
import time
import numpy as np
import random
import csv
import os
import multiprocessing as mp
import threading
import shutil

# =================================================================
# CONFIG
# =================================================================
NUM_ENVS        = 8        # Number of parallel environments (CPU cores).
ROBOTS_PER_ENV  = 12       # Robots running in each independent environment.
RUN_HOURS       = 1.0      # Number of Hours / Adjust according to number of robots and enviroments created.
SPACING         = 5.0      # Spacing between the robots keep large to prevent collision during the simulation.

# --- Renderer: "windowed" opens the pybullet GUI, "headless" runs in the background, suitable for parallel processing multiple experiments at the same time ---
RENDER_MODE = "headless"
_RENDER_MAP = {"windowed": p.GUI, "headless": p.DIRECT}

PHYSICS_FREQ    = 240
LOG_FREQ        = 60
DT              = 1.0 / PHYSICS_FREQ
LOG_INTERVAL    = int(PHYSICS_FREQ / LOG_FREQ)

FILE_PATH       = "merged_gaits.csv"
BASE_OUTPUT_PATH = "gait_dataset_pybullet"
SAVE_BUFFER_SIZE = 4000

# -------- Geometry / gait tuning --------
L1, L2          = 0.1000, 0.1000
BODY_OFFSET     = 0.0544

MIN_HEIGHT      = 0.14
MAX_HEIGHT      = 0.23
DEFAULT_HEIGHT  = 0.20

STOP_BLEND_SPEED = 0.08
KNEE_ASSEMBLY_OFFSET = math.pi / 2.0

GLOBAL_MAX_TORQUE   = 1.8
GLOBAL_MAX_VELOCITY = 6.0

DIRT_MODE_PROB = 0.50
CMD_MIN_DUR, CMD_MAX_DUR = 4.0, 6.0

DIRECTION_PARAMS = {
    "forward":  {"step_height": 0.0125, "urdf_x_offset": -0.025},
    "backward": {"step_height": 0.0125, "urdf_x_offset":  0.010},
    "sideways": {"step_height": 0.0150, "urdf_x_offset":  0.005},
    "diagonal": {"step_height": 0.0100, "urdf_x_offset": -0.015},
    "spin":     {"step_height": 0.0125, "urdf_x_offset":  0.020},
}

LEG_ORDER  = ["FL", "ML", "RL", "FR", "MR", "RR"]
LEG_PHASES = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]

LEG_ROOTS = {
    "FL": [ 0.12,  0.10], "ML": [ 0.00,  0.14], "RL": [-0.12,  0.10],
    "FR": [ 0.12, -0.10], "MR": [ 0.00, -0.14], "RR": [-0.12, -0.10]
}
IS_LEFT = [True, True, True, False, False, False] 

# Precompute structural NumPy arrays for vectorization
LEG_ROOTS_X = np.array([LEG_ROOTS[l][0] for l in LEG_ORDER]) # (6,)
LEG_ROOTS_Y = np.array([LEG_ROOTS[l][1] for l in LEG_ORDER]) # (6,)
LEG_PHASES_ARR = np.array(LEG_PHASES)                                # (6,)
IS_LEFT_ARR = np.array(IS_LEFT)                                      # (6,)

# =================================================================
# GAIT CSV LOOKUP
# =================================================================
df_master = pd.read_csv(FILE_PATH, skipinitialspace=True)
df_master.columns = df_master.columns.str.strip()
df_master["direction"] = df_master["direction"].astype(str).str.strip().str.lower()

DIRS = ["straight", "sideways", "diagonal", "spin"]
gait_tables = {}
for d in DIRS:
    sub = df_master[df_master["direction"] == d]
    if sub.empty:
        sub = df_master[df_master["direction"] == "straight"]
    sub = sub.reset_index(drop=True)
    gait_tables[d] = {
        "speed": sub["speed"].values,
        "freq": sub["frequency"].values,
        "amp": sub["step_amplitude"].values,
        "smin": float(sub["speed"].min()),
        "smax": float(sub["speed"].max()),
    }

def lookup_gait_vectorized(direction, norm_mag):
    tbl = gait_tables[direction]
    target = tbl["smin"] + norm_mag * (tbl["smax"] - tbl["smin"])
    diff = np.abs(tbl["speed"][:, np.newaxis] - target)
    idx = diff.argmin(axis=0)
    return tbl["freq"][idx], tbl["amp"][idx]

# =================================================================
# IK
# =================================================================
def solve_leg_ik_vectorized(tx, ty, tz, urdf_x_offset):
    x = tx + urdf_x_offset
    y = ty
    z_from_thigh = -(tz - BODY_OFFSET)

    hip = np.arctan2(y, -z_from_thigh)
    z_sag = -np.sqrt(y * y + z_from_thigh * z_from_thigh)

    dist_sq = x * x + z_sag * z_sag
    dist = np.sqrt(dist_sq)

    max_reach = (L1 + L2) * 0.99
    min_reach = abs(L1 - L2)
    dist = np.clip(dist, min_reach, max_reach)
    dist_sq = dist * dist

    cos_phi = (L1 * L1 + L2 * L2 - dist_sq) / (2 * L1 * L2)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    knee = np.pi - np.arccos(cos_phi)

    alpha = np.arctan2(z_sag, x)
    cos_beta = (L1 * L1 + dist_sq - L2 * L2) / (2 * L1 * dist)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    thigh = alpha - beta + np.pi / 2.0
    return hip, thigh, knee

# =================================================================
# BLENDED GAIT
# =================================================================
def blended_gait_vectorized(fwd, side, spin, norm_mag, dirt):
    f, s, r = np.abs(fwd), np.abs(side), np.abs(spin)

    diag_w = np.minimum(f, s) * 2.0
    straight_w = np.maximum(0.0, f - diag_w / 2.0)
    sideways_w = np.maximum(0.0, s - diag_w / 2.0)
    spin_w = r
    total_w = straight_w + sideways_w + diag_w + spin_w

    fr_str, am_str = lookup_gait_vectorized("straight", norm_mag)
    fr_sid, am_sid = lookup_gait_vectorized("sideways", norm_mag)
    fr_dia, am_dia = lookup_gait_vectorized("diagonal", norm_mag)
    fr_spi, am_spi = lookup_gait_vectorized("spin", r)

    sh_str = np.where(fwd >= 0.0, DIRECTION_PARAMS["forward"]["step_height"], DIRECTION_PARAMS["backward"]["step_height"])
    ux_str = np.where(fwd >= 0.0, DIRECTION_PARAMS["forward"]["urdf_x_offset"], DIRECTION_PARAMS["backward"]["urdf_x_offset"])

    sh_sid = DIRECTION_PARAMS["sideways"]["step_height"]
    ux_sid = DIRECTION_PARAMS["sideways"]["urdf_x_offset"]
    sh_dia = DIRECTION_PARAMS["diagonal"]["step_height"]
    ux_dia = DIRECTION_PARAMS["diagonal"]["urdf_x_offset"]
    sh_spi = DIRECTION_PARAMS["spin"]["step_height"]
    ux_spi = DIRECTION_PARAMS["spin"]["urdf_x_offset"]

    mask = total_w > 1e-3
    inv_tot = np.where(mask, 1.0 / total_w, 1.0)

    freq = (straight_w * fr_str + sideways_w * fr_sid + diag_w * fr_dia + spin_w * fr_spi) * inv_tot
    amp  = (straight_w * am_str + sideways_w * am_sid + diag_w * am_dia + spin_w * am_spi) * inv_tot
    sh   = (straight_w * sh_str + sideways_w * sh_sid + diag_w * sh_dia + spin_w * sh_spi) * inv_tot
    ux   = (straight_w * ux_str + sideways_w * ux_sid + diag_w * ux_dia + spin_w * ux_spi) * inv_tot

    freq = np.where(mask, freq, fr_str)
    amp  = np.where(mask, amp, am_str)
    sh   = np.where(mask, sh, sh_str)
    ux   = np.where(mask, ux, ux_str)

    dirt_mask = dirt > 0.5
    sh = np.where(dirt_mask, 0.020, sh)
    freq = np.where(dirt_mask, freq * 0.75, freq)
    amp = np.where(dirt_mask, amp * 0.75, amp)

    return {"freq": freq, "amp": amp, "step_h": sh, "urdf_x": ux}

def sample_commands():
    dirs = ["forward", "backward", "strafe", "spin"]
    selected = [random.choice(dirs)] if random.random() < 0.5 else random.sample(dirs, 2)
    fwd = side = spin = 0.0
    for d in selected:
        if d == "forward":   fwd += random.uniform(0.15, 1.0)
        elif d == "backward": fwd += random.uniform(-1.0, -0.15)
        elif d == "strafe":   side += random.choice([1.0, -1.0]) * random.uniform(0.15, 1.0)
        elif d == "spin":     spin += random.choice([1.0, -1.0]) * random.uniform(0.15, 1.0)
    side *= 0.7
    body_h = random.uniform(MIN_HEIGHT, MAX_HEIGHT)
    dirt = 1.0 if random.random() < DIRT_MODE_PROB else 0.0
    dur = random.uniform(CMD_MIN_DUR, CMD_MAX_DUR)
    return fwd, side, spin, body_h, dirt, dur

# =================================================================
# CONTINUOUS MERGER LOOP (RUNS IN BACKGROUND THREAD)
# =================================================================
def continuous_merger_loop(env_files, final_path, stop_event, interval=120):
    while not stop_event.is_set():
        stop_event.wait(timeout=interval)

        tmp_path = final_path + ".tmp"
        try:
            first_file_written = False
            with open(tmp_path, 'wb') as outfile:
                for f_path in env_files:
                    if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
                        with open(f_path, 'rb') as infile:
                            if not first_file_written:
                                shutil.copyfileobj(infile, outfile)
                                first_file_written = True
                            else:
                                infile.readline()  # Skip the header line
                                shutil.copyfileobj(infile, outfile)

            if first_file_written:
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(tmp_path, final_path)
                print(f"\n[Backup System] Auto-merged checkpoint saved via fast-stream!")
        except Exception as e:
            print(f"\n[Backup System] Warning: Skipping checkpoint merge due to file lock: {e}")

# =================================================================
# ENVIRONMENT PROCESS WORKER
# =================================================================
def run_environment(env_id, num_robots):
    output_path = f"{BASE_OUTPUT_PATH}_env_{env_id}.csv"

    render_mode_flag = _RENDER_MAP.get(RENDER_MODE, p.DIRECT)
    client_id = p.connect(render_mode_flag)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setTimeStep(DT, physicsClientId=client_id)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    p.changeDynamics(plane_id, -1, lateralFriction=1.0, restitution=0.0, physicsClientId=client_id)

    robots = []
    grid_cols = max(1, int(math.ceil(math.sqrt(num_robots))))
    for i in range(num_robots):
        gx = (i // grid_cols) * SPACING
        gy = (i %  grid_cols) * SPACING

        spawn_flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        r_id = p.loadURDF("HexaDog_ZBD.urdf", [gx, gy, DEFAULT_HEIGHT], flags=spawn_flags, physicsClientId=client_id)
        p.changeDynamics(r_id, -1, lateralFriction=1.0, physicsClientId=client_id)

        for j in range(p.getNumJoints(r_id, physicsClientId=client_id)):
            p.changeDynamics(r_id, j, lateralFriction=1.0, physicsClientId=client_id)

        robots.append(r_id)

    joint_maps = []
    joint_ids_flat = []
    for r_id in robots:
        mapping = {}
        for j in range(p.getNumJoints(r_id, physicsClientId=client_id)):
            j_info = p.getJointInfo(r_id, j, physicsClientId=client_id)
            mapping[j_info[1].decode()] = j
        joint_maps.append(mapping)

        r_ids = []
        for leg in LEG_ORDER:
            for suf in ["3", "2", "1"]:
                r_ids.append(mapping[f"{leg}{suf}"])
        joint_ids_flat.append(r_ids)

    # ---- Stateless schema: raw phase only ----
    headers = ['env', 'phase', 'joy_fwd', 'joy_side', 'joy_spin', 'body_height', 'dirt_mode',
               'lin_vel_x', 'lin_vel_y', 'lin_vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
    for leg in LEG_ORDER: headers += [f'{leg}3', f'{leg}2', f'{leg}1']

    with open(output_path, 'w', newline='') as f:
        csv.writer(f).writerow(headers)

    phase_accum      = np.zeros(num_robots)
    motion_blend     = np.zeros(num_robots)
    urdf_x_filt      = np.zeros(num_robots)
    body_height_curr = np.full(num_robots, DEFAULT_HEIGHT)

    cmd_fwd, cmd_side, cmd_spin, cmd_bh, cmd_dirt, cmd_dur = zip(*[sample_commands() for _ in range(num_robots)])
    cmd_fwd, cmd_side, cmd_spin, cmd_bh, cmd_dirt, cmd_dur = np.array(cmd_fwd), np.array(cmd_side), np.array(cmd_spin), np.array(cmd_bh), np.array(cmd_dirt), np.array(cmd_dur)
    cmd_timer = np.zeros(num_robots)

    data_buffer = []
    start_wall = time.time()
    physics_step = 0
    sim_time = 0.0
    URDF_X_ALPHA = 0.04
    HEIGHT_RATE = 0.06

    print(f"[Env {env_id}] Running {num_robots} robots (stateless schema). Output: {output_path}")

    try:
        while True:
            if time.time() - start_wall >= (RUN_HOURS * 3600.0):
                print(f"[Env {env_id}] Time budget reached.")
                break

            cmd_timer += DT
            for i in range(num_robots):
                if cmd_timer[i] >= cmd_dur[i]:
                    cmd_fwd[i], cmd_side[i], cmd_spin[i], cmd_bh[i], cmd_dirt[i], cmd_dur[i] = sample_commands()
                    cmd_timer[i] = 0.0

            norm_mag = np.minimum(1.0, np.sqrt(cmd_fwd**2 + cmd_side**2))
            active = (norm_mag > 0.02) | (np.abs(cmd_spin) > 0.02)
            dir_rad = np.where(norm_mag > 1e-3, np.arctan2(cmd_side, cmd_fwd), 0.0)

            g = blended_gait_vectorized(cmd_fwd, cmd_side, cmd_spin, norm_mag, cmd_dirt)

            urdf_x_filt = (1.0 - URDF_X_ALPHA) * urdf_x_filt + URDF_X_ALPHA * g["urdf_x"]
            h_err = cmd_bh - body_height_curr
            max_step = HEIGHT_RATE * DT
            body_height_curr += np.clip(h_err, -max_step, max_step)

            phase_accum = np.where(active, (phase_accum + g["freq"] * DT) % 1.0, 0.0)
            motion_blend += (active.astype(float) - motion_blend) * STOP_BLEND_SPEED

            p_leg = (phase_accum[:, np.newaxis] + LEG_PHASES_ARR[np.newaxis, :]) % 1.0

            s_phase = np.where(p_leg < 0.5, p_leg, p_leg - 0.5) * 2.0
            cycloid = s_phase - np.sin(2.0 * np.pi * s_phase) / (2.0 * np.pi)
            phase_mult = np.where(p_leg < 0.5, 1.0 - 2.0 * cycloid, -1.0 + 2.0 * cycloid)
            gait_z = np.where(p_leg < 0.5, g["step_h"][:, np.newaxis] * 0.5 * (1.0 - np.cos(2.0 * np.pi * s_phase)), 0.0)

            sign_x = np.where(np.cos(dir_rad) >= 0, 1.0, -1.0)[:, np.newaxis]
            sign_y = np.where(np.sin(dir_rad) >= 0, 1.0, -1.0)[:, np.newaxis]

            tx_trans = g["amp"][:, np.newaxis] * phase_mult * np.abs(np.cos(dir_rad))[:, np.newaxis] * sign_x * norm_mag[:, np.newaxis]
            ty_trans = g["amp"][:, np.newaxis] * phase_mult * np.abs(np.sin(dir_rad))[:, np.newaxis] * sign_y * norm_mag[:, np.newaxis]

            omega = -cmd_spin * g["amp"] * 2.0
            tx_spin = -LEG_ROOTS_Y[np.newaxis, :] * omega[:, np.newaxis] * phase_mult
            ty_spin =  LEG_ROOTS_X[np.newaxis, :] * omega[:, np.newaxis] * phase_mult

            tx = (tx_trans + tx_spin) * motion_blend[:, np.newaxis]
            ty = (ty_trans + ty_spin) * motion_blend[:, np.newaxis]
            tz = body_height_curr[:, np.newaxis] + gait_z

            h, th, kn = solve_leg_ik_vectorized(tx, ty, tz, urdf_x_filt[:, np.newaxis])

            left_mask = IS_LEFT_ARR[np.newaxis, :]
            th = np.where(left_mask, -th, th)
            kn = np.where(left_mask, -kn, kn)

            for i in range(num_robots):
                leg_idx = 0
                for leg in LEG_ORDER:
                    for suf in ["3", "2", "1"]:
                        val = h[i, leg_idx] if suf == "3" else (th[i, leg_idx] if suf == "2" else kn[i, leg_idx])
                        p.setJointMotorControl2(
                            robots[i], joint_maps[i][f"{leg}{suf}"],
                            p.POSITION_CONTROL, targetPosition=val,
                            force=GLOBAL_MAX_TORQUE, maxVelocity=GLOBAL_MAX_VELOCITY,
                            physicsClientId=client_id
                        )
                    leg_idx += 1

            if physics_step % LOG_INTERVAL == 0:
                for i in range(num_robots):
                    states = p.getJointStates(robots[i], joint_ids_flat[i], physicsClientId=client_id)
                    actual_joints = [state[0] for state in states]

                    lin_vel, ang_vel = p.getBaseVelocity(robots[i], physicsClientId=client_id)

                    row = [
                        env_id, phase_accum[i],
                        cmd_fwd[i], cmd_side[i], cmd_spin[i], body_height_curr[i], cmd_dirt[i],
                        lin_vel[0], lin_vel[1], lin_vel[2], ang_vel[0], ang_vel[1], ang_vel[2]
                    ] + actual_joints

                    data_buffer.append(row)

            p.stepSimulation(physicsClientId=client_id)
            sim_time += DT
            physics_step += 1

            if len(data_buffer) >= SAVE_BUFFER_SIZE:
                with open(output_path, 'a', newline='') as f:
                    csv.writer(f).writerows(data_buffer)
                data_buffer = []

            if env_id == 0 and physics_step % (PHYSICS_FREQ * 30) == 0:
                elapsed_min = (time.time() - start_wall) / 60.0
                print(f"[Info] wall {elapsed_min:5.1f} min | sim {sim_time/60:5.1f} min | buffer count {len(data_buffer)}")

    except KeyboardInterrupt:
        pass
    finally:
        if data_buffer:
            with open(output_path, 'a', newline='') as f:
                csv.writer(f).writerows(data_buffer)
        p.disconnect(physicsClientId=client_id)
        print(f"[Env {env_id}] Done. Saved to {output_path}")

# =================================================================
# MAIN SPOOLER
# =================================================================
if __name__ == '__main__':
    processes = []

    print(f"Spawning {NUM_ENVS} parallel environments running {ROBOTS_PER_ENV} robots each...")
    print(f"Total simulated tracking footprint: {NUM_ENVS * ROBOTS_PER_ENV} robots.")

    final_merged_path = "gait_dataset_pybullet_merged.csv"
    env_files = [f"{BASE_OUTPUT_PATH}_env_{i}.csv" for i in range(NUM_ENVS)]

    stop_merger = threading.Event()
    merger_thread = threading.Thread(
        target=continuous_merger_loop,
        args=(env_files, final_merged_path, stop_merger, 120),
        daemon=True
    )
    merger_thread.start()

    for env_id in range(NUM_ENVS):
        p_proc = mp.Process(target=run_environment, args=(env_id, ROBOTS_PER_ENV))
        processes.append(p_proc)
        p_proc.start()

    try:
        for p_proc in processes:
            p_proc.join()
    except KeyboardInterrupt:
        print("\n[Info] Main script interrupted by user. Shutting down cleanly...")
    finally:
        stop_merger.set()
        print("\n[Info] Simulations done. Performing final data sync...")
        time.sleep(2)

        try:
            first_file_written = False
            tmp_final_path = final_merged_path + ".final"

            print("Compiling ultimate dataset using high-speed streaming...")
            with open(tmp_final_path, 'wb') as outfile:
                for f_path in env_files:
                    if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
                        print(f"Streaming data from {f_path}...")
                        with open(f_path, 'rb') as infile:
                            if not first_file_written:
                                shutil.copyfileobj(infile, outfile)
                                first_file_written = True
                            else:
                                infile.readline()  # Skip header
                                shutil.copyfileobj(infile, outfile)

            if first_file_written:
                if os.path.exists(final_merged_path):
                    os.remove(final_merged_path)
                os.rename(tmp_final_path, final_merged_path)
                print("[Success] Done stitching ultimate dataset!")
            else:
                print("[Warning] No data found to merge.")

        except Exception as e:
            print(f"[Error] Failed to compile final dataset: {e}")

        print("Cleaning up temporary environment files...")
        for f_path in env_files:
            if os.path.exists(f_path):
                os.remove(f_path)
        print("Cleanup complete.")
