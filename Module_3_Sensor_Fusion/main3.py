#!/usr/bin/env python3
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import re
from io import StringIO
from scipy.spatial.transform import Rotation as R
import bisect
import pdb

# --- User‐tunable constants ---
TAG_SIZE            = 0.152
DEFAULT_SPACING     = 0.152
SPACING_EXCEPTION   = 0.178

# If you have a precomputed measurement‐noise covariance from your
# `estimate_covariance` routine, you can paste it here. Otherwise we
# just pick small diagonal values.
R_MEAS = np.array([[ 3.95854248e-03,  9.31191728e-04, -7.23132222e-04, -5.38995532e-04,
   2.62321142e-03,  2.25533503e-04],
 [ 9.31191728e-04,  3.97294235e-03, -2.02594591e-04, -3.30813450e-03,
  -8.39100363e-04,  2.62262109e-04],
 [-7.23132222e-04, -2.02594591e-04,  6.26121029e-04, -4.89942540e-04,
  -3.87337255e-04, -6.92359779e-05,],
 [-5.38995532e-04, -3.30813450e-03, -4.89942540e-04,  8.28882330e-03,
   8.38458145e-04,  4.47964321e-05,],
 [ 2.62321142e-03, -8.39100363e-04, -3.87337255e-04,  8.38458145e-04,
   8.27031387e-03, -6.12654779e-05,],
 [ 2.25533503e-04,  2.62262109e-04, -6.92359779e-05,  4.47964321e-05,
  -6.12654779e-05,  1.11024933e-04,]])

# Process‐noise covariance:  
#   small for velocities/orientation, larger for unmodeled accelerations
Q_PROC = np.diag([
    1e-4, 1e-4, 1e-4,    # pos
    1e-3, 1e-3, 1e-3,    # vel
    np.deg2rad(0.5)**2,  # roll
    np.deg2rad(0.5)**2,  # pitch
    np.deg2rad(0.5)**2   # yaw
])

# Camera‐to‐drone rigid‐body offset (from your original code)
CAM_TO_DRONE_TRANSLATION = np.array([-0.04, 0.0, -0.03]).reshape(3,1)
CAM_TO_DRONE_ROTATION    = R.from_euler('xyz', [np.pi,0,-np.pi/4]).as_matrix()


def read_parameters(path):
    """Read K, distortion, IMU‐camera translation/yaw, tag layout."""
    with open(path,'r') as f:
        txt = f.read()

    # --- Intrinsics ---
    cam_pat = r'% Camera Matrix.*?\[([^\]]+)\]'
    cm_str  = re.search(cam_pat, txt, re.S).group(1)
    K       = np.fromstring(cm_str.replace('...','').replace(';',''),
                            sep=' ').reshape(3,3)

    # --- Distortion ---
    dist_pat = r'% Distortion parameters.*?\n\[\s*([^\]]+)\s*\]'
    m = re.search(dist_pat, txt, re.S)
    dist = np.fromstring(m.group(1), sep=' ') if m else np.zeros(5)

    # --- Tag IDs layout ---
    tag_pat = r'% Tag ids:\s*\[([^\]]+)\]'
    t_str   = re.search(tag_pat, txt, re.S).group(1)
    t_clean = t_str.replace(',',' ').replace(';',' ')
    tag_mat = np.loadtxt(StringIO(t_clean), dtype=int)

    return K, dist, tag_mat


def get_tag_position(tag_matrix, tag_id):
    """Given a tag’s row/col in the printed layout, compute its world (x,y)."""
    row, col = np.where(tag_matrix==tag_id)
    row, col = int(row), int(col)
    # handle expanded spacing in middle columns
    if   col<=2: y = col*(TAG_SIZE+DEFAULT_SPACING)
    elif col<=5: y = col*(TAG_SIZE+DEFAULT_SPACING)+(SPACING_EXCEPTION-DEFAULT_SPACING)
    else:        y = col*(TAG_SIZE+DEFAULT_SPACING)+2*(SPACING_EXCEPTION-DEFAULT_SPACING)
    x = row*(TAG_SIZE+DEFAULT_SPACING)
    return x, y


def get_corners(tag_matrix, ids):
    """World‐frame corners of each tag (Z=0)."""
    out = {}
    for tid in ids:
        x,y = get_tag_position(tag_matrix, tid)
        out[tid] = {
            'bottom_left':  (x+TAG_SIZE, y),
            'bottom_right': (x+TAG_SIZE, y+TAG_SIZE),
            'top_right':    (x,         y+TAG_SIZE),
            'top_left':     (x,         y)
        }
    return out


def get_pixel_corners(p1,p2,p3,p4, ids):
    """Map pixel arrays into a dict like your original code."""
    out = {}
    for i, tid in enumerate(ids):
        out[tid] = {
            'bottom_left':  p1[:,i],
            'bottom_right': p2[:,i],
            'top_right':    p3[:,i],
            'top_left':     p4[:,i]
        }
    return out


def rotationMatrixToEulerAngles(Rm):
    """ZYX→(roll,pitch,yaw) in radians."""
    sy = np.sqrt(Rm[0,0]**2 + Rm[1,0]**2)
    singular = sy<1e-6
    if not singular:
        yaw   = np.arctan2( Rm[1,0], Rm[0,0])
        pitch = np.arctan2(-Rm[2,0], sy)
        roll  = np.arctan2( Rm[2,1], Rm[2,2])
    else:
        yaw   = np.arctan2(-Rm[1,2], Rm[1,1])
        pitch = np.arctan2(-Rm[2,0], sy)
        roll  = 0
    return np.array([roll,pitch,yaw])

def get_camera_pose(tag_pixel_corners, tag_world_corners, K, distortion_coeffs):
    """
    Returns (drone_position, R_est) or (None, None) if PnP cannot run.
    """
    object_points = []
    image_points  = []

    # collect 1-to-1 correspondences
    for tag_id, world_corners in tag_world_corners.items():
        pixel_corners = tag_pixel_corners[tag_id]
        for corner in ('top_left','top_right','bottom_right','bottom_left'):
            object_points.append((*world_corners[corner], 0.0))
            image_points .append(pixel_corners[corner])

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points  = np.asarray(image_points,  dtype=np.float64)

    # --- guard: need at least 4 points ---
    if object_points.shape[0] < 4:
        # print(f"Skipping PnP: only {object_points.shape[0]} points")
        return None, None

    # double-check they match
    if object_points.shape[0] != image_points.shape[0]:
        # print("Point count mismatch, skipping PnP")
        return None, None

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error as e:
        # print("OpenCV PnP error:", e)
        return None, None

    if not success:
        return None, None

    # convert to world→cam, then apply your drone offset
    R_cam2world, _ = cv2.Rodrigues(rvec)
    R_cam2world     = R_cam2world.T
    cam_pos         = (-R_cam2world @ tvec).reshape(3)
    drone_off       = -R_cam2world @ CAM_TO_DRONE_ROTATION.T @ CAM_TO_DRONE_TRANSLATION
    drone_pos       = cam_pos + drone_off.reshape(3)

    return drone_pos, R_cam2world

class EKF:
    """
    State: [px,py,pz, vx,vy,vz, roll,pitch,yaw]ᵀ (9×1)
    Process:  const‐vel + constant orientation
    Measurement: direct observe [px,py,pz,roll,pitch,yaw]
    """
    def __init__(self, Q, R, x0=None, P0=None):
        self.n = 9
        self.Q = Q
        self.R = R
        self.x = np.zeros((self.n,1)) if x0 is None else x0.reshape(self.n,1)
        self.P = np.eye(self.n)*1e-2 if P0 is None else P0

    def predict(self, dt):
        F = np.eye(self.n)
        # pos ← pos + v*dt
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt
        # vel, orientation stay constant
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        z: 6×1 measurement [px,py,pz, roll,pitch,yaw]ᵀ
        H: 6×9
        """
        H = np.zeros((6,self.n))
        H[0,0]=1; H[1,1]=1; H[2,2]=1
        H[3,6]=1; H[4,7]=1; H[5,8]=1

        z = z.reshape(6,1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def wrap_angle_to_zero(self, angle, tol=1e-2):
        """
        Wraps angles near ±pi (or multiples like 2pi) to 0, considering the sign.

        Args:
            angle (float or np.ndarray): Input angle(s) in radians.
            tol (float): Tolerance around pi or 2pi within which to wrap to 0.

        Returns:
            wrapped_angle: Angle after wrapping
        """

        # Normalize angle to [-pi, pi]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        # Set small angles around ±pi to zero
        wrapped_angle = np.where(np.abs(np.abs(angle) - np.pi) < tol, 0.0, angle)

        return wrapped_angle    


if __name__ == "__main__":
    # 1) Load camera params
    K, dist_coeffs, tag_matrix = read_parameters(
        "Module_2_PnP/data/parameters.txt"
    )

    # 2) Load one studentdata file
    mat = sio.loadmat(
        "Module_2_PnP/data/studentdata1.mat",
        simplify_cells=True
    )
    frames      = mat['data']
    gt_times    = mat['time'].flatten()    # 1D array of timestamps
    gt_vicon    = mat['vicon']             # shape (6, N): [x,y,z, roll,pitch,yaw]

    # 3) Initialize EKF
    ekf = EKF(Q=Q_PROC, R=R_MEAS)

    raw_states  = []
    filt_states = []
    times       = []

    # 4) Loop through frames, do predict + PnP + update
    prev_t = frames[0]['t']
    for frame in frames:
        t = frame['t']
        dt = t - prev_t
        prev_t = t

        # 4a) EKF predict
        ekf.predict(dt)

        # 4b) Build PnP measurement z
        ids = frame['id']
        p1  = frame['p1'].reshape(2,-1)
        p2  = frame['p2'].reshape(2,-1)
        p3  = frame['p3'].reshape(2,-1)
        p4  = frame['p4'].reshape(2,-1)

        world_c  = get_corners(tag_matrix,
                               ids if hasattr(ids,'__len__') else [ids])
        pixel_c  = get_pixel_corners(p1,p2,p3,p4,
                                     ids if hasattr(ids,'__len__') else [ids])
        dp, R_est = get_camera_pose(pixel_c, world_c, K, dist_coeffs)

        if dp is not None:
            roll,pitch,yaw = rotationMatrixToEulerAngles(R_est)
            z = np.hstack((dp, [ekf.wrap_angle_to_zero(roll-np.pi),pitch,yaw]))
            raw_states.append(z)
            ekf.update(z)
            filt_states.append(ekf.x.flatten())
            times.append(t)

    raw_states  = np.array(raw_states)    # shape (M,6)
    filt_states = np.array(filt_states)   # shape (M,9)
    times       = np.array(times)         # shape (M,)

    # 5) Align ground truth to our measurement times
    gt_states = []
    for t in times:
        idx = bisect.bisect_left(gt_times, t)
        if idx == len(gt_times):
            idx -= 1
        elif idx>0 and abs(gt_times[idx-1]-t) < abs(gt_times[idx]-t):
            idx -= 1
        gt_states.append(gt_vicon[:, idx])
    gt_states = np.array(gt_states)  # shape (M,6)

    # 6) 3D Trajectory Plot
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(gt_states[:,0],
            gt_states[:,1],
            gt_states[:,2],
            'c--', label='Ground Truth')
    ax.plot(filt_states[:,0],
            filt_states[:,1],
            filt_states[:,2],
            'k-',  label='EKF Estimate')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.set_title("3D Trajectory: Vicon vs EKF")
    plt.tight_layout()

    # 7) Time‐Series Comparison
    fig2, axs = plt.subplots(2, 3, figsize=(12,6), sharex=True)
    labels_pos = ['X (m)', 'Y (m)', 'Z (m)']
    labels_ang = ['Roll (°)', 'Pitch (°)', 'Yaw (°)']
    for i in range(3):
        # Position
        axs[0,i].plot(times, raw_states[:,i],    'r.', label='raw PnP')
        axs[0,i].plot(times, filt_states[:,i],   'k-', label='EKF estimate')
        axs[0,i].plot(times, gt_states[:,i],     'c--',label='Ground truth')
        axs[0,i].set_ylabel(labels_pos[i])
        axs[0,i].legend(loc='upper right')

        # Orientation
        axs[1,i].plot(times, np.rad2deg(raw_states[:,3+i]),    'r.', label='raw PnP')
        axs[1,i].plot(times, np.rad2deg(filt_states[:,6+i]),   'k-',  label='EKF estimate')
        axs[1,i].plot(times, np.rad2deg( gt_states[:,3+i]),    'c--', label='Ground truth')
        axs[1,i].set_ylabel(labels_ang[i])
        axs[1,i].legend(loc='upper right')

    axs[1,0].set_xlabel("Time (s)")
    axs[1,1].set_xlabel("Time (s)")
    axs[1,2].set_xlabel("Time (s)")
    plt.suptitle("Position and Orientation vs Time")
    plt.tight_layout(rect=[0,0.03,1,0.95])

    plt.show()