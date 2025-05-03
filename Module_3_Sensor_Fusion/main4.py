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

# Constants
TAG_SIZE            = 0.152
DEFAULT_SPACING     = 0.152
SPACING_EXCEPTION   = 0.178

# IMU gravity
GRAVITY = np.array([0, 0, -9.81])

R_MEAS = np.diag([
    0.2**2, 0.2**2, 0.2**2,
    np.deg2rad(2)**2,
    np.deg2rad(2)**2,
    np.deg2rad(2)**2
])

# Process noise including bias random-walk (15×15)
Q_PROC = np.diag([
    1e-4, 1e-4, 1e-4,        
    1e-3, 1e-3, 1e-3,        
    np.deg2rad(0.5)**2,      
    np.deg2rad(0.5)**2,      
    np.deg2rad(0.5)**2,      
    1e-6, 1e-6, 1e-6,        
    1e-6, 1e-6, 1e-6       
])

CAM_TO_DRONE_TRANSLATION = np.array([-0.04, 0.0, -0.03]).reshape(3,1)
CAM_TO_DRONE_ROTATION    = R.from_euler('xyz', [-np.pi,0,-np.pi/4]).as_matrix()

# Globals for live plotting
trajectory = []
gt_trajectory = []

# Helper: Euler-rate to body-rate mapping (Z-X-Y sequence)
def G_at(euler_angles):
    phi, theta, psi = euler_angles
    return np.array([
        [ np.cos(theta),               0, -np.cos(phi)*np.sin(theta)],
        [             0,               1,   np.sin(phi)             ],
        [ np.sin(theta),               0,  np.cos(phi)*np.cos(theta)]
    ])

def read_parameters(path):
    '''
    Read camera parameters and tag ids from a text file.
    '''
    txt = open(path,'r').read()
    cam_pat = r'% Camera Matrix.*?\[([^\]]+)\]'
    cm_str  = re.search(cam_pat, txt, re.S).group(1)
    K = np.fromstring(cm_str.replace('...','').replace(';',''),
                      sep=' ').reshape(3,3)
    dist_pat = r'% Distortion parameters.*?\n\[\s*([^\]]+)\s*\]'
    m = re.search(dist_pat, txt, re.S)
    dist = np.fromstring(m.group(1), sep=' ') if m else np.zeros(5)
    tag_pat = r'% Tag ids:\s*\[([^\]]+)\]'
    t_str   = re.search(tag_pat, txt, re.S).group(1)
    t_clean = t_str.replace(',',' ').replace(';',' ')
    tag_mat = np.loadtxt(StringIO(t_clean), dtype=int)
    return K, dist, tag_mat

def get_tag_position(tag_matrix, tag_id):
    '''
    Get the position of a tag in the tag matrix.
    '''
    row,col = np.where(tag_matrix==tag_id)
    row,col = int(row), int(col)
    if col<=2:
        y = col*(TAG_SIZE+DEFAULT_SPACING)
    elif col<=5:
        y = col*(TAG_SIZE+DEFAULT_SPACING)+(SPACING_EXCEPTION-DEFAULT_SPACING)
    else:
        y = col*(TAG_SIZE+DEFAULT_SPACING)+2*(SPACING_EXCEPTION-DEFAULT_SPACING)
    x = row*(TAG_SIZE+DEFAULT_SPACING)
    return x,y

def get_corners(tag_matrix, ids):
    '''
    Get the corners of the tags in the tag matrix.
    '''
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
    '''
    Get the pixel corners of the tags in the image.
    '''
    out = {}
    for i, tid in enumerate(ids):
        out[tid] = {
            'bottom_left':  p1[:,i],
            'bottom_right': p2[:,i],
            'top_right':    p3[:,i],
            'top_left':     p4[:,i]
        }
    return out

def get_camera_pose(tag_pix, tag_world, K, dist):
    '''
    Get the camera pose from the tag positions in the image and world.
    '''
    obj_pts, img_pts = [], []
    for tid, wc in tag_world.items():
        pc = tag_pix[tid]
        for corner in ('top_left','top_right','bottom_right','bottom_left'):
            obj_pts.append([*wc[corner],0.0])
            img_pts.append(pc[corner])
    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    if obj_pts.shape[0]<4 or obj_pts.shape[0]!=img_pts.shape[0]:
        return None, None
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts,img_pts,K,dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None, None
    if not ok: return None, None
    Rcw,_  = cv2.Rodrigues(rvec)
    Rcw    = Rcw.T
    cam_pos = (-Rcw @ tvec).reshape(3)
    off     = -Rcw @ CAM_TO_DRONE_ROTATION.T @ CAM_TO_DRONE_TRANSLATION
    drone_p = cam_pos + off.flatten()
    return drone_p, Rcw

def align_ground_truth(est, gt):
    '''
    Align the estimated and ground truth trajectories.
    '''
    est_ts = sorted(est.keys())
    gt_ts  = sorted(gt.keys())
    out = {}
    for t in est_ts:
        idx = bisect.bisect_left(gt_ts, t)
        if   idx==0: closest=gt_ts[0]
        elif idx==len(gt_ts): closest=gt_ts[-1]
        else:
            b,a = gt_ts[idx-1], gt_ts[idx]
            closest = b if abs(b-t)<abs(a-t) else a
        out[t] = gt[closest]
    return out

def visualize_camera_and_ground_truth(estimated, aligned_gt,
                                      frames, K, dist, tag_matrix,
                                      scale=0.2):
    '''
    Visualize the estimated camera pose and ground truth trajectory 
    and the position mse and orientations.
    '''
    global trajectory, gt_trajectory

    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    trajectory = []
    gt_trajectory = []

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    # error accumulators
    l2_errors = []
    timestamps = []

    # orientation accumulators
    roll_est_list, pitch_est_list, yaw_est_list = [], [], []
    roll_gt_list,  pitch_gt_list,  yaw_gt_list  = [], [], []

    index = 0
    for est_time, state_est in estimated.items():
        # get estimated position
        drone_position = np.array(state_est[0:3])

        # find the matching frame for this index
        if index >= len(frames):
            break
        frm = frames[index]
        ids = frm['id']
        p1 = frm['p1'].reshape(2, -1)
        p2 = frm['p2'].reshape(2, -1)
        p3 = frm['p3'].reshape(2, -1)
        p4 = frm['p4'].reshape(2, -1)

        world_c = get_corners(tag_matrix,
                              ids if hasattr(ids, '__len__') else [ids])
        pix_c   = get_pixel_corners(p1, p2, p3, p4,
                                    ids if hasattr(ids, '__len__') else [ids])

        # re-run PnP to get R_est
        _, R_est = get_camera_pose(pix_c, world_c, K, dist)
        if R_est is None:
            index += 1
            continue

        # ground truth
        gt = aligned_gt[est_time]
        gt_pos = np.array(gt[0:3])
        gt_roll, gt_pitch, gt_yaw = gt[3], gt[4], gt[5]

        # position error
        l2_err = np.linalg.norm(drone_position - gt_pos)
        l2_errors.append(l2_err)
        timestamps.append(est_time)

        # estimated camera-to-world rotation (compensate drone->camera)
        R_cam_world = R_est.T @ CAM_TO_DRONE_ROTATION.T
        roll_e, pitch_e, yaw_e = R.from_matrix(R_cam_world).as_euler('xyz')
        roll_est_list.append(np.degrees(roll_e))
        pitch_est_list.append(np.degrees(pitch_e))
        yaw_est_list.append(np.degrees(yaw_e))

        # ground-truth orientation in degrees
        roll_gt_list.append(np.degrees(gt_roll))
        pitch_gt_list.append(np.degrees(gt_pitch))
        yaw_gt_list.append(np.degrees(gt_yaw))

        # --- 3D plot ---
        ax.clear()
        # draw estimated axes
        x_est = R_cam_world @ np.array([scale, 0, 0]) + drone_position
        y_est = R_cam_world @ np.array([0, scale, 0]) + drone_position
        z_est = R_cam_world @ np.array([0, 0, scale]) + drone_position
        ax.scatter(*drone_position, c='m', s=50)
        ax.plot([drone_position[0], x_est[0]],
                [drone_position[1], x_est[1]],
                [drone_position[2], x_est[2]], 'r')
        ax.plot([drone_position[0], y_est[0]],
                [drone_position[1], y_est[1]],
                [drone_position[2], y_est[2]], 'g')
        ax.plot([drone_position[0], z_est[0]],
                [drone_position[1], z_est[1]],
                [drone_position[2], z_est[2]], 'b')

        # draw GT axes
        R_gt = R.from_euler('xyz', [gt_roll, gt_pitch, gt_yaw]).as_matrix()
        x_gt = R_gt @ np.array([scale, 0, 0]) + gt_pos
        y_gt = R_gt @ np.array([0, scale, 0]) + gt_pos
        z_gt = R_gt @ np.array([0, 0, scale]) + gt_pos
        ax.scatter(*gt_pos, c='c', s=50)
        ax.plot([gt_pos[0], x_gt[0]], [gt_pos[1], x_gt[1]], [gt_pos[2], x_gt[2]], 'r--')
        ax.plot([gt_pos[0], y_gt[0]], [gt_pos[1], y_gt[1]], [gt_pos[2], y_gt[2]], 'g--')
        ax.plot([gt_pos[0], z_gt[0]], [gt_pos[1], z_gt[1]], [gt_pos[2], z_gt[2]], 'b--')

        # plot trajectories
        trajectory.append(drone_position)
        gt_trajectory.append(gt_pos)
        if len(trajectory) > 1:
            t = np.array(trajectory)
            ax.plot(t[:,0], t[:,1], t[:,2], 'm-', label='Est. Traj')
        if len(gt_trajectory) > 1:
            g = np.array(gt_trajectory)
            ax.plot(g[:,0], g[:,1], g[:,2], 'c--', label='GT Traj')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.set_title(f"Cam Pose vs GT at t={est_time}")
        ax.legend()
        plt.draw(); plt.pause(0.001)

        index += 1

    # finalize interactive session
    plt.ioff()

    # Position error over time
    mean_l2 = np.mean(l2_errors)
    print(f"Mean L2 Error: {mean_l2:.4f}")
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(timestamps, l2_errors, label='L2 Pos. Error')
    plt.axhline(mean_l2, color='r', linestyle='--',
                label=f'Mean: {mean_l2:.4f}')
    plt.xlabel("Timestamp"); plt.ylabel("Position Error (m)")
    plt.title("Position Error over Time")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.ylim([0, 1.0])
    # plt.show()

    # Orientation comparison
    fig3, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(timestamps, roll_est_list, label='Roll Est.')
    axes[0].plot(timestamps, roll_gt_list,  label='Roll GT')
    axes[0].set_ylabel("Roll (°)"); axes[0].legend(); axes[0].grid(True)
    axes[0].set_ylim([-30, 30])

    axes[1].plot(timestamps, pitch_est_list, label='Pitch Est.')
    axes[1].plot(timestamps, pitch_gt_list,  label='Pitch GT')
    axes[1].set_ylabel("Pitch (°)"); axes[1].legend(); axes[1].grid(True)
    axes[1].set_ylim([-30, 30])

    axes[2].plot(timestamps, yaw_est_list, label='Yaw Est.')
    axes[2].plot(timestamps, yaw_gt_list,  label='Yaw GT')
    axes[2].set_ylabel("Yaw (°)"); axes[2].set_xlabel("Timestamp")
    axes[2].legend(); axes[2].grid(True); axes[2].set_ylim([-30, 30])

    fig3.suptitle("Estimated vs Ground Truth Orientation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

class EKF:
    def __init__(self, Q, R, x0=None, P0=None):
        self.Q = Q
        self.R = R
        self.x = np.zeros((15,1)) if x0 is None else x0.reshape(15,1)
        self.P = np.eye(15)*1e-2 if P0 is None else P0

    def predict(self, dt, u_omega, u_acc):
        '''
        Predict the next state using the IMU measurements.
        '''
        # unpack
        p = self.x[0:3,0]
        v = self.x[3:6,0]
        phi,theta,psi = self.x[6:9,0]
        bg = self.x[9:12,0]
        ba = self.x[12:15,0]

        Rbw = R.from_euler('xyz',[phi,theta,psi]).as_matrix()
        omega_corr = u_omega - bg
        acc_corr   = u_acc   - ba
        G = G_at([phi,theta,psi])

        # derivatives
        p_dot = v
        v_dot = Rbw @ acc_corr + GRAVITY
        q_dot = np.linalg.inv(G) @ omega_corr
        # biases random-walk
        bg_dot = np.zeros(3)
        ba_dot = np.zeros(3)

        # state Euler update
        self.x[0:3,0]  += p_dot * dt
        self.x[3:6,0]  += v_dot * dt
        self.x[6:9,0]  += q_dot * dt
        # biases remain

        # build Jacobian F
        F = np.eye(15)
        F[0:3,3:6] = np.eye(3)*dt
        F[3:6,12:15] = -Rbw * dt
        F[6:9,9:12]  = -np.linalg.inv(G)*dt

        # covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        '''
        Update the state using the PnP measurement.
        '''
        H = np.zeros((6,15))
        H[0:3,0:3] = np.eye(3)
        H[3:6,6:9] = np.eye(3)

        z = z.reshape(6,1)
        y = z - H @ self.x
        y[3:6,0] = (y[3:6,0]+np.pi)%(2*np.pi)-np.pi

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(15) - K @ H) @ self.P

    def wrap_angle_to_zero(self, angle, tol=1e-2):
        '''
        Wrap angle to zero (±π) using modulo operation.
        '''
        a = (angle+np.pi)%(2*np.pi) - np.pi
        mask = np.abs(np.abs(a)-np.pi)<tol
        a[mask]=0.0
        return a

class ParticleFilter:
    def __init__(self, num_particles, Q, R, init_mean, init_cov):
        self.np = num_particles
        self.Q  = Q
        self.R  = R
        self.particles = np.random.multivariate_normal(
            init_mean.flatten(), init_cov, size=self.np)
        self.weights   = np.ones(self.np)/self.np

    def predict(self, dt, u_omega, u_acc):
        '''
        Predict the next state using the IMU measurements.
        '''
        F = np.eye(15)
        F[0,3]=F[1,4]=F[2,5]=dt
        self.particles = (self.particles @ F.T
                          + np.random.multivariate_normal(np.zeros(15),
                                                          self.Q,
                                                          size=self.np))

        # add process noise
        self.particles += np.random.multivariate_normal(
            np.zeros(15), self.Q, size=self.np)

    def update(self, z):
        '''
        Update the state using the PnP measurement.
        '''
        hx = np.hstack((self.particles[:,0:3], self.particles[:,6:9]))
        y  = hx - z.reshape(1,6)
        y[:,3:6] = (y[:,3:6]+np.pi)%(2*np.pi)-np.pi

        Rinv = np.linalg.inv(self.R)
        exponent = -0.5 * np.sum((y @ Rinv) * y, axis=1)
        w = np.exp(exponent)
        self.weights = w/np.sum(w)

    def resample(self):
        '''
        Resample the particles based on their weights.
        '''
        cdf = np.cumsum(self.weights)
        start = np.random.random()/self.np
        positions = start + np.arange(self.np)/self.np
        idx = np.searchsorted(cdf, positions)
        self.particles = self.particles[idx]
        self.weights[:] = 1.0/self.np

    def estimate(self):
        '''
        Estimate the state using the particles and weights.
        '''
        max_idx = np.argmax(self.weights)
        x_max   = self.particles[max_idx]
        x_mean  = np.mean(self.particles, axis=0)
        x_wmean = np.average(self.particles, axis=0, weights=self.weights)
        return x_max, x_mean, x_wmean

if __name__ == "__main__":
    # ask which filter
    filt = input("Select filter type ('ekf' or 'pf'): ").strip().lower()
    if filt not in ('ekf','pf'):
        raise ValueError("Invalid filter type. Choose 'ekf' or 'pf'.")

    # select trajectory
    traj = input("Select trajectory (0-7): ").strip()

    # load all data
    K, dist_coeffs, tag_matrix = read_parameters("Module_3_Sensor_Fusion/data/parameters.txt")
    mat      = sio.loadmat("Module_3_Sensor_Fusion/data/studentdata" + traj + ".mat", simplify_cells=True)
    frames   = mat['data']
    gt_times = mat['time'].flatten()
    gt_vicon = mat['vicon']      # shape (6, N)

    # 1) find first measurement z0
    init_z     = None
    init_time  = None
    for frm in frames:
        ids = frm['id']
        p1, p2 = frm['p1'].reshape(2,-1), frm['p2'].reshape(2,-1)
        p3, p4 = frm['p3'].reshape(2,-1), frm['p4'].reshape(2,-1)
        world_c = get_corners(tag_matrix, ids if hasattr(ids,'__len__') else [ids])
        pix_c   = get_pixel_corners(p1,p2,p3,p4, ids if hasattr(ids,'__len__') else [ids])
        dp, R_est = get_camera_pose(pix_c, world_c, K, dist_coeffs)
        if dp is None:
            continue

        # convert R_est → roll,pitch,yaw
        r = R.from_matrix(R_est)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        # wrap roll exactly as in EKF
        roll = (roll - np.pi + np.pi) % (2*np.pi) - np.pi
        # build 15-dim init
        v0 = np.zeros(3)
        bg0 = np.zeros(3)
        ba0 = np.zeros(3)
        init_z = np.hstack((dp, v0, [roll,pitch,yaw], bg0, ba0))
        init_time = frm['t']
        break

    if init_z is None:
        raise RuntimeError("Never got a valid PnP measurement!")

    # prepare storage
    estimated     = {}
    ground_truth  = {}

    # seed ground truth at init_time
    idx = bisect.bisect_left(gt_times, init_time)
    if idx == len(gt_times):
        idx -= 1
    elif idx > 0 and abs(gt_times[idx-1]-init_time) < abs(gt_times[idx]-init_time):
        idx -= 1
    gt0 = gt_vicon[:, idx]

    # 2) instantiate filter at init_z
    if filt == 'ekf':
        ekf = EKF(Q_PROC, R_MEAS, x0=init_z, P0=np.eye(15)*1e-2)
        estimated[init_time]    = init_z
    else:
        # use EKF default P for PF init covariance
        tmp_ekf   = EKF(Q=Q_PROC, R=R_MEAS)
        init_cov  = tmp_ekf.P
        num_p     = int(input("Number of particles (e.g. 1000): "))
        pf = ParticleFilter(num_p, Q_PROC, R_MEAS, init_z, np.eye(15)*1e-2)
        estimated[init_time]    = init_z

    ground_truth[init_time] = gt0

    # 3) continue through the rest of the frames
    prev_t = init_time
    for frm in frames:
        t = frm['t']
        # skip everything up to and including init_time
        if t <= init_time:
            continue

        dt = t - prev_t
        prev_t = t

        if traj == '0':
            omg = frm['drpy']; acc = frm['acc']
        
        else:
            omg = frm['omg']; acc = frm['acc']

        # predict
        if filt=='ekf':
            ekf.predict(dt, omg, acc)
        else:
            pf.predict(dt, omg, acc)

        # PnP measurement
        ids = frm['id']
        p1, p2 = frm['p1'].reshape(2,-1), frm['p2'].reshape(2,-1)
        p3, p4 = frm['p3'].reshape(2,-1), frm['p4'].reshape(2,-1)
        world_c = get_corners(tag_matrix, ids if hasattr(ids,'__len__') else [ids])
        pix_c   = get_pixel_corners(p1,p2,p3,p4, ids if hasattr(ids,'__len__') else [ids])
        dp, R_est = get_camera_pose(pix_c, world_c, K, dist_coeffs)
        if dp is None:
            continue

        r = R.from_matrix(R_est)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        if filt == 'ekf':
            roll = ekf.wrap_angle_to_zero(np.array([roll-np.pi]))[0]
        else:
            roll = (roll - np.pi + np.pi) % (2*np.pi) - np.pi

        z = np.hstack((dp, [roll, pitch, yaw]))

        # update + (resample)
        if filt == 'ekf':
            ekf.update(z)
            state_est = ekf.x.flatten()
        else:
            pf.update(z)
            pf.resample()
            _, _, state_est = pf.estimate()

        estimated[t] = state_est

        # align ground truth
        idx = bisect.bisect_left(gt_times, t)
        if idx == len(gt_times):
            idx -= 1
        elif idx > 0 and abs(gt_times[idx-1]-t) < abs(gt_times[idx]-t):
            idx -= 1
        ground_truth[t] = gt_vicon[:, idx]

    # 4) visualize
    aligned_gt = align_ground_truth(estimated, ground_truth)
    visualize_camera_and_ground_truth(estimated, aligned_gt,
                                      frames, K, dist_coeffs, tag_matrix)

