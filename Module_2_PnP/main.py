import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2   
import re
from io import StringIO
import pdb
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import bisect

# global constants
TAG_SIZE = 0.152
DEFAULT_SPACING = 0.152
SPACING_EXCEPTION = 0.178

CAM_TO_DRONE_TRANSLATION = np.array([-0.04, 0.0, -0.03]).T
CAM_TO_DRONE_ROTATION = R.from_euler('xyz', [-np.pi, 0, -np.pi/4]).as_matrix()

def read_parameters(parameters):

     # --- Extract Camera Matrix ---
    camera_matrix_pattern = r'% Camera Matrix.*?\[([^\]]+)\]'
    camera_matrix_str = re.search(camera_matrix_pattern, parameters, re.S).group(1)
    K = np.fromstring(camera_matrix_str.replace('...', '').replace(';', ''), sep=' ').reshape(3, 3)
    print("Camera Matrix:\n", K)

    # Corrected distortion regex to handle line breaks and optional spaces
    distortion_pattern = r'% Distortion parameters.*?\n\[\s*([^\]]+)\s*\]'
    distortion_match = re.search(distortion_pattern, parameters, re.S)

    if distortion_match:
        distortion_str = distortion_match.group(1)
        distortion_coeffs = np.fromstring(distortion_str, sep=' ')
        print("Distortion Coefficients (k1, k2, p1, p2, k3):\n", distortion_coeffs)
    else:
        print("Distortion parameters not found!")

    # --- Extract Camera-IMU Calibration ---
    imu_pattern = r'% Camera-IMU Calibration.*?XYZ = \[([^\]]+)\];\s*Yaw = ([^\n]+);'
    imu_match = re.search(imu_pattern, parameters, re.S)
    imu_translation = np.fromstring(imu_match.group(1), sep=',')
    imu_yaw = float(eval(imu_match.group(2), {"pi": np.pi}))
    print("IMU Translation (XYZ):\n", imu_translation)
    print("IMU Yaw (rad):", imu_yaw)

    # --- Extract Tag IDs ---
    tag_pattern = r'% Tag ids:\s*\[([^\]]+)\]'
    tag_str = re.search(tag_pattern, parameters, re.S).group(1)
    tag_str_clean = tag_str.replace(',', ' ').replace(';', ' ')
    tag_matrix = np.loadtxt(StringIO(tag_str_clean), dtype=int)
    print("Tag IDs Matrix:\n", tag_matrix)

    return K, distortion_coeffs, imu_translation, imu_yaw, tag_matrix

# Load the camera parameters
parameters_path = 'Module_2_PnP/data/parameters.txt'
with open(parameters_path, 'r') as file:
    parameters = file.read()

K, distortion_coeffs, imu_translation, imu_yaw, tag_matrix = read_parameters(parameters)

trajectory = []
gt_trajectory = []

def align_ground_truth(estimated_dict, ground_truth_dict):
    """
    Aligns ground truth samples to each estimated sample based on the closest timestamp.

    Parameters:
    - estimated_dict: {timestamp: (rvec, tvec)}
    - ground_truth_dict: {timestamp: ground_truth_array}

    Returns:
    - aligned_ground_truth: {est_timestamp: ground_truth_array}
    """
    # Sorted timestamps
    est_timestamps = sorted(estimated_dict.keys())
    gt_timestamps = sorted(ground_truth_dict.keys())

    aligned_ground_truth = {}

    for est_time in est_timestamps:
        # Use bisect to find closest ground truth timestamp
        idx = bisect.bisect_left(gt_timestamps, est_time)

        if idx == 0:
            closest_gt_time = gt_timestamps[0]
        elif idx == len(gt_timestamps):
            closest_gt_time = gt_timestamps[-1]
        else:
            before = gt_timestamps[idx - 1]
            after = gt_timestamps[idx]
            # Pick the closest timestamp
            closest_gt_time = before if abs(before - est_time) < abs(after - est_time) else after

        aligned_ground_truth[est_time] = ground_truth_dict[closest_gt_time]

    return aligned_ground_truth

def visualize_camera_and_ground_truth(estimated, aligned_gt, scale=0.2):
    global trajectory, gt_trajectory

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
    for est_time, drone_position in estimated.items():
        # get estimated rotation
        R_est = None
        if index < len(data['data']):
            _, R_est = estimate_pose(data['data'][index])
        if drone_position is None or R_est is None:
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

        # camera‐to‐world rotation
        R_cam_world = R_est.T @ CAM_TO_DRONE_ROTATION.T
        roll_e, pitch_e, yaw_e = R.from_matrix(R_cam_world).as_euler('xyz')

        # store angles (in degrees)
        roll_est_list.append(np.degrees(roll_e))
        pitch_est_list.append(np.degrees(pitch_e))
        yaw_est_list.append(np.degrees(yaw_e))
        roll_gt_list.append(np.degrees(gt_roll))
        pitch_gt_list.append(np.degrees(gt_pitch))
        yaw_gt_list.append(np.degrees(gt_yaw))

        # --- 3D plot (same as before) ---
        ax.clear()
        # estimated axes
        x_est = R_cam_world @ np.array([scale,0,0]) + drone_position
        y_est = R_cam_world @ np.array([0,scale,0]) + drone_position
        z_est = R_cam_world @ np.array([0,0,scale]) + drone_position
        ax.scatter(*drone_position, c='m', s=50)
        ax.plot([drone_position[0], x_est[0]], [drone_position[1], x_est[1]], [drone_position[2], x_est[2]], 'r')
        ax.plot([drone_position[0], y_est[0]], [drone_position[1], y_est[1]], [drone_position[2], y_est[2]], 'g')
        ax.plot([drone_position[0], z_est[0]], [drone_position[1], z_est[1]], [drone_position[2], z_est[2]], 'b')

        # GT axes
        R_gt = R.from_euler('xyz',[gt_roll,gt_pitch,gt_yaw]).as_matrix()
        x_gt = R_gt @ np.array([scale,0,0]) + gt_pos
        y_gt = R_gt @ np.array([0,scale,0]) + gt_pos
        z_gt = R_gt @ np.array([0,0,scale]) + gt_pos
        ax.scatter(*gt_pos, c='c', s=50)
        ax.plot([gt_pos[0], x_gt[0]], [gt_pos[1], x_gt[1]], [gt_pos[2], x_gt[2]], 'r--')
        ax.plot([gt_pos[0], y_gt[0]], [gt_pos[1], y_gt[1]], [gt_pos[2], y_gt[2]], 'g--')
        ax.plot([gt_pos[0], z_gt[0]], [gt_pos[1], z_gt[1]], [gt_pos[2], z_gt[2]], 'b--')

        # trajectories
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
        plt.draw(); plt.pause(0.05)

        index += 1

    # position error plot (unchanged)
    mean_l2 = np.mean(l2_errors)
    print(f"Mean L2 Error: {mean_l2:.4f}")
    plt.ioff()
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(timestamps, l2_errors, label='L2 Pos. Error')
    plt.axhline(mean_l2, color='r', linestyle='--', label=f'Mean: {mean_l2:.4f}')
    plt.xlabel("Timestamp"); plt.ylabel("Position Error (m)")
    plt.title("Position Error over Time")
    plt.legend(); plt.grid(True); plt.tight_layout()
    # set y lim
    plt.ylim([0, 1.0])

    # --- Orientation comparison plot ---
    fig3, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    # Roll
    axes[0].plot(timestamps, roll_est_list, label='Roll Est.')
    axes[0].plot(timestamps, roll_gt_list,  label='Roll GT')
    axes[0].set_ylabel("Roll (°)")
    axes[0].legend(); axes[0].grid(True)
    # set axes limits
    axes[0].set_ylim([-30, 30])

    # Pitch
    axes[1].plot(timestamps, pitch_est_list, label='Pitch Est.')
    axes[1].plot(timestamps, pitch_gt_list,  label='Pitch GT')
    axes[1].set_ylabel("Pitch (°)")
    axes[1].legend(); axes[1].grid(True)
    axes[1].set_ylim([-30, 30])
    # Yaw
    axes[2].plot(timestamps, yaw_est_list, label='Yaw Est.')
    axes[2].plot(timestamps, yaw_gt_list,  label='Yaw GT')
    axes[2].set_ylabel("Yaw (°)"); axes[2].set_xlabel("Timestamp")
    axes[2].legend(); axes[2].grid(True)
    axes[2].set_ylim([-30, 30])

    fig3.suptitle("Estimated vs Ground Truth Orientation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix to Euler angles (yaw, pitch, roll) in radians.
    Rotation order is ZYX.
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1, 0], R[0, 0])   # Z axis
        pitch = np.arctan2(-R[2, 0], sy)     # Y axis
        roll = np.arctan2(R[2, 1], R[2, 2])  # X axis
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    return np.degrees([roll, pitch, yaw])


def get_camera_pose(tag_pixel_corners, tag_world_corners, K, distortion_coeffs):

    object_points = []
    image_points = []

    # print("len(tag_world_corners):", len(tag_world_corners))

    for tag_id in tag_world_corners:
        world_corners = tag_world_corners[tag_id]
        pixel_corners = tag_pixel_corners[tag_id]

        for corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            x, y = world_corners[corner]
            object_points.append([x, y, 0.0])  # Z is zero (all tags lie on the ground plane)
            image_points.append(pixel_corners[corner])

    # pdb.set_trace()
    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, distortion_coeffs)

    if success:

        return rvec, tvec

    else:
        print("PnP failed.")
        return None, None 

def get_pixel_corners(p1, p2, p3, p4, tag_id):

    tag_pixel_corners = {}
    for i in range(len(tag_id)):
        # pdb.set_trace()
        tag_pixel_corners[tag_id[i]] = {
            'bottom_left': p1[:, i],
            'bottom_right': p2[:, i],
            'top_right': p3[:, i],
            'top_left': p4[:, i],
        }

    return tag_pixel_corners

# Function to get the (x, y) position of a tag from its ID
def get_tag_position(tag_matrix, tag_id):
    # Find the row and column of the tag
    row, col = np.where(tag_matrix == tag_id)
    row, col = row[0], col[0]  # Extract the scalar values

    # X, Y coordinates in the world frame
    if col <= 2:
        y_position = col * (TAG_SIZE + DEFAULT_SPACING)
    elif 3 <= col <= 5:
        y_position = col * (TAG_SIZE + DEFAULT_SPACING) + (SPACING_EXCEPTION - DEFAULT_SPACING)
    else:  # for columns 6 and more
        y_position = col * (TAG_SIZE + DEFAULT_SPACING) + (SPACING_EXCEPTION - DEFAULT_SPACING) * 2
    
    x_position = row * (TAG_SIZE + DEFAULT_SPACING)
    
    return x_position, y_position

def get_corners(tag_matrix, tag_ids):
    corners = {}

    for tag_id in tag_ids:
        x, y = get_tag_position(tag_matrix, tag_id)

        # Each tag has 4 corners (top-left, top-right, bottom-left, bottom-right)
        corners[tag_id] = {
            'bottom_left': (x + TAG_SIZE, y),
            'bottom_right': (x + TAG_SIZE, y + TAG_SIZE),  
            'top_right': (x, y + TAG_SIZE),  
            'top_left': (x, y),  
        }

    return corners

def estimate_pose(frame_data, scale=0.2):

    rvec = None
    tvec = None

    # extract data
    frame = frame_data['img']
    tag_ids = frame_data['id']
    p1 = frame_data['p1'].reshape(2, -1)
    p2 = frame_data['p2'].reshape(2, -1)
    p3 = frame_data['p3'].reshape(2, -1)
    p4 = frame_data['p4'].reshape(2, -1)

    # get the corners of the tags
    if isinstance(tag_ids, int):
        tag_world_corners = get_corners(tag_matrix, [tag_ids])
        tag_pixel_corners = get_pixel_corners(p1, p2, p3, p4, [tag_ids])

    else:
        tag_world_corners = get_corners(tag_matrix, tag_ids)
        tag_pixel_corners = get_pixel_corners(p1, p2, p3, p4, tag_ids)

    if isinstance(tag_ids, int) or len(tag_ids) > 1:
        rvec, tvec = get_camera_pose(tag_pixel_corners, tag_world_corners, K, distortion_coeffs)
    
    if rvec is not None and tvec is not None:

        # ------------------------- Estimated Pose -------------------------
        # World wrt to cam
        R_est, _ = cv2.Rodrigues(rvec)
        # Cam wrt to world
        R_est = R_est.T
        camera_position = (-R_est @ tvec.reshape(3)).flatten()
        drone_offset = - R_est @ CAM_TO_DRONE_ROTATION.T @ CAM_TO_DRONE_TRANSLATION
        drone_position = camera_position + drone_offset

        return drone_position, R_est
    else:
        return None, None

def save_images(data):

    for i in range(len(data['data'])):
        img = data['data'][i]['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # save the image
        cv2.imwrite(f"Module_2_PnP/test/img_{i}.png", img)
    
    return None

def wrap_angle_to_zero(angle, tol=1e-2):

    # Normalize angle to [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    # Set small angles around ±pi to zero
    wrapped_angle = np.where(np.abs(np.abs(angle) - np.pi) < tol, 0.0, angle)

    return wrapped_angle

def estimate_covariance(estimated, aligned_gt):

    index = 0
    residuals = []
    for est_time, drone_position in estimated.items():

        R_est = None
        if index < len(data['data']):
            _, R_est = estimate_pose(data['data'][index])

        if drone_position is None or R_est is None:
            # print(f"Skipping frame {index} for covariance estimation due to PnP failure.")
            index += 1
            continue

        gt_data = aligned_gt[est_time]

        # Convert estimated rotation matrix to roll, pitch, yaw
        r = R.from_matrix(R_est)
        roll_est, pitch_est, yaw_est = r.as_euler('xyz', degrees=False)

        # Construct full estimated pose with extremely undesirable hack
        est_pose = np.array([drone_position[0], drone_position[1], drone_position[2], wrap_angle_to_zero(roll_est-np.pi), pitch_est, yaw_est])

        # Compute residual v_t = ground truth - estimated
        v_t = np.array(gt_data[:6]) - est_pose
        residuals.append(v_t)

        index += 1

    residuals = np.array(residuals)
    n = residuals.shape[0]

    if n < 2:
        raise ValueError("Not enough valid samples to compute covariance.")

    # Compute sample covariance matrix
    mean_residual = np.mean(residuals, axis=0, keepdims=True)
    centered_residuals = residuals - mean_residual
    R_cov = (centered_residuals.T @ centered_residuals) / (n - 1)

    print("Final Covariance matrix R:\n", R_cov)
    return R_cov

if __name__ == '__main__':

    estimated_all = {}
    ground_truth_all = {}

    for i in reversed(range(8)):

        print(f"Processing data/studentdata{i}.mat")
        # Load the data
        data_path = 'Module_2_PnP/data/studentdata' + str(i) + '.mat'
        data = sio.loadmat(data_path, simplify_cells=True)

        # save images
        # save_images(data)

        estimated = {}
        ground_truth = {}

        # estimate pose for each frame
        for i in range(len(data['data'])):  
            
            frame_data = data['data'][i]
            frame_timestamp = frame_data['t']
            drone_position, R_est = estimate_pose(frame_data)
            estimated[frame_timestamp] = drone_position

        for i in range(len(data['time'])):

            ground_truth_data = data['vicon'][:, i]
            ground_truth_timestamp = data['time'][i]
            ground_truth[ground_truth_timestamp] = ground_truth_data
            # print(ground_truth_timestamp)

        # Align ground truth to estimated
        aligned_gt = align_ground_truth(estimated, ground_truth)

        estimated_all = {**estimated_all, **estimated}
        ground_truth_all = {**ground_truth_all, **aligned_gt}


    # estimate the covariance
    estimate_covariance(estimated, aligned_gt)

    visualize_camera_and_ground_truth(estimated, aligned_gt)

