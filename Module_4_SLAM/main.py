import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats.distributions import chi2
from sklearn.metrics import mean_squared_error
import pdb

# Constants for obstacle extraction
MAX_DETECTION_RANGE = 75.0 
MIN_VALID_RANGE = 1.0    
ANGLE_INCREMENT = np.pi / 360 
ANGLE_MARGIN = 5 * np.pi / 306
RANGE_DIFF_THRESHOLD = 1.5 
ANGLE_DIFF_THRESHOLD = 10 * np.pi / 360 
CLUSTER_DISTANCE_THRESHOLD = 3.0
FINAL_DISTANCE_THRESHOLD = 1.0 
MIN_ANGLE_DIFF = 2 * np.pi / 360
ARROW_LEN = 5

class EKF_SLAM:
    def __init__(self, vehicle_geometry, std_devs):

        self.vehicle_geometry = vehicle_geometry
        self.std_devs = std_devs
        self.fig = None
        self.ax = None
        self.lat_long_trajectory = None

        self.trajectory = []

    def load_data(self, file_path):
        '''
        Load the data from a CSV file and preprocess it.
        '''
        data_in = pd.read_csv(file_path, index_col=0)
        # drop any row missing odometry or any laser beam
        laser_cols = [c for c in data_in.columns if c.startswith('laser_')]
        data_in = data_in.dropna(subset=['speed', 'steering'] + laser_cols, how='any')
        data_in.index = pd.to_timedelta(data_in.index, unit='s')
        return data_in

    def init_plot(self):
        '''
        Initialize the plot for visualizing the SLAM process.
        '''
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("EKF SLAM")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect('equal', adjustable='box')
        return {"fig": self.fig, "ax": self.ax}

    def extract_sensor_data(self, data_in):
        '''
        Extract sensor data from the loaded DataFrame.
        '''
        speed_data = data_in['speed']
        steering_data = data_in['steering']
        latitute = data_in['latitude']
        longitude = data_in['longitude']
        time_index = data_in.index
        laser_columns = [col for col in data_in.columns if col.startswith('laser_')]
        lidar_data = data_in[laser_columns]
        return latitute, longitude, speed_data, steering_data, lidar_data, time_index

    def motion_model(self, u, dt, state, vehicle_params):
        '''
        Compute the motion model and its Jacobian for the robot.
        u: control input (speed, steering angle)
        dt: time step
        state: current state of the robot
        vehicle_params: parameters of the vehicle
        '''
        vc = u[0] / (1 - np.tan(u[1]) * vehicle_params['H'] / vehicle_params['L'])
        motion = np.zeros([3], np.float32)
        motion[0] = vc * np.cos(state['x'][2]) \
                    - (vc / vehicle_params['L']) * np.tan(u[1]) * (vehicle_params['a'] * np.sin(state['x'][2]) \
                                                                   + vehicle_params['b'] * np.cos(state['x'][2]))
        motion[1] = vc * np.sin(state['x'][2]) \
                    + (vc / vehicle_params['L']) * np.tan(u[1]) * (vehicle_params['a'] * np.cos(state['x'][2]) \
                                                                   - vehicle_params['b'] * np.sin(state['x'][2]))
        motion[2] = (vc / vehicle_params['L']) * np.tan(u[1])
        motion = motion * dt

        G = np.zeros((3, 3), dtype=np.float32)
        G[0, 2] = - vc * np.sin(state['x'][2]) \
                  - (vc / vehicle_params['L']) * np.tan(u[1]) * (vehicle_params['a'] * np.cos(state['x'][2]) \
                                                                  - vehicle_params['b'] * np.sin(state['x'][2]))

        G[1, 2] = vc * np.cos(state['x'][2]) \
                  + (vc / vehicle_params['L']) * np.tan(u[1]) * (- vehicle_params['a'] * np.sin(state['x'][2]) \
                                                                  - vehicle_params['b'] * np.cos(state['x'][2]))

        G = G * dt

        return motion, G

    def predict(self, u, state, vehicle_params, dt):
        '''
        Predict the next state of the robot using the motion model.
        u: control input (speed, steering angle)
        state: current state of the robot and landmarks
        vehicle_params: parameters of the vehicle
        dt: time step
        '''
        n = state['x'].shape[0]

        # 1) build F to pick out the pose-subblock
        F = np.zeros((3, n))
        F[:, :3] = np.eye(3)

        # 2) compute 3x1 motion increment f and 3x3 Jacobian G
        f, G = self.motion_model(u, dt, state, vehicle_params)

        # 3) embed into full‐state transition
        G_full = np.eye(n) + F.T @ G @ F

        # 4) process noise only on the 3x3 pose block
        R3 = np.diag([
            self.std_devs['xy']**2,
            self.std_devs['xy']**2,
            self.std_devs['phi']**2
        ])

        # 5) update state and clamp heading
        state['x'][:3] += f
        state['x'][2] = self.clamp_angle(state['x'][2])

        # 6) full‐state covariance update
        state['P'] = G_full @ state['P'] @ G_full.T + F.T @ R3 @ F
        state['P'] = self.make_symmetric(state['P'])

        return state

    def clamp_angle(self, angle):
        '''
        Clamp the angle to the range [-pi, pi].
        '''
        return math.fmod(angle + np.pi, 2 * np.pi) - np.pi

    def make_symmetric(self, matrix):
        '''
        Make a matrix symmetric by averaging it with its transpose.
        '''
        return 0.5 * (matrix + matrix.T)
    
    def make_positive_definite(self, P):
        eigs = np.linalg.eigvals(P)
        if(np.all( eigs > 0)):
            return P
        else:
            offset = 1e-6 - min(0, eigs.min())

        return (P + offset * np.eye(P.shape[0]))
    
    def get_landmarks(self, laser_ranges):
        '''
        Extract landmarks from laser range data.
        laser_ranges: 1D array of laser range measurements
        Returns a list of tuples (range, angle, diameter) for each detected landmark.
        '''
        # Validate input
        if laser_ranges.ndim != 1:
            raise ValueError("laser_ranges must be a 1D array of measurements")

        num_points = laser_ranges.size
        angles = np.arange(num_points) * ANGLE_INCREMENT

        # 1) Filter out invalid/distant points
        valid_mask = laser_ranges < MAX_DETECTION_RANGE
        valid_indices = np.nonzero(valid_mask)[0]
        if valid_indices.size == 0:
            return []

        ranges = laser_ranges[valid_indices]
        angs = angles[valid_indices]

        # 2) Split into segments where jumps in range or angle occur
        range_jumps = np.abs(np.diff(ranges)) > RANGE_DIFF_THRESHOLD
        angle_jumps = np.diff(angs) > ANGLE_DIFF_THRESHOLD
        boundaries = np.flatnonzero(range_jumps | angle_jumps)

        segment_starts = np.concatenate(([0], boundaries + 1))
        segment_ends = np.concatenate((boundaries, [len(ranges) - 1]))

        # 3) Compute coordinates for segment endpoints
        x_start = ranges[segment_starts] * np.cos(angs[segment_starts])
        y_start = ranges[segment_starts] * np.sin(angs[segment_starts])
        x_end = ranges[segment_ends] * np.cos(angs[segment_ends])
        y_end = ranges[segment_ends] * np.sin(angs[segment_ends])

        num_segments = segment_starts.size
        keep = np.ones(num_segments, dtype=bool)

        # 4) Merge clusters: merge segments that are too close (shift by 1,2,3)
        for shift in (1, 2, 3):
            if num_segments > shift:
                dx = x_start[shift:] - x_end[:-shift]
                dy = y_start[shift:] - y_end[:-shift]
                dist2 = dx*dx + dy*dy
                close = dist2 < CLUSTER_DISTANCE_THRESHOLD**2
                idx = np.flatnonzero(close)
                keep[idx] = False
                keep[idx + shift] = False

        # 5) Remove segments too close in angle with overlapping distance
        if num_segments > 1:
            ang_diff = angs[segment_starts[1:]] - angs[segment_ends[:-1]]
            mask = ang_diff < MIN_ANGLE_DIFF
            back_behind = ranges[segment_starts[1:]] > ranges[segment_ends[:-1]]
            to_drop = np.where(mask & back_behind)[0] + 1
            keep[to_drop] = False

        candidates = np.flatnonzero(keep)
        if candidates.size == 0:
            return []

        # 6) Final cluster separation check
        dx_c = x_start[candidates] - x_end[candidates]
        dy_c = y_start[candidates] - y_end[candidates]
        dist2_c = dx_c*dx_c + dy_c*dy_c
        valid_c = dist2_c < FINAL_DISTANCE_THRESHOLD**2
        final_idxs = candidates[valid_c]
        if final_idxs.size == 0:
            return []

        # 7) Filter by minimum range and angular margins
        r1 = ranges[segment_starts[final_idxs]]
        r2 = ranges[segment_ends[final_idxs]]
        a1 = angs[segment_starts[final_idxs]]
        a2 = angs[segment_ends[final_idxs]]

        valid = (r1 > MIN_VALID_RANGE) & (a1 > ANGLE_MARGIN) & (a2 < (np.pi - ANGLE_MARGIN))
        final_idxs = final_idxs[valid]
        if final_idxs.size == 0:
            return []

        # 8) Compute obstacle properties
        r_mid = (r1[valid] + r2[valid]) / 2.0
        a_mid = ((a1[valid] + a2[valid]) / 2.0) - (np.pi / 2)

        return list(zip(r_mid.tolist(), a_mid.tolist()))
    
    def get_from_measurement_model(self, state, i):

        z_hat = np.zeros(2)
        H = np.zeros((2, state['x'].shape[0]))

        robot_x = state['x'][0]
        robot_y = state['x'][1]
        landmark_x = state['x'][3 + 2 * i]
        landmark_y = state['x'][3 + 2 * i + 1]

        z_hat[0] = np.sqrt((landmark_x - robot_x) ** 2 + (landmark_y - robot_y) ** 2)
        z_hat[1] = np.arctan2(landmark_y - robot_y, landmark_x - robot_x) - state['x'][2]

        H[0, 0] = (robot_x - landmark_x) / z_hat[0]
        H[0, 1] = (robot_y - landmark_y) / z_hat[0]
        H[0, 3 + 2 * i] = (landmark_x - robot_x) / z_hat[0]
        H[0, 3 + 2 * i + 1] = (landmark_y - robot_y) / z_hat[0]

        H[1, 0] = (landmark_y - robot_y) / (z_hat[0] ** 2)
        H[1, 1] = -(landmark_x - robot_x) / (z_hat[0] ** 2)
        H[1, 2] = -1
        H[1, 3 + 2 * i] = -(landmark_y - robot_y) / (z_hat[0] ** 2)
        H[1, 3 + 2 * i + 1] = (landmark_x - robot_x) / (z_hat[0] ** 2)

        # Normalize the bearing angle
        z_hat[1] = self.clamp_angle(z_hat[1])

        return z_hat, H
    
    def solve_cost_matrix_heuristic(self, M):
        result = []
        ordering = np.argsort(M.min(axis=1))
        for msmt in ordering:
            match = np.argmin(M[msmt, :])
            M[:, match] = 1e8
            result.append((msmt, match))
        return result
    
    def get_associations(self, obstacles, state, std_devs):
        
        if state["landmarks"] == 0:
            return [-1 for _ in obstacles]  # No landmarks yet

        alpha = chi2.ppf(0.9, 2)   # Acceptable match threshold
        beta = chi2.ppf(0.999, 2)   # New landmark threshold (more strict)
        
        landmarks = state["landmarks"]
        num_measurements = len(obstacles)

        # Predicted observations and innovation covariances
        z_hat = np.zeros([landmarks, 2])
        S = np.zeros([landmarks, 2, 2])
        Q = np.diag([std_devs["range"]**2, std_devs["bearing"]**2])

        for j in range(landmarks):
            z_hat[j], H = self.get_from_measurement_model(state, j)
            S[j] = H @ state["P"] @ H.T + Q     # innovation covariance

        # Build cost matrix
        M = alpha * np.ones((num_measurements, landmarks + num_measurements))  # Init with alpha
        for i, obs in enumerate(obstacles):
            obs_vector = np.array([obs[0], obs[1]])
            for j in range(landmarks):
                diff = obs_vector - z_hat[j]
                diff[1] = self.clamp_angle(diff[1])
                mahalanobis_dist = diff.T @ np.linalg.inv(S[j]) @ diff
                M[i, j] = mahalanobis_dist

        # Solve cost matrix heuristic
        matches = self.solve_cost_matrix_heuristic(np.copy(M))  # Copy M since it's modified
        matches.sort()

        # Interpret matches into associations
        associations = list(range(num_measurements))
        for k in range(num_measurements):
            obs_idx, matched_col = matches[k]

            if matched_col >= landmarks:
                if np.min(M[obs_idx, :landmarks]) > beta:
                    associations[obs_idx] = -1  # New landmark
                else:
                    associations[obs_idx] = -2  # Ambiguous
            else:
                associations[obs_idx] = matched_col  # Matched with landmark j

        return associations
    
    def init_landmark(self, state, obstacle):
        robot_x = state['x'][0]
        robot_y = state['x'][1]
        phi = state['x'][2]

        # Calculate landmark position
        landmark_x = robot_x + obstacle[0] * np.cos(phi + obstacle[1])
        landmark_y = robot_y + obstacle[0] * np.sin(phi + obstacle[1])

        # Initialize the landmark state
        new_landmark = np.array([landmark_x, landmark_y])
        state['x'] = np.concatenate((state['x'], new_landmark))

        # Create a new covariance block for the landmark
        landmark_cov = np.diag([std_devs['range']**2, std_devs['range']**2])

        # Extend the covariance matrix
        new_dim = state['x'].size
        new_cov = np.zeros((new_dim, new_dim))

        # Copy the old covariance matrix to the new block
        new_cov[0:new_dim-2, 0:new_dim-2] = state['P']

        # Add the new landmark covariance in the bottom-right corner
        new_cov[new_dim-2:, new_dim-2:] = landmark_cov

        # Optionally, add cross-covariance terms between the new landmark and the robot (set to zeros for simplicity)
        # new_cov[new_dim-2:, 0:3] = 0
        # new_cov[0:3, new_dim-2:] = 0

        # Update the state covariance
        state['P'] = new_cov
        state['landmarks'] += 1

        return state

    def get_update(self, state, associations, obstacles, vehicle_geometry, std_devs):
        '''
        # Perform a measurement update of the EKF state given a set of tree measurements.

        # obstacles is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

        # assoc is the data association for the given set of obstacles, i.e. obstacles[i] is an observation of the
        # ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
        # in the state for measurement i. If assoc[i] == -2, discard the measurement as 
        # it is too ambiguous to use.

        The diameter component of the measurement can be discarded.

        Returns the state.
        '''
        # Validate input
        associations = np.array(associations)
        measurements = np.array(obstacles)
        good_associations = associations[associations>-1]
        good_measurements = measurements[associations>-1]

        # Init matrices
        Q = np.diag([std_devs['range']**2, std_devs['bearing']**2])
        K = np.zeros((state['x'].shape[0], 2*len(good_associations)))
        residuals= np.zeros((2*good_measurements.shape[0]))
        H= np.zeros([2*good_associations.size, state['x'].shape[0]])
        
        for i in range(good_associations.size):
            # Get the predicted measurement and update the measurement matrix
            zhat, H_sub = self.get_from_measurement_model(state, good_associations[i])
            H[i:i+2, :] = H_sub
            
            # Compute the Kalman gain
            P_Ht = np.matmul(state['P'], H[i:i+2, :].T)
            S = np.matmul(H[i:i+2, :], np.matmul(state['P'], H[i:i+2, :].T)) + Q
            K[:, i:i+2] = np.matmul(P_Ht, np.linalg.inv(S))
            
            # Calculate the residuals
            residuals[i:i+2] = good_measurements[i] - zhat

        # Update state estimates
        state['x'] += np.matmul(K, residuals)
        state['x'][2] = self.clamp_angle(state['x'][2])

        # Update covariance matrix
        I_KH = np.eye(state['P'].shape[0]) - np.matmul(K, H)
        state['P'] = np.matmul(I_KH, state['P'])
        state['P'] = self.make_symmetric(state['P'])
        state['P'] = self.make_positive_definite(state['P'])

        # Handle new measurements (those with associations == -1)
        new_measurements = measurements[associations == -1]
        for i in range(new_measurements.shape[0]):
            state = self.init_landmark(state, new_measurements[i])

        return state
    
    def update_plot(self, landmarks, obstacles, robot):
        """
        Plot robot, landmarks and trajectory.
        Also draw a dashed red line from the robot to every obstacle
        measured in the current frame (obstacles = [range, bearing]).
        """
        # ── configuration ───────────────────────────────────────────────
        if not hasattr(self, "new_lm_thresh"):
            self.new_lm_thresh = 2.0     # distance (m) to call a landmark "new"

        # ── bookkeeping of seen landmarks ───────────────────────────────
        if not hasattr(self, "seen_landmarks"):
            self.seen_landmarks = np.empty((0, 2))

        cur_lms = np.asarray(landmarks) if landmarks is not None else np.empty((0, 2))

        new_mask = np.ones(len(cur_lms), dtype=bool)
        if self.seen_landmarks.size and cur_lms.size:
            d2      = np.square(cur_lms[:, None, :] - self.seen_landmarks[None, :, :]).sum(-1)
            min_d   = np.sqrt(d2.min(axis=1))
            new_mask = min_d > self.new_lm_thresh

        new_lms         = cur_lms[new_mask]
        self.seen_landmarks = np.vstack([self.seen_landmarks, new_lms])

        # ── plotting ────────────────────────────────────────────────────
        if self.ax is None:
            self.init_plot()

        self.ax.cla()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("EKF–SLAM: Robot trajectory")

        # robot
        rx, ry, phi = robot      # robot = (x, y, heading)
        self.ax.plot(rx, ry, 'bo', markersize=8, label="Robot")

        # heading arrow: from (rx,ry) in direction φ
        dx = ARROW_LEN * np.cos(phi)
        dy = ARROW_LEN * np.sin(phi)
        self.ax.arrow(rx, ry, dx, dy,
                    head_width=ARROW_LEN * 0.25,
                    head_length=ARROW_LEN * 0.25,
                    fc='blue', ec='blue')


        # all landmarks
        if cur_lms.size:
            self.ax.scatter(cur_lms[:, 0], cur_lms[:, 1],
                            marker='*', s=60, c='green', label="Landmarks")

        # trajectory so far
        if len(self.trajectory) > 1:
            xs, ys = zip(*self.trajectory)
            self.ax.plot(xs, ys, '-', linewidth=2, label="Trajectory")

        # new-landmark “discovery” lines
        for lx, ly in new_lms:
            self.ax.plot([rx, lx], [ry, ly], '--k', linewidth=1)

        # ── obstacles (range, bearing w.r.t robot) ─────────────────────
        if obstacles is not None and len(obstacles):
            obs_rb   = np.asarray(obstacles)         # shape (N,2)
            r, b     = obs_rb[:, 0], obs_rb[:, 1]    # ranges, bearings
            ox = rx + r * np.cos(phi + b)            # global X
            oy = ry + r * np.sin(phi + b)            # global Y

            # scatter & dashed line to every obstacle
            self.ax.scatter(ox, oy, marker='x', s=50, c='red', label="Obstacles")
            for x_o, y_o in zip(ox, oy):
                self.ax.plot([rx, x_o], [ry, y_o], 'r--', linewidth=1)

        self.ax.legend(loc="upper right")
        plt.pause(0.001)
            
    def calculate_mse(self):
        '''
        Calculate the Mean Squared Error (MSE) between the robot trajectory and the ground truth trajectory.
        '''
        if len(self.trajectory) > 1 and self.lat_long_trajectory is not None and len(self.lat_long_trajectory) > 1:
            # Convert trajectory list of tuples to a NumPy array
            robot_trajectory_array = np.array(self.trajectory)

            # Ensure both trajectories have the same number of points for MSE calculation
            min_len = min(robot_trajectory_array.shape[0], self.lat_long_trajectory.shape[0])
            robot_traj_trimmed = robot_trajectory_array[:min_len]
            lat_long_traj_trimmed = self.lat_long_trajectory[:min_len]

            if robot_traj_trimmed.shape[0] > 1:
                mse = mean_squared_error(robot_traj_trimmed, lat_long_traj_trimmed)
                return mse
            else:
                print("Warning: Not enough matching points in trajectories to calculate MSE.")
                return None
        else:
            print("Warning: Either robot trajectory or lat/long trajectory is empty or has only one point. Cannot calculate MSE.")
            return None

    def ekf_slam(self, sensor_data, vehicle_geometry, first_latitute, first_longitude):
        '''
        Main function to perform EKF SLAM.
        sensor_data: list of tuples containing sensor data
        vehicle_geometry: parameters of the vehicle
        latitute: latitude data (m)
        longitude: longitude data (m)
        '''
        last_odometry_time = None
        state = {
            'x': np.array([first_longitude, first_latitute, 0.0]),
            'P': np.diag([.1, .1, 0.01]),
            'landmarks': 0
        }

        # init plot
        plot_objects = self.init_plot()
        if not plot_objects:
            print("Failed to initialize plot.")
            return

        print("EKF SLAM processing sensor data...")

        for i, data in enumerate(sensor_data):
            t = data[1][0]
            if i % 100 == 0:
                print(f"Processing data point {i}/{len(sensor_data)} at time {t}")
                # print("Landmark estimates: ", state['landmarks'])

            if data[0] == 'odometry':
                # Process odometry data
                odometry = data[1][1:]
                if last_odometry_time is None:
                    last_odometry_time = t
                    continue
                dt = t - last_odometry_time

                # predict step
                state = self.predict(odometry, state, vehicle_geometry, dt)
                last_odometry_time = t

            elif data[0] == 'laser':

                laser_data = data[1][1:]
                obstacles = self.get_landmarks(laser_data)

                # data association
                data_association = self.get_associations(obstacles, state, std_devs)
                # print(data_association)
                
                # update state using lazer data 
                state = self.get_update(state, data_association, obstacles, vehicle_geometry, std_devs)

                self.trajectory.append(state['x'][0:2].copy())

                # Update the plot with landmarks
                landmarks = state['x'][3:].reshape(-1, 2)
                self.update_plot(landmarks, obstacles, state['x'][0:3])

        # Calculate MSE
        mse = self.calculate_mse()
        if mse is not None:
            print(f"MSE: {mse:.4f}")
        plt.show()

if __name__ == '__main__':

    vehicle_geometry = {
        "a": 3.78,
        "b": 0.50,
        "L": 2.83,
        "H": 0.76
    }

    std_devs = {
        "xy": 0.05,
        "phi": 0.5*np.pi/180,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    ekf_obj = EKF_SLAM(vehicle_geometry, std_devs)

    data = ekf_obj.load_data('Module_4_SLAM/victoria_park.csv')
    latitute, longitude, odo_speed, odo_steering, lidar_df, time_index = ekf_obj.extract_sensor_data(data)
    sensor_data = []

    for t in time_index:
        ts = t.total_seconds()

        odometry = np.array([ts,
                            odo_speed.loc[t],
                            odo_steering.loc[t]])
        sensor_data.append(('odometry',   odometry))

        laser_row = lidar_df.loc[t].values
        lazer = np.concatenate(([ts], laser_row))
        sensor_data.append(('laser', lazer))

    sensor_data.sort(key=lambda ev: ev[1][0])

    ekf_obj.ekf_slam(sensor_data, vehicle_geometry, latitute[0], longitude[0])
