import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats.distributions import chi2
from sklearn.metrics import mean_squared_error
import pdb

# Constants for obstacle extraction
MAX_DETECTION_RANGE = 80.0 
MIN_VALID_RANGE = 1.0    
ANGLE_INCREMENT = np.pi / 360 
ANGLE_MARGIN = 5 * np.pi / 306
RANGE_DIFF_THRESHOLD = 1.5 
ANGLE_DIFF_THRESHOLD = 10 * np.pi / 360 
CLUSTER_DISTANCE_THRESHOLD = 3.0
FINAL_DISTANCE_THRESHOLD = 1.0 
MIN_ANGLE_DIFF = 2 * np.pi / 360

class EKF_SLAM:
    def __init__(self, vehicle_geometry, robot_state, std_deviations):

        self.vehicle_geometry = vehicle_geometry
        self.robot_state = robot_state
        self.std_deviations = std_deviations
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

    def motion_model(self, u, dt, robot_state, vehicle_params):
        '''
        Compute the motion model and its Jacobian for the robot.
        u: control input (speed, steering angle)
        dt: time step
        robot_state: current state of the robot
        vehicle_params: parameters of the vehicle
        '''
        vc = u[0] / (1 - np.tan(u[1]) * vehicle_params['H'] / vehicle_params['L'])
        motion = np.zeros([3], np.float32)
        motion[0] = vc * np.cos(robot_state['x'][2]) \
                    - (vc / vehicle_params['L']) * np.tan(u[1]) * (vehicle_params['a'] * np.sin(robot_state['x'][2]) \
                                                                   + vehicle_params['b'] * np.cos(robot_state['x'][2]))
        motion[1] = vc * np.sin(robot_state['x'][2]) \
                    + (vc / vehicle_params['L']) * np.tan(u[1]) * (vehicle_params['a'] * np.cos(robot_state['x'][2]) \
                                                                   - vehicle_params['b'] * np.sin(robot_state['x'][2]))
        motion[2] = (vc / vehicle_params['L']) * np.tan(u[1])
        motion = motion * dt

        G = np.zeros((3, 3), dtype=np.float32)
        G[0, 2] = - vc * np.sin(robot_state['x'][2]) \
                  - (vc / vehicle_params['L']) * np.tan(u[1]) * (vehicle_params['a'] * np.cos(robot_state['x'][2]) \
                                                                  - vehicle_params['b'] * np.sin(robot_state['x'][2]))

        G[1, 2] = vc * np.cos(robot_state['x'][2]) \
                  + (vc / vehicle_params['L']) * np.tan(u[1]) * (- vehicle_params['a'] * np.sin(robot_state['x'][2]) \
                                                                  - vehicle_params['b'] * np.cos(robot_state['x'][2]))

        G = G * dt
        G = G + np.eye(3)

        return motion, G

    def predict(self, u, robot_state, vehicle_params, dt):
        '''
        Predict the next state of the robot using the motion model.
        u: control input (speed, steering angle)
        robot_state: current state of the robot and landmarks
        vehicle_params: parameters of the vehicle
        dt: time step
        '''
        n = robot_state['x'].shape[0]

        # 1) build F to pick out the pose-subblock
        F = np.zeros((3, n))
        F[:, :3] = np.eye(3)

        # 2) compute 3×1 motion increment f and 3×3 Jacobian G
        f, G = self.motion_model(u, dt, robot_state, vehicle_params)

        # 3) embed into full‐state transition
        G_full = np.eye(n) + F.T @ G @ F

        # 4) process noise only on the 3×3 pose block
        R3 = np.diag([
            self.std_deviations['xy']**2,
            self.std_deviations['xy']**2,
            self.std_deviations['phi']**2
        ])

        # 5) update state and clamp heading
        robot_state['x'][:3] += f
        robot_state['x'][2] = self.clamp_angle(robot_state['x'][2])

        # 6) full‐state covariance update
        robot_state['P'] = G_full @ robot_state['P'] @ G_full.T + F.T @ R3 @ F
        robot_state['P'] = self.make_symmetric(robot_state['P'])

        return robot_state

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
        diam = ((a2[valid] + ANGLE_INCREMENT) - a1[valid]) * (r1[valid] + r2[valid]) / 2.0

        return list(zip(r_mid.tolist(), a_mid.tolist(), diam.tolist()))

    def update_plot(self, landmarks, robot_xy, latitude, longitude):
        '''
        Update the plot with the current robot position and landmarks.
        '''
        if self.ax is None:
            self.init_plot()

        self.ax.cla()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("EKF–SLAM: Robot trajectory")

        # draw robot
        rx, ry = robot_xy
        self.ax.plot(rx, ry, 'bo', markersize=8, label='Robot')

        # draw landmarks
        if landmarks is not None and landmarks.size > 0:
            self.ax.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                marker='*',
                s=60,
                c='green',
                label='Landmarks'
            )

        # draw trajectory
        if len(self.trajectory) > 1:
            xs, ys = zip(*self.trajectory)
            self.ax.plot(xs, ys, '-', linewidth=2, label='Trajectory')

        # plot lat/long data
        if latitude is not None and longitude is not None:
            # Assuming latitude and longitude are pandas Series
            if isinstance(latitude, pd.Series) and isinstance(longitude, pd.Series):
                # Remove NaNs and align indices
                valid_lat = latitude.dropna()
                valid_lon = longitude.dropna()
                common_index = valid_lat.index.intersection(valid_lon.index)
                valid_lat = valid_lat[common_index]
                valid_lon = valid_lon[common_index]

                if not valid_lat.empty and not valid_lon.empty:
                    self.ax.plot(valid_lon.values, valid_lat.values, 'r-', linewidth=1, label='Lat/Long')
                    self.lat_long_trajectory = np.column_stack((valid_lon.values, valid_lat.values))
            else:
                print("Warning: latitude and longitude are not pandas Series. Cannot plot lat/long trajectory.")
                self.lat_long_trajectory = None
        else:
            self.lat_long_trajectory = None

        # auto-scale to include everything
        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.legend(loc='upper right')
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

    def ekf_slam(self, sensor_data, _, vehicle_geometry, latitute, longitude):
        '''
        Main function to perform EKF SLAM.
        sensor_data: list of tuples containing sensor data
        vehicle_geometry: parameters of the vehicle
        latitute: latitude data (m)
        longitude: longitude data (m)
        '''
        last_odometry_time = None
        robot_state = {
            'x': np.array([longitude[0], latitute[0], 0.0]), # Initialize at origin for simplicity in this example
            'P': np.diag([.1, .1, 0.01]),
            'num_landmarks': 0
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
                # print("Landmark estimates: ", robot_state['num_landmarks'])

            if data[0] == 'odometry':
                # Process odometry data
                odometry = data[1][1:]
                if last_odometry_time is None:
                    last_odometry_time = t
                    continue
                dt = t - last_odometry_time

                # predict step
                robot_state = self.predict(odometry, robot_state, vehicle_geometry, dt)
                last_odometry_time = t

            elif data[0] == 'laser':

                laser_data = data[1][1:]
                obstacles = self.get_landmarks(laser_data)

                # data association

                # update state using lazer data 

                self.trajectory.append(robot_state['x'][0:2].copy())

                # Update the plot with landmarks
                landmarks = robot_state['x'][3:].reshape(-1, 2)
                self.update_plot(landmarks, robot_state['x'][0:2], latitute, longitude)

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

    initial_robot_state = {
        "x": np.array([0.0, 0.0, 0.0]),
        "P": np.diag([.1, .1, 0.01]),
        "num_landmarks": 0
    }

    std_dev = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    ekf_obj = EKF_SLAM(vehicle_geometry, initial_robot_state, std_dev)

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

    ekf_obj.ekf_slam(sensor_data, initial_robot_state, vehicle_geometry, latitute, longitude)
