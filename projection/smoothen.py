from filterpy.kalman import KalmanFilter
import numpy as np

class TrajectorySmoother:
    def __init__(self):
        self.tracks = {}

    def update(self, detections, track_ids):
        smoothed_positions = []
        for detection, track_id in zip(detections, track_ids):
            if track_id not in self.tracks:
                self.tracks[track_id] = Filter(initial_detection=detection)
            else:
                self.tracks[track_id].update(detection)
            smoothed_positions.append(self.tracks[track_id].get_state())
        return smoothed_positions


class Filter:
    def __init__(self, dt=1, initial_detection=None):
        # Initialize a Kalman filter for 3D pose tracking of all joints
        self.dim = 2  # 2D Birdseye View Positions
        state_dim = 2 * self.dim  # Position and Velocity
        meas_dim = self.dim  # Just Position

        # Initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=meas_dim)

        # State transition matrix
        self.kf.F = np.eye(state_dim)
        for i in range(meas_dim):
            self.kf.F[i, i+meas_dim] = dt

        # Measurement function
        self.kf.H = np.zeros((meas_dim, state_dim))
        for i in range(meas_dim):
            self.kf.H[i, i] = 1

        # Measurement uncertainty
        self.kf.R = np.eye(meas_dim) * 20

        # Process uncertainty
        self.kf.Q = np.eye(state_dim) * 0.005
        for i in range(meas_dim):
            self.kf.Q[i+meas_dim, i+meas_dim] *= 0.25

        # Initial state uncertainty
        self.kf.P *= 20

        # Set initial state if initial keypoints are provided
        if initial_detection is not None and len(initial_detection) == meas_dim:
            self.kf.x[:meas_dim] = np.array(initial_detection).reshape(-1,1)  # Position
            self.kf.x[meas_dim:2*meas_dim] = 0  # Velocity

    def update(self, z):
        if z is not None:
            self.kf.update(z)
        self.kf.predict()

    def predict(self):
        self.kf.predict()
        return self.kf.x[:self.dim].flatten()

    def get_state(self):
        return self.kf.x[:self.dim].flatten()

