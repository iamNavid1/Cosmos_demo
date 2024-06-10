import numpy as np
from typing import List
from .kalman_filter import KalmanFilter

class TrajectoryFilter:
    """
    A class to filter out noise from detected object positions (trajectory) using Kalman filter
    """
    def __init__(self, dim=2, dt=1, max_staleness=90):
        """
        Initialize the smoother with the dimension of the position and the time step
        """
        self.dim = dim
        self.dt = dt
        self.max_staleness = max_staleness
        self.track_filters = {}

    def update(self, 
               track_ids: List[int],
               detections: List[np.ndarray]) -> List[np.ndarray]:
        """
        Update the Kalman filter for a given track_id with observed positions
        """
        updated_positions = []

        to_delete = []
        for id, trk in self.track_filters.items():
            trk.predict()
            if trk.stale_since() > self.max_staleness:
                to_delete.append(id)

        for id in to_delete:
            del self.track_filters[id]

        for i, track_id in enumerate(track_ids):
            if track_id > 0:
                z = np.array([detections[i].flatten() if len(detections[i]) != 0 else None])

                if track_id not in self.track_filters:
                    self.track_filters[track_id] = TrajectoryKalmanFilter(
                        dim=self.dim,
                        dt=self.dt,
                        initial_positions=z
                    )

                trajectory_kalman_filter = self.track_filters[track_id]
                trajectory_kalman_filter.update(z)
                current_estate = trajectory_kalman_filter.get_state()
                updated_positions.append(current_estate)

        return updated_positions
        

class TrajectoryKalmanFilter:
    """
    A class to maintain internal state of birdseye view positions of detected objects using Kalman filter
    """
    def __init__(self, dim=2, dt=1, initial_positions=None):
        self.dim = dim
        state_dim = 2 * self.dim # Position and Velocity
        meas_dim = self.dim # Just Position

        # Initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=meas_dim)
        # state transition matrix
        self.kf.F = np.eye(state_dim)
        for i in range(meas_dim):
            self.kf.F[i, i+meas_dim] = dt
        # Measurement function
        self.kf.H = np.zeros((meas_dim, state_dim))
        for i in range(meas_dim):
            self.kf.H[i, i] = 1
        # Current state uncertainty
        self.kf.P *= 10
        self.kf.P[meas_dim:, meas_dim:] *= 100 # Initial velocity uncertainty
        # Measurement uncertainty
        self.kf.R *= 5
        # Process uncertainty
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[meas_dim:, meas_dim:] *= 0.01
        # Set initial state if initial positions are provided
        if initial_positions is not None and len(initial_positions.flatten()) == meas_dim:
            self.kf.x[:meas_dim] = np.array(initial_positions).reshape(-1,1)

        self.time_since_update = 0

    def update(self, positions):
        """
        Update the Kalman filter with observed positions
        """
        if positions is not None:
            self.time_since_update = 0
            self.kf.update(positions.flatten())
        else:
            self.kf.update(None)

    def predict(self):
        """
        Update the Kalman filter without observed positions and return the predicted positions
        """
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:self.dim].flatten()
    
    def get_state(self):
        """
        Return the current state of the Kalman filter
        """
        return self.kf.x[:self.dim].flatten()
    
    def stale_since(self):
        """
        Return the number of frames since the last observed positions
        """
        return self.time_since_update
