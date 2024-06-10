import numpy as np
from .kalman_filter import KalmanFilter
from typing import List, Dict



class PoseKalmanFilter:
    """
    A class to maintain internal state of all joints of each pose instance using Kalman filter
    """
    def __init__(self, num_joints=17, dim=2, dt=1, initial_keypoints=None):
        # Initialize a 3D pose tracker using initial keypoints
        self.num_joints = num_joints
        self.dim = dim
        state_dim = (self.num_joints * self.dim + 2) * 2  # Position and Velocity + anchor point
        meas_dim = self.num_joints * self.dim + 2  # Just Position + anchor point

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
        # Current state uncertainty
        self.kf.P *= 10
        self.kf.P[meas_dim:, meas_dim:] *= 100 # Initial velocity uncertainty
        # Measurement uncertainty
        self.kf.R *= 5
        # Process uncertainty
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[meas_dim:, meas_dim:] *= 0.01
        # Set initial state if initial keypoints are provided
        if initial_keypoints is not None and len(initial_keypoints.flatten()) == meas_dim:
            self.kf.x[:meas_dim] = initial_keypoints.reshape(-1,1)  # <- Position; velocity is 0

        self.time_since_update = 0
        # self.age = 0

    def update(self, keypoints):
        """
        Update the Kalman filter with observed keypoints
        """
        if keypoints is not None:
            self.time_since_update = 0
            self.kf.update(keypoints.flatten())
        else:
            self.kf.update(None)
        
        # self.age += 1

    def predict(self):
        """
        Update the Kalman filter without observed keypoints and return the predicted keypoints
        """
        self.kf.predict()
        self.time_since_update += 1
        # self.age += 1
        return self.kf.x[:self.num_joints*self.dim].reshape((self.num_joints, self.dim))
    
    def get_state(self):
        """
        Return the current estimation of the keypoints
        """
        return self.kf.x[:self.num_joints*self.dim].reshape((self.num_joints, self.dim))
    
    # def get_age(self):
    #     """
    #     Return the age of the Kalman filter
    #     """
    #     return self.age
    
    def stale_since(self):
        """
        Return the number of frames since the last observed keypoints
        """
        return self.time_since_update
        
    def is_confirmed(self):
        """
        Return whether the Kalman filter is confirmed
        """
        return self.time_since_update == 0
    
    
class PoseFilter:
    """
    A class to filter out noisy keypoints by tracking them using Kalman filter
    """
    def __init__(self, num_joints=17, dim=2, dt=1, max_staleness=90):
        """
        Initialize the PoseFilter with the number of joints and dimensions
        """
        self.num_joints = num_joints
        self.dim = dim
        self.dt = dt
        self.max_staleness = max_staleness
        self.track_filters = {}

    def update(self, 
               track_ids: List[int],
               poses: List[np.ndarray],
               anchors: List[np.ndarray],
               confirmed: List[bool]) -> List[np.ndarray]:
        """
        Update the Kalman filter for a given track_id with observed keypoints
        """
        updated_keypoints = []

        to_delete = []
        for id, trk in self.track_filters.items():
            trk.predict()
            if trk.stale_since() > self.max_staleness:
                to_delete.append(id)
        
        for id in to_delete:
            del self.track_filters[id]
        
        for i, track_id in enumerate(track_ids):
            if track_id > 0: 

                z = np.array([np.concatenate((poses[i].flatten(), 
                                              anchors[i].flatten())) 
                                              if poses[i] is not None else None])

                if track_id not in self.track_filters:
                    self.track_filters[track_id] = PoseKalmanFilter(
                        num_joints=self.num_joints,
                        dim=self.dim,
                        dt=self.dt,
                        initial_keypoints=z
                    )

                pose_kalman_filter = self.track_filters[track_id]

                if confirmed[i]:
                    pose_kalman_filter.update(z)
                else:
                    # Increase the measurement uncertainty for unconfirmed keypoints temporarily
                    # factor = 10 if self.dim == 2 else 15
                    pose_kalman_filter.kf.R *= 10
                    pose_kalman_filter.update(z)
                    pose_kalman_filter.kf.R /= 10

                current_state = pose_kalman_filter.get_state()
                updated_keypoints.append(current_state)

        return np.array(updated_keypoints)
    


                



    





# class PoseKalmanFilter:
#     def __init__(self, num_joints=17, dt=1, initial_keypoints=None):
#         # Initialize a Kalman filter for 3D pose tracking of all joints
#         self.num_joints = num_joints
#         self.dim = 3  # Assuming 3D keypoints
#         state_dim = self.num_joints * 2 * self.dim  # Position and Velocity
#         meas_dim = self.num_joints * self.dim  # Just Position

#         # Initialize the Kalman filter
#         self.kf = KalmanFilter(dim_x=state_dim, dim_z=meas_dim)

#         # State transition matrix
#         self.kf.F = np.eye(state_dim)
#         for i in range(meas_dim):
#             self.kf.F[i, i+meas_dim] = dt

#         # Measurement function
#         self.kf.H = np.zeros((meas_dim, state_dim))
#         for i in range(meas_dim):
#             self.kf.H[i, i] = 1

#         # Measurement uncertainty
#         self.kf.R = np.eye(meas_dim) * 25

#         # Process uncertainty
#         self.kf.Q = np.eye(state_dim) * 1
#         for i in range(meas_dim):
#             self.kf.Q[i+meas_dim, i+meas_dim] *= 0.1

#         # Initial state uncertainty
#         self.kf.P *= 2

#         # Set initial state if initial keypoints are provided
#         if initial_keypoints is not None and len(initial_keypoints.flatten()) == meas_dim:
#             self.kf.x[:meas_dim] = initial_keypoints.reshape(-1,1)  # Position
#             self.kf.x[meas_dim:2*meas_dim] = 0  # Velocity

#     def update(self, z):
#         self.kf.predict()
#         if z is not None:
#             self.kf.update(z)

#     def get_state(self):
#         return self.kf.x[:self.num_joints * self.dim].reshape((self.num_joints, self.dim))






# class PoseFilter:
#     def __init__(self):
#         self.track_filters = {}

#     def update(self, track_id, keypoints):
#         if track_id not in self.track_filters:
#             self.track_filters[track_id] = PoseKalmanFilter(
#                 num_joints=len(keypoints),
#                 initial_keypoints=keypoints if np.all(~np.isnan(keypoints)) else None
#             )

#         kalman_filter = self.track_filters[track_id]
#         if np.all(~np.isnan(keypoints)):
#             kalman_filter.update(keypoints.flatten())
#         else:
#             kalman_filter.update(None)

#         return kalman_filter.get_state()

#     def get_state(self, tracklets, detected_keypoints):
#         filtered_keypoints = {}
#         for tracklet in tracklets:
#             track_id = int(tracklet[2])
#             if track_id in detected_keypoints:
#                 keypoints = detected_keypoints[track_id]
#                 filtered_keypoints[track_id] = self.update(track_id, keypoints)
#         return filtered_keypoints

#     def reset(self):
#         self.track_filters = {}