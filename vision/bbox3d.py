import numpy as np
from projection.coordTrans import Cam2Bird
from collections import deque
import math
from scipy.ndimage import convolve1d


class BBox3D:
    
    def __init__(self, track_dic): 
        self.track_dic = track_dic
        self.pos_bird = {}
        self.check = {}
        for key in track_dic:
            self.pos_bird[key] = track_dic[key]['pos_bird']
            if len(track_dic[key]['pos_bird']) > 1:
                if track_dic[key]['pos_bird'][-1] is not None and track_dic[key]['pos_bird'][-2] is not None:
                    self.check[key] = True
                else:
                    self.check[key] = False
            else:
                self.check[key] = False
        self.projection = Cam2Bird()


    @staticmethod
    def get_theta(pos):
        last_pos = pos[-60:]
        filtered_points = np.array([item for item in last_pos if item is not None])
        filtered_points = filtered_points[::-1]

        if len(filtered_points) == 2:
            return math.atan2(filtered_points[1][1] - filtered_points[0][1], filtered_points[1][0] - filtered_points[0][0])
        
        # Calculate the median of x and y values
        median_x, median_y = np.median(filtered_points, axis=0)

        # Center the points
        filtered_points = filtered_points.astype(float)
        filtered_points[:, 0] -= median_x
        filtered_points[:, 1] -= median_y

        window_size = int(len(filtered_points) * 0.15)+1  
        kernel = np.ones(window_size) / window_size
        smoothed_x = convolve1d(filtered_points[:, 0], weights=kernel, mode='nearest')
        smoothed_y = convolve1d(filtered_points[:, 1], weights=kernel, mode='nearest')

        # Split into 2 parts
        split_index = int(len(smoothed_x) / 3)
        x2, y2 = smoothed_x[:split_index], smoothed_y[:split_index]
        x1, y1 = smoothed_x[2 * split_index:], smoothed_y[2 * split_index:]

        # get the median of the two parts
        x2_median, y2_median = np.median(x2), np.median(y2)
        x1_median, y1_median = np.median(x1), np.median(y1)

        theta = math.atan2(y2_median - y1_median, x2_median - x1_median)

        return theta
        

    @staticmethod
    def get_bottom(center, theta):
        xc, yc = center
        bottom = np.array([[xc-12, yc-7], 
                           [xc+12, yc-7], 
                           [xc+12, yc+7], 
                           [xc-12, yc+7]])
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        rotated_bottom = np.matmul(R, (bottom - [xc, yc]).T).T + [xc, yc]
        rotated_bottom = (rotated_bottom + .5).astype(int)
        return rotated_bottom
        

    @staticmethod
    def transform_bottom(bottom_bird, center_cam, projection):
        homography_matrix = np.linalg.inv(projection.find_homography(center_cam))
        # transform the bottom from bird's eye view to camera view
        bottom_cam = np.concatenate((bottom_bird, np.ones((4,1))), axis=1)
        bottom_cam = np.matmul(homography_matrix, bottom_cam.T).T
        bottom_cam /= bottom_cam[:,2].reshape(-1,1)
        bottom_cam = (bottom_cam[:,:2] + .5).astype(int)
        return bottom_cam    


    def get_3dbox(self):
        desired_order = ['timestamp', 'frame_count', 'pos_cam', 'pos_bird', 'bbox2D', 'bbox2D', 'bbox3D', 'theta', 'pose3D']
        for key in self.pos_bird:
            if 'bbox3D' not in self.track_dic[key]:
                self.track_dic[key]['bbox3D'] = []
                self.track_dic[key]['theta'] = []
            if self.check[key] is True:
                center_bird = self.pos_bird[key][-1]
                theta = self.get_theta(self.pos_bird[key])
                bottom_bird = self.get_bottom(center_bird, theta)
                center_cam = np.array(self.track_dic[key]['pos_cam'][-1])
                bottom_cam = self.transform_bottom(bottom_bird, center_cam, self.projection)                
                height = self.track_dic[key]['bbox2D'][-1][1] - self.track_dic[key]['bbox2D'][-1][3]
                top_cam = bottom_cam.copy()
                top_cam[:,1] += int(.9 * height)
                self.track_dic[key]['bbox3D'].append(np.concatenate((top_cam, bottom_cam), axis=0).tolist())
                self.track_dic[key]['theta'].append(theta)
            elif len(self.track_dic[key]['bbox3D'])==0:
                self.track_dic[key]['bbox3D'].append(None)
                self.track_dic[key]['theta'].append(None)
            elif self.track_dic[key]['bbox3D'][-1] != None:
                self.track_dic[key]['bbox3D'].append(None)
                self.track_dic[key]['theta'].append(None)

        self.track_dic = {
            outer_key: {key: inner_dict[key] for key in desired_order if key in inner_dict}
            for outer_key, inner_dict in self.track_dic.items()
        }
        return self.track_dic