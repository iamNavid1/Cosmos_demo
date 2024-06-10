import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from projection.smoothen import TrajectoryFilter
import os
import glob


class Cam2Bird:
    
    def __init__(self):
        homographies_path =  "./projection/Homography_Matrix"
        regions_path = "./projection/ROI_Vertices"
        self.h_matrices, self.regions, self.centeroids = self.load(homographies_path, regions_path)
        self.filter_trajectory = TrajectoryFilter() # Smoothing the trajectory
        

    @staticmethod
    def load(homographies_path, regions_path):
        h_files = glob.glob(os.path.join(homographies_path, "*.csv"))
        r_files = glob.glob(os.path.join(regions_path, "*.csv"))
        h_matrices = []
        regions = []
        centeroids = []
        for h_file, r_file in zip(h_files, r_files):
            h_matrix = pd.read_csv(h_file, header=None).values
            h_matrices.append(h_matrix)
            region = pd.read_csv(r_file, header=None).values
            regions.append(region)
            centeroid = np.mean(region, axis=0)
            centeroids.append(centeroid)
        return h_matrices, regions, centeroids


    @staticmethod
    def is_inside_region(x, y, vertices):
        point = Point(x, y)
        polygon = Polygon(vertices)
        inside = point.intersects(polygon)
        return inside
    
    def find_homography(self, CamCoordinate):
        for region_id, vertices in enumerate(self.regions):
            if self.is_inside_region(CamCoordinate[0], CamCoordinate[1], vertices):
                return self.h_matrices[region_id]

        # If the point is not inside any of the regions due to detection inacuracies:
        # Find the closest region centeroid to the point
        distances = []
        for centeroid in self.centeroids:
            distance = np.linalg.norm(np.array(CamCoordinate[:2]) - centeroid)
            distances.append(distance)
        region_id = np.argmin(distances)
        return self.h_matrices[region_id]

    def add_transformation(self, CamCoordinates):
        destinations = []
        track_ids = []
        for CamCoordinate in CamCoordinates:
            h = self.find_homography(CamCoordinate[7:9])
            source = np.array([[CamCoordinate[7]], [CamCoordinate[8]], [1]])
            destination = np.dot(h, source)
            destination /= destination[2]
            destination = (destination[:2] + .5).astype(int).flatten()
            destinations.append(destination)
            track_ids.append(CamCoordinate[2])
        # Smoothing the positions using Kalman Filter
        smoothed_positions = self.filter_trajectory.update(track_ids, destinations)
        # Combining original and smoothed positions
        self.positions = (np.concatenate((np.array(CamCoordinates, dtype='object'), np.array(smoothed_positions)), axis=1)).tolist()
        return self.positions
    
