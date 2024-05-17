import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from projection.smoothen import TrajectorySmoother
import os
import glob


class Cam2Bird:
    
    def __init__(self):
        homographies_path =  "./projection/Homography_Matrix"
        regions_path = "./projection/ROI_Vertices"
        self.h_matrices, self.regions = self.load(homographies_path, regions_path)
        self.smoother = TrajectorySmoother()
        

    @staticmethod
    def load(homographies_path, regions_path):
        h_files = glob.glob(os.path.join(homographies_path, "*.csv"))
        r_files = glob.glob(os.path.join(regions_path, "*.csv"))
        h_matrices = []
        regions = []
        for h_file, r_file in zip(h_files, r_files):
            h_matrix = pd.read_csv(h_file, header=None).values
            h_matrices.append(h_matrix)
            region = pd.read_csv(r_file, header=None).values
            regions.append(region)
        return h_matrices, regions


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
        if self.is_inside_region(CamCoordinate[0], CamCoordinate[1], [[0-100,720+100],[0-100,330],[1280+100,330],[1280+100,720+100]]):
            return self.h_matrices[0]
        if self.is_inside_region(CamCoordinate[0], CamCoordinate[1], [[0-100,330],[0-100,220],[1280+100,220],[1280+100,330]]):
            return self.h_matrices[1]
        if self.is_inside_region(CamCoordinate[0], CamCoordinate[1], [[0-100,220],[0-100,0-100],[640,0-100],[640,220]]):
            return self.h_matrices[5]
        if self.is_inside_region(CamCoordinate[0], CamCoordinate[1], [[640,220],[640,0-100],[1280+100,0-100],[1280+100,220]]):
            return self.h_matrices[3]
        raise ValueError("Point is not inside any region or close to any region!")


    def add_transformation(self, CamCoordinates):
        destinations = []
        track_ids = []
        for CamCoordinate in CamCoordinates:
            h = self.find_homography(CamCoordinate[7:9])
            source = np.array([[CamCoordinate[7]], [CamCoordinate[8]], [1]])
            destination = np.dot(h, source)
            destination /= destination[2]
            destination = (destination[:2] + .5).astype(int).flatten()
            destinations.append(destination.tolist())
            track_ids.append(CamCoordinate[2])
        # Smoothing the positions using Kalman Filter
        smoothed_positions = self.smoother.update(destinations, track_ids)
        # Combining original and smoothed positions
        self.positions = (np.concatenate((np.array(CamCoordinates, dtype='object'), np.array(smoothed_positions)), axis=1)).tolist()
        return self.positions
    
