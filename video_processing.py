import os
import cv2 as cv
import numpy as np
import argparse
import datetime
from vision.detectors import ObjectDetection, PoseEstimation
from vision.loader import ResourceLoader
from vision.bbox3d import BBox3D
from vision.util import TrackHistoryUpdate, VizBird, Viz3Dbbox
from projection.coordTrans import Cam2Bird

class VideoProcessor:
    """
    Main class to process the video stream for pedestrian tracking and pose estimation
    """
    def __init__(self, args):

        args.fps = 4
        args.resolution = '1920x1080'
        args.num_instances =5
        args.plot_size = 600
        args.pose_viz = True


        self.args = args

        # Import resources
        self.loader = ResourceLoader(args)
        self.detection_model, self.classes_list = self.loader.load_yolo()
        self.tracker = self.loader.load_tracker()
        self.pose_model, self.pose_lifter, self.visualizer = self.loader.load_pose()
        self.birdseye = self.loader.load_background()

        # Initialize the object detector, pose estimator, and projection
        object_detector = ObjectDetection(self.detection_model, self.classes_list, self.tracker, args)
        pose_estimator = PoseEstimation(self.pose_model, self.pose_lifter, self.visualizer, args)
        projection = Cam2Bird()

        # Frame properties for displaying and saving
        self.cam_width, self.cam_height = map(int, args.resolution.split('x'))
        self.birdseye_height, self.birdseye_width = self.birdseye.shape[:2]
        self.total_width = self.cam_width + self.birdseye_width
        self.total_height = max(self.cam_height, self.birdseye_height) + args.plot_size
        if args.num_instances * args.plot_size > self.total_width:
            args.num_instances = self.total_width // args.plot_size
        self.combined_frame = np.ones((self.total_height, self.total_width, 3), dtype=np.uint8) * 255

        plot_order = [0] * args.num_instances    # A list to Maintain the order of pose3d plots
        track_dic = {}                           # A dictionary to store the track history
        frame_idx = 1                            # Frame index

    def process_frame(self, frame):
        # Detect people in the frame
        frame, detections, confirmed = self.object_detector.get_pedestrains(frame, self.frame_idx)
        if len(detections) > 0:
            detections = self.projection.add_transformation(detections)

        # Get the pose estimation of the detected pedestrains
        pose2d_frame, pose3d_dic, pose_det = self.pose_estimator.get_pose(frame, self.frame_idx, detections, confirmed)
        frame = pose2d_frame

        # Update the track history and generate the 3D bounding box
        self.track_dic = TrackHistoryUpdate(self.track_dic, detections, pose_det, self.frame_idx) 
        self.track_dic = BBox3D(self.track_dic).get_3dbox()

        # Visualize the 3D bounding box of the detected objects
        frame = Viz3Dbbox(self.track_dic, frame)

        # Visualize the bird's eye view of the detected objects
        birdseye_copy = self.birdseye.copy()
        VizBird(self.track_dic, birdseye_copy, self.args)

        self.frame_idx += 1

        # Write the outputs to the frame
        self.combined_frame.fill(255)
        self.combined_frame[:self.cam_height, :self.cam_width] = frame
        self.combined_frame[:self.birdseye_height, self.cam_width:] = birdseye_copy
        y_offset = max(self.cam_height, self.birdseye_height)

        availability = [True] * self.args.num_instances
        included = []
        for i, track_id in enumerate(self.plot_order):
            if track_id in pose3d_dic:
                self.combined_frame[y_offset:, i * self.args.plot_size:(i + 1) * self.args.plot_size] = pose3d_dic[track_id]
                included.append(track_id)
                availability[i] = False

        for i, free in enumerate(availability):
            if free:
                for key in pose3d_dic:
                    if key not in included:
                        self.combined_frame[y_offset:, i * self.args.plot_size:(i + 1) * self.args.plot_size] = pose3d_dic[key]
                        included.append(key)
                        self.plot_order[i] = key if key > 0 else self.plot_order[i]
                        break

        return self.combined_frame
