import os
import numpy as np
import pandas as pd
import cv2 as cv
from ultralytics import YOLO
import vision.constants as c 
from vision.tracker import OCSORT
from pathlib import Path
from mmpose.apis import init_model
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.models.pose_estimators import PoseLifter
from mmpose.registry import VISUALIZERS

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  

class ResourceLoader:
    
    def __init__(self): 
        pass


    def load_yolo(self):          # Load YOLO model and class names
        try:
            model = YOLO(c.yoloWeights)
            model.to('cuda')
            # torch.cuda.set_device(0)
            with open(c.ClassesListPath, "r") as f:
                classes_list = [line.strip() for line in f.readlines()]
            return model, classes_list
        except FileNotFoundError as e:
            print(f"Error loading YOLO: {e}")
            return None, None
        except Exception as e:
            print(f"An error occurred while loading YOLO: {e}")
            return None, None


    def load_tracker(self):
        try:
            tracker = OCSORT(
                # model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
                # device='mps',
                # fp16=True,
                asso_func='iou',
                det_thresh = .2,
                min_hits=3,
                max_age = 90,
                fps = int(c.fps))
            return tracker
        except FileNotFoundError as e:
            print(f"Error loading tracker: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while loading tracker: {e}")
            return None


    def load_pose(self):
        try:
            pose_estimator = init_model(
                c.PoseModelPath,
                'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
                device='cuda')
            assert isinstance(pose_estimator, TopdownPoseEstimator), 'Only "TopDown"' \
                'model is supported for the 1st stage (2D pose detection)'

            det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
            det_dataset_skeleton = pose_estimator.dataset_meta.get('skeleton_links', None)
            det_dataset_link_color = pose_estimator.dataset_meta.get('skeleton_link_colors', None)

            pose_lifter = init_model(
                c.PoseLifterPath,
                'https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth',
                device='cuda')
            assert isinstance(pose_lifter, PoseLifter), \
                'Only "PoseLifter" model is supported for the 2nd stage (2D-to-3D lifting)'

            pose_lifter.cfg.visualizer.radius = 3
            pose_lifter.cfg.visualizer.line_width = 1
            pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
            pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
            pose_lifter.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
            visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)

            # the dataset_meta is loaded from the checkpoint
            visualizer.set_dataset_meta(pose_lifter.dataset_meta)
            
            return pose_estimator, pose_lifter, visualizer

        except FileNotFoundError as e:
            print(f"Error loading Pose: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while loading Pose: {e}")
            return None
        

    def load_background(self):
        try:
            BirdsEye = cv.imread(c.BirdsEyeViewPath)
            return BirdsEye
        except FileNotFoundError as e:
            print(f"Error loading images: {e}")
            return None, None
        except Exception as e:
            print(f"An error occurred while loading background image: {e}")
            return None, None
    
