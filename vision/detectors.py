import numpy as np
import cv2 as cv
import mmcv
import time
import vision.constants as c
from shapely.geometry import Point, Polygon
from .util import get_color
from .filters import PoseFilter
from mmpose.apis import (convert_keypoint_definition, 
                        extract_pose_sequence,
                        inference_pose_lifter_model, inference_topdown)
from mmpose.structures import PoseDataSample, merge_data_samples, split_instances

class ObjectDetection:
    
    def __init__(self, model, classes_list, tracker, args): 
        self.model = model
        self.classes_list = classes_list
        self.tracker = tracker
        width, height = map(int, args.resolution.split('x'))
        mask = np.array([[int(vertex[0]*width), int(vertex[1]*height)] 
                         for vertex in c.frame_mask])
        self.roi = Polygon(mask)


    def get_pedestrains(self, frame, frame_idx):
        # results = self.model(self.frame, device=c.device, conf=c.conf)
        self.frame = frame.copy()
        results = self.model(self.frame, conf=c.conf, verbose=False)      
        self.result = results[0]
        # Store detected objects and their confidence levels
        bboxes = np.array(self.result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(self.result.boxes.cls.cpu(), dtype="int")
        confs = np.round(np.array(self.result.boxes.conf.cpu(), dtype="float"), 2)
        
        result_ = []
        detections = []
        person_mask = (classes == c.PersonIndex)
        for bbox, cls, conf in zip(bboxes[person_mask], classes[person_mask], confs[person_mask]):
            (x, y, x2, y2) = bbox
            result_.append([x, y, x2, y2, conf, cls])

        # track the detected objects
        result_ = np.asarray(result_)
        if result_.shape[0] == 0:
            result_ = np.array([[-1, -1, -1, -1, -1, -1]])  # set a dummy value for the bounding box to avoid errors in the tracker

        # update the tracker with new detections
        tracks, certainty = self.tracker.update(result_, self.frame)
        current_time = time.time()

        for track in tracks:
            xyxy = track[0:4].astype('int') # float64 to int
            
            # check if the bbox falls within the region of interest
            if not self.roi.contains(Point((xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2)):
                continue

            id = track[4].astype('int') # float64 to int
            center =  [int((xyxy[0] + xyxy[2])/2 +.5), int((xyxy[3]-.1*(xyxy[3]-xyxy[1]))+.5)]
            # cv.circle(self.frame, [int(center[0]), int(center[1])], 5, (255, 0, 0), -1)

            detections.append([current_time, frame_idx, id, xyxy[0], xyxy[1], xyxy[2], xyxy[3], center[0], center[1]])
            color = get_color(id)
            # cv.rectangle(
            #     self.frame,
            #     (xyxy[0], xyxy[1]),
            #     (xyxy[2], xyxy[3]),
            #     color,
            #     2
            # )
            cv.putText(
                self.frame,
                f'id: {id}',
                (xyxy[0], xyxy[1]-20),
                cv.FONT_HERSHEY_PLAIN,
                1.5,
                color,
                1,
                cv.LINE_AA
            )
        # sort the detections by the track id
        detections = sorted(detections, key=lambda x: x[2])
        return self.frame, detections, certainty


class PoseEstimation:
        
        def __init__(self, pose_estimator, pose_lifter, visualizer, args):
            self.pose_estimator = pose_estimator
            self.pose_lifter = pose_lifter
            self.visualizer = visualizer
            self.pose_est_results_list = []
            self.num_instances = args.num_instances
            self.plot_size = args.plot_size
            self.filter2D = PoseFilter(dim=2, fps=args.fps)
            self.filter3D = PoseFilter(dim=3, fps=args.fps)


        def get_pose(self, frame, frame_idx, detections, confirmed):
            self.frame = frame.copy()
            self.frame_idx = frame_idx
            if len(detections) == 0: # set a dummy value for the bounding box to avoid errors
                self.detections = [[-1, -1, -1, -1, -1, -1, -1]]
            else:
                self.detections = detections
            self.confirmed = confirmed

            pose2d_frame, pose3d_dic, pred_3d_keypoints = self.process_one_image()
            return pose2d_frame, pose3d_dic, pred_3d_keypoints


        def process_one_image(self):

            visualize_frame = mmcv.bgr2rgb(self.frame)

            bboxes = [[p[3], p[4], p[5], p[6]] for p in self.detections]
            track_ids = [p[2] for p in self.detections]
            
            pose_lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
            pose_lift_dataset_name = self.pose_lifter.dataset_meta['dataset_name']
            pose_det_dataset_name = self.pose_estimator.dataset_meta['dataset_name']

            pose_est_results = inference_topdown(self.pose_estimator, self.frame, bboxes)

            # ------------------------------------------ KALMAN FILTER ------------------------------------------
            keypoints = [item.pred_instances.keypoints[0] for item in pose_est_results]

            score_confidence = [True if np.mean(item.pred_instances.keypoint_scores) > 0.33 
                                else False for item in pose_est_results]
            
            self.confirmed = [score_confidence and track_confirmed for 
                         score_confidence, track_confirmed in zip(score_confidence, self.confirmed)]
            
            anchors = [np.array([bbox[0], bbox[1]]) for bbox in bboxes]

            updated_keypoints = self.filter2D.update(track_ids, keypoints, anchors, self.confirmed)

            if len(updated_keypoints) > 0:
                for i, data_sample in enumerate(pose_est_results):
                    data_sample.pred_instances.keypoints[0] = updated_keypoints[i]
            # ----------------------------------------------------------------------------------------------------

            # convert 2d pose estimation results into the format for pose-lifting
            # such as changing the keypoint order, flipping the keypoint, etc.
            pose_est_results_converted = []
            for i, data_sample in enumerate(pose_est_results):
                pred_instances = data_sample.pred_instances.cpu().numpy()
                keypoints = pred_instances.keypoints
                # calculate area and bbox
                if 'bboxes' in pred_instances:
                    areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                    for bbox in pred_instances.bboxes])
                    pose_est_results[i].pred_instances.set_field(areas, 'areas')
                else:
                    areas, bboxes = [], []
                    for keypoint in keypoints:
                        xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                        xmax = np.max(keypoint[:, 0])
                        ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                        ymax = np.max(keypoint[:, 1])
                        areas.append((xmax - xmin) * (ymax - ymin))
                        bboxes.append([xmin, ymin, xmax, ymax])
                    pose_est_results[i].pred_instances.areas = np.array(areas)
                    pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

                # track id
                if len(track_ids) == 0:
                    track_id = -1
                else:
                    track_id = track_ids[i]
                pose_est_results[i].set_field(track_id, 'track_id')
                
                # convert keypoints for pose-lifting
                pose_est_result_converted = PoseDataSample()
                pose_est_result_converted.set_field(
                    pose_est_results[i].pred_instances.clone(), 'pred_instances')
                pose_est_result_converted.set_field(
                    pose_est_results[i].gt_instances.clone(), 'gt_instances')
                keypoints = convert_keypoint_definition(keypoints,
                                                        pose_det_dataset_name,
                                                        pose_lift_dataset_name)
                pose_est_result_converted.pred_instances.set_field(
                    keypoints, 'keypoints')
                pose_est_result_converted.set_field(pose_est_results[i].track_id,
                                                    'track_id')
                pose_est_results_converted.append(pose_est_result_converted)

            self.pose_est_results_list.append(pose_est_results_converted.copy())

            # Second stage: Pose lifting
            # extract and pad input pose2d sequence
            pose_seq_2d = extract_pose_sequence(
                self.pose_est_results_list,
                frame_idx=self.frame_idx,
                causal=pose_lift_dataset.get('causal', False),
                seq_len=pose_lift_dataset.get('seq_len', 1),
                step=pose_lift_dataset.get('seq_step', 1))

            # conduct 2D-to-3D pose lifting
            pose_lift_results = inference_pose_lifter_model(
                self.pose_lifter,
                pose_seq_2d,
                image_size=visualize_frame.shape[:2],
                norm_pose_2d=True)

            # print("3d keypoint_scores are:", pose_lift_results[0].pred_instances.keypoint_scores)

            # post-processing
            for idx, pose_lift_result in enumerate(pose_lift_results):
                pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

                pred_instances = pose_lift_result.pred_instances
                keypoints = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores
                if keypoint_scores.ndim == 3:
                    keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                    pose_lift_results[
                        idx].pred_instances.keypoint_scores = keypoint_scores
                if keypoints.ndim == 4:
                    keypoints = np.squeeze(keypoints, axis=1)

                keypoints = keypoints[..., [0, 2, 1]]
                keypoints[..., 0] = -keypoints[..., 0]
                keypoints[..., 2] = -keypoints[..., 2]

                # rebase height (z-axis)
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

                pose_lift_results[idx].pred_instances.keypoints = keypoints

            pose_lift_results = sorted(
                pose_lift_results, key=lambda x: x.get('track_id', 1e4))
            track_ids = [item.track_id for item in pose_lift_results]

            # # ------------------------------------------ KALMAN FILTER ------------------------------------------
            keypoints = [item.pred_instances.keypoints[0] for item in pose_lift_results]
            
            updated_keypoints = self.filter3D.update(track_ids, keypoints, anchors, self.confirmed)

            if len(updated_keypoints) > 0:
                for i, data_sample in enumerate(pose_lift_results):
                    data_sample.pred_instances.keypoints[0] = updated_keypoints[i]
            # # ----------------------------------------------------------------------------------------------------

            pred_3d_data_samples = merge_data_samples(pose_lift_results)
            det_data_sample = merge_data_samples(pose_est_results)
            pred_3d_keypoints = pred_3d_data_samples.get('pred_instances', None).get('keypoints')
            
            # if self.pose_viz:
            pose2d_frame, pose3d_dic = self.visualizer.add_datasample(
                visualize_frame,
                data_sample=pred_3d_data_samples,
                det_data_sample=det_data_sample,
                track_ids=track_ids,
                dataset_2d=pose_det_dataset_name,
                dataset_3d=pose_lift_dataset_name,
                draw_bbox=False,
                kpt_thr=0.2,
                num_instances=self.num_instances,
                plot_size=self.plot_size,
                show_kpt_idx=False)            
            # else:
            #     pose2d_frame = self.frame
            #     pose3d_dic = {}
                                    
            return pose2d_frame, pose3d_dic, pred_3d_keypoints

