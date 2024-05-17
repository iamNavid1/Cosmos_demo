import numpy as np
import cv2 as cv
import colorsys
import vision.constants as c
import mmcv
from mmpose.apis import (convert_keypoint_definition, 
                        extract_pose_sequence,
                        inference_pose_lifter_model, inference_topdown)
from mmpose.structures import PoseDataSample, merge_data_samples, split_instances

class ObjectDetection:
    
    def __init__(self, model, timestamp, classes_list, tracker=None): 
        self.model = model
        self.timestamp = timestamp
        self.classes_list = classes_list
        self.tracker = tracker


    @staticmethod
    def get_color(number):
        " Converts an integer number to a color "
        # change these however you want to
        hue = number*30 % 180
        saturation = number*103 % 256
        value = number*50% 256
        
        # expects normalized values
        color = colorsys.hsv_to_rgb (hue/179, saturation/255, value/255)
        return [int(c*255) for c in color]


    def get_pedestrains(self, frame, frame_idx):
        # results = self.model(self.frame, device=c.device, conf=c.conf)
        self.frame = frame.copy()
        results = self.model(self.frame, conf=c.conf)                        
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
            if self.tracker is None:
                color = (0, 255, 0)
                cv.rectangle(self.frame, (x, y), (x2, y2), color, 2, cv.LINE_AA)
                label = f"{self.classes_list[cls]}: {conf:.2f}"
                cv.putText(self.frame, label, (x, y-8), cv.FONT_HERSHEY_PLAIN, 1.25, color, 2, cv.LINE_AA)
                center =  [int((x + x2)/2 +.5), y2]
                # cv.circle(self.frame, center, 5, (255, 0, 0), -1, cv.LINE_AA)

        if self.tracker is not None:
            result_ = np.asarray(result_)
            if result_.shape[0] == 0:
                result_ = np.array([[-1, -1, -1, -1, -1, -1]])
            # update the tracker with the new detections
            tracks = self.tracker.update(result_, self.frame)
            current_time = self.timestamp[frame_idx-1]
            for track in tracks:
                xyxy = track[0:4].astype('int') # float64 to int
                # check if the bounding box is more than 20px outside the frame dimensions
                if xyxy[0]+20 < 0 or xyxy[0]-20 > self.frame.shape[1] or xyxy[1]+20 < 0 or xyxy[1]-20 > self.frame.shape[0]:
                    continue
                id = track[4].astype('int') # float64 to int
                conf = track[5]
                cls = track[6].astype('int') # float64 to int
                # ind = track[7].astype('int') # float64 to int

                center =  [int((xyxy[0] + xyxy[2])/2 +.5), int((xyxy[3]-.1*(xyxy[3]-xyxy[1]))+.5)]
                # cv.circle(self.frame, [int(center[0]), int(center[1])], 5, (255, 0, 0), -1)

                # if center[0] < 0 or center[0] > self.frame.shape[1] or center[1] < 0 or center[1] > self.frame.shape[0]:
                #     continue
                detections.append([current_time, frame_idx, id, xyxy[0], xyxy[1], xyxy[2], xyxy[3], center[0], center[1]])
                color = self.get_color(id*50)
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
                    (xyxy[0], xyxy[1]-10),
                    cv.FONT_HERSHEY_PLAIN,
                    1.2,
                    color,
                    1,
                    cv.LINE_AA
                )
            # sort the detections by the track id
            detections = sorted(detections, key=lambda x: x[2])
        return self.frame, detections


class PoseEstimation:
        
        def __init__(self, pose_estimator, pose_lifter, visualizer, pose_viz=False):
            self.pose_estimator = pose_estimator
            self.pose_lifter = pose_lifter
            self.visualizer = visualizer
            self.pose_est_results_list = []
            self.pose_viz = pose_viz


        def get_pose(self, frame, frame_idx, detections):
            self.frame = frame.copy()
            self.frame_idx = frame_idx
            self.detections = detections

            det_frame, pose_frame, pred_3d_keypoints = self.process_one_image()
            return det_frame, pose_frame, pred_3d_keypoints


        def process_one_image(self):

            visualize_frame = mmcv.bgr2rgb(self.frame)
                    
            bboxes = [[p[3], p[4], p[5], p[6]] for p in self.detections]
            track_ids = [p[2] for p in self.detections]
            
            pose_lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
            pose_lift_dataset_name = self.pose_lifter.dataset_meta['dataset_name']

            pose_est_results = inference_topdown(self.pose_estimator, self.frame, bboxes)

            pose_det_dataset_name = self.pose_estimator.dataset_meta['dataset_name']
            pose_est_results_converted = []

            # convert 2d pose estimation results into the format for pose-lifting
            # such as changing the keypoint order, flipping the keypoint, etc.
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

            # ------------------------------------------ KALMAN FILTER ------------------------------------------
            # detected_keypoints = {item.track_id: item.pred_instances.keypoints[0] for item in pose_lift_results}
            # filtered_keypoints, track_filters = process_pose_estimates_with_kalman(positions, detected_keypoints, track_filters)
            # for item in pose_lift_results:
            #     item.pred_instances.keypoints[0] = filtered_keypoints[item.track_id]
            # ---------------------------------------------------------------------------------------------------

            pred_3d_data_samples = merge_data_samples(pose_lift_results)
            det_data_sample = merge_data_samples(pose_est_results)
            pred_3d_keypoints = pred_3d_data_samples.get('pred_instances', None).get('keypoints')
            
            if self.pose_viz:
                det_frame, pose_frame = self.visualizer.add_datasample(
                    'result',
                    visualize_frame,
                    data_sample=pred_3d_data_samples,
                    det_data_sample=det_data_sample,
                    track_ids=track_ids,
                    draw_gt=False,
                    dataset_2d=pose_det_dataset_name,
                    dataset_3d=pose_lift_dataset_name,
                    show=False,
                    draw_bbox=False,
                    kpt_thr=0.1,
                    num_instances=8,
                    wait_time=0,
                    show_kpt_idx=False)
                # frame_vis = self.visualizer.get_image()
                # mmcv.imwrite(frame_vis, 'NEW.jpg')
            
            else:
                det_frame = self.frame
                pose_frame = None
                                    
            return det_frame, pose_frame, pred_3d_keypoints

