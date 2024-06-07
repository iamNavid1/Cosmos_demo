import numpy as np
import os
import cv2 as cv
import time
import datetime
import argparse
import datetime
from vision.detectors import ObjectDetection, PoseEstimation
from vision.loader import ResourceLoader
from vision.bbox3d import BBox3D
from vision.util import TrackHistoryUpdate, VizBird, Viz3Dbbox
from projection.coordTrans import Cam2Bird
import json
import csv


# Main function to stream the video, detect people, track on birdseye view, and save the output 
def Main(input_path, combined_display, pose_viz):

    # Import resources
    loader = ResourceLoader()
    detection_model, classes_list = loader.load_yolo()        # YOLO model and its classes 
    tracker = loader.load_tracker()                           # Tracker model
    pose_model, pose_lifter, visualizer = loader.load_pose()  # Pose estimation model, its lifter, and visualizer
    cam, birdseye, timestamp1 = loader.load_background()      # Background images at camera view and birdseye view

    object_detector = ObjectDetection(detection_model, timestamp1, classes_list, tracker)
    pose_estimator = PoseEstimation(pose_model, pose_lifter, visualizer, pose_viz)
    projection = Cam2Bird()

    # Frame properties for displaying and saving
    cam_height, cam_width = cam.shape[:2]
    birdseye_height, birdseye_width = birdseye.shape[:2]
    padding = 25
    combined_frame_width = cam_width + birdseye_width + padding
    combined_frame_height = max(cam_height, birdseye_height)
    combined_frame = np.zeros((combined_frame_height, combined_frame_width, 3), dtype=np.uint8)
    
    # Create output path to save
    directory_path = os.path.dirname(input_path)
    file_name, extension = os.path.splitext(os.path.basename(input_path))
    cam_output_path = os.path.join(directory_path, file_name + "_cam" + extension)
    birdseye_output_path = os.path.join(directory_path, file_name + "_birdseye" + extension)
    combined_output_path = os.path.join(directory_path, file_name + "_combined" + extension)
    pose_output_path = os.path.join(directory_path, file_name + "_pose" + extension)

    # Capture the video
    rtsp_str = f"rtsp://{args.usr_name}:{args.usr_pwd}@{args.rtsp_url}?videoencodec=h264&resolution={args.resolution}&fps={args.fps}&date=1&clock=1"
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Failed to open the video stream")
        return
    
    # Get video properties for saving
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    if combined_display:
        # cv.namedWindow("Combined View", cv.WINDOW_NORMAL)
        # cv.moveWindow("Combined View", 0, 0)
        combined_out = cv.VideoWriter(combined_output_path, fourcc, fps, (combined_frame_width, combined_frame_height))
        if pose_viz:
            pose_out = cv.VideoWriter(pose_output_path, fourcc, fps, (cam_width, cam_height))

    else:
        # cv.namedWindow("Camera View", cv.WINDOW_NORMAL)
        # cv.moveWindow("Camera View", 0, 0)
        # cv.namedWindow("Bird's Eye View", cv.WINDOW_NORMAL)
        # cv.moveWindow("Bird's Eye View", 423, 0)
        cam_out = cv.VideoWriter(cam_output_path, fourcc, fps, (cam_width, cam_height))
        birdseye_out = cv.VideoWriter(birdseye_output_path, fourcc, fps, (birdseye_width, birdseye_height))
        if pose_viz:
            pose_out = cv.VideoWriter(pose_output_path, fourcc, fps, (cam_width, cam_height))

    fps_ = []
    track_dic = {}
    frame_idx = 1
    # stop_frame = 180
    # stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    # cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx-1)

    
    while True:
        
        # Keep track of the progress
        with open('progress.txt', 'w') as f:
            f.write(f"{str(datetime.datetime.now())}\n")

        start = datetime.datetime.now()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read the frame")
            break
          
        # Detect people in the frame
        frame, detections = object_detector.get_pedestrains(frame, frame_idx)
        if len(detections) > 0:
            detections = projection.add_transformation(detections)
        
        # Get the pose estimation of the detected people
        pose_frame_2d, pose_frame_3d, pose_det = pose_estimator.get_pose(frame, frame_idx, detections)
        frame = pose_frame_2d

        # Update the track history and generate the 3D bounding box
        track_dic = TrackHistoryUpdate(track_dic, detections, pose_det, frame_idx)
        track_dic = BBox3D(track_dic).get_3dbox()

        # Visualize the 3D bounding box of the detected objects
        frame = Viz3Dbbox(track_dic, frame)

        # Visualize the bird's eye view of the detected objects
        birdseye_copy = birdseye.copy()
        VizBird(track_dic, birdseye_copy)

        frame_idx += 1
        
        end = datetime.datetime.now()
        fps = float(1/((end - start).total_seconds()))
        fps_ = np.append(fps_, fps)
        
        # Display and write the output frame to the video file
        if combined_display:
            combined_frame[:cam_height, :cam_width] = frame
            combined_frame[:birdseye_height, cam_width + padding:] = birdseye_copy
            cv.putText(combined_frame, f"FPS: {fps:.1f}", (50, 50),
                cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv.LINE_AA)
            # cv.imshow("Combined View", combined_frame)
            combined_out.write(combined_frame)
            if pose_viz:
                pose_out.write(pose_frame_3d)
        else:
            cv.putText(frame, f"FPS: {fps:.1f}", (50, 50),
                cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv.LINE_AA)
            # cv.imshow("Camera View", frame)
            # cv.imshow("Bird's Eye View", birdseye_copy)
            cam_out.write(frame)
            birdseye_out.write(birdseye_copy)
            if pose_viz:
                pose_out.write(pose_frame_3d)

        if cv.waitKey(1) & 0xFF in (ord("q"),27):
            break
          
                
    print(f"Average FPS: {np.mean(fps_):.2f}")

    # # save the track_dic as a json file
    # with open('track_dic.json', 'w') as outfile:
    #     json.dump(track_dic, outfile)

    # # save the full as a csv file
    # csv_file = 'track_dic.csv'
    # with open(csv_file, 'w', newline='') as csvfile:
    #     fieldnames = ['track id', 'timestamp', 'frame_count', 'pos_cam', 'pos_bird', 'bbox2D', 'bbox3D', 'theta', 'pose3D']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for id, data in track_dic.items():
    #         list_lengths = len(data['bbox3D'])
    #         # min_length = min(list_lengths)
    #         for i in range(list_lengths):
    #             writer.writerow({
    #                 'track id': id,
    #                 'timestamp': data['timestamp'][i],
    #                 'frame_count': data['frame_count'][i],
    #                 'pos_cam': data['pos_cam'][i],
    #                 'pos_bird': data['pos_bird'][i],
    #                 'bbox2D': data['bbox2D'][i],
    #                 'bbox3D': data['bbox3D'][i],
    #                 'theta': data['theta'][i],
    #                 'pose3D': data['pose3D'][i]
    #             })

    cap.release()
    if combined_display:
        combined_out.release()
        if pose_viz:
            pose_out.release()
    else:
        cam_out.release()
        birdseye_out.release()
        if pose_viz:
            pose_out.release()
    cv.destroyAllWindows()

    
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Pedestrian Tracking Program")
    parser.add_argument('--input', type=str, default='test1.mp4', help="Path to input file")
    parser.add_argument('--combined_display', action='store_true', default=False, help="Whether to combine the outputs into one window")
    parser.add_argument('--pose_viz', action='store_true', default=True, help="Whether to visualize the pose estimation")
    parser.add_argument('--usr_name', type=str, default="cosmosuser", help='User name for the progress file')
    parser.add_argument('--usr_pwd', type=str, default="cosmos101", help='Password for the progress file')
    parser.add_argument('--rtsp_url', type=str, default="cam1-md1.sb1.cosmos-lab.org/axis-media/media.amp", help='RTSP URL for the progress file')
    parser.add_argument('--resolution', type=str, default='1280x720', help='Resolution for the progress file')
    parser.add_argument('--fps', type=int, default=10, help='FPS for the progress file')
    args = parser.parse_args()
    Main(args.input, args.combined_display, args.pose_viz)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
