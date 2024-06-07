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
def Main(args, input_path, pose_viz):

    # Import resources
    loader = ResourceLoader()
    detection_model, classes_list = loader.load_yolo()        # YOLO model and its classes 
    tracker = loader.load_tracker()                           # Tracker model
    pose_model, pose_lifter, visualizer = loader.load_pose()  # Pose estimation model, its lifter, and visualizer
    birdseye = loader.load_background()      # Background images at camera view and birdseye view

    object_detector = ObjectDetection(detection_model, classes_list, tracker)
    pose_estimator = PoseEstimation(pose_model, pose_lifter, visualizer, args.num_instances, args.plot_size, pose_viz)
    projection = Cam2Bird()

    # Frame properties for displaying and saving
    cam_width, cam_height = map(int, args.resolution.split('x'))
    birdseye_height, birdseye_width = birdseye.shape[:2]
    total_width = cam_width + birdseye_width
    total_height = max(cam_height, birdseye_height) + args.plot_size
    if args.num_instances * args.plot_size > total_width:
        args.num_instances = total_width // args.plot_size
    combined_frame = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Create output path to save
    directory_path = os.path.dirname(input_path)
    file_name, extension = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(directory_path, file_name + "_output" + extension)

    # Capture the video
    rtsp_str = f"rtsp://{args.usr_name}:{args.usr_pwd}@{args.rtsp_url}?videoencodec=h264&resolution={args.resolution}&fps={args.fps}&date=1&clock=1"
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Failed to open the video stream")
        return
    
    # Get video properties for saving
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter(output_path, fourcc, fps, (total_width, total_height))

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
        pose2d_frame, pose3d_list, pose_det = pose_estimator.get_pose(frame, frame_idx, detections)
        frame = pose2d_frame

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
        combined_frame[:cam_height, :cam_width] = frame
        combined_frame[:birdseye_height, cam_width:] = birdseye_copy
        y_offset = max(cam_height, birdseye_height)
        for i, pose in enumerate(pose3d_list):
            x_offset = i * args.plot_size
            combined_frame[y_offset:, x_offset:x_offset+args.plot_size] = pose

        cv.putText(combined_frame, f"FPS: {fps:.1f}", (50, 50),
            cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv.LINE_AA)
        # cv.imshow("Combined View", combined_frame)
        out.write(combined_frame)


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
    out.release()
    cv.destroyAllWindows()

    
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Pedestrian Tracking Program")
    parser.add_argument('--input', type=str, default='test1.mp4', help="Path to input file")
    parser.add_argument('--pose_viz', action='store_true', default=True, help="Whether to visualize the pose estimation")
    parser.add_argument('--usr_name', type=str, default="cosmosuser", help='User name for the progress file')
    parser.add_argument('--usr_pwd', type=str, default="cosmos101", help='Password for the progress file')
    parser.add_argument('--rtsp_url', type=str, default="cam1-md1.sb1.cosmos-lab.org/axis-media/media.amp", help='RTSP URL for the progress file')
    parser.add_argument('--resolution', type=str, default='1280x720', help='Resolution for the progress file')
    parser.add_argument('--fps', type=int, default=10, help='FPS for the progress file')
    parser.add_argument('--num_instances', type=int, default=5, help='Number of instances for displaying 3d pose estimation')
    parser.add_argument('--plot_size', type=int, default=300, help='Size of the plot for displaying 3d pose estimation')
    args = parser.parse_args()
    Main(args, args.input, args.pose_viz)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
