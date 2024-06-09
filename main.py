import numpy as np
import os
import cv2 as cv
import datetime
import argparse
import datetime
from vision.detectors import ObjectDetection, PoseEstimation
from vision.loader import ResourceLoader
from vision.bbox3d import BBox3D
from vision.util import TrackHistoryUpdate, VizBird, Viz3Dbbox
from projection.coordTrans import Cam2Bird


# Main function to stream the video, detect people, track on birdseye view, and save the output 
def Main(args):

    # Import resources
    loader = ResourceLoader()
    detection_model, classes_list = loader.load_yolo()        # YOLO model and its classes 
    tracker = loader.load_tracker()                           # Tracker model
    pose_model, pose_lifter, visualizer = loader.load_pose()  # Pose estimation model, its lifter, and visualizer
    birdseye = loader.load_background()      # Background images at camera view and birdseye view

    # Initialize the object detector, pose estimator, and projection
    object_detector = ObjectDetection(detection_model, classes_list, tracker, args.resolution)
    pose_estimator = PoseEstimation(pose_model, pose_lifter, visualizer, args.num_instances, args.plot_size, args.pose_viz)
    projection = Cam2Bird()

    # Frame properties for displaying and saving
    cam_width, cam_height = map(int, args.resolution.split('x'))
    birdseye_height, birdseye_width = birdseye.shape[:2]
    total_width = cam_width + birdseye_width
    total_height = max(cam_height, birdseye_height) + args.plot_size
    if args.num_instances * args.plot_size > total_width:
        args.num_instances = total_width // args.plot_size
    combined_frame = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    # A list to Maintain the order of pose3d plots
    # plot_order = [None] * args.num_instances
    plot_order = [0] * args.num_instances
    
    # Create output path to save
    directory_path = os.path.dirname(args.input)
    file_name, extension = os.path.splitext(os.path.basename(args.input))
    output_path = os.path.join(directory_path, file_name + "_output" + extension)

    # Capture the video
    rtsp_str = f"rtsp://{args.usr_name}:{args.usr_pwd}@{args.rtsp_url}?videoencodec=h264&resolution={args.resolution}&fps={args.fps}&date=1&clock=1"
    cap = cv.VideoCapture(args.input)
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
        frame, detections, confirmed = object_detector.get_pedestrains(frame, frame_idx)
        if len(detections) > 0:
            detections = projection.add_transformation(detections)
        
        # Get the pose estimation of the detected people
        pose2d_frame, pose3d_dic, pose_det = pose_estimator.get_pose(frame, frame_idx, detections, confirmed)
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
        combined_frame.fill(255)
        combined_frame[:cam_height, :cam_width] = frame
        combined_frame[:birdseye_height, cam_width:] = birdseye_copy
        y_offset = max(cam_height, birdseye_height)

        availability = [True] * args.num_instances
        included = []
        for i, track_id in enumerate(plot_order):
            if track_id in pose3d_dic:
                combined_frame[y_offset:, i*args.plot_size:(i+1)*args.plot_size] = pose3d_dic[track_id]
                included.append(track_id)
                availability[i] = False

        for i, free in enumerate(availability):
            if free:
                for key in pose3d_dic:
                    if key not in included:
                        combined_frame[y_offset:, i*args.plot_size:(i+1)*args.plot_size] = pose3d_dic[key]
                        included.append(key)
                        plot_order[i] = key if key > 0 else plot_order[i]
                        break

        # cv.putText(combined_frame, f"FPS: {fps:.1f}", (50, 50),
        #     cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv.LINE_AA)
        # cv.imshow("Combined View", combined_frame)
        out.write(combined_frame)

        cv.imwrite(f"frame.jpg", combined_frame)

        if cv.waitKey(1) & 0xFF in (ord("q"),27):
            break
          
    # print(f"Average FPS: {np.mean(fps_):.2f}")

    cap.release()
    out.release()
    cv.destroyAllWindows()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Tracking Program")
    parser.add_argument('--input', type=str, default='test2.mp4', help="Path to input file")
    parser.add_argument('--pose_viz', action='store_true', default=True, help="Whether to visualize the pose estimation")
    parser.add_argument('--usr_name', type=str, default="cosmosuser", help='User name for the progress file')
    parser.add_argument('--usr_pwd', type=str, default="cosmos101", help='Password for the progress file')
    parser.add_argument('--rtsp_url', type=str, default="cam1-md1.sb1.cosmos-lab.org/axis-media/media.amp", help='RTSP URL for the progress file')
    parser.add_argument('--resolution', type=str, default='1920x1080', help='Resolution for the progress file')
    parser.add_argument('--fps', type=int, default=10, help='FPS for the progress file')
    parser.add_argument('--num_instances', type=int, default=5, help='Number of instances for displaying 3d pose estimation')
    parser.add_argument('--plot_size', type=int, default=600, help='Size of the plot for displaying 3d pose estimation')
    args = parser.parse_args()
    Main(args)
