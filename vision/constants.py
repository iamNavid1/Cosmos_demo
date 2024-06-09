BirdsEyeViewPath = "./data/BirdsEye.png"            # Empty background for birdseye view
CamViewPath =  "./data/Cam.png"                     # Empty background for camera view
Cam1TSPath = "./data/Cam1Timestamps_aligned.csv"    # Timestamps for camera 1
PoseModelPath = "./data/PoseConfigs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"               # Pose estimation model
PoseLifterPath = "./data/PoseConfigs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py" # Pose estimation model
yoloWeights = "./data/YoloConfigs/yolov9e.pt"                          # YOLO v9 large size
ClassesListPath = "./data/classes.txt"         # List of COCO classes for YOLO detections
PersonIndex = 0                                     # Index of person on COCO classes list
device = "cuda"                                     # Apple Metal Performance Shader for GPU computation
conf = .5                                           # YOLO model detection confidence 
frame_mask = [[0.2979, 0.9815], [0.2490, 0.7306], [0.2146, 0.4852], [0.2042, 0.2778],       # Normalized margins to ignore the detections if outside
    [0.5266, 0.2056], [0.6083, 0.2000], [0.6479, 0.2463], [0.7010, 0.2444], 
    [0.9807, 0.4454], [0.9719, 0.8213], [0.9057, 0.9269], [0.6318, 0.9194]]
fps = 30                                           # Frames per second