from collections import deque
import cv2 as cv
import colorsys
from vision.bbox3d import BBox3D
import numpy as np


def TrackHistoryUpdate(track_dic, detections, pose_detections):
    """
    Update the tracker history with new detections.
    """
    timestamp = []
    frame_count = []
    ids = []
    pos_cam = []
    pos_bird = []
    bbox2D = []
    pose3D = []

    for det, pose_det in zip(detections, pose_detections):
        timestamp.append(float(det[0]))
        frame_count.append(int(det[1]))
        ids.append(int(det[2]))
        bbox2D.append([int(x) for x in det[3:7]])
        pos_cam.append([int(x) for x in det[7:9]])
        pos_bird.append([int(x) for x in det[9:11]])
        pose3D.append(pose_det.tolist())


    if len(track_dic) != 0:
        for key in track_dic:
            if key in ids:
                track_dic[key]['timestamp'].append(timestamp[ids.index(key)])
                track_dic[key]['frame_count'].append(frame_count[ids.index(key)])
                track_dic[key]['pos_cam'].append(pos_cam[ids.index(key)])
                track_dic[key]['pos_bird'].append(pos_bird[ids.index(key)])
                track_dic[key]['bbox2D'].append(bbox2D[ids.index(key)])
                track_dic[key]['pose3D'].append(pose3D[ids.index(key)])
            elif track_dic[key]['timestamp'][-1] is not None:
                track_dic[key]['timestamp'].append(None)
                track_dic[key]['frame_count'].append(None)
                track_dic[key]['pos_cam'].append(None)
                track_dic[key]['pos_bird'].append(None)
                track_dic[key]['bbox2D'].append(None)
                track_dic[key]['pose3D'].append(None)

    for id in ids:
        if id not in track_dic:
            track_dic[id] = {
                'timestamp': [],
                'frame_count': [],
                'pos_cam': [],
                'pos_bird': [],
                'bbox2D': [],
                'pose3D': []
            }
            track_dic[id]['timestamp'].append(timestamp[ids.index(id)])
            track_dic[id]['frame_count'].append(frame_count[ids.index(id)])
            track_dic[id]['pos_cam'].append(pos_cam[ids.index(id)])
            track_dic[id]['pos_bird'].append(pos_bird[ids.index(id)])
            track_dic[id]['bbox2D'].append(bbox2D[ids.index(id)])
            track_dic[id]['pose3D'].append(pose3D[ids.index(id)])

    # track_dic = FullBBox3D(track_dic).get_3dbox()
    return track_dic


def delete_track(track_dic):
    """
    Delete tracks from the tracker history whose readings are None for 60 frame
    """
    # delete the track whose pos_cam is None for 90 frames
    track_dic_ = track_dic.copy()
    for key in track_dic:
        if track_dic[key]['pos_cam'].count(None) == 90:
            del track_dic_[key]
    
    return track_dic_


def get_color(number):
    " Converts an integer number to a color "
    # change these however you want to
    hue = number*30 % 180
    saturation = number*103 % 256
    value = number*50% 256
    
    # expects normalized values
    color = colorsys.hsv_to_rgb (hue/179, saturation/255, value/255)
    return [int(c*255) for c in color]


def VizBird(track_dic, frame):
    """
    Visualize the bird's eye view of the detected objects.
    """
    for key in track_dic:
        if track_dic[key]['pos_bird'][-1] is not None:
            points = track_dic[key]['pos_bird'][-180:]
            filtered_points = [pt for pt in points if pt is not None]
            color = get_color(key*50)
            for i in range(len(filtered_points) - 1):
                # cv.line(birdseye_copy, [int(filtered_points[i][0]), int(filtered_points[i][1])], [int(filtered_points[i+1][0]), int(filtered_points[i+1][1])], (255, 0, 0), 1)
                cv.line(frame, [filtered_points[i][0], filtered_points[i][1]], [int(filtered_points[i+1][0]), int(filtered_points[i+1][1])], color, 2, cv.LINE_AA)
                cv.circle(frame, [track_dic[key]['pos_bird'][-1][0], track_dic[key]['pos_bird'][-1][1]], 6, color, -1, cv.LINE_AA)
                cv.circle(frame, [track_dic[key]['pos_bird'][-1][0], track_dic[key]['pos_bird'][-1][1]], 6, (255, 255, 255), 1, cv.LINE_AA)


def Viz3Dbbox(track_dic, frame):
    """
    Visualize the 3D bounding box of the detected objects.
    """
    new_frame = frame.copy()
    for key in track_dic:
        vertices = track_dic[key]['bbox3D'][-1]
        if vertices is not None:
            color = get_color(key*50)
            overlay = new_frame.copy()
            cv.fillPoly(overlay, [np.array(vertices[:4]).reshape((-1, 1, 2))], color, cv.LINE_AA)
            cv.fillPoly(overlay, [np.array(vertices[4:]).reshape((-1, 1, 2))], color, cv.LINE_AA)
            for i in range(4):
                cv.fillPoly(overlay, [np.array([vertices[i], vertices[(i+1)%4], vertices[(i+1)%4+4], vertices[i+4]]).reshape((-1, 1, 2))], color, cv.LINE_AA)
            new_frame = cv.addWeighted(overlay, 0.55, new_frame, 0.45, 0)

            #draw two closed polygones using the first and second 4 points of the vertices
            # cv.circle(new_frame, vertices[0], 6, color, -1)
            cv.polylines(new_frame, [np.array(vertices[:4]).reshape((-1, 1, 2))], True, color, 1, cv.LINE_AA)
            cv.polylines(new_frame, [np.array(vertices[4:]).reshape((-1, 1, 2))], True, color, 1, cv.LINE_AA)
            #draw lines between the corresponding points of the two polygons
            for i in range(4):
                cv.line(new_frame, vertices[i], vertices[i+4], color, 1, cv.LINE_AA)
            cv.line(new_frame, vertices[0], vertices[7], color, 1, cv.LINE_AA)
            cv.line(new_frame, vertices[3], vertices[4], color, 1, cv.LINE_AA)
            # cv.putText(frame, f"{key}", (int((track_dic[key]['bbox2D'][0][0]+track_dic[key]['bbox2D'][0][2])/2+.5), track_dic[key]['bbox2D'][0][1]-12), cv.FONT_HERSHEY_PLAIN, 1.25, color, 2, cv.LINE_AA)
    return new_frame
