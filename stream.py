#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi
import cv2
import socket
import argparse
from video_processing import VideoProcessor, initial_load

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    """
    """
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        rtsp_addr = f"rtsp://{args.usr_name}:{args.usr_pwd}@{args.rtsp_url}?videoencodec=h264&resolution={args.resolution}&fps={args.fps}"
        self.cap = cv2.VideoCapture(rtsp_addr)
        self.number_frames = 0
        self.fps = args.fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds

        self.cam_width, self.cam_height = initial_load(args)
        
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             f'caps=video/x-raw,format=BGR,width={self.total_width},height={self.total_height},framerate={self.fps}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' 
        self.processor = VideoProcessor(args)

    def on_need_data(self, src, length):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = self.processor.process_frame(frame)
                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                # write float with 2 decimal points
                print(f'pushed buffer, frame {self.number_frames}, durations {float(self.duration) / Gst.SECOND:.4f} s')
                if retval != Gst.FlowReturn.OK:
                    print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(args.port))
        self.get_mount_points().add_factory(args.stream_uri, self.factory)
        self.attach(None)
        print(f"Stream Address: rtsp://{self.get_ip()}:{args.port}{args.stream_uri}")

    def get_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip


parser = argparse.ArgumentParser(description="Pedestrian Tracking Program")
parser.add_argument('--usr_name', type=str, help='User name for the rtsp stream')
parser.add_argument('--usr_pwd', type=str, help='Password for the rtsp stream')
parser.add_argument('--rtsp_url', type=str, help='RTSP URL for the progress file')
parser.add_argument('--resolution', type=str, default='1920x1080', help='Resolution for the rtsp stream')
parser.add_argument("--fps", default=4, help="fps of the rtsp stream", type=int)
parser.add_argument("--port", default=8554, help="port to stream video", type=int)
parser.add_argument("--stream_uri", default="/video_stream", help="rtsp video stream uri")
parser.add_argument('--num_instances', type=int, default=5, help='Number of instances for displaying 3d pose estimation')
parser.add_argument('--plot_size', type=int, default=600, help='Size of the plot for displaying 3d pose estimation')
args = parser.parse_args()

# initializing the threads and running the stream on loop.
GObject.threads_init()
Gst.init(None)
server = GstServer()
loop = GObject.MainLoop()
loop.run()
