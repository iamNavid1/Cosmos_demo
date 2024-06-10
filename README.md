## Installation

Developed and tested on Ubuntu 22.04 with Python 3.9.16, this repository is optimized for Debian-based Linux distros. Follow the steps below to get started:

1. Install GStreamer-1.0 and related plugins (if not already installed)
    ```sh
    sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
    ```

2. Install RTSP server (if not already installed)
    ```sh
    sudo apt-get install libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp
    ```

3. Create a new environment and activate it (preferably with Python 3.9.16)
    ```sh
    conda create -n vision python=3.9.16
    ```

4. Clone the repository:
    ```sh
    git clone https://github.com/iamNavid1/Cosmos_demo.git
    ```

5. Navigate to the project directory:
    ```sh
    cd Cosmos_demo
    ```

4. Run the setup script:
    ```sh
    ./setup.sh
    ```

This script will install the Python dependencies for this pipeline.

## Usage
1. **Preparation:**
Prepare your RTSP camera address and the credentials to connect.

2. **Run:**
Run the main stream script with the desired options:
    ```sh
    python stream.py --usr_name <rtsp_camera_user_name> \
                     --usr_pwd <rtsp_camera_password> \
                     --rtsp_url <rtsp_camera_address> \
                     [--resolution <rtsp_camera_resolution>] \
                     [--fps <rtsp_camera_fps>] \
                     [--port <port_to_stream>] \
                     [--uri <stream_uri>] \
                     [--num_instances <3d_pose_instances>]
                     [--plot_size <3d_pose_plot_size>]
    ```

   - `--usr_name`: Required username for the RTSP camera.
   - `--usr_pwd`: Required password for the RTSP camera.
   - `--rtsp_url`: Required RTSP URL of the camera.
   - `--resolution`: (Optional) Desired resolution for the stream. Default: 1920x1080
   - `--fps`: (Optional) Frames per second for the stream. Default: 4 (to ensure real-time operation)
   - `--port`: (Optional) Port to stream the output. Default: 8554
   - `--stream_uri`: (Optional) URI for the stream. Default: /video_stream
   - `--num_instances`:  (Optional) Number of 3D pose estimation instances to visualize. Default: 5
   - `--plot_size`: (Optional) Size of the 3D pose plots (square). Default: 600

   Example:
    ```sh
    python stream.py --usr_name MY_USR_NME --usr_pwd MY_PSWRD --rtsp_url CAM_ADD    
    ```

3. **Wait for the Stream URL:**
After running the script, wait a few seconds until you see the address to view the output video feed. It will look like `rtsp://<server-ip-address>:8554/<stream_uri>`.

4. **Access the Stream:**
Open your preferred media player or browser that supports RTSP streaming (e.g., VLC Media Player) and enter the provided RTSP URL to view the live video feed with 3D pose estimations.
For testing purposes, we have included a Python file named rtsp_test.py to help you verify and view the RTSP stream. To run this test script, simply execute the following command in a new terminal:
    ```sh
    python rtsp_test.py   
    ```

    
