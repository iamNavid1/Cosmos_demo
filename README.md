## Installation

1. Create a new environment and activate it (preferably with Python 3.9.16)
    ```sh
    conda create -n vision python=3.9.16
    ```

2. Clone the repository:
    ```sh
    git clone https://github.com/iamNavid1/Cosmos.git
    ```

3. Navigate to the project directory:
    ```sh
    cd Cosmos
    ```

4. Run the setup script:
    ```sh
    ./setup.sh
    ```

This script will install the Python dependencies for this pipeline.

## Usage
1. Prepare your input video file and place it in a suitable directory. There is a sample video named "test1.mp4" in the main directory.

2. Run the main script with the desired options:
    ```sh
    python main.py --input <path_to_your_video_file> [--combined_display] [--pose_viz]
    ```

   - `--input`: Specify the path to your input video file.
   - `--combined_display`: Optional flag to combine the camera and bird's eye view into one window and save it as a single video file.
   - `--pose_viz`: Optional flag to visualize the pose estimation and save it as a separate video file.

   For example:
    ```sh
    python main.py --input ./test1.mp4 --pose_viz
    ```
    or:
    ```sh
    python main.py --input ./test1.mp4 --combined_display --pose_viz
    ```


3. After running the script, you will find the output video files in the same directory as your input file, with additional suffixes indicating the type of output (e.g., `_cam`, `_birdseye`, `_combined`, `_pose`).

4. The script will also generate two additional files:
    - `track_dic.json`: A JSON file containing the tracking data.
    - `track_dic.csv`: A CSV file containing the same tracking data for easier analysis.

5. You can monitor the progress of the processing in the `progress.txt` file, which will be updated with timestamps corresponding to each processed frame.




