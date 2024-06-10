import cv2
import time
import argparse

def connect_to_stream(address, max_retries=5, retry_interval=3):
    """Attempt to connect to the RTSP stream with retry logic."""
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(address)
        if cap.isOpened():
            return cap
        else:
            print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
    print(f"Failed to connect to the RTSP stream after {max_retries} attempts.")
    return None

def main(address):
    cap = connect_to_stream(address)

    if cap is None:
        print("Unable to connect to the RTSP stream. Exiting.")
        return

    cv2.namedWindow("Cosmos Demo", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Cosmos Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to read frame from the stream.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Tracking Program")
    parser.add_argument('--stream_address', required=True, type=str, help='RTSP stream address')
    args = parser.parse_args()
    main(args.stream_address)
