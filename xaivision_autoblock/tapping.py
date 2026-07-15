import cv2
import time
import argparse
import os

def main(rtsp_url, output_file, target_fps):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Get frame dimensions and FPS from the RTSP stream
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("Warning: Could not get FPS from the stream. Defaulting to 20 FPS.")
        original_fps = 20.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame skip interval
    frame_interval = int(original_fps / target_fps)

    # VideoWriter object
    writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'avc1'
    out = cv2.VideoWriter(output_file, writer_fourcc, target_fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Failed to open VideoWriter for file: {output_file}")
        cap.release()
        return

    # Capture and save frames
    print("Starting to capture frames...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from RTSP stream.")
            break

        # Write the frame to the output file if within the frame interval
        if frame_count % frame_interval == 0:
            out.write(frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    print("Finished capturing frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture RTSP stream and save to MP4 file')
    parser.add_argument('rtsp_url', type=str, help='RTSP stream URL')
    parser.add_argument('output_file', type=str, help='Output MP4 file')
    parser.add_argument('target_fps', type=float, help='Target FPS for output video')

    args = parser.parse_args()
    main(args.rtsp_url, args.output_file, args.target_fps)

