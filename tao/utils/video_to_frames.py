#!/usr/bin/env python3
import os
import cv2
import argparse

def extract_frames(video_path, output_dir, interval_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Cannot read FPS from video")
    frame_interval = int(fps * interval_sec)

    os.makedirs(output_dir, exist_ok=True)
    count = saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            fname = os.path.join(output_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames to '{output_dir}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract frames from video at regular intervals"
    )
    p.add_argument("video", help="Path to input video file")
    p.add_argument(
        "--output_dir",
        default="images",
        help="Directory to save extracted frames",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Time interval between frames (in seconds)",
    )
    args = p.parse_args()
    extract_frames(args.video, args.output_dir, args.interval)
