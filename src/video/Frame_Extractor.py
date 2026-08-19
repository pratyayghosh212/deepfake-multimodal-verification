import cv2
import json
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    interval: float = 1.0
):
    """
    Extract one frame every `interval` seconds.

    Returns metadata containing:
    - frame filename
    - timestamp
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise ValueError("Could not determine video FPS.")

    # Calculate duration
    duration = total_frames / fps

    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    metadata = []

    current_time = 0.0
    frame_number = 0

    # Extract frames
    while current_time < duration:

        # Move to the required timestamp
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000
        )

        success, frame = cap.read()

        if not success:
            break

        # Create filename
        filename = f"frame_{frame_number:05d}.jpg"
        output_path = output_dir / filename

        # Save frame
        cv2.imwrite(str(output_path), frame)

        # Store metadata
        metadata.append({
            "frame_id": frame_number,
            "filename": filename,
            "timestamp": round(current_time, 3)
        })

        print(
            f"Saved {filename} "
            f"at {current_time:.2f}s"
        )

        frame_number += 1
        current_time += interval

    # Release video
    cap.release()

    # Save metadata
    metadata_path = output_dir / "frames.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved {len(metadata)} frames.")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":

    extract_frames(
        video_path="data/videos/test.mp4",
        output_dir="data/frames",
        interval=1.0
    )