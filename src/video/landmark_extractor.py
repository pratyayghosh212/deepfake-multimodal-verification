import json
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_landmarks(
    frames_dir,
    output_path,
    model_path
):
    """
    Extract facial landmarks from timestamped frames.

    Timestamps are read from frames.json rather than
    being calculated from the frame index.
    """

    frames_dir = Path(frames_dir)

    # --------------------------------------------------
    # Load frame metadata
    # --------------------------------------------------

    metadata_path = frames_dir / "frames.json"

    if not metadata_path.exists():
        print(
            f"Could not find frame metadata: "
            f"{metadata_path}"
        )
        return

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        frame_metadata = json.load(file)

    # Create filename -> timestamp mapping
    timestamp_map = {
        item["filename"]: item["timestamp"]
        for item in frame_metadata
    }

    print(
        f"Loaded timestamp metadata for "
        f"{len(timestamp_map)} frames."
    )

    # --------------------------------------------------
    # Find frames
    # --------------------------------------------------

    frame_files = sorted(
        frames_dir.glob("*.jpg")
    )

    if not frame_files:
        print("No frames found.")
        return

    # --------------------------------------------------
    # MediaPipe model configuration
    # --------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Loading Face Landmarker...")

    detector = vision.FaceLandmarker.create_from_options(
        options
    )

    results = []

    # --------------------------------------------------
    # Process frames
    # --------------------------------------------------

    for frame_path in frame_files:

        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            print(
                f"Could not read {frame_path}"
            )
            continue

        # OpenCV: BGR -> RGB
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        detection_result = detector.detect(
            mp_image
        )

        # --------------------------------------------------
        # Get REAL timestamp from frames.json
        # --------------------------------------------------

        if frame_path.name not in timestamp_map:

            print(
                f"Warning: no timestamp found for "
                f"{frame_path.name}"
            )

            continue

        timestamp = timestamp_map[
            frame_path.name
        ]

        frame_result = {
            "filename": frame_path.name,
            "timestamp": timestamp,
            "faces": []
        }

        # --------------------------------------------------
        # Face landmarks
        # --------------------------------------------------

        if detection_result.face_landmarks:

            for face_landmarks in (
                detection_result.face_landmarks
            ):

                landmarks = []

                for landmark in face_landmarks:

                    landmarks.append({
                        "x": round(
                            landmark.x,
                            6
                        ),
                        "y": round(
                            landmark.y,
                            6
                        ),
                        "z": round(
                            landmark.z,
                            6
                        )
                    })

                frame_result["faces"].append(
                    landmarks
                )

        results.append(
            frame_result
        )

        print(
            f"{frame_path.name} | "
            f"{timestamp:.3f}s | "
            f"faces: "
            f"{len(frame_result['faces'])}"
        )

    detector.close()

    # --------------------------------------------------
    # Save landmarks
    # --------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        f"Saved landmarks to: {output_path}"
    )


if __name__ == "__main__":

    extract_landmarks(
        frames_dir="data/frames",
        output_path=(
            "data/visual_analysis/"
            "landmarks.json"
        ),
        model_path=(
            "models/face_landmarker.task"
        )
    )