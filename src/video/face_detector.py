

import cv2
import json
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def detect_faces(
    frames_dir: str,
    output_path: str,
    model_path: str
):
    """
    Detect faces in extracted video frames.

    For every frame, we store:
    - filename
    - timestamp
    - face bounding box
    - detection confidence
    """

    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    model_path = Path(model_path)

    # --------------------------------------------------
    # Check required paths
    # --------------------------------------------------

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Face detector model not found: {model_path}"
        )

    # Create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Create MediaPipe Face Detector
    # --------------------------------------------------

    print("Loading face detector...")

    detector = vision.FaceDetector.create_from_model_path(
        str(model_path)
    )

    print("Face detector loaded.")

    # --------------------------------------------------
    # Load frame metadata
    # --------------------------------------------------

    metadata_path = frames_dir / "frames.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"frames.json not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        frame_metadata = json.load(f)

    # --------------------------------------------------
    # Process frames
    # --------------------------------------------------

    results = []

    print(
        f"\nProcessing {len(frame_metadata)} frames...\n"
    )

    for frame_info in frame_metadata:

        filename = frame_info["filename"]
        timestamp = frame_info["timestamp"]

        frame_path = frames_dir / filename

        # Load image
        image = cv2.imread(str(frame_path))

        if image is None:
            print(
                f"WARNING: Could not read {frame_path}"
            )
            continue

        # --------------------------------------------------
        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        # --------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # Convert NumPy image → MediaPipe Image
        # --------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # --------------------------------------------------
        # Run face detector
        # --------------------------------------------------

        detection_result = detector.detect(
            mp_image
        )

        faces = []

        # Image dimensions
        height, width, _ = image.shape

        # --------------------------------------------------
        # Extract detected faces
        # --------------------------------------------------

        for detection in detection_result.detections:

            bounding_box = detection.bounding_box

            x = bounding_box.origin_x
            y = bounding_box.origin_y
            w = bounding_box.width
            h = bounding_box.height

            # Make sure coordinates don't go outside image
            x = max(0, x)
            y = max(0, y)

            w = min(w, width - x)
            h = min(h, height - y)

            # Detection confidence
            confidence = 0.0

            if detection.categories:

                confidence = (
                    detection.categories[0].score
                )

            faces.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "confidence": round(
                    confidence,
                    4
                )
            })

        # --------------------------------------------------
        # Store result for this frame
        # --------------------------------------------------

        frame_result = {
            "filename": filename,
            "timestamp": timestamp,
            "faces": faces
        }

        results.append(frame_result)

        print(
            f"{filename} | "
            f"{timestamp:.2f}s | "
            f"faces detected: {len(faces)}"
        )

    # --------------------------------------------------
    # Release detector
    # --------------------------------------------------

    detector.close()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\n--------------------------------")
    print("Face detection completed.")
    print(f"Frames processed: {len(results)}")
    print(f"Results saved to: {output_path}")
    print("--------------------------------")


# ======================================================
# PROGRAM ENTRY POINT
# ======================================================

if __name__ == "__main__":

    detect_faces(
        frames_dir="data/frames",
        output_path="data/visual_analysis/faces.json",
        model_path="models/blaze_face_short_range.tflite"
    )