import json
import math
from pathlib import Path


# =========================================================
# JSON utility
# =========================================================

def load_json(file_path):
    """
    Load JSON data from a file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# Distance calculation
# =========================================================

def distance(a, b):
    """
    Calculate 2D Euclidean distance between
    two facial landmarks.
    """

    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# =========================================================
# Mouth openness
# =========================================================

def calculate_mouth_openness(landmarks):
    """
    Estimate mouth openness using facial landmarks.

    MediaPipe Face Landmarker uses a fixed landmark
    topology. We use two points around the upper
    and lower lip and normalize the distance by
    the mouth width.

    This produces a scale-independent mouth-opening
    value.
    """

    if len(landmarks) < 14:
        return 0.0

    # -----------------------------------------------------
    # Mouth landmarks
    #
    # These indices correspond to points around
    # the inner mouth region in MediaPipe Face Mesh.
    # -----------------------------------------------------

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # -----------------------------------------------------
    # Vertical mouth opening
    # -----------------------------------------------------

    vertical_distance = distance(
        upper_lip,
        lower_lip
    )

    # -----------------------------------------------------
    # Horizontal mouth width
    # -----------------------------------------------------

    horizontal_distance = distance(
        left_mouth,
        right_mouth
    )

    # Prevent division by zero
    if horizontal_distance == 0:
        return 0.0

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    openness = (
        vertical_distance /
        horizontal_distance
    )

    return openness


# =========================================================
# Extract mouth movement
# =========================================================

def build_mouth_features(
    landmark_data
):
    """
    Convert facial landmark data into
    timestamped mouth movement features.
    """

    features = []

    previous_openness = None

    for frame in landmark_data:

        timestamp = frame["timestamp"]
        filename = frame["filename"]
        faces = frame["faces"]

        # -------------------------------------------------
        # No face
        # -------------------------------------------------

        if not faces:

            features.append({
                "timestamp": timestamp,
                "frame": filename,
                "face_detected": False,
                "mouth_openness": 0.0,
                "mouth_movement": 0.0
            })

            previous_openness = None

            continue

        # -------------------------------------------------
        # First detected face
        # -------------------------------------------------

        landmarks = faces[0]

        # -------------------------------------------------
        # Calculate mouth openness
        # -------------------------------------------------

        openness = calculate_mouth_openness(
            landmarks
        )

        # -------------------------------------------------
        # Calculate frame-to-frame movement
        # -----------------------------------------------------

        if previous_openness is None:

            movement = 0.0

        else:

            movement = abs(
                openness -
                previous_openness
            )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        features.append({
            "timestamp": timestamp,
            "frame": filename,
            "face_detected": True,
            "mouth_openness": round(
                openness,
                6
            ),
            "mouth_movement": round(
                movement,
                6
            )
        })

        previous_openness = openness

    return features


# =========================================================
# Speech activity
# =========================================================

def is_speech_active(
    timestamp,
    transcript
):
    """
    Determine whether Whisper detected speech
    at a particular timestamp.
    """

    for segment in transcript["segments"]:

        if (
            segment["start"]
            <= timestamp
            <= segment["end"]
        ):

            return True

    return False


# =========================================================
# Align mouth movement with speech
# =========================================================

def build_lipsync_analysis(
    mouth_features,
    transcript
):
    """
    Compare mouth movement with speech activity.

    The goal is not to claim that a video is fake.

    Instead, we produce evidence describing whether
    mouth movement and speech occur together.
    """

    results = []

    for item in mouth_features:

        timestamp = item["timestamp"]

        speech_active = is_speech_active(
            timestamp,
            transcript
        )

        mouth_movement = item[
            "mouth_movement"
        ]

        # -------------------------------------------------
        # Classify local synchronization
        # -------------------------------------------------

        if not item["face_detected"]:

            status = "no_face"

        elif not speech_active:

            if mouth_movement > 0.015:

                status = "mouth_movement_without_speech"

            else:

                status = "no_speech"

        else:

            if mouth_movement > 0.005:

                status = "speech_with_mouth_movement"

            else:

                status = "speech_without_mouth_movement"

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        results.append({
            "timestamp": timestamp,

            "frame": item["frame"],

            "face_detected": item[
                "face_detected"
            ],

            "mouth_openness": item[
                "mouth_openness"
            ],

            "mouth_movement": mouth_movement,

            "speech_active": speech_active,

            "sync_status": status
        })

    return results


# =========================================================
# Summary
# =========================================================

def calculate_summary(results):
    """
    Calculate simple statistics for the
    entire video.
    """

    speech_with_movement = 0
    speech_without_movement = 0
    movement_without_speech = 0

    total_speech_frames = 0

    for item in results:

        if item["speech_active"]:

            total_speech_frames += 1

            if (
                item["mouth_movement"]
                > 0.005
            ):

                speech_with_movement += 1

            else:

                speech_without_movement += 1

        else:

            if (
                item["mouth_movement"]
                > 0.015
            ):

                movement_without_speech += 1

    if total_speech_frames > 0:

        synchronization_ratio = (
            speech_with_movement /
            total_speech_frames
        )

    else:

        synchronization_ratio = 0.0

    return {
        "total_frames": len(results),

        "speech_frames": total_speech_frames,

        "speech_with_mouth_movement":
            speech_with_movement,

        "speech_without_mouth_movement":
            speech_without_movement,

        "mouth_movement_without_speech":
            movement_without_speech,

        "synchronization_ratio": round(
            synchronization_ratio,
            4
        )
    }


# =========================================================
# Save JSON
# =========================================================

def save_json(
    data,
    output_path
):
    """
    Save data to JSON.
    """

    output_path = Path(
        output_path
    )

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
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    landmarks_path = (
        "data/visual_analysis/"
        "landmarks.json"
    )

    transcript_path = (
        "data/transcripts/"
        "transcript.json"
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_path = (
        "data/visual_analysis/"
        "lipsync_analysis.json"
    )

    # -----------------------------------------------------
    # Load landmarks
    # -----------------------------------------------------

    print(
        "Loading facial landmarks..."
    )

    landmark_data = load_json(
        landmarks_path
    )

    print(
        f"Frames loaded: "
        f"{len(landmark_data)}"
    )

    # -----------------------------------------------------
    # Load transcript
    # -----------------------------------------------------

    print(
        "Loading transcript..."
    )

    transcript = load_json(
        transcript_path
    )

    print(
        f"Speech segments: "
        f"{len(transcript['segments'])}"
    )

    # -----------------------------------------------------
    # Mouth features
    # -----------------------------------------------------

    print()
    print(
        "Extracting mouth movement..."
    )

    mouth_features = build_mouth_features(
        landmark_data
    )

    # -----------------------------------------------------
    # Lip-sync analysis
    # -----------------------------------------------------

    print(
        "Analyzing speech-mouth synchronization..."
    )

    results = build_lipsync_analysis(
        mouth_features,
        transcript
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = calculate_summary(
        results
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    output = {
        "summary": summary,
        "frames": results
    }

    save_json(
        output,
        output_path
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print()
    print(
        "--------------------------------"
    )

    print(
        "Lip-sync analysis completed."
    )

    print(
        f"Total frames: "
        f"{summary['total_frames']}"
    )

    print(
        f"Speech frames: "
        f"{summary['speech_frames']}"
    )

    print(
        f"Speech + mouth movement: "
        f"{summary['speech_with_mouth_movement']}"
    )

    print(
        f"Speech without mouth movement: "
        f"{summary['speech_without_mouth_movement']}"
    )

    print(
        f"Mouth movement without speech: "
        f"{summary['mouth_movement_without_speech']}"
    )

    print(
        f"Synchronization ratio: "
        f"{summary['synchronization_ratio']}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print(
        "--------------------------------"
    )