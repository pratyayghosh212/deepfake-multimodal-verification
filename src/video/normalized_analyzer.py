import json
import math
from pathlib import Path


# ---------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------

def distance(a, b):
    """2D Euclidean distance."""

    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]

    return math.sqrt(dx * dx + dy * dy)


def normalize_landmarks(landmarks):
    """
    Normalize facial landmarks.

    Step 1:
        Remove translation by centering around
        the face centroid.

    Step 2:
        Remove scale differences by dividing by
        the face size.
    """

    if not landmarks:
        return []

    # -----------------------------------------------------
    # Calculate centroid
    # -----------------------------------------------------

    center_x = sum(
        point["x"] for point in landmarks
    ) / len(landmarks)

    center_y = sum(
        point["y"] for point in landmarks
    ) / len(landmarks)

    # -----------------------------------------------------
    # Center the landmarks
    # -----------------------------------------------------

    centered = []

    for point in landmarks:

        centered.append({
            "x": point["x"] - center_x,
            "y": point["y"] - center_y,
            "z": point["z"]
        })

    # -----------------------------------------------------
    # Calculate face scale
    #
    # Use the maximum distance from the centroid.
    # -----------------------------------------------------

    scale = max(
        math.sqrt(
            point["x"] ** 2 +
            point["y"] ** 2
        )
        for point in centered
    )

    # Prevent division by zero
    if scale == 0:
        scale = 1.0

    # -----------------------------------------------------
    # Scale normalization
    # -----------------------------------------------------

    normalized = []

    for point in centered:

        normalized.append({
            "x": point["x"] / scale,
            "y": point["y"] / scale,
            "z": point["z"] / scale
        })

    return normalized


# ---------------------------------------------------------
# Compare two normalized faces
# ---------------------------------------------------------

def compare_landmarks(previous, current):
    """
    Calculate how much facial geometry changed
    between two consecutive frames.
    """

    count = min(
        len(previous),
        len(current)
    )

    if count == 0:
        return {
            "mean_deformation": 0.0,
            "std_deformation": 0.0,
            "max_deformation": 0.0
        }

    changes = []

    for i in range(count):

        dx = (
            current[i]["x"] -
            previous[i]["x"]
        )

        dy = (
            current[i]["y"] -
            previous[i]["y"]
        )

        dz = (
            current[i]["z"] -
            previous[i]["z"]
        )

        change = math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )

        changes.append(change)

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    mean_change = (
        sum(changes) /
        len(changes)
    )

    variance = sum(
        (value - mean_change) ** 2
        for value in changes
    ) / len(changes)

    std_change = math.sqrt(
        variance
    )

    max_change = max(changes)

    return {
        "mean_deformation": round(
            mean_change,
            6
        ),
        "std_deformation": round(
            std_change,
            6
        ),
        "max_deformation": round(
            max_change,
            6
        )
    }


# ---------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------

def analyze_normalized_landmarks(
    input_path,
    output_path
):

    input_path = Path(input_path)
    output_path = Path(output_path)

    # -----------------------------------------------------
    # Load landmarks
    # -----------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        frames = json.load(file)

    print(
        f"Loaded {len(frames)} frames."
    )

    results = []

    previous_normalized = None

    # -----------------------------------------------------
    # Process frames
    # -----------------------------------------------------

    for frame in frames:

        timestamp = frame["timestamp"]
        filename = frame["filename"]
        faces = frame["faces"]

        # -------------------------------------------------
        # No face
        # -------------------------------------------------

        if not faces:

            previous_normalized = None

            results.append({
                "timestamp": timestamp,
                "filename": filename,
                "face_detected": False,
                "mean_deformation": 0.0,
                "std_deformation": 0.0,
                "max_deformation": 0.0
            })

            print(
                f"{filename} | no face"
            )

            continue

        # -------------------------------------------------
        # Use first detected face
        # -------------------------------------------------

        current_landmarks = faces[0]

        # -------------------------------------------------
        # Normalize current face
        # -------------------------------------------------

        current_normalized = (
            normalize_landmarks(
                current_landmarks
            )
        )

        # -------------------------------------------------
        # First face
        # -------------------------------------------------

        if previous_normalized is None:

            result = {
                "timestamp": timestamp,
                "filename": filename,
                "face_detected": True,
                "landmark_count": len(
                    current_normalized
                ),
                "mean_deformation": 0.0,
                "std_deformation": 0.0,
                "max_deformation": 0.0
            }

            results.append(result)

            previous_normalized = (
                current_normalized
            )

            print(
                f"{filename} | baseline"
            )

            continue

        # -------------------------------------------------
        # Compare normalized geometry
        # -------------------------------------------------

        comparison = compare_landmarks(
            previous_normalized,
            current_normalized
        )

        result = {
            "timestamp": timestamp,
            "filename": filename,
            "face_detected": True,
            "landmark_count": len(
                current_normalized
            ),
            **comparison
        }

        results.append(result)

        print(
            f"{filename} | "
            f"mean deformation: "
            f"{comparison['mean_deformation']:.6f} | "
            f"max: "
            f"{comparison['max_deformation']:.6f}"
        )

        previous_normalized = (
            current_normalized
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

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
        f"Saved normalized analysis to:"
    )
    print(output_path)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    analyze_normalized_landmarks(
        input_path=(
            "data/visual_analysis/"
            "landmarks.json"
        ),
        output_path=(
            "data/visual_analysis/"
            "normalized_landmark_analysis.json"
        )
    )