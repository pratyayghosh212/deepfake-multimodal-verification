import json
from pathlib import Path


# =========================================================
# JSON utilities
# =========================================================

def load_json(file_path):
    """Load a JSON file."""

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# Speech evidence
# =========================================================

def build_speech_evidence(transcript):
    """
    Convert Whisper transcript segments
    into standardized speech evidence objects.
    """

    evidence = []

    for segment in transcript["segments"]:

        evidence_item = {
            "id": (
                f"speech_"
                f"{segment['segment_id']:04d}"
            ),

            "type": "speech",

            "start": segment["start"],
            "end": segment["end"],

            "text": segment["text"],

            "source": "whisper_transcription",

            "language": transcript[
                "language"
            ],

            "language_probability": transcript[
                "language_probability"
            ]
        }

        evidence.append(
            evidence_item
        )

    return evidence


# =========================================================
# Temporal alignment
# =========================================================

def find_speech_at_timestamp(
    timestamp,
    speech_evidence
):
    """
    Find speech segments that overlap
    a particular visual timestamp.

    A speech segment overlaps a visual frame
    when:

        start <= timestamp <= end
    """

    matched_speech = []

    for speech in speech_evidence:

        if (
            speech["start"]
            <= timestamp
            <= speech["end"]
        ):

            matched_speech.append({
                "id": speech["id"],
                "start": speech["start"],
                "end": speech["end"],
                "text": speech["text"],
                "language": speech["language"]
            })

    return matched_speech


# =========================================================
# Visual evidence
# =========================================================

def build_visual_evidence(
    face_data,
    temporal_data,
    model_data,
    speech_evidence
):
    """
    Combine all currently available visual signals.

    Sources:
        1. MediaPipe face detection
        2. Normalized landmark temporal analysis
        3. Pretrained ViT deepfake model
        4. Timestamp-aligned speech
    """

    evidence = []

    # -----------------------------------------------------
    # Lookup tables
    # -----------------------------------------------------

    temporal_lookup = {
        item["filename"]: item
        for item in temporal_data
    }

    model_lookup = {
        item["filename"]: item
        for item in model_data
    }

    # -----------------------------------------------------
    # Process every frame
    # -----------------------------------------------------

    for index, frame in enumerate(
        face_data
    ):

        filename = frame["filename"]
        timestamp = frame["timestamp"]
        faces = frame["faces"]

        # -------------------------------------------------
        # Temporal analysis
        # -------------------------------------------------

        temporal = temporal_lookup.get(
            filename,
            {}
        )

        # -------------------------------------------------
        # Visual model
        # -------------------------------------------------

        model = model_lookup.get(
            filename,
            {}
        )

        # -------------------------------------------------
        # Find speech occurring at this timestamp
        # -------------------------------------------------

        aligned_speech = (
            find_speech_at_timestamp(
                timestamp,
                speech_evidence
            )
        )

        # -------------------------------------------------
        # Build evidence item
        # -------------------------------------------------

        evidence_item = {
            "id": (
                f"visual_"
                f"{index:04d}"
            ),

            "type": "visual",

            "timestamp": timestamp,

            "frame": filename,

            "source": [
                "mediapipe_face_detection",
                "normalized_landmark_analysis",
                "vit_deepfake_detector"
            ],

            # ---------------------------------------------
            # Face information
            # ---------------------------------------------

            "face_count": len(faces),

            "faces": faces,

            # ---------------------------------------------
            # Temporal information
            # ---------------------------------------------

            "temporal": {
                "mean_deformation": temporal.get(
                    "mean_deformation",
                    0.0
                ),

                "std_deformation": temporal.get(
                    "std_deformation",
                    0.0
                ),

                "max_deformation": temporal.get(
                    "max_deformation",
                    0.0
                )
            },

            # ---------------------------------------------
            # Deepfake model information
            # ---------------------------------------------

            "deepfake_model": {
                "predicted_label": model.get(
                    "predicted_label"
                ),

                "real_probability": model.get(
                    "real_probability",
                    0.0
                ),

                "fake_probability": model.get(
                    "fake_probability",
                    0.0
                )
            },

            # ---------------------------------------------
            # Temporal multimodal alignment
            # ---------------------------------------------

            "aligned_speech": aligned_speech
        }

        evidence.append(
            evidence_item
        )

    return evidence


# =========================================================
# Save evidence
# =========================================================

def save_evidence(
    evidence,
    output_path
):
    """Save unified evidence to JSON."""

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
            evidence,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Input files
    # -----------------------------------------------------

    transcript_path = (
        "data/transcripts/"
        "transcript.json"
    )

    faces_path = (
        "data/visual_analysis/"
        "faces.json"
    )

    temporal_path = (
        "data/visual_analysis/"
        "normalized_landmark_analysis.json"
    )

    model_path = (
        "data/visual_analysis/"
        "visual_model_analysis.json"
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_path = (
        "data/evidence/"
        "evidence.json"
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

    # -----------------------------------------------------
    # Load face detection
    # -----------------------------------------------------

    print(
        "Loading face analysis..."
    )

    face_data = load_json(
        faces_path
    )

    # -----------------------------------------------------
    # Load temporal analysis
    # -----------------------------------------------------

    print(
        "Loading temporal analysis..."
    )

    temporal_data = load_json(
        temporal_path
    )

    # -----------------------------------------------------
    # Load visual model analysis
    # -----------------------------------------------------

    print(
        "Loading visual model analysis..."
    )

    model_data = load_json(
        model_path
    )

    # -----------------------------------------------------
    # Build speech evidence
    # -----------------------------------------------------

    print(
        "Building speech evidence..."
    )

    speech_evidence = (
        build_speech_evidence(
            transcript
        )
    )

    # -----------------------------------------------------
    # Build visual evidence
    # -----------------------------------------------------

    print(
        "Building visual evidence..."
    )

    visual_evidence = (
        build_visual_evidence(
            face_data,
            temporal_data,
            model_data,
            speech_evidence
        )
    )

    # -----------------------------------------------------
    # Combine evidence
    # -----------------------------------------------------

    unified_evidence = (
        speech_evidence +
        visual_evidence
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    unified_evidence.sort(
        key=lambda item:
        item.get(
            "start",
            item.get(
                "timestamp",
                0
            )
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_evidence(
        unified_evidence,
        output_path
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()

    print(
        f"Created "
        f"{len(unified_evidence)} "
        f"evidence items."
    )

    print(
        f"Speech evidence: "
        f"{len(speech_evidence)}"
    )

    print(
        f"Visual evidence: "
        f"{len(visual_evidence)}"
    )

    # Count aligned visual frames
    aligned_count = sum(
        1
        for item in visual_evidence
        if item["aligned_speech"]
    )

    print(
        f"Visual frames with "
        f"aligned speech: "
        f"{aligned_count}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )