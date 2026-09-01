import json
from pathlib import Path


# =========================================================
# JSON utilities
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
# Audio alignment
# =========================================================

def find_audio_evidence(
    timestamp,
    audio_data
):
    """
    Find audio windows that contain
    the given video timestamp.
    """

    matches = []

    for window in audio_data:

        start = window["start"]
        end = window["end"]

        if start <= timestamp <= end:

            matches.append({
                "start": start,
                "end": end,
                "rms": window.get(
                    "rms",
                    0.0
                ),
                "zcr": window.get(
                    "zcr",
                    0.0
                ),
                "spectral_centroid": window.get(
                    "spectral_centroid",
                    0.0
                )
            })

    return matches


# =========================================================
# Speech alignment
# =========================================================

def find_speech_evidence(
    timestamp,
    transcript
):
    """
    Find speech segments that contain
    the given video timestamp.
    """

    matches = []

    for segment in transcript["segments"]:

        start = segment["start"]
        end = segment["end"]

        if start <= timestamp <= end:

            matches.append({
                "segment_id": segment["segment_id"],
                "start": start,
                "end": end,
                "text": segment["text"]
            })

    return matches


# =========================================================
# Temporal multimodal alignment
# =========================================================

def build_temporal_alignment(
    visual_data,
    audio_data,
    transcript
):
    """
    Align visual, audio and speech evidence
    using timestamps.

    Each visual frame becomes one aligned
    multimodal evidence item.
    """

    aligned = []

    for frame in visual_data:

        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        timestamp = frame["timestamp"]

        # -------------------------------------------------
        # Frame filename
        #
        # Standardized evidence uses "frame".
        # Raw face data uses "filename".
        #
        # Supporting both makes this function robust.
        # -------------------------------------------------

        filename = frame.get(
            "frame",
            frame.get("filename")
        )

        # -------------------------------------------------
        # Find matching audio
        # -------------------------------------------------

        audio_matches = find_audio_evidence(
            timestamp,
            audio_data
        )

        # -------------------------------------------------
        # Find matching speech
        # -------------------------------------------------

        speech_matches = find_speech_evidence(
            timestamp,
            transcript
        )

        # -------------------------------------------------
        # Build aligned evidence
        # -------------------------------------------------

        aligned_item = {
            "id": (
                f"alignment_"
                f"{len(aligned):04d}"
            ),

            "timestamp": timestamp,

            "frame": filename,

            # ---------------------------------------------
            # Visual evidence
            # ---------------------------------------------

            "visual": {
                "face_count": frame.get(
                    "face_count",
                    0
                ),

                "temporal": frame.get(
                    "temporal",
                    {}
                ),

                "deepfake_model": frame.get(
                    "deepfake_model",
                    {}
                )
            },

            # ---------------------------------------------
            # Audio evidence
            # ---------------------------------------------

            "audio": audio_matches,

            # ---------------------------------------------
            # Speech evidence
            # ---------------------------------------------

            "speech": speech_matches
        }

        aligned.append(
            aligned_item
        )

    return aligned


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
    # Input files
    # -----------------------------------------------------

    visual_path = (
        "data/evidence/"
        "evidence.json"
    )

    audio_path = (
        "data/audio/"
        "audio_analysis.json"
    )

    transcript_path = (
        "data/transcripts/"
        "transcript.json"
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_path = (
        "data/evidence/"
        "temporal_alignment.json"
    )

    # -----------------------------------------------------
    # Load visual evidence
    # -----------------------------------------------------

    print(
        "Loading visual evidence..."
    )

    evidence = load_json(
        visual_path
    )

    visual_data = [
        item
        for item in evidence
        if item.get("type") == "visual"
    ]

    print(
        f"Visual evidence items: "
        f"{len(visual_data)}"
    )

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    print(
        "Loading audio analysis..."
    )

    audio_data = load_json(
        audio_path
    )

    print(
        f"Audio windows: "
        f"{len(audio_data)}"
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
    # Build alignment
    # -----------------------------------------------------

    print()
    print(
        "Building temporal multimodal alignment..."
    )

    aligned_data = build_temporal_alignment(
        visual_data,
        audio_data,
        transcript
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_json(
        aligned_data,
        output_path
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print(
        "--------------------------------"
    )

    print(
        "Temporal alignment completed."
    )

    print(
        f"Aligned frames: "
        f"{len(aligned_data)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print(
        "--------------------------------"
    )